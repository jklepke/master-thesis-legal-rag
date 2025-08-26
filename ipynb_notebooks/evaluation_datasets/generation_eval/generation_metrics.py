import os
import json
import random
import logging
from pathlib import Path
from uuid import uuid4
import pandas as pd
import openai
from openai import OpenAI
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datasets import Dataset
import time
from collections import defaultdict

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Langchain Imports
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI

# Ragas Imports
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample
from ragas.metrics import (
    answer_correctness,
    ResponseRelevancy,
    Faithfulness,
)

from bert_score import score as bertscore

# == CONFIG ==

load_dotenv()
ragas_token = os.getenv("RAGAS_APP_TOKEN")
os.environ["RAGAS_APP_TOKEN"] = ragas_token
openai.api_key = os.environ['OPENAI_API_KEY']

# LLM Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
client = OpenAI(api_key=openai.api_key)

# Embedding Setup
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# Response Relevancy Config
response_relevancy_scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=embedding_model)

# Faithfulness Config
faithfulness_scorer = Faithfulness(llm=evaluator_llm)


# == CONFIG - END ==

def calculate_bleu_score(generated_response: str, ground_truth: str) -> float:
    try:
        generated_response_tokens = nltk.word_tokenize(generated_response)
        ground_truth_tokens = nltk.word_tokenize(ground_truth)

        smoothie = SmoothingFunction().method4
        return sentence_bleu([ground_truth_tokens], generated_response_tokens, smoothing_function=smoothie)
    
    except Exception as e:
        print(f"[BLEU ERROR] Could not calculate BLEU: {e}")
        return 0.0

def calculate_mean_bleu_score(eval_dataset, group_key: str = "question_context"):
    print(f"→ Calculation: Mean BLEU Score")

    bleu_scores = []
    grouped_scores = defaultdict(list)

    for entry in tqdm(eval_dataset):
        generated = entry["generated_response"]
        reference = entry["ground_truth"]
        score = calculate_bleu_score(generated, reference)
        bleu_scores.append(score)

        group = entry.get(group_key, "undefined")
        grouped_scores[group].append(score)

    # global mean
    mean_bleu = round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else 0.0

    # grouped means
    mean_group_bleu = {
        group: round(sum(scores) / len(scores), 4) if scores else 0.0
        for group, scores in grouped_scores.items()
    }

    return mean_bleu, mean_group_bleu

def calculate_rouge_scores(entry):
    try:
        ground_truth_tokens = word_tokenize(entry["ground_truth"])
        generated_response_tokens = word_tokenize(entry["generated_response"])

        ground_truth_text = " ".join(ground_truth_tokens)
        generated_response_text = " ".join(generated_response_tokens)
        
        r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = r_scorer.score(ground_truth_text, generated_response_text)
        return scores
    
    except Exception as e:
        print(f"[ROUGE ERROR] Could not calculate ROUGE: {e}")
        return {
            'rouge1': rouge_scorer.Score(0.0, 0.0, 0.0),
            'rouge2': rouge_scorer.Score(0.0, 0.0, 0.0),
            'rougeL': rouge_scorer.Score(0.0, 0.0, 0.0),
        } 
    

def calculate_mean_rouge_scores(eval_dataset, group_key="question_context"):
    print(f"→ Calculation: Mean ROUGE Scores (Rouge1, Rouge2, RougeL)")
    rouge_totals = defaultdict(list)
    grouped_scores = defaultdict(lambda: defaultdict(list))

    for entry in tqdm(eval_dataset):
        scores = calculate_rouge_scores(entry)
        group = entry.get(group_key, "undefined")

        for key in ['rouge1', 'rouge2', 'rougeL']:
            val = scores[key].fmeasure
            rouge_totals[key].append(val)
            grouped_scores[group][key].append(val)

    def mean(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0

    mean_rouge = {f"mean_{k}": mean(v) for k, v in rouge_totals.items()}
    grouped_rouge = {
        group: {f"mean_{k}": mean(v) for k, v in score_dict.items()}
        for group, score_dict in grouped_scores.items()
    }

    return mean_rouge, grouped_rouge
    
def calculate_meteor_scores(generated_response: str, ground_truth: str):
    try:
        ground_truth_tokens = nltk.word_tokenize(ground_truth)
        generated_response_tokens = nltk.word_tokenize(generated_response)
        return meteor_score([ground_truth_tokens], generated_response_tokens)
    
    except Exception as e:
        print(f"[METEOR ERROR] Could not calculate METEOR: {e}")
        return 0.0

def calculate_mean_meteor_score(eval_dataset, group_key="question_context"):
    print("→ Calculation: Mean METEOR Score")
    
    meteor_scores = []
    grouped_scores = defaultdict(list)
    
    for entry in tqdm(eval_dataset):
        generated_response = entry["generated_response"]
        ground_truth = entry["ground_truth"]
        meteor = calculate_meteor_scores(generated_response, ground_truth)
        meteor_scores.append(meteor)
        
        # Fallback group if key missing
        group = entry.get(group_key, "undefined")
        grouped_scores[group].append(meteor)
    
    def mean(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0

    mean_meteor = mean(meteor_scores)
    
    grouped_meteor = {
        group: mean(score_list) for group, score_list in grouped_scores.items()
    }

    return mean_meteor, grouped_meteor


def calculate_mean_bert_score(eval_dataset, group_key="question_context", lang="de", model_type="bert-base-multilingual-cased"):
    print("→ Calculation: BERTScore (Precision, Recall, F1)")

    try:
        refs = [e["ground_truth"] for e in eval_dataset]
        gens = [e["generated_response"] for e in eval_dataset]

        P, R, F1 = bertscore(gens, refs, lang=lang, model_type=model_type, rescale_with_baseline=False)

        mean_bert = {
            "mean_bert_precision": round(P.mean().item(), 4),
            "mean_bert_recall": round(R.mean().item(), 4),
            "mean_bert_f1": round(F1.mean().item(), 4),
        }
    except Exception as e:
        print(f"⚠️ BERTScore (global) failed: {e}")
        mean_bert = {
            "mean_bert_precision": 0.0,
            "mean_bert_recall": 0.0,
            "mean_bert_f1": 0.0,
        }

    # Grouped calculation
    grouped = defaultdict(lambda: {"refs": [], "gens": []})
    for entry in eval_dataset:
        group = entry.get(group_key, "undefined")
        grouped[group]["refs"].append(entry["ground_truth"])
        grouped[group]["gens"].append(entry["generated_response"])

    grouped_bert = {}
    for group, data in grouped.items():
        try:
            if not data["refs"] or not data["gens"]:
                continue
            P_g, R_g, F1_g = bertscore(data["gens"], data["refs"], lang=lang, model_type=model_type, rescale_with_baseline=False)
            grouped_bert[group] = {
                "mean_bert_precision": round(P_g.mean().item(), 4),
                "mean_bert_recall": round(R_g.mean().item(), 4),
                "mean_bert_f1": round(F1_g.mean().item(), 4),
            }
        except Exception as e:
            print(f"BERTScore failed for group '{group}': {e}")
            grouped_bert[group] = {
                "mean_bert_precision": 0.0,
                "mean_bert_recall": 0.0,
                "mean_bert_f1": 0.0,
            }

    return mean_bert, grouped_bert

def calculate_answer_correctness_scores(eval_dataset):
    
    data = {
        "question": [],
        "answer": [],
        "ground_truth": []
    }

    for entry in eval_dataset:
        data["question"].append(entry["query"])
        data["answer"].append(entry["generated_response"])
        data["ground_truth"].append(entry["ground_truth"])

    try:
        dataset = Dataset.from_dict(data)
        scores_df = evaluate(dataset, metrics=[answer_correctness]).to_pandas()
        return scores_df["answer_correctness"].tolist()
    
    except Exception as e:
        print(f"Answer Correctness evaluation failed: {e}")
        return [0.0 for _ in eval_dataset]

def calculate_mean_answer_correctness(eval_dataset):
    print("→ Calculation: Answer Correctness (RAGAS)")

    scores = calculate_answer_correctness_scores(eval_dataset)

    # Attach score to each entry
    for entry, score in zip(eval_dataset, scores):
        entry["answer_correctness_score"] = score

    # Fallback if "question_context" is missing
    grouped_scores = {}
    for entry in eval_dataset:
        group = entry.get("question_context", "undefined")
        grouped_scores.setdefault(group, []).append(entry["answer_correctness_score"])

    def mean(values): return round(sum(values) / len(values), 4) if values else 0.0

    # Mean global + grouped
    mean_all = mean(scores)
    mean_grouped = {
        group: mean(score_list) for group, score_list in grouped_scores.items()
    }

    return {
        "mean_answer_correctness": mean_all,
        "grouped_mean_answer_correctness": mean_grouped,
        "all_scores": scores
    }

async def calculate_response_relevancy_ragas(dataset, batch_size=10, sleep_time=60):
    id_sample_map = {
        entry["query_id"]: SingleTurnSample(
            user_input=entry["query"],
            response=entry["generated_response"],
            retrieved_contexts=entry["retrieved_chunk_contexts"]
        )
        for entry in dataset
    }

    scores = {}
    ids = list(id_sample_map.keys())
    samples = list(id_sample_map.values())

    for i in tqdm(range(0, len(samples), batch_size), desc="→ Response Relevancy (LLM)"):
        batch_ids = ids[i:i+batch_size]
        batch_samples = samples[i:i+batch_size]

        # async call with exception handling
        batch_results = await asyncio.gather(*[
            response_relevancy_scorer.single_turn_ascore(sample)
            for sample in batch_samples
        ], return_exceptions=True)

        for entry_id, result in zip(batch_ids, batch_results):
            if isinstance(result, Exception):
                print(f"Error for ID {entry_id}: {result}")
                scores[entry_id] = 0.0  # fallback score
            else:
                scores[entry_id] = result

        if i + batch_size < len(samples):
            time.sleep(sleep_time)

    return scores


async def calculate_mean_response_relevancy_ragas(eval_dataset):
    print("→ Calculation: Response Relevancy with Ragas")

    scores_dict = await calculate_response_relevancy_ragas(eval_dataset)
    
    # Attach scores to original dataset using ID
    for entry in eval_dataset:
        entry_id = entry["query_id"]
        entry["response_relevancy_score"] = scores_dict.get(entry_id, 0.0)  

    # Group by question_context
    grouped_scores = {}
    for entry in eval_dataset:
        group = entry.get("question_context", "undefined")
        grouped_scores.setdefault(group, []).append(entry["response_relevancy_score"])

    def mean(values): return round(sum(values) / len(values), 4) if values else 0.0

    # Overall mean score
    mean_overall = mean(list(scores_dict.values()))

    # Mean scores per group (e.g., per question type)
    mean_grouped = {
        group: mean(score_list) for group, score_list in grouped_scores.items()
    }

    return {
        "mean_response_relevancy": mean_overall,
        "grouped_mean_response_relevancy": mean_grouped,
        "all_scores": scores_dict
    }



async def calculate_faithfulness_ragas(dataset, batch_size=10, sleep_time=60):
    id_sample_map = {
        entry["query_id"]: SingleTurnSample(
            user_input=entry["query"],
            response=entry["generated_response"],
            retrieved_contexts=entry["retrieved_chunk_contexts"]
        )
        for entry in dataset
    }

    scores = {}
    ids = list(id_sample_map.keys())
    samples = list(id_sample_map.values())

    for i in tqdm(range(0, len(samples), batch_size), desc="→ Faithfulness (LLM)"):
        batch_ids = ids[i:i + batch_size]
        batch_samples = samples[i:i + batch_size]

        batch_results = await asyncio.gather(*[
            faithfulness_scorer.single_turn_ascore(sample)
            for sample in batch_samples
        ], return_exceptions=True)

        for entry_id, result in zip(batch_ids, batch_results):
            if isinstance(result, Exception):
                print(f"Error for ID {entry_id}: {result}")
                scores[entry_id] = 0.0
            else:
                scores[entry_id] = result

        if i + batch_size < len(samples):
            time.sleep(sleep_time)

    return scores



async def calculate_mean_faithfulness_ragas(eval_dataset):
    print("→ Calculation: Faithfulness with Ragas")

    # Compute scores per entry
    scores_dict = await calculate_faithfulness_ragas(eval_dataset)

    # Attach scores to each entry based on ID
    for entry in eval_dataset:
        entry_id = entry["query_id"]
        entry["faithfulness_score"] = scores_dict.get(entry_id, 0.0)  # fallback 0.0

    # Grouping by question_context (or 'undefined')
    grouped_scores = {}
    for entry in eval_dataset:
        group = entry.get("question_context", "undefined")
        grouped_scores.setdefault(group, []).append(entry["faithfulness_score"])

    def mean(values): return round(sum(values) / len(values), 4) if values else 0.0

    # Global mean
    mean_overall = mean(list(scores_dict.values()))

    # Group-wise mean
    mean_grouped = {
        group: mean(score_list) for group, score_list in grouped_scores.items()
    }

    return {
        "mean_faithfulness": mean_overall,
        "grouped_mean_faithfulness": mean_grouped,
        "all_scores": scores_dict
    }



async def evaluate_and_save_results(json_filename, model_name="Not defined", evaluation_mode="experimental"):
    with open(f"eval_datasets/{json_filename}", "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    if model_name == "Not defined":
        model_name = os.path.splitext(os.path.basename(json_filename))[0]
        
    final_eval = ""

    # == Classic Metrics ==
    print("→ Calculating classic metrics")

    mean_bleu, grouped_bleu = calculate_mean_bleu_score(eval_dataset)
    mean_rouge, grouped_rouge = calculate_mean_rouge_scores(eval_dataset)
    mean_meteor, grouped_meteor = calculate_mean_meteor_score(eval_dataset)
    mean_bert, grouped_bert = calculate_mean_bert_score(eval_dataset)
    answer_correctness = calculate_mean_answer_correctness(eval_dataset)

    metrics = {
        "model": model_name,
        "mean_bleu": mean_bleu,
        "mean_meteor": mean_meteor,
        **mean_rouge,
        **mean_bert,
        "mean_answer_correctness": answer_correctness["mean_answer_correctness"]
    }

    grouped_metrics = {
        "bleu": grouped_bleu,
        "meteor": grouped_meteor,
        "rouge1": {},
        "rouge2": {},
        "rougeL": {},
        "bert_precision": {},
        "bert_recall": {},
        "bert_f1": {},
        "answer_correctness": answer_correctness["grouped_mean_answer_correctness"]
    }
    
    # Iteriere über alle Gruppen
    for group_name, scores in grouped_rouge.items():
        grouped_metrics["rouge1"][group_name] = scores.get("mean_rouge1", 0.0)
        grouped_metrics["rouge2"][group_name] = scores.get("mean_rouge2", 0.0)
        grouped_metrics["rougeL"][group_name] = scores.get("mean_rougeL", 0.0)

    for group_name, scores in grouped_bert.items():
        grouped_metrics["bert_precision"][group_name] = scores.get("mean_bert_precision", 0.0)
        grouped_metrics["bert_recall"][group_name] = scores.get("mean_bert_recall", 0.0)
        grouped_metrics["bert_f1"][group_name] = scores.get("mean_bert_f1", 0.0)
    

    # == RAGAS Metrics ==
    if evaluation_mode == "final_eval":
        print("→ Calculating LLM-based metrics (RAGAS)")
        ragas_response = await calculate_mean_response_relevancy_ragas(eval_dataset)
        ragas_faithfulness = await calculate_mean_faithfulness_ragas(eval_dataset)

        metrics.update({
            "mean_response_relevancy": ragas_response["mean_response_relevancy"],
            "mean_faithfulness": ragas_faithfulness["mean_faithfulness"]
        })

        grouped_metrics["response_relevancy"] = ragas_response["grouped_mean_response_relevancy"]
        grouped_metrics["faithfulness"] = ragas_faithfulness["grouped_mean_faithfulness"]
        
        final_eval = "_final"

    # == Save Overall Means ==
    df_results = pd.DataFrame([metrics])
    df_results.to_csv(f"eval_results/{model_name}_generation_evaluation{final_eval}.csv", index=False, encoding="utf-8")

    # == Save Grouped Means if available ==
    if grouped_metrics and any(grouped_metrics.values()):
        # Transpose grouped dict to shape: group -> metric -> value
        combined_grouped = {}
        for metric, group_dict in grouped_metrics.items():
            for group, value in group_dict.items():
                combined_grouped.setdefault(group, {})[f"mean_{metric}"] = value

        grouped_df = pd.DataFrame.from_dict(combined_grouped, orient="index")
        grouped_df.index.name = "question_context"
        grouped_df.to_csv(f"eval_results/{model_name}_generation_grouped{final_eval}.csv", encoding="utf-8")

    return df_results


def run_generation_evaluation(json_filename, model_name="Not defined", evaluation_mode: str = "experimental"):
    return asyncio.run(evaluate_and_save_results(json_filename, model_name, evaluation_mode))