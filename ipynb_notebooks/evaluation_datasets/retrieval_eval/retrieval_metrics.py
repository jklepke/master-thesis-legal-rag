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
from typing import List
import time

# Langchain Imports
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI

# Ragas Imports
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from ragas.testset.persona import Persona
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample
from ragas.metrics import (
    LLMContextPrecisionWithReference, 
    LLMContextPrecisionWithoutReference,
    NonLLMContextPrecisionWithReference,
    LLMContextRecall,
    NonLLMContextRecall,
    ContextEntityRecall
)

# == SETUP ==

load_dotenv()
ragas_token = os.getenv("RAGAS_APP_TOKEN")
os.environ["RAGAS_APP_TOKEN"] = ragas_token
openai.api_key = os.environ['OPENAI_API_KEY']

# LLM Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
client = OpenAI(api_key=openai.api_key)

# Embedding Setup
embedding_model = OpenAIEmbeddings()

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# context precision config
context_pprecision_with_ref = LLMContextPrecisionWithReference(llm=evaluator_llm)
context_pprecision_without_ref = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
context_precision_with_ref_NonLLM = NonLLMContextPrecisionWithReference()

# context recall config
context_recall = LLMContextRecall(llm=evaluator_llm)
context_recall_NonLLM = NonLLMContextRecall()

# context entities config
context_entity_recall_scorer = ContextEntityRecall(llm=evaluator_llm)


def calculate_recall_at_k(entry) -> float:
    """
    Recall@k = Number of correctly retrieved relevant documents / number of actually relevant documents
    """
    try:
        # get Ground-Truth-IDs of the chunks
        relevant_chunk_ids = set(entry.get("ground_truth_chunk_ids", []))
        if not relevant_chunk_ids:
            return 0.0
        
        # get all retrieved chunk ids
        retrieved_chunk_ids = set(entry.get("retrieved_chunk_ids", []))
        
        # Number of correct retrieved chunks (intersection between ground-truth and retrieved chunks)
        correct_hits = relevant_chunk_ids.intersection(retrieved_chunk_ids)
        
        # Calculation of Recall@k
        return len(correct_hits) / len(relevant_chunk_ids)
    
    except Exception as e:
        print(f"[ERROR] Recall@k failed for entry: {e}")
        return 0.0


def calculate_mean_recall_at_k(eval_dataset) -> float:
    print(f"→ Calculation: Mean Recall@k")
    recalls = [calculate_recall_at_k(entry) for entry in tqdm(eval_dataset)]
    return round(sum(recalls) / len(recalls), 4)

def calculate_reciprocal_rank(entry):
    try:
        relevant_chunk_ids = set(entry.get("ground_truth_chunk_ids", []))
        retrieved_chunk_ids = entry.get("retrieved_chunk_ids", [])
        
        for idx, chunk_id in enumerate(retrieved_chunk_ids):
            if chunk_id in relevant_chunk_ids:
                return 1 / (idx + 1)  # Returns the reciprocal value of the position (1/rank).
        return 0.0
    except Exception as e:
        print(f"[ERROR] Reciprocal Rank failed for entry: {e}")
        return 0.0

def calculate_mean_reciprocal_rank(eval_dataset) -> float:
    print(f"→ Calculation: Mean Reciprocal Rank")
    reciprocal_ranks = [calculate_reciprocal_rank(entry) for entry in tqdm(eval_dataset)]
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)

def calculate_average_precision(entry):
    try:
        relevant_chunk_ids = set(entry.get("ground_truth_chunk_ids", []))
        retrieved_chunk_ids = entry.get("retrieved_chunk_ids", [])
        
        if not relevant_chunk_ids:
            return 0.0

        hits = 0
        precision_sum = 0.0
        
        for idx, chunk_id in enumerate(retrieved_chunk_ids):
            if chunk_id in relevant_chunk_ids:
                hits += 1
                precision = hits / (idx + 1)  # Precision@k
                precision_sum += precision

        return precision_sum / len(relevant_chunk_ids)
    
    except Exception as e:
        print(f"[ERROR] Average Precision failed for entry: {e}")
        return 0.0

def calculate_mean_average_precision(eval_dataset) -> float:
    print(f"→ Calculation: Mean Average Precision")
    average_precisions = [calculate_average_precision(entry) for entry in tqdm(eval_dataset)]
    return round(sum(average_precisions) / len(average_precisions), 4)


async def calculate_context_precision_ragas(dataset, use_reference=True, LLM_based=True, batch_size=10, sleep_time=60):
    scores = []

    if LLM_based:
        samples = [
            SingleTurnSample(
                user_input=entry["query"],
                response=entry["generated_response"],
                reference=entry["ground_truth"],
                retrieved_contexts=entry["retrieved_chunk_contexts"]
            )
            for entry in dataset
        ]

        scorer = context_pprecision_with_ref if use_reference else context_pprecision_without_ref

        for i in tqdm(range(0, len(samples), batch_size), desc="→ Context Precision (LLM)"):
            batch = samples[i:i + batch_size]
            batch_scores = await asyncio.gather(*[
                scorer.single_turn_ascore(sample) for sample in batch
            ], return_exceptions=True)

            for idx, result in enumerate(batch_scores):
                if isinstance(result, Exception):
                    logging.error(f"[ERROR] Sample {i+idx} failed: {repr(result)}")
                    scores.append(0.0)
                else:
                    scores.append(result)

            if i + batch_size < len(samples):
                time.sleep(sleep_time)

    else:
        samples = [
            SingleTurnSample(
                retrieved_contexts=entry["retrieved_chunk_contexts"],
                reference_contexts=entry["ground_truth_chunk_contexts"]
            )
            for entry in dataset
        ]

        for i in tqdm(range(0, len(samples), batch_size), desc="→ Context Precision (Non-LLM)"):
            batch = samples[i:i + batch_size]
            batch_scores = await asyncio.gather(*[
                context_precision_with_ref_NonLLM.single_turn_ascore(sample) for sample in batch
            ], return_exceptions=True)

            for idx, result in enumerate(batch_scores):
                if isinstance(result, Exception):
                    logging.error(f"[ERROR] Sample {i+idx} failed: {repr(result)}")
                    scores.append(0.0)
                else:
                    scores.append(result)

            # if i + batch_size < len(samples):
            #     time.sleep(sleep_time)

    return scores

async def calculate_mean_context_precision_ragas(eval_dataset, use_reference, LLM_based) -> float:
    print(f"→ Calculation: Mean Context Precision Using Ragas {'with' if use_reference else 'without'} Reference and {'LLM-Based' if LLM_based else 'NonLLM-Based'}")
    mean_context_precision = await calculate_context_precision_ragas(eval_dataset, use_reference, LLM_based)  
    return round(sum(mean_context_precision) / len(mean_context_precision), 4)

async def calculate_context_recall_ragas(dataset, LLM_based=True, batch_size=10, sleep_time=60):
    scores = []

    if LLM_based:
        samples = [
            SingleTurnSample(
                user_input=entry["query"],
                response=entry["generated_response"],
                reference=entry["ground_truth"],
                retrieved_contexts=entry["retrieved_chunk_contexts"]
            )
            for entry in dataset
        ]

        for i in tqdm(range(0, len(samples), batch_size), desc="→ Context Recall (LLM)"):
            batch = samples[i:i+batch_size]
            batch_scores = await asyncio.gather(*[
                context_recall.single_turn_ascore(sample) for sample in batch
            ], return_exceptions=True)

            for idx, result in enumerate(batch_scores):
                if isinstance(result, Exception):
                    logging.error(f"[Context Recall LLM] Sample {i+idx} failed: {repr(result)}")
                    scores.append(0.0)
                else:
                    scores.append(result)

            if i + batch_size < len(samples):
                time.sleep(sleep_time)

    else:
        samples = [
            SingleTurnSample(
                retrieved_contexts=entry["retrieved_chunk_contexts"],
                reference_contexts=entry["ground_truth_chunk_contexts"]
            )
            for entry in dataset
        ]

        for i in tqdm(range(0, len(samples), batch_size), desc="→ Context Recall (Non-LLM)"):
            batch = samples[i:i+batch_size]
            batch_scores = await asyncio.gather(*[
                context_recall_NonLLM.single_turn_ascore(sample) for sample in batch
            ], return_exceptions=True)

            for idx, result in enumerate(batch_scores):
                if isinstance(result, Exception):
                    logging.error(f"[Context Recall Non-LLM] Sample {i+idx} failed: {repr(result)}")
                    scores.append(0.0)
                else:
                    scores.append(result)

            # if i + batch_size < len(samples):
            #     time.sleep(sleep_time)

    return scores


async def calculate_mean_context_recall_ragas(eval_dataset, LLM_based) -> float:
    print(f"→ Calculation: Mean Context Recall Using Ragas with {'LLM-Based' if LLM_based else 'NonLLM-Based'}")
    mean_context_recall = await calculate_context_recall_ragas(eval_dataset, LLM_based)
    return round(sum(mean_context_recall) / len(mean_context_recall), 4)

async def calculate_context_entity_recall_ragas(dataset, batch_size=10, sleep_time=60):
    samples = [
        SingleTurnSample(
            reference=entry["ground_truth"],
            retrieved_contexts=entry["retrieved_chunk_contexts"]
        )
        for entry in dataset
    ]

    scores = []
    total_batches = (len(samples) + batch_size - 1) // batch_size

    for i in tqdm(range(total_batches), desc="→ Context Entity Recall (Batched)"):
        batch = samples[i * batch_size: (i + 1) * batch_size]
        batch_scores = await asyncio.gather(*[
            context_entity_recall_scorer.single_turn_ascore(sample) for sample in batch
        ], return_exceptions=True)

        for idx, result in enumerate(batch_scores):
            if isinstance(result, Exception):
                logging.error(f"[Context Entity Recall] Sample {i*batch_size + idx} failed: {repr(result)}")
                scores.append(0.0)
            else:
                scores.append(result)

        if i < total_batches - 1:
            time.sleep(sleep_time)

    return scores

async def calculate_mean_context_entity_recall_ragas(eval_dataset) -> float:
    print(f"→ Calculation: Mean Context Entity Recall (Batched)")
    scores = await calculate_context_entity_recall_ragas(eval_dataset, batch_size=5, sleep_time=5)
    return round(sum(scores) / len(scores), 4)

async def evaluate_and_save_results(json_filename, model_name: str = "Not defined", evaluation_mode: str = "experimental"):

    with open(f"eval_datasets/{json_filename}", "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    if model_name == "Not defined":
        model_name = os.path.basename(json_filename).replace(".json", "")
        
    metrics = {}
    
    final_eval = ""
    
    if evaluation_mode=="experimental":

        metrics = {
            "model": model_name,
            "mean_recall_at_k": calculate_mean_recall_at_k(eval_dataset),
            "mean_reciprocal_rank": calculate_mean_reciprocal_rank(eval_dataset),
            "mean_average_precision": calculate_mean_average_precision(eval_dataset),
            "context_precision_ragas_with_reference_NonLLM": await calculate_mean_context_precision_ragas(eval_dataset, use_reference=True, LLM_based=False),
            "context_recall_ragas_NonLLM": await calculate_mean_context_recall_ragas(eval_dataset, LLM_based=False),
        }
        
    elif evaluation_mode=="final_eval":
        
        metrics = {
            "model": model_name,
            "mean_recall_at_k": calculate_mean_recall_at_k(eval_dataset),
            "mean_reciprocal_rank": calculate_mean_reciprocal_rank(eval_dataset),
            "mean_average_precision": calculate_mean_average_precision(eval_dataset),
            "context_precision_ragas_with_reference_LLM": await calculate_mean_context_precision_ragas(eval_dataset, use_reference=True, LLM_based=True),
            "context_precision_ragas_with_reference_NonLLM": await calculate_mean_context_precision_ragas(eval_dataset, use_reference=True, LLM_based=False),
            "context_recall_ragas_LLM": await calculate_mean_context_recall_ragas(eval_dataset, LLM_based=True),
            "context_recall_ragas_NonLLM": await calculate_mean_context_recall_ragas(eval_dataset, LLM_based=False),
            #"context_entity_recall_ragas": await calculate_mean_context_entity_recall_ragas(eval_dataset),
        } 
        
        final_eval = "_final"

    df_results = pd.DataFrame([metrics])

    output_path = f"eval_results/{model_name}_retrieval_evaluation{final_eval}.csv"
    df_results.to_csv(output_path, index=False, encoding="utf-8")

    return df_results

def run_retrieval_evaluation(json_filename, model_name="Not defined", evaluation_mode: str = "experimental"):
    return asyncio.run(evaluate_and_save_results(json_filename, model_name, evaluation_mode))