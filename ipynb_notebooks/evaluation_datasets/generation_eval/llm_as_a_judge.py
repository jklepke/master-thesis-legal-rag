# Imports
import json
import openai
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any


# Configurations
# Load environment variables. Assumes that the project directory contains a .env file with API keys
load_dotenv()

# Set the OpenAI API key from the environment variables
# Make sure to update "OPENAI_API_KEY" to match the variable name in your .env file
openai.api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=openai.api_key)


JUDGE_PROMPT = """
You are an expert evaluator for AI-generated answers. Please assess each answer using the following three dimensions, each rated on a 5-point scale. For each criterion, use the detailed scoring definitions below. Respond only with a valid JSON object.

--- Scoring Guidelines ---

1. Answer Relevance
Measures how well the answer addresses the question’s intent and topical scope.

Score 1: The answer is off-topic, unrelated to the question, or completely misses the user’s intent.  
Score 2: The answer is only marginally related to the question; it shows limited topical relevance.  
Score 3: The answer is somewhat relevant but lacks completeness or includes off-topic elements.  
Score 4: The answer is largely relevant and addresses most of the question, with minor omissions.  
Score 5: The answer is highly relevant, directly addresses the question, and provides helpful insights.

---

2. Faithfulness
Evaluates whether the content is factually correct and free of hallucinations or misleading claims.

Score 1: The answer contains factual errors, hallucinations, or unsupported claims.  
Score 2: Some parts are accurate, but major factual inaccuracies or fabricated content are present.  
Score 3: Mostly factually correct, but contains minor errors, ambiguities, or uncertain claims.  
Score 4: Factually sound with only negligible issues or unclear phrasing.  
Score 5: Fully factually accurate, free of hallucinations or unsupported statements.

---

3. Answer Correctness
Assesses the semantic and conceptual overlap between the generated and reference answer.

Score 1: No meaningful overlap with the reference answer; key points are missing or contradicted.  
Score 2: Limited alignment; only a few aspects of the reference are addressed.  
Score 3: Partial overlap; key points are present but not fully developed or precise.  
Score 4: High degree of alignment with the reference, though not a perfect match in all details.  
Score 5: Strong match with the reference in meaning, structure, and coverage of key points.

---

Return your evaluation in the following JSON format:

{
  "Answer Relevance": <1–5>,
  "Faithfulness": <1–5>,
  "Answer Correctness": <1–5>,
  "Justification": {
    "Answer Relevance": "<short explanation>",
    "Faithfulness": "<short explanation>",
    "Answer Correctness": "<short explanation>"
  }
}
"""

RE_EVAL_PROMPT = """
You are now acting as a second-level evaluator, critically reassessing prior LLM evaluations. Your goal is to *double-check* the evaluation of an AI-generated answer to a question, and to correct or refine the initial rating **if necessary**.

Your perspective should be independent and critical. Focus especially on:

- whether any details were overlooked,
- if the prior rating was too generous or too strict,
- or if additional nuances were missed.

Use the **same rating scheme** as before, and return your updated scores. If you agree with the original, you may keep the same values – but only after verification.

"""

EVALUATION_SCHEME = """
--- Scoring Guidelines ---

1. Answer Relevance
Measures how well the answer addresses the question’s intent and topical scope.

Score 1: The answer is off-topic, unrelated to the question, or completely misses the user’s intent.  
Score 2: The answer is only marginally related to the question; it shows limited topical relevance.  
Score 3: The answer is somewhat relevant but lacks completeness or includes off-topic elements.  
Score 4: The answer is largely relevant and addresses most of the question, with minor omissions.  
Score 5: The answer is highly relevant, directly addresses the question, and provides helpful insights.

---

2. Faithfulness
Evaluates whether the content is factually correct and free of hallucinations or misleading claims.

Score 1: The answer contains factual errors, hallucinations, or unsupported claims.  
Score 2: Some parts are accurate, but major factual inaccuracies or fabricated content are present.  
Score 3: Mostly factually correct, but contains minor errors, ambiguities, or uncertain claims.  
Score 4: Factually sound with only negligible issues or unclear phrasing.  
Score 5: Fully factually accurate, free of hallucinations or unsupported statements.

---

3. Answer Correctness
Assesses the semantic and conceptual overlap between the generated and reference answer.

Score 1: No meaningful overlap with the reference answer; key points are missing or contradicted.  
Score 2: Limited alignment; only a few aspects of the reference are addressed.  
Score 3: Partial overlap; key points are present but not fully developed or precise.  
Score 4: High degree of alignment with the reference, though not a perfect match in all details.  
Score 5: Strong match with the reference in meaning, structure, and coverage of key points.

---

Return only a JSON object in the following format:

{
  "Answer Relevance": <1–5>,
  "Faithfulness": <1–5>,
  "Answer Correctness": <1–5>,
  "Justification": {
    "Answer Relevance": "<short explanation>",
    "Faithfulness": "<short explanation>",
    "Answer Correctness": "<short explanation>"
  }
}

Critically re-evaluate now. If the initial rating was too lenient, adjust it downward. Otherwise, confirm or refine it only if justified. Do not change scores unless you find a clear reason.
"""



def evaluate_item(query, generated_answer, ground_truth):
    prompt = JUDGE_PROMPT + f"""

Question: {query}

Generated Answer: {generated_answer}

Reference Answer: {ground_truth}

Evaluate now:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}
    
    
def re_evaluate_item(entry):
    prompt = RE_EVAL_PROMPT + f"""

Question: {entry['query']}

Generated Answer: {entry['generated_response']}

Reference Answer: {entry['ground_truth']}

Initial Scores:
Answer Relevance: {entry['Answer Relevance']}
Faithfulness: {entry['Faithfulness']}
Answer Correctness: {entry['Answer Correctness']}
""" + EVALUATION_SCHEME

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        content = response.choices[0].message.content.strip()
        return {
            **entry,
            "ReEval": json.loads(content)
        }
    except Exception as e:
        return {**entry, "ReEvalError": str(e)}



def evaluate_with_retry(entry, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return {
                **entry,
                **evaluate_item(entry["query"], entry["generated_response"], entry["ground_truth"])
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"Failed on query_id {entry['query_id']} – {e}")
                return {**entry, "error": str(e)}
            
            
def re_evaluate_with_retry(entry, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return {
                **entry,
                "ReEval": re_evaluate_item(entry)["ReEval"]  
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"Re-Eval failed on query_id {entry['query_id']} – {e}")
                return {**entry, "ReEvalError": str(e)}



def run_llm_judge_parallel(input_path, output_path, max_workers=4):
    with open(input_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    results = [None] * len(entries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(evaluate_with_retry, entry): i
            for i, entry in enumerate(entries)
        }

        for future in tqdm(as_completed(future_to_idx), total=len(entries), desc="Evaluating with GPT-4o-mini"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                results[idx] = {**entries[idx], "error": str(e)}

    # Save output
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)

    print(f"Evaluated results written to: {output_path}")
    return output_path


def run_llm_rejudge_parallel(input_path, output_path, max_workers=4):
    with open(input_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    results = [None] * len(entries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(re_evaluate_item, entry): i
            for i, entry in enumerate(entries)
        }

        for future in tqdm(as_completed(future_to_idx), total=len(entries), desc="Re-evaluating"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {**entries[idx], "ReEvalError": str(e)}

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)

    print(f"Re-evaluated results written to: {output_path}")
    return output_path



def plot_score_boxplot(df: pd.DataFrame, score_cols: list, output_dir: str = "/mnt/data") -> str:
    """
    Erstellt einen Boxplot der Bewertungsdimensionen und speichert ihn als PNG.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    boxplot_path = os.path.join(output_dir, "score_boxplot.png")

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[score_cols])
    plt.title("Score Distribution per Dimension")
    plt.ylabel("Score (1–5)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()

    return boxplot_path

def plot_histograms(df, score_cols, output_dir="/mnt/data"):
    """Erstellt Histogramme der Scores pro Dimension."""
    fig, axes = plt.subplots(1, len(score_cols), figsize=(14, 4))
    for i, col in enumerate(score_cols):
        sns.histplot(df[col], bins=5, discrete=True, ax=axes[i])
        axes[i].set_title(f"Histogram – {col}")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "score_histograms.png")
    plt.savefig(hist_path)
    plt.close()
    return hist_path

def plot_grouped_bar_by_context(df, score_cols, output_dir="/mnt/data"):
    """Erstellt gruppierte Balkendiagramme pro Fragekontext."""
    group_means = df.groupby("question_context")[score_cols].mean().reset_index()
    melted = pd.melt(group_means, id_vars="question_context", var_name="Dimension", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="question_context", y="Score", hue="Dimension")
    plt.title("Mittelwerte je Dimension und Kontext")
    plt.xticks(rotation=45)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "grouped_bar_by_context.png")
    plt.savefig(bar_path)
    plt.close()
    return bar_path

def calculate_and_visualize_scores_of_evaluation_scheme(json_path: str, output_file_name: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Falls ReEval existiert, ersetze die Bewertungen im Haupt-Dict
    for entry in data:
        if "ReEval" in entry:
            for dim in ["Answer Relevance", "Faithfulness", "Answer Correctness"]:
                entry[dim] = entry["ReEval"].get(dim, entry.get(dim))

    df = pd.DataFrame(data)

    score_cols = ["Answer Relevance", "Faithfulness", "Answer Correctness"]
 
    
    # Overall statistic
    overall_stats = {}
    for col in score_cols:
        mean_val = df[col].mean()
        overall_stats[col] = {
            "mean": round(mean_val, 3),
            "std": round(df[col].std(), 3),
            "min": int(df[col].min()),
            "max": int(df[col].max()),
            "percent": round((mean_val / 5) * 100, 3)
        }

        
    # Grouped statistics
    grouped_stats = {}
    grouped = df.groupby("question_context")
    for name, group in grouped:
        grouped_stats[name] = {
            "count": len(group)
        }
        for col in score_cols:
            mean_val = group[col].mean()
            grouped_stats[name][col] = {
                "mean": round(mean_val, 3),
                "std": round(group[col].std(), 3),
                "percent": round((mean_val / 5) * 100, 3)
            }

    output_dir = "eval_results"
    summary_path = os.path.join(output_dir, output_file_name)

    # Visualization
    boxplot_path = plot_score_boxplot(df, score_cols, summary_path)
    histogram_path = plot_histograms(df, score_cols, summary_path)
    grouped_bar_path = plot_grouped_bar_by_context(df, score_cols, summary_path)
    
    evaluation_summary = {
        "overall": overall_stats,
        "by_question_context": grouped_stats
    }
    
    with open(f"{summary_path}/summary.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
        
    overall_df = pd.DataFrame.from_dict(overall_stats, orient="index")

    records = []
    for context, stats in grouped_stats.items():
        for dimension, values in stats.items():
            if dimension == "count":
                continue
            record = {
                "question_context": context,
                "dimension": dimension,
                **values
            }
            record["count"] = stats["count"]
            records.append(record)

    grouped_df = pd.DataFrame(records)

    # Speichern als zwei getrennte Excel-Dateien
    overall_path = os.path.join(summary_path, "eval_overall_stats.xlsx")
    grouped_path = os.path.join(summary_path, "eval_grouped_stats.xlsx")

    overall_df.to_excel(overall_path)
    grouped_df.to_excel(grouped_path, index=False)    

    return {
        "overall": overall_stats,
        "by_question_context": grouped_stats,
        "boxplot_path": boxplot_path,
        "histogram_path": histogram_path,
        "grouped_bar_path": grouped_bar_path,
        "summary_json_path": summary_path
    }