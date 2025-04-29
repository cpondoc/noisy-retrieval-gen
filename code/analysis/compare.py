"""
Analyzing our predictions
"""

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import os
import pandas as pd
from tqdm import tqdm
import json


def load_data():
    """
    Load in benchmark corpus, queries, and qfrels
    """
    # Load in corpus
    benchmark = "scidocs"
    new_data = load_dataset(
        f"mteb/{benchmark}",
        "corpus",
    )
    corpus = new_data["corpus"]

    docs = []
    for text in corpus["text"]:
        docs.append(text)

    # Load in queries
    new_data = load_dataset(
        f"mteb/{benchmark}",
        "queries",
    )
    queries = new_data["queries"]

    # Load in qfrels
    last_data = load_dataset(f"mteb/{benchmark}")
    qfrels = last_data["test"]

    return corpus, queries, qfrels, docs


def find_correct_answers(corpus, queries, qfrels):
    """
    Find correct answers for a retrieval benchmark
    """
    # Find best matches per query-id
    df_qrels = pd.DataFrame(qfrels)
    best_matches = df_qrels.loc[df_qrels.groupby("query-id")["score"].idxmax()]

    # Convert to dictionary for fast lookup
    query_dict = {q["_id"]: q["text"] for q in queries}
    corpus_dict = {c["_id"]: c["text"] for c in corpus}

    # Retrieve the actual text pairs
    best_matches["query_text"] = best_matches["query-id"].map(query_dict)
    best_matches["corpus_text"] = best_matches["corpus-id"].map(corpus_dict)
    return best_matches


def analyze_mteb_predictions(file_path):
    """
    Look at result from MTEB run
    """
    # Load the predictions file (replace with actual path)
    with open(file_path, "r") as f:
        predictions = json.load(f)

    # Convert predictions to a DataFrame for easier processing
    pred_list = []
    for query_id, docs in predictions.items():
        best_doc_id = max(docs, key=docs.get)  # Get corpus ID with highest score
        best_score = docs[best_doc_id]
        pred_list.append(
            {"query-id": query_id, "corpus-id": best_doc_id, "pred_score": best_score}
        )

    df_preds = pd.DataFrame(pred_list)
    return df_preds


def compare_mteb_predictions(df1, df2):
    """
    Compare two MTEB prediction dataframes and return a list of query and corpus IDs where they differ.
    """
    # Merge, find difference, extract list
    df_comparison = df1.merge(df2, on="query-id", suffixes=("_1", "_2"))
    mismatches = df_comparison[
        df_comparison["corpus-id_1"] != df_comparison["corpus-id_2"]
    ]
    mismatch_list = list(
        zip(
            mismatches["query-id"], mismatches["corpus-id_1"], mismatches["corpus-id_2"]
        )
    )
    return mismatch_list


def get_noisy_docs():
    """
    Get all of the noisy stuff.
    """
    # Load data from HF
    dataset = load_dataset(
        "cpondoc/noisy-scidocs-6258", ignore_verifications=True, keep_in_memory=True
    )
    train_data = dataset["train"]

    # Convert dataset dictionary into list of dictionaries
    processed_data = []
    for i in range(len(train_data["text"])):  # Iterate through index positions
        doc = {"id": f"doc_{i}", "text": train_data["text"][i]}
        processed_data.append(doc)

    return processed_data


def find_mismatched_text(mismatch_list, corpus, queries, noisy_corpus):
    """
    Given a list of mismatched query and corpus ID tuples, retrieve the actual text
    from both clean and noisy corpora using a unified dictionary.
    """
    # Convert datasets to dictionaries for fast lookup
    query_dict = {q["_id"]: q["text"] for q in queries}
    corpus_dict = {c["_id"]: c["text"] for c in corpus}

    # Add noisy corpus data into corpus_dict (overwrites or extends clean data)
    for doc in noisy_corpus:
        corpus_dict[doc["id"]] = doc["text"]

    mismatched_texts = []

    for query_id, corpus_id_1, corpus_id_2 in mismatch_list:
        query_text = query_dict.get(query_id, "Query not found")
        corpus_text_1 = corpus_dict.get(corpus_id_1, "Document not found")
        corpus_text_2 = corpus_dict.get(corpus_id_2, "Document not found")

        mismatched_texts.append(
            {
                "query-id": query_id,
                "query-text": query_text,
                "corpus-id-1": corpus_id_1,
                "corpus-text-1": corpus_text_1,
                "corpus-id-2": corpus_id_2,
                "corpus-text-2": corpus_text_2,
            }
        )

    return mismatched_texts


def save_mismatch_results(mismatch_results):
    """
    Save examples of mismatch results to text file
    """
    # Define the folder to save text files
    save_folder = "data/analysis/SCIDOCS/6258"
    os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

    for result in mismatch_results:
        # Check if `corpus-id-2` starts with "doc" (indicating a noisy retrieval)
        if result["corpus-id-2"].startswith("doc"):
            # Define the file path
            file_path = os.path.join(save_folder, f"{result['query-id']}.txt")

            # Prepare the formatted text
            content = (
                f"Query: {result['query-text']}\n\n"
                f"Vanilla Retrieved ({result['corpus-id-1']}): {result['corpus-text-1']}\n\n"
                f"Noisy Retrieved ({result['corpus-id-2']}): {result['corpus-text-2']}"
            )

            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"Saved mismatch result to {file_path}")


if __name__ == "__main__":
    """
    Run analysis code
    """
    # Get initial correct answers
    corpus, queries, qfrels, docs = load_data()
    correct_answers = find_correct_answers(corpus, queries, qfrels)

    # Look at ground truth + a noisy run
    normal_ir = analyze_mteb_predictions(
        "calibration/SCIDOCS/fineweb/1/0/SCIDOCS_default_predictions.json"
    )
    noisy_ir = analyze_mteb_predictions(
        "calibration/SCIDOCS/fineweb/1/6000/SCIDOCS_default_predictions.json"
    )


    # Most recent run
    mismatch_list = compare_mteb_predictions(normal_ir, noisy_ir)
    noisy_corpus = get_noisy_docs()
    mismatch_results = find_mismatched_text(
        mismatch_list, corpus, queries, noisy_corpus
    )

    # Find statistics
    total_num = len(mismatch_results)
    num_noisy = 0
    for result in mismatch_results:
        num_noisy += int(result["corpus-id-2"].startswith("doc"))
    print(f"Percentage affected by noisy documents: {float(num_noisy / total_num)}")

    # Analyze
    save_mismatch_results(mismatch_results)
