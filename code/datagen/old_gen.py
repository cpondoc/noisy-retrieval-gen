"""
Playing around with noisy benchmark creation
"""

import openai
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import os
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

TOPICS_PROMPT = f"""
Please take the following topics and cluster them into a list of 10-15 topics. These topics should be short and concise,
but not just limited to one or two words. Please return only the 10-15 topics, one on each line.

Topics:
"""


def read_queries(file_path: str) -> str:
    """
    Reads a text file and returns its contents as a single string with each line separated by a newline.
    """
    with open(f"{file_path}/queries.txt", "r", encoding="utf-8") as file:
        return "".join(file.readlines())


def load_data(dataset_name: str):
    """
    Load in NFCorpus corpus, queries, and qfrels
    """
    # Load in corpus
    corpus = load_dataset(dataset_name, "corpus")
    corpus = corpus["corpus"]

    docs = []
    for text in corpus["text"]:
        docs.append(text)

    # Load in queries
    queries = load_dataset(
        dataset_name,
        "queries",
    )
    queries = queries["queries"]

    # Load in qfrels
    qfrels = load_dataset(dataset_name)
    qfrels = qfrels["test"]

    return corpus, queries, qfrels, docs


def write_queries(queries, file_path):
    """
    Write to a separate text file
    """
    with open(f"{file_path}/queries.txt", "w", encoding="utf-8") as file:
        for query in queries["text"]:  # Assuming queries have a "text" field
            file.write(query + "\n")


def chunk_text(text, max_tokens=8000, encoding_name="cl100k_base"):
    """Splits text into chunks that fit within the token limit of the specified encoding."""
    encoding = tiktoken.get_encoding(encoding_name)  # Explicitly get the tokenizer
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = encoding.decode(tokens[i : i + max_tokens])
        chunks.append(chunk)

    return chunks


def generate_topics(
    queries_str,
    model="gpt-3.5-turbo-0125",
    max_tokens=100,
    output_dir="data/FiQA-2018/topics",
):
    """
    Use an LLM to generate topics while handling large inputs.
    Saves each response to a text file in the specified directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use the correct tokenizer based on model
    encoding_name = (
        "cl100k_base" if "gpt-4" in model or "gpt-3.5" in model else "p50k_base"
    )
    chunks = chunk_text(queries_str, max_tokens=8000, encoding_name=encoding_name)

    responses = []

    print(f"Saving responses to: {output_dir}")  # Debugging print

    for i, chunk in enumerate(chunks):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": f"{TOPICS_PROMPT}\n{chunk}"}],
                max_tokens=max_tokens,
                temperature=1.1,
            )
            response_text = response.choices[0].message.content
            responses.append(response_text)

            # Save each response to a separate text file
            file_path = os.path.join(output_dir, f"response_{i+1}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response_text)

            print(f"Saved: {file_path}")  # Debugging print
        except Exception as e:
            print(f"Error in processing chunk {i+1}: {e}")

    return responses  # Returns a list of responses


# Add this to your main block to run the additional generation
if __name__ == "__main__":
    """
    Run all main functions
    """
    # Load in the data, save the queries
    corpus, queries, qfrels, docs = load_data("mteb/nfcorpus")
    # write_queries(queries, "data/NFCorpus")

    # # Generate topics
    # queries_str = read_queries("data/NFCorpus")
    # topics = generate_topics(
    #     TOPICS_PROMPT, queries_str, output_dir="data/SCIDOCS/topics"
    # )
