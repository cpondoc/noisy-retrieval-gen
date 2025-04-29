"""
Annotation on data examples using GPT-4
"""

import os
import csv
import re
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Build static prompt
base_prompt = """
Below is an extract from a web page. Evaluate whether the page is noisy based on its relevance, coherence, and focus on the intended topic using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract contains some information related to the intended topic but also includes significant off-topic material such as advertisements, promotional content, or irrelevant navigation links.
- Add another point if the extract frequently mixes relevant and irrelevant content, resulting in disorganized or incoherent writing that makes it difficult to follow the intended topic.
- Award a third point if the extract mostly discusses the intended topic but contains occasional noise, such as minor off-topic tangents, casual language, or redundant information that moderately impacts clarity.
- Grant a fourth point if the extract is generally clean and focused, with only minor distracting elements that do not significantly affect the overall understanding or relevance to the topic.
- Bestow a fifth point if the extract is completely clean, coherent, highly focused, and free from any irrelevant, distracting, or incoherent material, providing a clear and uninterrupted discussion of the intended topic.

The extract:
{extract}

After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Noise score: <total points>"
"""

def get_score(doc):
    """
    Use GPT-4 via OpenAI SDK >= 1.0.0 to evaluate the noise score.
    """
    prompt = base_prompt.format(extract=doc)

    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0.7,
            max_tokens=300,
        )
        output_text = response.choices[0].text

        match = re.search(r"Noise score:\s*(\d+)", output_text)
        return int(match.group(1)) if match else 0

    except Exception as e:
        print(f"Error processing doc: {e}")
        return 0

def load_data():
    """
    Load and combine NFCorpus and noisy-nf-10771 data.
    """
    # Load NFCorpus
    nfcorpus = load_dataset("mteb/nfcorpus", "corpus", split="corpus")
    nfcorpus = [{"_id": doc["_id"], "text": doc["text"]} for doc in nfcorpus]

    # Load noisy dataset
    noisy_ds = load_dataset("cpondoc/noisy-nf-10771", split="train", ignore_verifications=True)

    def clean_metadata(example):
        lines = example["text"].splitlines()
        lines = [
            line for line in lines
            if not line.startswith(("URL:", "TOPIC SET:", "TOPIC:", "SIMILARITY:"))
        ]
        example["text"] = "\n".join(lines).strip()
        return example

    noisy_ds = noisy_ds.map(clean_metadata)
    noisy_docs = [{"_id": f"noisy_{i}", "text": doc["text"]} for i, doc in enumerate(noisy_ds)]

    return nfcorpus + noisy_docs

def save_annotations(corpus, output_path="annotations.csv"):
    """
    Annotate documents and append each row to CSV immediately.
    """
    file_exists = os.path.exists(output_path)

    with open(output_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["_id", "score"])

        for doc in tqdm(corpus, desc="Annotating corpus"):
            score = get_score(doc["text"])
            writer.writerow([doc["_id"], score])

    print(f"Annotations saved to {output_path}")

if __name__ == "__main__":
    corpus = load_data()
    save_annotations(corpus)
