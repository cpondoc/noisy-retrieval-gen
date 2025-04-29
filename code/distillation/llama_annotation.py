# llama_annotation.py

import os
import csv
import re
import modal
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv

# Load local .env
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Setup Modal volume for output
output_volume = modal.Volume.from_name("llama-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "huggingface_hub",
        "python-dotenv",
        "accelerate"
    )
    .env({"HUGGINGFACE_TOKEN": HUGGINGFACE_TOKEN})
)

# App with volume mount
app = modal.App(
    name="llama3-annotation",
    image=image,
    volumes={"/outputs": output_volume},
)

# Static prompt
base_prompt = """
Below is an extract from a web page. Evaluate whether the page is noisy based on its relevance, coherence, and focus on the intended topic using the additive 5-point scoring system described below...

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

@app.function(gpu="A100", timeout=60 * 60 * 24)
def run_annotation():
    # Authenticate to Hugging Face
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Llama 3 8B Instruct
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ["HUGGINGFACE_TOKEN"],
        trust_remote_code=True,
        resume_download=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ["HUGGINGFACE_TOKEN"],
        trust_remote_code=True,
        resume_download=True,
    )

    # Load datasets
    nfcorpus = load_dataset("mteb/nfcorpus", "corpus", split="corpus")
    nfcorpus = [{"_id": doc["_id"], "text": doc["text"]} for doc in nfcorpus]

    noisy_ds = load_dataset("cpondoc/noisy-nf-10771", split="train")

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

    corpus = nfcorpus + noisy_docs

    # Save output to persistent volume
    output_path = "/outputs/annotations.csv"
    file_exists = os.path.exists(output_path)

    with open(output_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["_id", "score"])

        batch_size = 8  # ⚡ Batch 8 samples at a time
        for i in tqdm(range(0, len(corpus), batch_size), desc="Annotating corpus"):
            batch = corpus[i:i+batch_size]

            batch_prompts = [base_prompt.format(extract=doc["text"]) for doc in batch]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=False,
            )

            outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for doc, output_text in zip(batch, outputs_text):
                match = re.search(r"Noise score:\s*(\d+)", output_text)
                score = int(match.group(1)) if match else 0

                writer.writerow([doc["_id"], score])

            f.flush()  # flush after each batch for safety

    print(f"Annotations saved to {output_path}")
