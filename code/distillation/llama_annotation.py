# llama_annotation.py

import os
import csv
import re
import modal
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mitigate memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup Modal volume for output
output_volume = modal.Volume.from_name("llama-output", create_if_missing=True)

# Modal image and dependencies
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
)

# Define Modal app with volume + secret
app = modal.App(
    name="llama3-annotation",
    image=image,
    volumes={"/outputs": output_volume},
    secrets=[modal.Secret.from_name("hf-token")],
)

# Scoring prompt
base_prompt = """Below is an extract from a web page. Evaluate whether the page is noisy based on its relevance, coherence, and focus on the intended topic using the additive 5-point scoring system described below...

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

def load_corpus():
    from datasets import load_dataset

    # Load TREC-COVID corpus from BeIR
    mteb_ds = load_dataset("BeIR/trec-covid", "corpus")["corpus"]

    # Take only the first half of the dataset
    quarter_len = len(mteb_ds) // 2
    mteb_ds = mteb_ds.select(range(quarter_len))

    # Convert to list of {"_id", "text"}
    corpus = [{"_id": doc["_id"], "text": doc["text"]} for doc in mteb_ds]
    return corpus

@app.function(gpu="A100-80GB", timeout=60 * 60 * 24)
def run_annotation():
    """
    Run model distillation + annotation loop
    """
    from huggingface_hub import login

    HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
    if not HUGGINGFACE_TOKEN:
        raise RuntimeError("Missing HUGGINGFACE_TOKEN from Modal secret environment")

    login(token=HUGGINGFACE_TOKEN)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HUGGINGFACE_TOKEN,
        trust_remote_code=True,
        resume_download=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HUGGINGFACE_TOKEN,
        trust_remote_code=True,
        resume_download=True,
    )

    corpus = load_corpus()

    # Resume support
    output_path = "/outputs/trec-covid.csv"
    already_done = set()

    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            already_done = set(row["_id"] for row in reader)

    def safe_generate(batch_prompts, batch_size):
        while batch_size > 0:
            try:
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        temperature=0.7,
                        do_sample=False,
                    )
                return tokenizer.batch_decode(outputs, skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size //= 2
                print(f"⚠️ OOM – retrying with batch size {batch_size}")
        raise RuntimeError("OOM even with batch_size = 1")

    with open(output_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not already_done:
            writer.writerow(["_id", "score"])

        batch_size = 12
        for i in tqdm(range(0, len(corpus), batch_size), desc="Annotating corpus"):
            batch = [doc for doc in corpus[i:i+batch_size] if doc["_id"] not in already_done]
            if not batch:
                continue

            batch_prompts = [base_prompt.format(extract=doc["text"]) for doc in batch]
            outputs_text = safe_generate(batch_prompts, batch_size)

            for doc, output_text in zip(batch, outputs_text):
                match = re.search(r"Noise score:\s*(\d+)", output_text)
                score = int(match.group(1)) if match else 0
                writer.writerow([doc["_id"], score])
                already_done.add(doc["_id"])

            f.flush()

    print(f"Annotations saved to {output_path}")
