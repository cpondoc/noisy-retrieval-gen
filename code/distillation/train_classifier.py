# train_classifier_modal.py

import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from modal import App, gpu, method, Image, Volume

# === Modal Setup ===
app = App(name="bert-covid-regression-half")
volume = Volume.from_name("noisy-covid-regression-half", create_if_missing=True)
image = (
    Image.debian_slim()
    .pip_install(
        "transformers>=4.39.0",
        "datasets",
        "evaluate",
        "scikit-learn",
        "pandas",
        "numpy",
        "accelerate",
        "bitsandbytes",
    )
    .add_local_python_source("_remote_module_non_scriptable")
)

# === Constants ===
DATASET_NAME = "BeIR/trec-covid"
TARGET_COLUMN = "score"
CACHE_DIR = "/vol/cache"
ANNOTATION_PATH = "/vol/half-trec-covid.csv"
BASE_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"
CHECKPOINT_DIR = "/vol/checkpoints/half-trec-covid-noise"

# === Metrics for Regression ===
def compute_metrics(eval_pred):
    from sklearn.metrics import mean_squared_error, r2_score
    preds, labels = eval_pred
    preds = preds.flatten()
    return {
        "mse": mean_squared_error(labels, preds),
        "r2": r2_score(labels, preds),
    }

@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={"/vol": volume},
    timeout=60 * 60 * 4,
)
def train_classifier():
    # === Load TREC-COVID corpus ===
    def load_corpus():
        mteb_ds = load_dataset("BeIR/trec-covid", "corpus", cache_dir=CACHE_DIR)["corpus"]
        quarter_len = len(mteb_ds) // 2
        mteb_ds = mteb_ds.select(range(quarter_len))
        return [{"_id": doc["_id"], "text": doc["text"]} for doc in mteb_ds]

    covid_dataset = load_corpus()

    # === Load annotations ===
    annotations_df = pd.read_csv(ANNOTATION_PATH)
    annotations_df[TARGET_COLUMN] = annotations_df[TARGET_COLUMN].clip(0, 5)

    # === Merge annotations with text ===
    covid_df = pd.DataFrame(covid_dataset)
    covid_merged = covid_df.merge(annotations_df, on="_id", how="inner")
    covid_merged = covid_merged.rename(columns={"score": TARGET_COLUMN})

    # === Convert to HuggingFace dataset ===
    dataset = Dataset.from_pandas(covid_merged)

    # === Split into train/test sets ===
    dataset = dataset.train_test_split(train_size=0.9, seed=42)

    # === Load model and tokenizer ===
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=1,  # ← Regression
        classifier_dropout=0.0,
        hidden_dropout_prob=0.0,
        output_hidden_states=False,
    )
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # === Preprocessing function ===
    def preprocess(examples):
        texts = [str(t) for t in examples["text"]]
        batch = tokenizer(texts, truncation=True, padding=True)
        batch["labels"] = [float(score) for score in examples[TARGET_COLUMN]]
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # === Training Arguments ===
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        learning_rate=3e-4,
        num_train_epochs=20,
        seed=0,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=64,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="mse",  # ← Use regression metric
        greater_is_better=False,      # ← Lower MSE is better
        bf16=True,
    )

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # === Train and Save ===
    trainer.train()
    trainer.save_model(os.path.join(CHECKPOINT_DIR, "final"))

if __name__ == "__main__":
    app.run()
