# train_classifier_modal.py

import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from modal import App, gpu, method, Image, Volume

# === Modal Setup ===
app = App(name="bert-nfcorpus-classifier")
volume = Volume.from_name("noisy-nfcorpus-vol", create_if_missing=True)
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
DATASET_NAME = "mteb/nfcorpus"
TARGET_COLUMN = "score"
CACHE_DIR = "/vol/cache"
ANNOTATION_PATH = "/vol/code/distillation/annotations.csv"
BASE_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"
CHECKPOINT_DIR = "/vol/checkpoints/bert_snowflake_regression"
OUTPUT_MODEL_NAME = "HuggingFaceTB/fineweb-edu-scorer"

def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    preds = np.round(logits).astype(int).clip(0, 5)
    return {"f1_macro": f1_score(labels, preds, average="macro")}

@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={"/vol": volume},
    timeout=60 * 60 * 4,
)
def train_classifier():
    # === Load dataset ===
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)
    full_data = load_dataset(DATASET_NAME, "corpus", split="corpus")
    corpus_df = pd.DataFrame(full_data)

    # === Load annotations ===
    annotations_df = pd.read_csv(ANNOTATION_PATH)
    annotations_df[TARGET_COLUMN] = annotations_df[TARGET_COLUMN].clip(0, 5).astype(int)

    # === Merge annotations into dataset ===
    dataset_df = pd.DataFrame(dataset)
    merged_df = dataset_df.merge(annotations_df, left_on="corpus-id", right_on="_id", how="inner")
    merged_df = merged_df.drop(columns=["score_x", "_id"])
    merged_df = merged_df.rename(columns={"score_y": "score"})

    # === Merge text from corpus ===
    merged_df = merged_df.merge(corpus_df[["_id", "text"]], left_on="corpus-id", right_on="_id", how="left")
    merged_df = merged_df.drop(columns=["_id"])

    # === Convert to HuggingFace dataset ===
    dataset = Dataset.from_pandas(merged_df)
    dataset = dataset.map(lambda x: {TARGET_COLUMN: np.clip(int(x[TARGET_COLUMN]), 0, 5)}, num_proc=8)
    dataset = dataset.cast_column(TARGET_COLUMN, ClassLabel(names=[str(i) for i in range(6)]))
    dataset = dataset.train_test_split(train_size=0.9, seed=42, stratify_by_column=TARGET_COLUMN)

    # === Load model and tokenizer ===
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=1,
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

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[TARGET_COLUMN])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # === Training arguments and Trainer setup ===
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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(CHECKPOINT_DIR, "final"))

if __name__ == "__main__":
    app.run()
