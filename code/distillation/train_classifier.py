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
app = App(name="bert-nfcorpus-k-fold")
volume = Volume.from_name("noisy-nfcorpus-k-fold", create_if_missing=True)
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
TARGET_COLUMN = "score"
CACHE_DIR = "/vol/cache"
ANNOTATION_PATH = "/vol/nfcorpus-k-fold-training.csv"
BASE_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"
CHECKPOINT_DIR = "/vol/checkpoints/nfcorpus-k-fold"

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
    import os
    import pandas as pd
    from datasets import load_dataset, Dataset
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
        AutoModelForSequenceClassification,
    )

    # === Load annotations first ===
    # Load CSV without assuming anything weird about index
    annotations_df = pd.read_csv(ANNOTATION_PATH, index_col=False)

    # Print a few rows to debug
    print(annotations_df.head())
    print("Columns:", annotations_df.columns.tolist())

    # Ensure the column exists
    assert TARGET_COLUMN in annotations_df.columns, f"Missing column: {TARGET_COLUMN}"

    # Drop any bad rows, then cast
    annotations_df = annotations_df[pd.to_numeric(annotations_df[TARGET_COLUMN], errors="coerce").notnull()]
    annotations_df[TARGET_COLUMN] = annotations_df[TARGET_COLUMN].astype(float).clip(0, 5)
    # annotations_df = pd.read_csv(ANNOTATION_PATH, header=0)
    # annotations_df[TARGET_COLUMN] = annotations_df[TARGET_COLUMN].astype(float).clip(0, 5)
    annotation_ids = set(annotations_df["_id"].astype(str))

    # === Load NFCorpus documents filtered by annotation IDs ===
    def load_corpus(annotation_ids):
        mteb_ds = load_dataset("BeIR/nfcorpus", "corpus", cache_dir=CACHE_DIR)["corpus"]
        return [
            {"_id": doc["_id"], "text": doc["text"]}
            for doc in mteb_ds
            if doc["_id"] in annotation_ids
        ]

    # === Load and clean noisy documents filtered by annotation IDs ===
    def load_noisy_dataset(annotation_ids, subset_size=None):
        print(f"Loading in {str(subset_size) if subset_size else 'full dataset'}!")
        dataset = load_dataset("cpondoc/noisy-nf-10771", keep_in_memory=True)

        def remove_metadata(example):
            lines = example["text"].splitlines()
            lines = [
                line for line in lines
                if not line.startswith(("URL:", "TOPIC SET:", "TOPIC:", "SIMILARITY:"))
            ]
            cleaned_text = "\n".join(lines).strip()
            example["text"] = cleaned_text
            return example

        dataset = dataset.map(remove_metadata)
        train_data = dataset["train"]

        if subset_size:
            train_data = train_data.select(range(min(subset_size, len(train_data))))

        processed_data = [
            {"_id": f"doc_{i}", "text": train_data["text"][i]}
            for i in range(len(train_data["text"]))
        ]

        return [doc for doc in processed_data if doc["_id"] in annotation_ids]
    covid_dataset = load_corpus(annotation_ids) + load_noisy_dataset(annotation_ids)

    # === Merge annotations with text ===
    covid_df = pd.DataFrame(covid_dataset)
    covid_merged = covid_df.merge(annotations_df, on="_id", how="inner")
    covid_merged = covid_merged.rename(columns={"score": TARGET_COLUMN})

    # === Convert to HuggingFace dataset ===
    dataset = Dataset.from_pandas(covid_merged)

    # === Train/test split ===
    dataset = dataset.train_test_split(train_size=0.9, seed=42)

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

    # === Tokenization function ===
    def preprocess(examples):
        texts = [str(t) for t in examples["text"]]
        batch = tokenizer(texts, truncation=True, padding=True)
        batch["labels"] = [float(score) for score in examples[TARGET_COLUMN]]
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # === Training arguments ===
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
        metric_for_best_model="mse",
        greater_is_better=False,
        bf16=True,
    )

    # === Train model ===
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
