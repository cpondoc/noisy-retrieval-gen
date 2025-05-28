"""
Script to use the Fineweb classifier as a pure reranker.
"""
import torch
from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer

# Define reranker, set of base models that we can change
BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"
QUALITY_MODELS = {
    "HuggingFaceTB/fineweb-edu-classifier": "fineweb",
    "gpt2": "gpt2",
    "nvidia/domain-classifier": "nvidia-dc",
    # "nfcorpus-trained-4-30": "nfcorpus-trained-4-30",
    "nfcorpus-all-noise-classifier": "nfcorpus-all-noise-classifier",
    # "nfcorpus-k-fold-classifier" : "nfcorpus-k-fold-classifier", 
}

# Also define the set of tasks
TASKS = ["NFCorpus"]
tasks = mteb.get_tasks(tasks=TASKS, languages=["eng"])

# Iterate through each model, set up the initial encoder
for subset_size in [10771]:
    for quality_p in [0, 0.995, 0.99, 0.999, 0.9]:
        for key, value in QUALITY_MODELS.items():
            dual_encoder = SentenceTransformer(BASE_MODEL).to('cuda' if torch.cuda.is_available() else 'cpu')
            eval_splits = ["test"]

            # Run eval by first doing the results, then reranking
            for task in TASKS:
                evaluation = MTEB(tasks=[task])
                evaluation.run(
                    dual_encoder,
                    eval_splits=eval_splits,
                    save_predictions=True,
                    output_folder=f"k-fold/{task}/{value}/{str(quality_p)}/{str(subset_size)}/",
                    subset_size=subset_size,
                    quality_p=quality_p,
                    quality_classifier=key,
                    classifier_normalization="softmax_entropy",
                    quality_batch_size=8,
                )