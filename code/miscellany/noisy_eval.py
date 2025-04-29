"""
Separate script for running model on noisy benchmark
"""

import torch
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define model and task
BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"
TASK = "NFCorpus"

# Move model to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dual_encoder = SentenceTransformer(BASE_MODEL).to(device)

# Run evaluation for task
evaluation = MTEB(tasks=[TASK], verbosity=1)
evaluation.run(
    dual_encoder,
    eval_splits=["test"],
    save_predictions=True,
    output_folder=f"noisy_results/no-rerank/{TASK}/6000-examples/{BASE_MODEL}/{TASK}/",
)
