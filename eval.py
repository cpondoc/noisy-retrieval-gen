"""
Separate script for running model on non-edited benchmark
"""
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define model and task
BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"
TASK = "FiQA2018"

dual_encoder = SentenceTransformer(BASE_MODEL)

# Run evaluation for task
evaluation = MTEB(tasks=[TASK], verbosity=1)
evaluation.run(
    dual_encoder,
    eval_splits=["test"],
    save_predictions=True,
    output_folder=f"results/{BASE_MODEL}/{TASK}/",
)
