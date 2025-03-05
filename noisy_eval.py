from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define model and task
BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"
TASK = "NFCorpus"  # Set NFCorpus as the evaluation task

dual_encoder = SentenceTransformer(BASE_MODEL)

# Run evaluation for NFCorpus
evaluation = MTEB(tasks=[TASK], verbosity=1)
evaluation.run(
    dual_encoder,
    eval_splits=["test"],
    save_predictions=True,
    output_folder=f"noisy_results/noisy_cc/5663-examples/{BASE_MODEL}/{TASK}/",
)
