print("1")
from mteb import MTEB
print("2")
from sentence_transformers import SentenceTransformer
print("3")

# Define model and tasks
BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"
TASKS = ["ArguAna", "FiQA2018", "QuoraRetrieval"]

print("Getting model...")
dual_encoder = SentenceTransformer(BASE_MODEL)
print("Retrieving model....")

# Run evaluation for each task
for task in TASKS:
    evaluation = MTEB(tasks=[task],  verbosity=1)
    evaluation.run(
        dual_encoder,
        eval_splits=["test"],
        save_predictions=True,
        output_folder=f"results/{BASE_MODEL}/{task}/",
    )
