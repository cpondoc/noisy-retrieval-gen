import modal

app = modal.App("covid-in-domain-reranker")

# Path to your local custom mteb
LOCAL_MTEB_DIR = "./mteb"

# Create or attach to a persisted volume for output storage
output_volume = modal.Volume.from_name("mteb-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("modal_requirements.txt")
    .env({
        "PYTHONPATH": "/mteb"
    })
    .add_local_dir(LOCAL_MTEB_DIR, "/mteb")
)

@app.function(
    gpu="A100",
    image=image,
    timeout=60 * 60 * 4,
    volumes={"/outputs": output_volume}
)
def run_reranker():
    import torch
    from mteb import MTEB
    import mteb
    from sentence_transformers import SentenceTransformer

    BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"
    QUALITY_MODELS = {
        "trec-covid-k-fold-classifier" : "trec-covid-k-fold-classifier",    
        "HuggingFaceTB/fineweb-edu-classifier": "fineweb",
        "gpt2": "gpt2",
        "nvidia/domain-classifier": "nvidia-dc",
    }

    TASKS = ["TRECCOVID"]
    tasks = mteb.get_tasks(tasks=TASKS, languages=["eng"])
    eval_splits = ["test"]

    for subset_size in [1442]:
        for quality_p in [0.999, 0.995, 0.99, 0.9]:
            for key, value in QUALITY_MODELS.items():
                print(f"Running task={TASKS[0]}, quality_p={quality_p}, subset_size={subset_size}")
                dual_encoder = SentenceTransformer(BASE_MODEL).to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

                evaluation = MTEB(tasks=TASKS)
                evaluation.run(
                    dual_encoder,
                    eval_splits=eval_splits,
                    save_predictions=True,
                    output_folder=f"/outputs/modal_test/{TASKS[0]}/{value}/{quality_p}/{subset_size}/",
                    subset_size=subset_size,
                    quality_p=quality_p,
                    quality_classifier=key,
                    quality_batch_size=16,
                )
