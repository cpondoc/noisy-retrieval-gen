import os
from modal import App, Volume, Image, Secret

# === Modal setup ===
volume = Volume.from_name("noisy-trec-covid-k-fold", create_if_missing=False)

image = (
    Image.debian_slim()
    .pip_install("huggingface_hub")
)

app = App(
    name="hf-model-uploader",
    image=image,
    volumes={"/vol": volume},
    secrets=[Secret.from_name("hf-token")],  # pulls HF_TOKEN from Modal secret
)

REPO_ID = "cpondoc/trec-covid-k-fold-classifier"
FOLDER_PATH = "/vol/checkpoints/trec-covid-k-fold/final"

@app.function()
def upload_model_to_hf():
    from huggingface_hub import login, upload_folder, create_repo
    import os

    token = os.environ["HF_TOKEN"]
    login(token=token, add_to_git_credential=False)

    # ✅ Create the Hugging Face repo if it doesn't exist
    create_repo(
        repo_id=REPO_ID,
        exist_ok=True,
        private=False  # Set to True if you want it private
    )

    # ✅ Upload the model
    upload_folder(
        repo_id=REPO_ID,
        folder_path=FOLDER_PATH,
        path_in_repo="",
        commit_message="Upload model from Modal",
    )

if __name__ == "__main__":
    app.run()
