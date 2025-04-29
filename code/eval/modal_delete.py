import modal
import shutil
import os

app = modal.App("delete-specific-folder")

output_volume = modal.Volume.from_name("mteb-outputs", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/outputs": output_volume},
    timeout=60,
)
def delete_specific_folder():
    import os
    path = "/outputs/gpt2.zip"
    if os.path.exists(path):
        os.remove(path)
        print("Deleted gpt2.zip")
    else:
        print("File not found")
