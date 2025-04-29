# zip_all_outputs.py

import modal
import shutil
import os

app = modal.App("zip-entire-mteb")

volume = modal.Volume.from_name("mteb-outputs", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/outputs": volume},
    timeout=60,
)
def zip_entire_volume():
    source_dir = "/outputs/modal_test/NFCorpus/gpt2"
    zip_path = "/outputs/gpt2.zip"

    # Avoid zipping the zip file itself (if re-running)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', source_dir)
    print(f"✅ Zipped entire volume to: {zip_path}")
