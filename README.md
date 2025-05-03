# Sifting through the Noise: Navigating Noisy Information Retrieval Settings
Work exploring how to add lightweight heuristics for quality in the face of noisy data.

## Set-Up
First, create virtual environment and install requirements:
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

You should then install `mteb` locally, to adjust the noisy dataset:
```sh
pip install mteb/
```

Finally, you can run scripts for evaluation, analysis, etc.
```sh
python3 noisy_eval.py
```

## To Run

### Distillation

First, to run the LLM annotation code, run:
```bash
modal run code/distillation/llama_annotation.py::run_annotation
```

Then, to train the model, run:
```bash
modal run code/distillation/train_classifier.py
```

Finally, to upload the model weights, run:
```bash
modal run code/distillation/model_upload.py::upload_model_to_hf
```

### Evaluation

Run the main script:
```bash
python3 code/eval/rerank.py
```