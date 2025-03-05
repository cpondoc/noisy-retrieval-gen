# Noisy Retrieval Benchmark Generation
Trying to look into how to create retrieval benchmarks that are noisy...

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

## To Update with Noisy Dataset
In `AbsTaskRetrieval.py`, edit `get_noisy_docs` to use the correct Hugging Face dataset.
