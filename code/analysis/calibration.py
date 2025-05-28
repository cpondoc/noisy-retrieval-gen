
import os
import json
import matplotlib.pyplot as plt

root_dir = "calibration/analysis/NFCorpus/gpt2"
results = {}

print(f"Scanning directory: {root_dir}")

# Traverse the directory tree
for prob in os.listdir(root_dir):
    prob_path = os.path.join(root_dir, prob)
    if not os.path.isdir(prob_path):
        continue

    results[prob] = []

    for noisy_size in os.listdir(prob_path):
        num_noisy_size = int(noisy_size)
        if 0 < num_noisy_size < 1000:
            continue

        model_base = os.path.join(prob_path, noisy_size, "Snowflake__snowflake-arctic-embed-m")
        if not os.path.isdir(model_base):
            continue

        # Get the subfolder inside Snowflake (assume just one)
        subdirs = [d for d in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, d))]
        if not subdirs:
            continue

        model_hash_dir = os.path.join(model_base, subdirs[0])
        json_path = os.path.join(model_hash_dir, "NFCorpus.json")

        if not os.path.isfile(json_path):
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                main_score = data["scores"]["test"][0]["main_score"]
                results[prob].append((int(noisy_size), main_score))
                print(f"[✓] {prob} / {noisy_size} → {main_score}")
        except Exception as e:
            print(f"Failed to read {json_path}: {e}")

# Plotting and Saving
if not results:
    print("No results found. Exiting.")
else:
    plt.figure(figsize=(10, 6))
    for prob, values in results.items():
        if not values:
            continue
        values.sort()
        x = [v[0] for v in values]
        y = [v[1] for v in values]
        plt.plot(x, y, marker='o', label=f"Prob {prob}")

    plt.xlabel("Noisy Data Size")
    plt.ylabel("Main Score")
    plt.title("Main Score vs Noisy Data Size (by Probability)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    save_path = "data/analysis/NFCorpus/calibration-gpt2-modified-perplexity.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")


