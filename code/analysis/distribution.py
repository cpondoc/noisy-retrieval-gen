import numpy as np
import matplotlib.pyplot as plt
import os

def plot_score_distribution(task: str, model: str, num_batches: int = 4):
    # Prepare paths
    base_path = f"data/distributions/{task}/{model}"
    
    # Load all batches
    scores = []
    for i in range(num_batches):
        file_path = os.path.join(base_path, f"quality_scores_batch_{i}.txt")
        batch_scores = np.loadtxt(file_path)
        scores.append(batch_scores)

    # Combine all scores
    scores = np.concatenate(scores)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel(f"Softmax of {model} Softmax Entropy Score")
    plt.ylabel("Frequency")
    plt.title(f"Softmax of {model} Softmax Entropy Score Distribution of {task}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save
    output_path = os.path.join(base_path, "quality.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

# Example usage
plot_score_distribution("TREC-COVID", "fineweb")
