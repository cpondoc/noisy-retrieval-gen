import matplotlib.pyplot as plt

# Define the list of evals
evals = ["FiQA-2018", "NFCorpus", "SCIDOCS"]

plt.figure(figsize=(8, 6))

for eval in evals:
    file_path = f"data/{eval}/queries.txt"  # Replace with your actual file path
    
    word_counts = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            words = line.strip().split()
            word_counts.append(len(words))
    
    plt.hist(word_counts, bins=range(min(word_counts), max(word_counts) + 2), edgecolor='black', alpha=0.5, label=eval)

plt.xlabel("Number of Words per Line")
plt.ylabel("Frequency")
plt.title("Histogram of Word Counts per Query Across Evals")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the figure
plt.savefig("figures/prompts/all_evals.png", dpi=300)
plt.show()