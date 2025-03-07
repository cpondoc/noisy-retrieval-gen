import matplotlib.pyplot as plt
import numpy as np

# Data from your input
examples = [0, 250, 500, 1116, 2000, 3000, 4000, 5000, 5663]

# Original data (Vanilla Embedding Model)
accuracy_vanilla = [
    0.36746,
    0.35765,
    0.35047,
    0.33236,
    0.32092,
    0.31016,
    0.30518,
    0.29965,
    0.29676,
]

# New data (FineWeb-Edu Reranking)
accuracy_fineweb = [
    0.34746,
    0.34426,
    0.34326,
    0.3365,
    0.32874,
    0.32265,
    0.3179,
    0.31389,
    0.31222,
]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot original data
plt.plot(
    examples,
    accuracy_vanilla,
    marker="o",
    linestyle="-",
    color="blue",
    linewidth=2,
    markersize=8,
    label="Vanilla Embedding Model"
)

# Plot new data
plt.plot(
    examples,
    accuracy_fineweb,
    marker="s",
    linestyle="-",
    color="green",
    linewidth=2,
    markersize=8,
    label="FineWeb-Edu Reranking"
)

# Add title and labels with increased font size
plt.title("NFCorpus Accuracy with Noisy Examples", fontsize=16, fontweight="bold")
plt.xlabel("Number of Noisy Examples", fontsize=14)
plt.ylabel("NFCorpus Accuracy", fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend
plt.legend(fontsize=12)

# Customize tick parameters
plt.tick_params(axis="both", which="major", labelsize=12)

# Add data points annotation for both series
for i, (x, y1, y2) in enumerate(zip(examples, accuracy_vanilla, accuracy_fineweb)):
    plt.annotate(
        f"{y1:.5f}",
        (x, y1),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        color="blue"
    )
    plt.annotate(
        f"{y2:.5f}",
        (x, y2),
        textcoords="offset points",
        xytext=(0, -20),
        ha="center",
        fontsize=9,
        color="green"
    )

# Set appropriate y-axis limits to better visualize both series
plt.ylim(0.29, 0.38)

# Save the figure
plt.tight_layout()
plt.savefig("NFCorpus/comparison.png", dpi=300)

# Show the plot
plt.show()