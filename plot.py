import matplotlib.pyplot as plt
import numpy as np

# Data from your input
examples = [0, 250, 500, 1116, 2000, 3000, 4000, 5000, 5663]
accuracy = [
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

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(
    examples,
    accuracy,
    marker="o",
    linestyle="-",
    color="blue",
    linewidth=2,
    markersize=8,
)

# Add title and labels with increased font size
plt.title("NFCorpus Accuracy with Noisy Examples", fontsize=16, fontweight="bold")
plt.xlabel("Number of Noisy Examples", fontsize=14)
plt.ylabel("NFCorpus Accuracy", fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Customize tick parameters
plt.tick_params(axis="both", which="major", labelsize=12)

# Add data points annotation
for i, (x, y) in enumerate(zip(examples, accuracy)):
    plt.annotate(
        f"{y:.5f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=10,
    )

# Set appropriate y-axis limits to better visualize the drop
plt.ylim(0.29, 0.38)

# Show the plot
plt.tight_layout()
plt.show()
