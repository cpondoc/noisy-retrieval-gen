import matplotlib.pyplot as plt
import numpy as np

# Full list of example counts
examples = [0, 250, 500, 1116, 2000, 3000, 4000, 5000, 5663, 6000, 7000, 8000, 9000, 10000, 10771]

# Vanilla Embedding Model data
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
    0.30014,
    0.29653,
    0.29055,
    0.28916,
    0.28715,
    0.28472,
]

# FineWeb-Edu Reranking with a=0.99 data
accuracy_fineweb_099 = [
    0.34746,
    0.34426,
    0.34326,
    0.3365,
    0.32874,
    0.32265,
    0.3179,
    0.31389,
    0.31222,
    0.31224,
    0.30984,
    0.30778,
    0.30514,
    0.30425,
    0.30255,
]

# FineWeb-Edu Reranking with a=0.999 data
accuracy_fineweb_0999 = [
    0.36775,
    0.35858,
    0.35469,
    0.33846,
    0.32566,
    0.31549,
    0.31045,
    0.30528,
    0.30215,
    0.30484,
    0.30092,
    0.29578,
    0.29429,
    0.29283,
    0.29084,
]

# New dataset
new_data_examples = [250, 500, 1116, 2000, 3000, 4000]
new_data_accuracy = [0.35723, 0.35085, 0.33265, 0.32108, 0.31172, 0.30606]

# Create the plot
plt.figure(figsize=(12, 7))

# Plot vanilla data
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

# Plot fineweb a=0.99 data
plt.plot(
    examples,
    accuracy_fineweb_099,
    marker="s",
    linestyle="-",
    color="green",
    linewidth=2,
    markersize=8,
    label="FineWeb-Edu Reranking (a=0.99)"
)

# Plot fineweb a=0.999 data
plt.plot(
    examples,
    accuracy_fineweb_0999,
    marker="^",
    linestyle="-",
    color="red",
    linewidth=2,
    markersize=8,
    label="FineWeb-Edu Reranking (a=0.999)"
)

# Plot new data
plt.plot(
    new_data_examples,
    new_data_accuracy,
    marker="d",
    linestyle="-",
    color="purple",
    linewidth=2,
    markersize=8,
    label="NVIDIA Domain Classifier (a=0.99)"
)

# Add title and labels with increased font size
plt.title("NFCorpus Accuracy with Noisy Examples", fontsize=16, fontweight="bold")
plt.xlabel("Number of Noisy Examples", fontsize=14)
plt.ylabel("NFCorpus Accuracy", fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend with updated labels
plt.legend(fontsize=12, loc="best", framealpha=0.9)

# Customize tick parameters
plt.tick_params(axis="both", which="major", labelsize=12)

# Set appropriate y-axis limits to better visualize all series
plt.ylim(0.28, 0.38)

# Save the figure
plt.tight_layout()
plt.savefig("figures/NFCorpus/comparison_all_models.png", dpi=300)

# Show the plot
plt.show()