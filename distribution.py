import numpy as np
import matplotlib.pyplot as plt

# Load cosine similarity scores from a text file
cosine_similarities = np.loadtxt('nfcorpus_fineweb_edu.txt')
# cosine_similarities = np.clip(cosine_similarities, -1, 1)  # Ensure values are in valid range

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(cosine_similarities, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel("Fineweb Educational Quality Score")
plt.ylabel("Frequency")
plt.title("Fineweb Educational Quality Score Distribution of NFCorpus")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
