import os
import re
import matplotlib.pyplot as plt

def extract_similarity(text):
    # Regex to find SIMILARITY: float_number
    match = re.search(r"SIMILARITY:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None

def process_folder(folder_path):
    similarities = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                sim = extract_similarity(content)
                if sim is not None:
                    similarities.append(sim)

    if not similarities:
        print("No similarity scores found.")
        return

    # Plot histogram
    plt.hist(similarities, bins=20, edgecolor='black')
    plt.title("Histogram of Similarity Scores")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("similarity_histogram.png")
    plt.show()

    # Print average
    avg_sim = sum(similarities) / len(similarities)
    print(f"Average similarity: {avg_sim:.4f}")

# Replace with your actual folder path
if __name__ == "__main__":
    folder = "data/analysis/SCIDOCS/6258"
    process_folder(folder)
