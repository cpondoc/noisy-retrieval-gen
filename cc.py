"""
Sift through Common Crawl and find data that pertains to multiple sets of topics
"""

from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import os
import gzip
import requests
from warcio.archiveiterator import ArchiveIterator
import trafilatura
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Path to the file containing WARC URLs
WARC_LIST_PATH = "data/common-crawl/warcs.txt"
WARC_DOWNLOAD_DIR = "data/common-crawl/zip/"  # Directory for downloaded WARC files

# Path to the files containing different topic sets with separate base directories
TOPIC_SETS = [
    {
        "name": "fiqa",
        "path": "data/FiQA-2018/topics/general.txt",
        "articles_dir": "data/FiQA-2018/articles/",  # Separate base directory for set1
    },
    {
        "name": "scidocs",
        "path": "data/SCIDOCS/topics/general.txt",
        "articles_dir": "data/SCIDOCS/articles/",  # Separate base directory for set2
    },
    # Add more topic sets as needed
]

# Minimum similarity threshold to consider an article as matching a topic
SIMILARITY_THRESHOLD = 0.4


def is_english(text, probability_threshold=0.9):
    """
    Checks if the provided text is in English.

    Args:
        text (str): The text to analyze
        probability_threshold (float): Minimum probability to consider text as English (0.0 to 1.0)

    Returns:
        tuple: (is_english (bool), probability (float), detected_language (str))
    """
    if not text or len(text.strip()) == 0:
        return False, 0.0, "unknown"

    try:
        # Get language probabilities
        language_probs = detect_langs(text)

        # Check if English is the top language
        if language_probs and language_probs[0].lang == "en":
            probability = language_probs[0].prob
            return probability >= probability_threshold, probability, "en"
        else:
            # Find English in the results if it exists
            for lang_prob in language_probs:
                if lang_prob.lang == "en":
                    return lang_prob.prob >= probability_threshold, lang_prob.prob, "en"

            # English not found, return the top language
            top_lang = language_probs[0].lang
            return False, 0.0, top_lang

    except LangDetectException as e:
        # Handle errors (like empty text)
        return False, 0.0, f"error: {str(e)}"


def read_topics(filepath: str):
    """Read topics from a file and return as a list."""
    if not os.path.exists(filepath):
        print(f"❌ Error: Topics file not found: {filepath}")
        return []

    with open(filepath, "r") as file:
        topics = [line.strip() for line in file if line.strip()]

    print(f"📄 Loaded {len(topics)} topics from {filepath}")
    return topics


def create_topic_folders(topic_set_configs: list):
    """Create folders for each topic in each topic set using their separate base directories."""
    topic_folders = {}
    topic_sets = {}

    for config in topic_set_configs:
        set_name = config["name"]
        topics = read_topics(config["path"])
        base_dir = config["articles_dir"]

        if not topics:
            continue

        topic_sets[set_name] = topics
        topic_folders[set_name] = {}

        # Create base directory for this topic set
        os.makedirs(base_dir, exist_ok=True)
        print(f"📁 Created/verified base directory: {base_dir}")

        # Create a folder for each topic in this set
        for topic in topics:
            # Create a valid folder name
            folder_name = re.sub(r"[^\w\s-]", "", topic).replace(" ", "_").lower()
            folder_path = os.path.join(base_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            print(f"📁 Created/verified topic folder: {folder_path}")
            topic_folders[set_name][topic] = {
                "folder_name": folder_name,
                "base_dir": base_dir,
            }

    return topic_sets, topic_folders


def embed_topics(topic_sets: dict):
    """
    Embed topics from multiple sets using SentenceTransformer.
    Returns a mapping from set names to {topic: embedding} dictionaries.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    topic_embeddings = {}

    for set_name, topics in topic_sets.items():
        embeddings = model.encode(topics)
        topic_embeddings[set_name] = {
            topic: embedding.tolist() for topic, embedding in zip(topics, embeddings)
        }
        print(f"🧠 Embedded {len(topics)} topics for set '{set_name}'")

    return topic_embeddings


def ensure_directory_exists(filepath: str):
    """Ensure the directory for the given filepath exists."""
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)


def read_warc_list(filepath: str):
    """Read the list of WARC URLs from a file."""
    if not os.path.exists(filepath):
        print(f"❌ Error: WARC list file not found: {filepath}")
        return []

    with open(filepath, "r") as file:
        warc_urls = [line.strip() for line in file if line.strip()]

    print(f"📄 Loaded {len(warc_urls)} WARC URLs from {filepath}")
    return warc_urls


def download_warc(url: str):
    """Download a WARC file if it does not exist."""
    filename = os.path.join(WARC_DOWNLOAD_DIR, os.path.basename(url))

    if os.path.exists(filename):
        print(f"✅ File already exists: {filename}")
        return filename

    ensure_directory_exists(filename)  # Ensure the directory exists

    print(f"⬇️ Downloading {filename}...")
    response = requests.get(url, stream=True)
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            file.write(chunk)
    print(f"✅ Download complete: {filename}")
    return filename


def compute_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity using NumPy."""
    # Convert embeddings to NumPy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Compute the dot product and magnitudes
    dot_product = np.dot(embedding1, embedding2)
    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)

    # Return the cosine similarity
    return dot_product / (magnitude1 * magnitude2)


def create_safe_filename(url: str):
    """Create a safe filename from URL."""
    # Remove protocol and special characters
    safe_name = re.sub(r"^https?://", "", url)
    safe_name = re.sub(r"[^\w\s-]", "_", safe_name)
    # Limit filename length
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    return safe_name + ".txt"


def find_best_topic_match(article_embedding, topic_embeddings_set):
    """Find the best matching topic in a topic set."""
    best_similarity = -1
    best_topic = None

    for topic, topic_embedding in topic_embeddings_set.items():
        sim = compute_cosine_similarity(article_embedding, topic_embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_topic = topic

    return best_topic, best_similarity


def extract_warc(
    filename: str, topic_sets: dict, topic_embeddings: dict, topic_folders: dict
):
    """Extract URLs and raw text content from the WARC file and match against multiple topic sets."""
    print(f"\n📖 Extracting content from {filename}...\n")

    # Initialize the model for article embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    with gzip.open(filename, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == "response":
                url = record.rec_headers.get_header("WARC-Target-URI")
                html_content = record.content_stream().read().decode(errors="ignore")

                # Extract raw text using trafilatura, check for English
                raw_text = trafilatura.extract(html_content)
                if not raw_text or not is_english(raw_text)[0]:
                    continue

                # Embed the article's raw text
                article_embedding = model.encode([raw_text])[0]

                # Check the article against each topic set
                matches_found = False

                for set_name, topic_embeddings_set in topic_embeddings.items():
                    best_topic, best_similarity = find_best_topic_match(
                        article_embedding, topic_embeddings_set
                    )

                    # Save if similarity is above threshold
                    if best_similarity >= SIMILARITY_THRESHOLD:
                        matches_found = True
                        print(f"🌐 URL: {url}")
                        print(
                            f"📝 Extracted Text:\n{raw_text[:300]}..."
                        )  # First 300 chars
                        print(
                            f"🏆 Set: {set_name}, Best Topic: {best_topic} (Similarity: {best_similarity:.4f})"
                        )

                        # Get topic folder info
                        topic_info = topic_folders[set_name][best_topic]
                        base_dir = topic_info["base_dir"]
                        folder_name = topic_info["folder_name"]

                        # Create the file path in the appropriate directory
                        file_name = create_safe_filename(url)
                        file_path = os.path.join(base_dir, folder_name, file_name)

                        # Only save if the file doesn't exist
                        if not os.path.exists(file_path):
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(f"URL: {url}\n\n")
                                f.write(f"TOPIC SET: {set_name}\n")
                                f.write(f"TOPIC: {best_topic}\n")
                                f.write(f"SIMILARITY: {best_similarity:.4f}\n\n")
                                f.write(raw_text)
                            print(f"💾 Saved article to {file_path}")
                        else:
                            print(f"⏭️ File already exists: {file_path}")

                if matches_found:
                    print("-" * 80)

    # Delete the .gz file after extraction
    os.remove(filename)
    print(f"🗑️ Deleted {filename} after extraction.")


if __name__ == "__main__":
    # Create directories
    os.makedirs(WARC_DOWNLOAD_DIR, exist_ok=True)

    # Create folder structure for topics and get topic sets
    topic_sets, topic_folders = create_topic_folders(TOPIC_SETS)

    if not topic_sets:
        print("❌ No valid topic sets found. Exiting.")
        exit(1)

    # Embed all topics from all sets
    topic_embeddings = embed_topics(topic_sets)

    # Read in URLs, iterate and check
    warc_urls = read_warc_list(WARC_LIST_PATH)
    for warc_url in warc_urls[15:]:
        warc_file = download_warc(f"https://data.commoncrawl.org/{warc_url}")
        extract_warc(warc_file, topic_sets, topic_embeddings, topic_folders)
