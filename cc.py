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
ARTICLES_DIR = "data/common-crawl/articles/"  # Base directory for saved articles

# Path to the file containing topics
TOPICS_FILE_PATH = "data/topics/manual-topics.txt"


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
        if language_probs and language_probs[0].lang == 'en':
            probability = language_probs[0].prob
            return probability >= probability_threshold, probability, 'en'
        else:
            # Find English in the results if it exists
            for lang_prob in language_probs:
                if lang_prob.lang == 'en':
                    return lang_prob.prob >= probability_threshold, lang_prob.prob, 'en'
            
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


def create_topic_folders(topics: list):
    """Create a folder for each topic in the articles directory."""
    for topic in topics:
        # Create a valid folder name (replace spaces with underscores, remove special characters)
        folder_name = re.sub(r'[^\w\s-]', '', topic).replace(' ', '_').lower()
        folder_path = os.path.join(ARTICLES_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 Created/verified topic folder: {folder_path}")
    
    return {topic: re.sub(r'[^\w\s-]', '', topic).replace(' ', '_').lower() for topic in topics}


def embed_topics(topics: list):
    """
    Embed a bunch of topics using SentenceTransformer and return a mapping from
    topics to their respective embeddings.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    embeddings = model.encode(topics)

    return {topic: embedding.tolist() for topic, embedding in zip(topics, embeddings)}


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
    safe_name = re.sub(r'^https?://', '', url)
    safe_name = re.sub(r'[^\w\s-]', '_', safe_name)
    # Limit filename length
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    return safe_name + ".txt"


def extract_warc(filename: str, topics: list, topic_embeddings: dict, topic_folders: dict):
    """Extract URLs and raw text content from the WARC file using trafilatura."""
    print(f"\n📖 Extracting content from {filename}...\n")

    # Initialize the model for article embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    with gzip.open(filename, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == "response":
                url = record.rec_headers.get_header("WARC-Target-URI")
                html_content = record.content_stream().read().decode(errors="ignore")

                # Extract raw text using trafilatura, check for english
                raw_text = trafilatura.extract(html_content)
                if raw_text and is_english(
                    raw_text
                )[0]:
                    # Embed the article's raw text
                    article_embedding = model.encode([raw_text])[0]

                    # Find the most similar topic based on cosine similarity
                    best_similarity = -1  # Initialize with a low similarity
                    best_topic = None

                    for topic, topic_embedding in topic_embeddings.items():
                        sim = compute_cosine_similarity(
                            article_embedding, topic_embedding
                        )
                        if sim > best_similarity:
                            best_similarity = sim
                            best_topic = topic
                    
                    # Check for best similarity:
                    if best_similarity >= 0.4:
                        print(f"🌐 URL: {url}")
                        print(f"📝 Extracted Text:\n{raw_text[:500]}...")  # Print first 500 chars
                        print(
                            f"🏆 Best Topic: {best_topic} (Similarity: {best_similarity:.4f})"
                        )
                        
                        # Save the content to the appropriate topic folder
                        folder_name = topic_folders[best_topic]
                        file_name = create_safe_filename(url)
                        file_path = os.path.join(ARTICLES_DIR, folder_name, file_name)
                        
                        # Only save if the file doesn't exist
                        if not os.path.exists(file_path):
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(f"URL: {url}\n\n")
                                f.write(f"SIMILARITY: {best_similarity:.4f}\n\n")
                                f.write(raw_text)
                            print(f"💾 Saved article to {file_path}")
                        else:
                            print(f"⏭️ File already exists: {file_path}")
                        
                        print("-" * 80)

    # Delete the .gz file after extraction
    os.remove(filename)
    print(f"🗑️ Deleted {filename} after extraction.")


if __name__ == "__main__":
    """
    Embed all of the topics, then return
    """
    # Read topics
    topics = read_topics(TOPICS_FILE_PATH)
    
    # Create a folder for each topic
    topic_folders = create_topic_folders(topics)
    
    # Embed topics
    topic_embeddings = embed_topics(topics)

    # Read in URLs, iterate and check
    warc_urls = read_warc_list(WARC_LIST_PATH)
    for warc_url in warc_urls[9:]:
        warc_file = download_warc(f"https://data.commoncrawl.org/{warc_url}")
        extract_warc(warc_file, topics, topic_embeddings, topic_folders)