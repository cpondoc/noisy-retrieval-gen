"""
Iterate through all of the common crawl, check.
"""
import os
import gzip
import requests
from warcio.archiveiterator import ArchiveIterator
import trafilatura

# Path to the file containing WARC URLs
WARC_LIST_PATH = "data/common-crawl/warcs.txt"
WARC_DOWNLOAD_DIR = "data/common-crawl/zip/"  # Directory for downloaded WARC files

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

def extract_warc(filename: str):
    """Extract URLs and raw text content from the WARC file using trafilatura."""
    print(f"\n📖 Extracting content from {filename}...\n")

    with gzip.open(filename, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == "response":
                url = record.rec_headers.get_header("WARC-Target-URI")
                html_content = record.content_stream().read().decode(errors="ignore")

                # Extract raw text using trafilatura
                raw_text = trafilatura.extract(html_content)

                if raw_text:  # Ensure valid extraction
                    print(f"🌐 URL: {url}")
                    print(f"📝 Extracted Text:\n{raw_text[:500]}...")  # Print first 500 chars
                    print("-" * 80)

    # Delete the .gz file after extraction
    os.remove(filename)
    print(f"🗑️ Deleted {filename} after extraction.")

if __name__ == "__main__":
    warc_urls = read_warc_list(WARC_LIST_PATH)

    for warc_url in warc_urls:
        warc_file = download_warc(warc_url)
        extract_warc(warc_file)
