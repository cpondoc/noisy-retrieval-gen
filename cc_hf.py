import os
import glob
import random
from datasets import Dataset
import pandas as pd
from collections import Counter


def process_text_files(base_dir, sample_size=None):
    """
    Process text files in the directory structure and optionally sample them
    while maintaining the original topic distribution.

    Args:
        base_dir: Base directory containing topic subdirectories
        sample_size: Number of articles to sample (if None, returns all articles)

    Returns:
        A list of dictionaries with article_id, topic and text fields
    """
    all_data = []
    topic_counts = {}

    # Get all topic directories
    topic_dirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    # First collect all articles to determine distribution
    for topic in topic_dirs:
        topic_path = os.path.join(base_dir, topic)

        # Find all text files in the topic directory
        text_files = glob.glob(os.path.join(topic_path, "*.txt"))
        topic_counts[topic] = len(text_files)

        for file_path in text_files:
            # Extract filename without extension as article_id
            article_id = os.path.basename(file_path).split(".")[0]

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

                # Skip the first 4 lines and join the rest
                if len(lines) > 4:
                    content = "".join(lines[4:])
                else:
                    content = ""

            # Add to our dataset
            all_data.append({"article_id": article_id, "topic": topic, "text": content})

    # If no sample size specified or sample size >= total articles, return all data
    if sample_size is None or sample_size >= len(all_data):
        return all_data

    # Perform stratified sampling
    sampled_data = stratified_sample(all_data, topic_counts, sample_size)

    return sampled_data


def stratified_sample(data, topic_counts, sample_size):
    """
    Perform stratified sampling to maintain the original topic distribution.

    Args:
        data: List of article dictionaries
        topic_counts: Dictionary with topic counts
        sample_size: Number of articles to sample

    Returns:
        A list of sampled article dictionaries
    """
    total_articles = sum(topic_counts.values())

    # Calculate how many articles to sample from each topic
    topic_samples = {}
    for topic, count in topic_counts.items():
        # Calculate the proportion of this topic in the original dataset
        proportion = count / total_articles
        # Calculate how many articles to sample from this topic
        topic_samples[topic] = max(1, round(sample_size * proportion))

    # Adjust sample sizes to match the exact requested sample_size
    # This is needed because of rounding
    total_sampled = sum(topic_samples.values())

    if total_sampled != sample_size:
        # Find difference
        diff = sample_size - total_sampled

        # Distribute the difference among topics proportionally
        topics_sorted = sorted(
            topic_counts.keys(),
            key=lambda t: topic_counts[t] / total_articles,
            reverse=(diff > 0),
        )

        for topic in topics_sorted:
            if diff == 0:
                break
            elif diff > 0:
                # Need to add samples
                if topic_samples[topic] < topic_counts[topic]:
                    topic_samples[topic] += 1
                    diff -= 1
            else:
                # Need to remove samples
                if topic_samples[topic] > 1:
                    topic_samples[topic] -= 1
                    diff += 1

    # Group articles by topic
    articles_by_topic = {}
    for article in data:
        topic = article["topic"]
        if topic not in articles_by_topic:
            articles_by_topic[topic] = []
        articles_by_topic[topic].append(article)

    # Sample from each topic
    sampled_data = []
    for topic, num_to_sample in topic_samples.items():
        # Ensure we don't try to sample more than available
        num_to_sample = min(num_to_sample, len(articles_by_topic[topic]))
        # Random sampling without replacement
        sampled_data.extend(random.sample(articles_by_topic[topic], num_to_sample))

    return sampled_data


def main():
    # Base directory path
    base_dir = "data/common-crawl/articles"

    # Number of articles to sample
    sample_size = 5000

    # Process and sample text files
    data = process_text_files(base_dir)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # Print topic distribution
    topic_distribution = df["topic"].value_counts()
    print("Topic Distribution:")
    for topic, count in topic_distribution.items():
        print(f"  {topic}: {count} articles ({count/len(df)*100:.2f}%)")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Save to Hugging Face
    dataset_name = "cpondoc/noisy-nf-35"
    dataset.push_to_hub(dataset_name)

    print(
        f"\nSuccessfully processed {len(data)} articles across {len(set(df['topic']))} topics."
    )
    print(f"Dataset uploaded to Hugging Face as {dataset_name}")


if __name__ == "__main__":
    main()
