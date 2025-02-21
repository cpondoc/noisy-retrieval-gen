"""
Playing around with noisy benchmark creation
"""
import openai
import numpy as np
from datasets import load_dataset
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def calculate_text_stats(texts):
    # Get the length of each text
    lengths = np.array([len(text) for text in texts])

    # Calculate mean, median, and lengths using numpy
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)

    # Return the results
    return {
        "mean_length": mean_length,
        "median_length": median_length,
        "lengths": lengths,
        "total_texts": len(texts)
    }

def load_data():
    """
    Load in NFCorpus corpus, queries, and qfrels
    """
    # Load in corpus
    new_data = load_dataset(
        "mteb/nfcorpus",
        "corpus",
    )
    corpus = new_data["corpus"]

    docs = []
    for text in corpus["text"]:
        docs.append(text)

    # Load in queries
    new_data = load_dataset(
        "mteb/nfcorpus",
        "queries",
    )
    queries = new_data["queries"]

    # Load in qfrels
    last_data = load_dataset(
        "mteb/nfcorpus"
    )
    qfrels = last_data["test"]
    
    return corpus, queries, qfrels, docs

def write_queries():
    """
    Write to a separate text file
    """
    # Load in queries
    dataset = load_dataset("mteb/nfcorpus", "queries")
    queries = dataset["queries"]

    # Write queries to a text file
    with open("queries.txt", "w", encoding="utf-8") as file:
        for query in queries["text"]:  # Assuming queries have a "text" field
            file.write(query + "\n")

def chat_with_gpt(prompt, model="gpt-3.5-turbo", max_tokens=100):
    """
    ChatGPT wrapper baby
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=1.1,
    )
    return response.choices[0].message.content

def generate_text(topic, target_length, model="gpt-3.5-turbo"):
    """
    Generates text for a given topic, limiting the length based on mean/median.
    """
    max_tokens = target_length // 4  # Approximate token conversion
    prompt = f"Write an article, at a middle-school level, on the following topic: {topic}. Also make it a bit messy, which means it can include some irrelavnt information and some typos every now and then."
    return chat_with_gpt(prompt, model, max_tokens)

def generate_articles_for_topics(topics, num_articles=3, target_length=1500, model="gpt-3.5-turbo"):
    """
    Generates a specified number of articles for each topic and saves them in separate folders.
    """
    base_folder = "articles"

    # Ensure the base "articles" folder exists
    os.makedirs(base_folder, exist_ok=True)

    articles = {}
    
    for topic in tqdm(topics):
        # Create a folder for each topic, replacing spaces with underscores for safety
        topic_folder = os.path.join(base_folder, topic.replace(" ", "_"))
        os.makedirs(topic_folder, exist_ok=True)

        topic_articles = []
        
        for i in range(num_articles):
            article = generate_text(topic, target_length, model)

            # Save each article as a separate text file
            article_filename = os.path.join(topic_folder, f"article_{i+1}.txt")
            with open(article_filename, "w", encoding="utf-8") as f:
                f.write(article)

            topic_articles.append(article)
        
        articles[topic] = topic_articles

    return articles

def read_topics_into_n_strings(file_path, n):
    """
    Reads a text file and splits its contents into `n` separate strings.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]  # Remove empty lines

    # Split lines into N parts as evenly as possible
    split_strings = ["\n".join(lines[i::n]) for i in range(n)]
    
    return tuple(split_strings)  # Returns N separate strings

def cluster_macro_topics():
    """
    Return responses for macro topics
    """
    file_path = "queries.txt"  # Replace with your actual file path
    num_strings = 3  # Change this to however many separate strings you want
    topic_strings = read_topics_into_n_strings(file_path, num_strings)

    prompts = []
    for i, topic_string in enumerate(topic_strings, 1):
        prompts.append(f"""
Please take the following topics and cluster them into a list of 10 topics. These topics should be short and concise,
but not just limited to one or two words. Please return only the 10 topics, one on each line.

{topic_string}
""")

    responses = []
    for prompt in prompts:
        response = chat_with_gpt(prompt)
        responses.append(response)
    print(responses)

# Run only when executed directly
if __name__ == "__main__":
    """
    Make prompt then give.
    """
    # Get topics
    target_length = 1500
    topics = read_topics_into_n_strings("topics-pass-2.txt", 1)
    topics_list = topics[0].split("\n")

    # Generatic articles
    generate_articles_for_topics(topics_list)
