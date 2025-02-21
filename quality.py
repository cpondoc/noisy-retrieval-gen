import numpy as np
import os
import fasttext

def fasttext_english_filter(content: str):
    """
    Using similar Fineweb techniques to just filter out noise.
    """
    # Define pre-trained fasttext model
    model_path = "models/lid.176.bin"
    model = fasttext.load_model(model_path)

    # Inference, and return (can tune 0.6 probability)
    content = content.replace('\n', ' ')
    label, probability = model.predict(content, k=1)
    
    # Convert probability to numpy array using np.asarray() instead of np.array() with copy=False
    probability = np.asarray(probability)
    
    return {"label": label, "probability": probability}

def process_articles(base_folder="articles"):
    """
    Reads articles from folders inside the 'articles' directory,
    passes each through the fasttext_english_filter, and retrieves probabilities.
    """
    articles_with_probs = {}

    # Traverse through each folder within the "articles" directory
    for topic_folder in os.listdir(base_folder):
        topic_folder_path = os.path.join(base_folder, topic_folder)
        
        # Check if it's a directory
        if os.path.isdir(topic_folder_path):
            topic_probs = []
            
            # Read all text files inside the topic folder
            for filename in os.listdir(topic_folder_path):
                if filename.endswith(".txt"):
                    article_path = os.path.join(topic_folder_path, filename)
                    
                    # Open and read the article
                    with open(article_path, "r", encoding="utf-8") as file:
                        content = file.read()

                    # Pass the article content through the filtering function
                    probs = fasttext_english_filter(content)
                    topic_probs.append({filename: probs})

            # Store the results for this topic
            articles_with_probs[topic_folder] = topic_probs

    return articles_with_probs

if __name__ == "__main__":
    # Run the function
    articles_probs = process_articles()

    # Fix for accessing the probabilities:
    for topic, article_probs in articles_probs.items():
        print(f"\nTopic: {topic}")
        for article_dict in article_probs:
            for article, probs in article_dict.items():
                print(f"Article: {article}, Probabilities: {probs}")
