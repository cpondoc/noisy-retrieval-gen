import os
import glob
from datasets import Dataset
import pandas as pd

def process_text_files(base_dir):
    """
    Process all text files in the directory structure.
    
    Args:
        base_dir: Base directory containing topic subdirectories
        
    Returns:
        A list of dictionaries with topic and content fields
    """
    all_data = []
    
    # Get all topic directories
    topic_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for topic in topic_dirs:
        topic_path = os.path.join(base_dir, topic)
        
        # Find all text files in the topic directory
        text_files = glob.glob(os.path.join(topic_path, "*.txt"))
        
        for file_path in text_files:
            # Extract filename without extension as article_id
            article_id = os.path.basename(file_path).split('.')[0]
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Skip the first 4 lines and join the rest
                if len(lines) > 4:
                    content = ''.join(lines[4:])
                else:
                    content = ''
            
            # Add to our dataset
            all_data.append({
                'article_id': article_id,
                'topic': topic,
                'text': content
            })
    
    return all_data

def main():
    # Base directory path
    base_dir = "data/common-crawl/articles"
    
    # Process all text files
    data = process_text_files(base_dir)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Save to Hugging Face
    dataset.push_to_hub("cpondoc/noisy-cc-227")
    
    print(f"Successfully processed {len(data)} articles across {len(set(df['topic']))} topics.")
    print(f"Dataset uploaded to Hugging Face as 'cpondoc/noisy-cc-227'")

if __name__ == "__main__":
    main()