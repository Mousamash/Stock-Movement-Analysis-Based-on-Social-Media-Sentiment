import re
import pandas as pd
from src.utils.helpers import ensure_directory_exists
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"\s+", " ", text)     # Remove extra whitespace
    return text.strip()

def process_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df['cleaned_text'] = df['title'].fillna('') + " " + df['selftext'].fillna('')
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    
    # Ensure the directory exists
    ensure_directory_exists(os.path.dirname(output_file))
    
    df.to_csv(output_file, index=False)

# Example usage
if __name__ == "__main__":
    process_data('data/raw/reddit_data.csv', 'data/processed/cleaned_data.csv') 