from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Example usage
if __name__ == "__main__":
    text = "The stock market is doing great!"
    print(analyze_sentiment(text)) 