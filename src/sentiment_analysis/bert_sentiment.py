from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BertSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        
    def analyze_sentiment(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert prediction to sentiment score (-1 to 1)
            sentiment_score = predictions[0].tolist()
            # Assuming FinBERT output is [negative, neutral, positive]
            return sentiment_score[2] - sentiment_score[0]  # positive - negative
        except Exception as e:
            print(f"Error in BERT sentiment analysis: {e}")
            return 0  # neutral sentiment as fallback 