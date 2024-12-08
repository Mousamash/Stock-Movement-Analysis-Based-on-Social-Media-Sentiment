import pandas as pd
from src.data_scraping.reddit_scraper import scrape_reddit_data
from src.data_processing.clean_data import process_data
from src.sentiment_analysis.bert_sentiment import BertSentimentAnalyzer
from src.model.train_model import train_model
from src.utils.helpers import ensure_directory_exists
from sklearn.metrics import precision_score, recall_score
import time
import yfinance as yf
import ta

# Initialize BERT analyzer
bert_analyzer = BertSentimentAnalyzer()

def get_current_price(ticker):
    try:
        stock_data = yf.download(ticker, period='1d', progress=False)
        if not stock_data.empty:
            current_price = float(stock_data['Close'].iloc[-1])
            return round(current_price, 2)
        else:
            print(f"No price data found for {ticker}")
            return 'N/A'
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return 'N/A'

def get_technical_indicators(stock):
    try:
        # Get historical data
        data = yf.download(stock, period="1mo")
        
        # Add technical indicators
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        
        # Get latest values
        latest = data.iloc[-1]
        
        return {
            'RSI': latest['RSI'],
            'MACD': latest['MACD'],
            'MA20': latest['MA20']
        }
    except Exception as e:
        print(f"Error calculating technical indicators for {stock}: {e}")
        return None

def calculate_predicted_price(current_price, movement, technical_data=None):
    try:
        if isinstance(current_price, str) and current_price.startswith('$'):
            current_price = float(current_price[1:])
        
        # Incorporate technical indicators if available
        if technical_data:
            rsi = technical_data['RSI']
            # Adjust movement based on RSI
            if rsi > 70:  # Overbought
                movement = max(0.3, movement - 0.2)
            elif rsi < 30:  # Oversold
                movement = min(0.7, movement + 0.2)
        
        if movement > 0.5:
            percent_increase = (movement - 0.5) * 10
            predicted_price = current_price * (1 + percent_increase/100)
        else:
            percent_decrease = (0.5 - movement) * 10
            predicted_price = current_price * (1 - percent_decrease/100)
            
        return f"${predicted_price:.2f}"
    except:
        return "N/A"

def main():
    # Expanded list of stock names with their common names
    stock_names = [
        "AAPL",      # Apple
        "TSLA",      # Tesla
        "GOOGL",     # Google
        "MSFT",      # Microsoft
        "AMZN",      # Amazon
        "META",      # Meta (Facebook)
        "NVDA",      # NVIDIA
        "AMD"        # AMD
    ]
    
    # Get real-time prices using yfinance
    actual_prices = {}
    print("\nFetching current stock prices...")
    for stock in stock_names:
        price = get_current_price(stock)
        actual_prices[stock] = price
        print(f"{stock}: ${price}")
    
    # DataFrame to store results
    results = []
    
    # Scrape data for each stock
    for stock in stock_names:
        print(f"\n{'='*50}")
        print(f"Processing data for {stock}...")
        print(f"{'='*50}")
        
        try:
            # Scrape Reddit data with expanded search terms
            search_terms = {
                "AAPL": ["AAPL", "Apple"],
                "TSLA": ["TSLA", "Tesla"],
                "GOOGL": ["GOOGL", "Google"],
                "MSFT": ["MSFT", "Microsoft"],
                "AMZN": ["AMZN", "Amazon"],
                "META": ["META", "Facebook"],
                "NVDA": ["NVDA", "NVIDIA"],
                "AMD": ["AMD"]
            }
            
            # Use both ticker and company name for searching
            data = scrape_reddit_data(search_terms.get(stock, [stock]))
            
            if data.empty:
                print(f"No Reddit data found for {stock}")
                continue
                
            print(f"Found {len(data)} posts for {stock}")
            
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            continue
        
        # Add a delay to avoid rate limiting
        time.sleep(5)
        
        try:
            # Clean data
            process_data('data/raw/reddit_data.csv', 'data/processed/cleaned_data.csv')
            
            # Load cleaned data
            df = pd.read_csv('data/processed/cleaned_data.csv')
            
            if df.empty:
                print(f"No cleaned data available for {stock}")
                continue
            
            # Analyze sentiment using BERT
            sentiments = []
            for text in df['cleaned_text']:
                try:
                    sentiment_value = bert_analyzer.analyze_sentiment(text)
                    sentiments.append(sentiment_value)
                except Exception as e:
                    print(f"Error analyzing sentiment for text: {e}")
                    sentiments.append(0)
            
            df['sentiment'] = sentiments
            
            # Get technical indicators
            technical_data = get_technical_indicators(stock)
            
            # Calculate sentiment scores (1 for positive, 0 for negative)
            df['label'] = [1 if s > 0 else 0 for s in df['sentiment']]
            
            # Train model and get predictions
            model = train_model(df[['sentiment']].values, df['label'].values)
            predictions = model.predict(df[['sentiment']].values)
            
            # Calculate metrics
            accuracy = model.score(df[['sentiment']].values, df['label'].values)
            precision = precision_score(df['label'], predictions)
            recall = recall_score(df['label'], predictions)
            
            # Calculate predicted price movement
            predicted_movement = sum(df['label']) / len(df['label'])
            
            # Get current price without the '$' symbol for calculation
            current_price = actual_prices[stock]
            
            # Calculate predicted price with technical indicators
            predicted_price = calculate_predicted_price(current_price, predicted_movement, technical_data)
            
            # Store the results in a dictionary
            result = {
                "Stock": stock,
                "Actual Price": f"${actual_prices[stock]}" if actual_prices[stock] != 'N/A' else 'N/A',
                "Predicted Price Movement": f"{predicted_movement:.3f}",
                "Predicted Price": predicted_price,
                "Model Accuracy": f"{accuracy:.3f}",
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}"
            }
            
            # Append the result to our results list
            results.append(result)
            
            print(f"Successfully processed {stock}")
            print(f"Current price: ${current_price}")
            print(f"Predicted price: {predicted_price}")
            print(f"Predicted price movement: {predicted_movement:.3f}")
            print(f"Model accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            
        except Exception as e:
            print(f"Error in processing pipeline for {stock}: {e}")
            print(f"Full error details: ", e.__class__.__name__)
            continue
    
    if not results:
        print("\nNo results were generated. Please check the error messages above.")
    else:
        try:
            # Convert results list to DataFrame
            results_df = pd.DataFrame(results)
            
            # Reorder columns
            results_df = results_df[[
                "Stock", 
                "Actual Price", 
                "Predicted Price Movement",
                "Predicted Price",
                "Model Accuracy",
                "Precision",
                "Recall"
            ]]
            
            # Save to Excel with formatting
            with pd.ExcelWriter('data/final/stock_predictions.xlsx', engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Predictions')
                
            print("\nResults saved to data/final/stock_predictions.xlsx")
            print("\nFinal Results:")
            print(results_df.to_string(index=False))
        except Exception as e:
            print(f"Error saving results to Excel: {e}")
            print("\nFinal Results:")
            print(pd.DataFrame(results))

if __name__ == "__main__":
    main()