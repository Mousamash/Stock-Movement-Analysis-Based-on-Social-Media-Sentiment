# Stock Movement Analysis Based on Social Media Sentiment

## Overview

This project aims to develop a machine learning model that predicts stock movements by analyzing sentiment from social media platforms, specifically Reddit. The model scrapes data from relevant subreddits, performs sentiment analysis using BERT, and incorporates technical indicators to enhance prediction accuracy.

### Key Features

- **Data Scraping**: Collects posts from Reddit subreddits focused on stock discussions.
- **Sentiment Analysis**: Utilizes BERT (FinBERT) for advanced sentiment analysis of the scraped text.
- **Technical Analysis**: Integrates technical indicators such as RSI, MACD, and moving averages.
- **Price Prediction**: Predicts stock price movements based on sentiment and technical indicators.
- **Results Export**: Saves predictions, actual prices, and model metrics to an Excel file.

## Requirements

To run this project, you need the following Python packages:

- `pandas`
- `numpy`
- `yfinance`
- `praw`
- `transformers`
- `torch`
- `scikit-learn`
- `openpyxl`
- `ta` (Technical Analysis Library)
- `python-dotenv`
- `textblob`

You can install the required packages using the following command:
pip install -r requirements.txt


## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd stock-movement-analysis
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the root directory of the project and add your Reddit API credentials:
   ```plaintext
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

## Usage

1. **Run the Main Script**:
   Execute the main script to start the analysis:
   ```bash
   python main.py
   ```

2. **View Results**:
   After execution, the results will be saved in an Excel file located at `data/final/stock_predictions.xlsx`. This file will contain:
   - Stock symbol
   - Actual price
   - Predicted price movement
   - Predicted price
   - Model accuracy
   - Precision
   - Recall



## Code Structure

- **`main.py`**: The main script that orchestrates data scraping, sentiment analysis, and price prediction.
- **`requirements.txt`**: Lists all the required Python packages for the project.
- **`src/data_scraping/reddit_scraper.py`**: Contains functions for scraping data from Reddit.
- **`src/data_processing/clean_data.py`**: Handles data cleaning and preprocessing.
- **`src/sentiment_analysis/bert_sentiment.py`**: Implements sentiment analysis using BERT.
- **`src/model/train_model.py`**: Contains functions for training the prediction model.
- **`src/utils/helpers.py`**: Utility functions for directory management and other helper tasks.

  ## Final OutPut
![capx](https://github.com/user-attachments/assets/1619e34d-551c-422f-9047-739c9a33ea33)

## Future Improvements

- **Real-time Data**: Implement real-time data fetching and analysis.
- **Additional Data Sources**: Integrate data from Twitter and financial news articles.
- **Advanced Models**: Experiment with more complex models like LSTM for time-series forecasting.
- **User Interface**: Develop a more interactive web interface for easier interaction with the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BERT](https://github.com/google-research/bert) for its powerful language understanding capabilities.
- [yfinance](https://pypi.org/project/yfinance/) for easy access to financial data.
- [PRAW](https://praw.readthedocs.io/en/latest/) for accessing Reddit's API.
- [Technical Analysis Library](https://github.com/bukosabino/ta) for calculating technical indicators.
