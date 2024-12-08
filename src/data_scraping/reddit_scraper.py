import os
import praw
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def authenticate_reddit():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

def scrape_reddit_data(search_terms, limit=100):
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    
    reddit = authenticate_reddit()
    subreddits = ['stocks', 'investing', 'wallstreetbets']
    all_posts = []
    
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            for term in search_terms:
                try:
                    posts = subreddit.search(term, limit=limit//len(search_terms))
                    
                    for post in posts:
                        all_posts.append({
                            "title": post.title,
                            "selftext": post.selftext,
                            "subreddit": subreddit_name,
                            "search_term": term,
                            "score": post.score,
                            "num_comments": post.num_comments,
                            "upvote_ratio": post.upvote_ratio,
                            "created_utc": post.created_utc
                        })
                    
                    print(f"Successfully scraped data from r/{subreddit_name} for term '{term}'")
                    
                except Exception as e:
                    print(f"Error searching for term '{term}' in r/{subreddit_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error accessing r/{subreddit_name}: {e}")
            continue
            
    if not all_posts:
        print(f"No data found for search terms: {search_terms}")
        return pd.DataFrame()  # Return empty DataFrame instead of raising exception
        
    df = pd.DataFrame(all_posts)
    
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    
    df.to_csv('data/raw/reddit_data.csv', index=False)
    return df

# Example usage
if __name__ == "__main__":
    data = scrape_reddit_data("AAPL")
    print(f"Total posts collected: {len(data)}") 