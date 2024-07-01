# 
import praw
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from client_id import client_id, client_secret

nltk.download('vader_lexicon')


reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent="tickerhungary",

)
sia = SentimentIntensityAnalyzer()

sentiments = {"positive": 0, "neutral": 0, "negative": 0}

def get_sentiment_vader(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

def process_submission(submission_id):
    try:
        submission = reddit.submission(id=submission_id)
        local_sentiments = {"positive": 0, "neutral": 0, "negative": 0, "title_count": 0, "com_count": 0}
        
        if 'NVDA' in submission.title.upper():
            local_sentiments["title_count"] += 1
            sentiment = get_sentiment_vader(submission.title)
            local_sentiments[sentiment] += 1
        
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            if 'NVDA' in comment.body.upper():
                local_sentiments["com_count"] += 1
                sentiment = get_sentiment_vader(comment.body)
                local_sentiments[sentiment] += 1
        
        return local_sentiments
    except Exception as e:
        logging.error(f"Error processing submission {submission_id}: {e}")
        return None

def main():
    submission_ids = [submission.id for submission in reddit.subreddit("wallstreetbets").new(limit=10)]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # futures dictionary mapping Future objects to submission IDs
        futures = {executor.submit(process_submission, submission_id): submission_id for submission_id in submission_ids}
        
        for future in as_completed(futures):
            result = future.result()  # This is correct; we get the result of the future here
            if result:
                sentiments["positive"] += result["positive"]
                sentiments["neutral"] += result["neutral"]
                sentiments["negative"] += result["negative"]
            else:
                logging.info(f"Skipping result for submission {futures[future]} due to error.")
    
    print(f"Titles mentioning 'NVDA': {sum(result['title_count'] for result in [future.result() for future in futures if future.result()])}")
    print(f"Comments mentioning 'NVDA': {sum(result['com_count'] for result in [future.result() for future in futures if future.result()])}")
    print(f"Sentiment counts: {sentiments}")


if __name__ == "__main__":
    main()