"""
Sentiment analysis module for WhatsApp Chat Analyzer
"""
import re
import pandas as pd
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK resources
def download_nltk_resources():
    """Download necessary NLTK resources if not already present"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    
    try:
        nltk.data.find('corpora/movie_reviews')
    except LookupError:
        nltk.download('movie_reviews')
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def sentiment_analysis(text, method="textblob"):
    """
    Perform sentiment analysis on a given text using multiple methods
    
    Parameters:
    text (str): Text to analyze
    method (str): Method to use ('textblob', 'vader', or 'combined')
    
    Returns:
    float: Sentiment polarity score (-1 to 1)
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    # Clean the text by removing URLs, media references and emojis
    text = re.sub(r'https?://\S+|www\.\S+|<Media omitted>|image omitted|video omitted|audio omitted|document omitted', '', text)
    
    if method == "textblob":
        # TextBlob analysis
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    elif method == "vader":
        # VADER analysis
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores['compound']  # Compound score ranges from -1 to 1
    
    else:  # combined (default)
        # Use both and average the results (normalized to -1 to 1 scale)
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        analyzer = SentimentIntensityAnalyzer()
        vader_score = analyzer.polarity_scores(text)['compound']
        
        # Average the two scores, giving slightly more weight to VADER
        return (textblob_score + vader_score * 1.5) / 2.5

def categorize_sentiment(score):
    """
    Categorize sentiment scores into Positive, Negative, or Neutral
    
    Parameters:
    score (float): Sentiment score between -1 and 1
    
    Returns:
    str: Sentiment category
    """
    if score > 0.15:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def emotion_detection(text):
    """
    Detect emotion in text (basic implementation)
    
    Parameters:
    text (str): Text to analyze
    
    Returns:
    str: Detected emotion
    """
    if not isinstance(text, str) or not text.strip():
        return 'Neutral'
        
    # Define emotion keywords
    emotion_keywords = {
        'Joy': ['happy', 'joy', 'delighted', 'pleased', 'glad', 'cheerful', 'exciting', 'exciting', '😊', '😃', '😄', '❤️', '♥️'],
        'Anger': ['angry', 'furious', 'irritated', 'annoyed', 'mad', 'frustrate', 'rage', '😠', '😡', '🤬'],
        'Sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'heartbroken', 'disappointed', 'upset', '😢', '😭', '😔', '😞', '💔'],
        'Fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'horror', '😨', '😱', '😰'],
        'Surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'unexpected', 'wow', '😮', '😲', '😯']
    }
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count occurrences of emotion keywords
    emotion_counts = {}
    for emotion, keywords in emotion_keywords.items():
        count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        emotion_counts[emotion] = count
    
    # Get the dominant emotion (if any)
    max_emotion = max(emotion_counts.items(), key=lambda x: x[1])
    
    # If no emotion detected, use sentiment analysis for basic categorization
    if max_emotion[1] == 0:
        sentiment_score = sentiment_analysis(text)
        if sentiment_score > 0.2:
            return 'Joy'
        elif sentiment_score < -0.2:
            return 'Sadness'
        else:
            return 'Neutral'
    
    return max_emotion[0]

def analyze_sentiment(df, method="combined"):
    """
    Analyze sentiment in messages
    
    Parameters:
    df (DataFrame): DataFrame with messages
    method (str): Method to use for sentiment analysis
    
    Returns:
    dict: Dictionary with sentiment analysis results
    """
    if len(df) == 0:
        return {}
    
    # Add sentiment column
    df_copy = df.copy()
    df_copy['sentiment'] = df_copy['message'].apply(lambda x: sentiment_analysis(x, method))
    df_copy['sentiment_category'] = df_copy['sentiment'].apply(categorize_sentiment)
    df_copy['emotion'] = df_copy['message'].apply(emotion_detection)
    
    # Sentiment statistics
    sentiment_stats = {
        'average_sentiment': df_copy['sentiment'].mean(),
        'sentiment_distribution': df_copy.groupby('sentiment_category').size().to_dict(),
        'emotion_distribution': df_copy.groupby('emotion').size().to_dict()
    }
    
    # Sentiment by user
    user_sentiments = {}
    for user in df_copy['author'].unique():
        # Create a proper copy of the filtered data
        user_data = df_copy.loc[df_copy['author'] == user].copy()
        user_sentiments[user] = {
            'average': user_data['sentiment'].mean(),
            'positive': (user_data['sentiment_category'] == 'Positive').sum() / len(user_data) * 100,
            'negative': (user_data['sentiment_category'] == 'Negative').sum() / len(user_data) * 100,
            'neutral': (user_data['sentiment_category'] == 'Neutral').sum() / len(user_data) * 100,
            'most_common_emotion': user_data['emotion'].value_counts().idxmax()
        }
    
    sentiment_stats['user_sentiments'] = user_sentiments
    
    return sentiment_stats

def sentiment_time_frames(df, method="combined"):
    """
    Analyze sentiment time frames in messages
    
    Parameters:
    df (DataFrame): DataFrame with messages
    method (str): Method to use for sentiment analysis
    
    Returns:
    DataFrame: Sentiment trends over time
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate sentiment scores
    df_copy['sentiment'] = df_copy['message'].apply(lambda x: sentiment_analysis(x, method))
    df_copy['sentiment_category'] = df_copy['sentiment'].apply(categorize_sentiment)
    
    # Create a date column for grouping
    df_copy['date'] = df_copy['date'].dt.date
    
    # Group by date and sentiment category
    sentiment_over_time = df_copy.groupby(['date', 'sentiment_category']).size().unstack(fill_value=0)
    
    # Add a rolling average of sentiment
    df_copy['date_dt'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values('date_dt')
    df_copy['sentiment_rolling_avg'] = df_copy['sentiment'].rolling(window=20, min_periods=1).mean()
    
    # Group by date for average sentiment
    avg_sentiment = df_copy.groupby('date')['sentiment'].mean().reset_index()
    rolling_sentiment = df_copy.groupby('date')['sentiment_rolling_avg'].mean().reset_index()
    
    return {
        'category_counts': sentiment_over_time,
        'average_by_date': avg_sentiment,
        'rolling_average': rolling_sentiment
    }
