"""
Enhanced sentiment analysis module for WhatsApp Chat Analyzer with Hinglish support
"""
import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Try to import optional dependencies
HINGLISH_SUPPORT = False
TRANSFORMER_SUPPORT = False

try:
    from indic_transliteration.sanscript import transliterate, SCHEMES
    from indic_translator import Translator
    translator = Translator()
    HINGLISH_SUPPORT = True
except ImportError:
    print("Warning: indic_transliteration or indic_translator not available. Hinglish support will be limited.")
    # Define dummy functions
    def transliterate(text, from_scheme, to_scheme):
        return text
    
    class DummyTranslator:
        def translate(self, text, src_lang, tgt_lang):
            return text
    
    translator = DummyTranslator()
    SCHEMES = type('obj', (object,), {
        'IAST': None,
        'DEVANAGARI': None
    })

try:
    from transformers import pipeline
    sentiment_pipeline = None  # We'll initialize this lazily to save memory
    TRANSFORMER_SUPPORT = True
except ImportError:
    print("Warning: transformers not available. Transformer-based sentiment analysis will not be available.")
    # Define dummy functions
    def pipeline(*args, **kwargs):
        return None
    
    sentiment_pipeline = None

# Common Hindi words for language detection
HINDI_MARKERS = set([
    'hai', 'tha', 'thi', 'hain', 'kya', 'main', 'mein', 'ko', 'ki', 'ka', 'ke',
    'aur', 'par', 'mera', 'tera', 'apna', 'yeh', 'woh', 'accha', 'nahi', 'nahin',
    'kyun', 'kaise', 'kahan', 'jab', 'tab', 'abhi', 'kabhi', 'saath', 'baat',
    'kuch', 'bahut', 'thoda', 'jyada', 'kam', 'zyada', 'lekin', 'phir', 'sirf',
    'bas', 'hi', 'toh', 'haan', 'na', 'ya', 'raha', 'rahi', 'karna', 'karenge',
    'karoge', 'bhai', 'yaar', 'dost', 'ghar', 'aaj', 'kal', 'subah', 'shaam', 
    'raat', 'din', 'samay', 'waqt', 'jaldi', 'der', 'pehle', 'baad'
])

def initialize_sentiment_model():
    """Initialize the Hugging Face sentiment pipeline (lazy loading)"""
    global sentiment_pipeline
    if sentiment_pipeline is None and TRANSFORMER_SUPPORT:
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Error initializing sentiment model: {e}")
    return sentiment_pipeline

def detect_language(text):
    """
    Detect if text is likely Hinglish based on presence of Hindi markers
    
    Parameters:
    text (str): Text to analyze
    
    Returns:
    str: 'hindi' if text contains Hindi markers, 'english' otherwise
    """
    if not isinstance(text, str) or not text.strip():
        return 'english'
    
    # Clean text and convert to lowercase
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned_text.split()
    
    # Check if any words are in the Hindi markers set
    for word in words:
        if word in HINDI_MARKERS:
            return 'hindi'
    
    return 'english'

def translate_hinglish_to_english(text):
    """
    Translate Hinglish (Romanized Hindi) to English
    
    Parameters:
    text (str): Hinglish text to translate
    
    Returns:
    str: Translated English text
    """
    if not HINGLISH_SUPPORT:
        return text
        
    try:
        # Transliterate from Roman script to Devanagari
        devanagari = transliterate(text, SCHEMES.IAST, SCHEMES.DEVANAGARI)
        
        # Translate from Hindi to English
        english = translator.translate(devanagari, src_lang="hi", tgt_lang="en")
        return english
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def process_message(text, use_transformer=False):
    """
    Process a single message for sentiment analysis, with Hinglish support
    
    Parameters:
    text (str): Text message to process
    use_transformer (bool): Whether to use the Hugging Face transformer model
    
    Returns:
    dict: Dictionary containing original text, processed text, sentiment, and score
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'original': text,
            'processed': text,
            'sentiment': 'NEUTRAL',
            'score': 0.0
        }

    # Detect language
    lang = detect_language(text)
    processed_text = text
    
    # Translate if Hinglish
    if lang == 'hindi':
        processed_text = translate_hinglish_to_english(text)
    
    # Clean the text by removing URLs, media references
    processed_text = re.sub(r'https?://\S+|www\.\S+|<Media omitted>|image omitted|video omitted|audio omitted|document omitted', '', processed_text)
    
    # Perform sentiment analysis
    if use_transformer and TRANSFORMER_SUPPORT:
        # Use Hugging Face transformer for sentiment analysis
        model = initialize_sentiment_model()
        try:
            if model:
                result = model(processed_text[:512])  # Limit text size for transformer
                sentiment = result[0]['label']
                score = result[0]['score']
                
                # Map LABEL_0/LABEL_1 to standard format
                if sentiment == 'LABEL_0':
                    sentiment = 'NEGATIVE'
                    score = 1 - score  # Invert score for consistency
                else:
                    sentiment = 'POSITIVE'
            else:
                # Fallback to VADER if model initialization failed
                analyzer = SentimentIntensityAnalyzer()
                scores = analyzer.polarity_scores(processed_text)
                score = scores['compound']
                sentiment = 'POSITIVE' if score > 0.05 else 'NEGATIVE' if score < -0.05 else 'NEUTRAL'
        except Exception as e:
            print(f"Transformer error: {e}")
            # Fallback to VADER
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(processed_text)
            score = scores['compound']
            sentiment = 'POSITIVE' if score > 0.05 else 'NEGATIVE' if score < -0.05 else 'NEUTRAL'
    else:
        # Use VADER for sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(processed_text)
        score = scores['compound']
        sentiment = 'POSITIVE' if score > 0.05 else 'NEGATIVE' if score < -0.05 else 'NEUTRAL'
    
    return {
        'original': text,
        'processed': processed_text,
        'sentiment': sentiment,
        'score': score
    }

def process_batch(batch, use_transformer=False):
    """Process a batch of messages"""
    return [process_message(msg, use_transformer) for msg in batch]

def analyze_sentiment_enhanced(df, method="vader", batch_size=100, use_multiprocessing=True):
    """
    Analyze sentiment in messages with Hinglish support
    
    Parameters:
    df (DataFrame): DataFrame with messages
    method (str): Method to use ('vader', 'transformer', or 'combined')
    batch_size (int): Size of batches for parallel processing
    use_multiprocessing (bool): Whether to use multiprocessing for large datasets
    
    Returns:
    dict: Dictionary with sentiment analysis results and DataFrame with sentiment data
    """
    if len(df) == 0:
        return {}
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Determine whether to use transformer based on method
    use_transformer = (method == "transformer" or method == "combined")
    
    # For small datasets, process directly
    if len(df) < batch_size or not use_multiprocessing:
        results = []
        # Show progress bar
        for _, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="Analyzing sentiment"):
            result = process_message(row['message'], use_transformer=use_transformer)
            results.append(result)
    else:
        # For larger datasets, use parallel processing with batches
        messages = df_copy['message'].tolist()
        batches = [messages[i:i + batch_size] for i in range(0, len(messages), batch_size)]
        
        # Using serial processing instead of parallel to avoid pickling issues
        results = []
        for batch in tqdm(batches, desc="Processing message batches"):
            batch_result = process_batch(batch, use_transformer)
            results.extend(batch_result)
    
    # Add results to DataFrame
    df_copy['processed_text'] = [r['processed'] for r in results]
    df_copy['sentiment_score'] = [r['score'] for r in results]
    df_copy['sentiment_label'] = [r['sentiment'] for r in results]
    
    # Normalize scores to -1 to 1 range if using transformer
    if use_transformer:
        # Transform POSITIVE/NEGATIVE scores to -1 to 1 range
        df_copy['sentiment_score'] = df_copy.apply(
            lambda row: row['sentiment_score'] if row['sentiment_label'] == 'POSITIVE' else -row['sentiment_score'],
            axis=1
        )
    
    # Categorize sentiment
    df_copy['sentiment_category'] = df_copy['sentiment_score'].apply(categorize_sentiment)
    
    # Add emotion detection
    df_copy['emotion'] = df_copy['processed_text'].apply(emotion_detection)
    
    # Calculate overall statistics
    sentiment_stats = {
        'average_sentiment': df_copy['sentiment_score'].mean(),
        'sentiment_distribution': df_copy.groupby('sentiment_category').size().to_dict(),
        'emotion_distribution': df_copy.groupby('emotion').size().to_dict(),
        'language_distribution': {
            'english': sum(detect_language(msg) == 'english' for msg in df_copy['message']),
            'hinglish': sum(detect_language(msg) == 'hindi' for msg in df_copy['message'])
        }
    }
    
    # Calculate sentiment by user
    user_sentiments = {}
    for user in df_copy['author'].unique():
        user_data = df_copy.loc[df_copy['author'] == user].copy()
        user_sentiments[user] = {
            'average': user_data['sentiment_score'].mean(),
            'positive': (user_data['sentiment_category'] == 'Positive').sum() / len(user_data) * 100,
            'negative': (user_data['sentiment_category'] == 'Negative').sum() / len(user_data) * 100,
            'neutral': (user_data['sentiment_category'] == 'Neutral').sum() / len(user_data) * 100,
            'most_common_emotion': user_data['emotion'].value_counts().idxmax(),
            'hinglish_percentage': sum(detect_language(msg) == 'hindi' for msg in user_data['message']) / len(user_data) * 100
        }
    
    sentiment_stats['user_sentiments'] = user_sentiments
    
    return sentiment_stats, df_copy

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
        'Joy': ['happy', 'joy', 'delighted', 'pleased', 'glad', 'cheerful', 'exciting', 'exciting', 
                '😊', '😃', '😄', '❤️', '♥️', 'love', 'awesome', 'amazing', 'wonderful', 'great'],
        'Anger': ['angry', 'furious', 'irritated', 'annoyed', 'mad', 'frustrate', 'rage', 
                  '😠', '😡', '🤬', 'hate', 'idiot', 'stupid', 'terrible'],
        'Sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'heartbroken', 'disappointed', 'upset', 
                    '😢', '😭', '😔', '😞', '💔', 'sorry', 'unfortunate', 'regret', 'miss'],
        'Fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'horror', 
                '😨', '😱', '😰', 'scary', 'frightened', 'panic', 'dread'],
        'Surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'unexpected', 'wow', 
                    '😮', '😲', '😯', 'sudden', 'unbelievable', 'incredible']
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
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        sentiment_score = scores['compound']
        
        if sentiment_score > 0.2:
            return 'Joy'
        elif sentiment_score < -0.2:
            return 'Sadness'
        else:
            return 'Neutral'
    
    return max_emotion[0]

def sentiment_time_frames_enhanced(df, method="vader"):
    """
    Analyze sentiment time frames in messages with Hinglish support
    
    Parameters:
    df (DataFrame): DataFrame with messages and sentiment analysis
    method (str): Method used for sentiment analysis
    
    Returns:
    dict: Sentiment trends over time
    """
    # Check if sentiment analysis has been run on the dataframe
    if 'sentiment_score' not in df.columns:
        # Run sentiment analysis if not already done
        sentiment_stats, df = analyze_sentiment_enhanced(df, method=method)
    else:
        df_copy = df.copy()
    
    # Create a date column for grouping
    df_copy = df.copy()
    df_copy['date'] = df_copy['date'].dt.date
    
    # Rename sentiment_score to sentiment for UI compatibility
    if 'sentiment_score' in df_copy.columns and 'sentiment' not in df_copy.columns:
        df_copy['sentiment'] = df_copy['sentiment_score']
    
    # Group by date and sentiment category
    sentiment_over_time = df_copy.groupby(['date', 'sentiment_category']).size().unstack(fill_value=0)
    
    # Add a rolling average of sentiment
    df_copy['date_dt'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values('date_dt')
    df_copy['sentiment_rolling_avg'] = df_copy['sentiment_score'].rolling(window=20, min_periods=1).mean()
    
    # Group by date for average sentiment
    # Use 'sentiment' column for UI compatibility
    avg_sentiment = df_copy.groupby('date')['sentiment'].mean().reset_index()
    rolling_sentiment = df_copy.groupby('date')['sentiment_rolling_avg'].mean().reset_index()
    
    # Language usage over time
    df_copy['is_hinglish'] = df_copy['message'].apply(lambda x: 1 if detect_language(x) == 'hindi' else 0)
    language_usage = df_copy.groupby('date')['is_hinglish'].mean().reset_index()
    language_usage['is_hinglish'] = language_usage['is_hinglish'] * 100  # Convert to percentage
    
    return {
        'category_counts': sentiment_over_time,
        'average_by_date': avg_sentiment,
        'rolling_average': rolling_sentiment,
        'language_usage': language_usage
    }
