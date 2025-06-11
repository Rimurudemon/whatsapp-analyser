"""
Utility functions for WhatsApp Chat Analyzer
"""
import streamlit as st
import nltk

@st.cache_resource
def download_nltk_resources():
    """Download NLTK resources required for the application"""
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

def get_freq_label(time_freq):
    """Convert frequency code to human-readable label"""
    return {
        'D': 'Daily',
        'W': 'Weekly',
        'M': 'Monthly'
    }.get(time_freq, 'Daily')

def get_window_size(time_freq):
    """Get appropriate window size for rolling averages based on time frequency"""
    return {
        'D': 7,   # 7-day average for daily data
        'W': 4,   # 4-week average for weekly data
        'M': 3,   # 3-month average for monthly data
    }.get(time_freq, 7)
