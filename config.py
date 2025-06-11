"""
Configuration settings for the WhatsApp Chat Analyzer
"""

# App settings
APP_TITLE = "WhatsApp Chat Analyzer 📊"
APP_DESCRIPTION = "Upload your WhatsApp chat export to analyze conversation patterns and statistics"
APP_ICON = "💬"
APP_LAYOUT = "wide"

# Data settings
DEFAULT_MIN_MESSAGES = 10
DEFAULT_DAYS_TO_ANALYZE = 30
MAX_DAYS_TO_ANALYZE = 500
MIN_DAYS_TO_ANALYZE = 7

# Time series settings
DEFAULT_TIME_FREQ = "D"
TIME_FREQ_OPTIONS = [
    ("Daily", "D"),
    ("Weekly", "W"),
    ("Monthly", "M")
]

# User analysis settings
MAX_USERS_TO_COMPARE = 5

# Sentiment analysis
SENTIMENT_METHOD_OPTIONS = [
    "Combined (TextBlob + VADER)",
    "TextBlob",
    "VADER",
    "Enhanced (Hinglish Support)",
    "Enhanced + Transformer"
]

SENTIMENT_METHOD_MAP = {
    "TextBlob": "textblob",
    "VADER": "vader",
    "Combined (TextBlob + VADER)": "combined",
    "Enhanced (Hinglish Support)": "enhanced_vader",
    "Enhanced + Transformer": "enhanced_transformer"
}

# Chart colors
CHART_COLORS = {
    'blue': '#636EFA',
    'green': '#00CC96',
    'purple': '#AB63FA',
    'orange': '#FFA15A',
    'red': '#EF553B'
}

SENTIMENT_COLORS = {
    'Positive': '#00CC96', 
    'Neutral': '#636EFA',
    'Negative': '#EF553B'
}
