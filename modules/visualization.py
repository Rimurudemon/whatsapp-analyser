"""
Visualization module for WhatsApp Chat Analyzer
"""
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

def generate_word_cloud(df, user=None):
    """
    Generate word cloud for a specific user or all users
    
    Parameters:
    df (DataFrame): DataFrame with messages
    user (str): Username to generate word cloud for (None for all users)
    
    Returns:
    WordCloud: Generated word cloud object
    """
    if len(df) == 0:
        return None
    
    # Filter by user if specified
    if user and user != "All Users":
        filtered_df = df[df['author'] == user].copy()
        messages = filtered_df['message'].str.cat(sep=' ')
    else:
        messages = df['message'].str.cat(sep=' ')
    
    # Remove URLs and media references
    messages = re.sub(r'https?://\S+|www\.\S+|<Media omitted>', '', messages)
    
    # Remove emojis and special characters
    messages = re.sub(r'[^\w\s]', '', messages)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=stop_words,
        min_font_size=10,
        max_words=100
    ).generate(messages)
    
    return wordcloud

def plot_word_cloud(wordcloud):
    """
    Plot a word cloud
    
    Parameters:
    wordcloud (WordCloud): WordCloud object to plot
    
    Returns:
    matplotlib.figure.Figure: Figure containing the word cloud
    """
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    return fig
