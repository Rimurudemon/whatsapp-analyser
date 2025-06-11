"""
Emoji analysis module for WhatsApp Chat Analyzer
"""
from collections import Counter
import emoji

def analyze_emojis(df):
    """
    Analyze emoji usage in messages
    
    Parameters:
    df (DataFrame): DataFrame with messages
    
    Returns:
    dict: Dictionary with emoji analysis results
    """
    if len(df) == 0:
        return {}
    
    # Extract emojis from messages
    def extract_emojis(text):
        if not isinstance(text, str):
            return []
        return [c for c in text if c in emoji.EMOJI_DATA]
    
    df_copy = df.copy()
    df_copy['emojis'] = df_copy['message'].apply(extract_emojis)
    df_copy['emoji_count'] = df_copy['emojis'].apply(len)
    
    # Count emojis
    all_emojis = []
    for emoji_list in df_copy['emojis']:
        all_emojis.extend(emoji_list)
    
    emoji_counts = Counter(all_emojis)
    
    # Emojis per user
    user_emoji_counts = df_copy.groupby('author')['emoji_count'].sum().sort_values(ascending=False)
    
    # Most common emoji by user
    user_most_common_emoji = {}
    for author in df_copy['author'].unique():
        # Create a proper copy of the filtered data
        author_df = df_copy.loc[df_copy['author'] == author].copy()
        author_emojis = []
        for emoji_list in author_df['emojis']:
            author_emojis.extend(emoji_list)
        
        if author_emojis:
            user_most_common_emoji[author] = Counter(author_emojis).most_common(1)[0][0]
    
    return {
        'total_emojis': len(all_emojis),
        'unique_emojis': len(emoji_counts),
        'most_common_emojis': emoji_counts.most_common(10),
        'user_emoji_counts': user_emoji_counts,
        'user_most_common_emoji': user_most_common_emoji
    }
