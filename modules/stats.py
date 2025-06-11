"""
Statistics module for WhatsApp Chat Analyzer
"""
import pandas as pd

def get_basic_stats(df):
    """Calculate basic statistics from the chat DataFrame"""
    if len(df) == 0:
        return {}
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Total messages
    total_messages = len(df_copy)
    
    # Messages per user
    messages_per_user = df_copy['author'].value_counts()
    
    # Characters per user
    df_copy.loc[:, 'message_length'] = df_copy['message'].apply(len)
    chars_per_user = df_copy.groupby('author')['message_length'].sum().sort_values(ascending=False)
    
    # Average message length per user
    avg_length_per_user = df_copy.groupby('author')['message_length'].mean().sort_values(ascending=False)
    
    # Date range
    date_range = (df_copy['date'].max() - df_copy['date'].min()).days + 1
    
    # Messages per day
    df_copy.loc[:, 'day'] = df_copy['date'].dt.date
    messages_per_day = df_copy.groupby('day').size()
    avg_messages_per_day = messages_per_day.mean()
    
    # Most active day
    most_active_day = messages_per_day.idxmax()
    most_active_day_count = messages_per_day.max()
    
    # Time analysis
    df_copy.loc[:, 'hour'] = df_copy['date'].dt.hour
    hourly_activity = df_copy.groupby('hour').size()
    peak_hour = hourly_activity.idxmax()
    
    # Most active user
    most_active_user = messages_per_user.idxmax()
    most_active_user_count = messages_per_user.max()
    
    # Most verbose user (most characters)
    most_verbose_user = chars_per_user.idxmax()
    most_verbose_chars = chars_per_user.max()
    
    # Day of week analysis
    df_copy.loc[:, 'day_of_week'] = df_copy['date'].dt.day_name()
    day_of_week_counts = df_copy.groupby('day_of_week').size()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_counts = day_of_week_counts.reindex(day_order)
    
    return {
        'total_messages': total_messages,
        'messages_per_user': messages_per_user,
        'chars_per_user': chars_per_user,
        'avg_length_per_user': avg_length_per_user,
        'date_range': date_range,
        'avg_messages_per_day': avg_messages_per_day,
        'most_active_day': most_active_day,
        'most_active_day_count': most_active_day_count,
        'hourly_activity': hourly_activity,
        'peak_hour': peak_hour,
        'most_active_user': most_active_user,
        'most_active_user_count': most_active_user_count,
        'most_verbose_user': most_verbose_user,
        'most_verbose_chars': most_verbose_chars,
        'day_of_week_counts': day_of_week_counts
    }
