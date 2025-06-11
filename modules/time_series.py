"""
Time series analysis module for WhatsApp Chat Analyzer
"""
import pandas as pd
import numpy as np
from datetime import timedelta

def generate_time_series(df, users=None, freq='D'):
    """
    Generate time series of message counts
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    users (list): List of users to include (None for all)
    freq (str): Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns:
    DataFrame with time series data
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    # Create a copy of the DataFrame first to avoid modifying the original
    df_copy = df.copy()
    
    # Filter by users if specified
    if users:
        df_copy = df_copy[df_copy['author'].isin(users)].copy()
    
    # Set date as index for resampling
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Create a DataFrame with time series for each user
    time_series = {}
    
    # Overall time series
    all_messages = df_copy.set_index('date').resample(freq).size()
    time_series['All'] = all_messages
    
    # Per-user time series
    for user in df_copy['author'].unique():
        user_df = df_copy.loc[df_copy['author'] == user].copy()
        user_messages = user_df.set_index('date').resample(freq).size()
        time_series[user] = user_messages
    
    # Combine all time series
    result = pd.DataFrame(time_series)
    result = result.fillna(0)
    
    return result

def analyze_chat_patterns(df, conversation_gap_hours=3):
    """
    Analyze chat patterns like response times, conversation initiators, etc.
    
    Parameters:
    df (DataFrame): DataFrame with messages
    conversation_gap_hours (int): Hours of inactivity to define a new conversation
    
    Returns:
    dict: Dictionary with chat pattern analysis results
    """
    if len(df) == 0:
        return {}
    
    # Create a copy to avoid modifying the original
    df_sorted = df.copy().sort_values('date')
    
    # Calculate response times between users
    def calculate_response_times(df_sorted):
        response_analysis = {}
        users = df_sorted['author'].unique()
        
        for i, user1 in enumerate(users):
            for user2 in users[i+1:]:
                # Filter messages by these two users
                user_messages = df_sorted[df_sorted['author'].isin([user1, user2])].copy().sort_values('date')
                
                # Calculate response times
                response_times_1to2 = []
                response_times_2to1 = []
                
                for i in range(1, len(user_messages)):
                    curr_msg = user_messages.iloc[i]
                    prev_msg = user_messages.iloc[i-1]
                    
                    if curr_msg['author'] != prev_msg['author']:
                        # Calculate time difference in minutes
                        time_diff = (curr_msg['date'] - prev_msg['date']).total_seconds() / 60
                        
                        # Only consider responses within 24 hours (1440 minutes)
                        if time_diff <= 1440:
                            if curr_msg['author'] == user1:
                                response_times_1to2.append(time_diff)
                            else:
                                response_times_2to1.append(time_diff)
                
                # Store the results
                if response_times_1to2:
                    response_analysis[f"{user1} → {user2}"] = response_times_1to2
                if response_times_2to1:
                    response_analysis[f"{user2} → {user1}"] = response_times_2to1
                    
        return response_analysis
    
    # Find conversation initiators
    def find_conversation_initiators(df_sorted, conversation_gap=timedelta(hours=conversation_gap_hours)):
        df_conv = df_sorted.copy()
        df_conv['time_diff'] = df_conv['date'].diff()
        df_conv['new_conversation'] = df_conv['time_diff'] > conversation_gap
        
        # Count initiators
        initiators = df_conv.loc[df_conv['new_conversation'], 'author'].value_counts()
        
        return initiators
    
    # Create activity heatmap data
    def create_activity_heatmap(df):
        heatmap_df = df.copy()
        heatmap_df['day_of_week'] = heatmap_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        heatmap_df['hour'] = heatmap_df['date'].dt.hour
        
        # Create a heatmap of activity
        day_hour_counts = heatmap_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        
        # Create a pivot table for the heatmap
        pivot_data = day_hour_counts.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
        
        # Reindex to ensure all days and hours appear
        pivot_data = pivot_data.reindex(range(7), fill_value=0)
        pivot_data = pivot_data.reindex(columns=range(24), fill_value=0)
        
        return pivot_data
    
    response_times = calculate_response_times(df_sorted)
    initiators = find_conversation_initiators(df_sorted, conversation_gap=timedelta(hours=conversation_gap_hours))
    activity_heatmap = create_activity_heatmap(df)
    
    return {
        'response_times': response_times,
        'initiators': initiators,
        'activity_heatmap': activity_heatmap
    }
