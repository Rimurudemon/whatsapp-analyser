"""
Media analysis module for WhatsApp Chat Analyzer
"""
import re
import pandas as pd

def analyze_media(df):
    """
    Analyze media content from messages
    
    Parameters:
    df (DataFrame): DataFrame with messages
    
    Returns:
    dict: Dictionary with media analysis results
    """
    if len(df) == 0:
        return {}
    
    # Count image, video, audio, and document messages
    image_pattern = r'<Media omitted>|image omitted|\.jpg|\.jpeg|\.png|\.gif'
    video_pattern = r'video omitted|\.mp4|\.mov|\.avi'
    audio_pattern = r'audio omitted|\.mp3|\.ogg|\.m4a'
    document_pattern = r'document omitted|\.pdf|\.doc|\.docx|\.xls|\.xlsx|\.ppt|\.pptx'
    
    df_copy = df.copy()
    df_copy['has_image'] = df_copy['message'].str.contains(image_pattern, case=False)
    df_copy['has_video'] = df_copy['message'].str.contains(video_pattern, case=False)
    df_copy['has_audio'] = df_copy['message'].str.contains(audio_pattern, case=False)
    df_copy['has_document'] = df_copy['message'].str.contains(document_pattern, case=False)
    
    media_counts = {
        'images': df_copy['has_image'].sum(),
        'videos': df_copy['has_video'].sum(),
        'audio': df_copy['has_audio'].sum(),
        'documents': df_copy['has_document'].sum()
    }
    
    # Media per user
    media_per_user = {
        'images': df_copy.loc[df_copy['has_image']].groupby('author').size().sort_values(ascending=False),
        'videos': df_copy.loc[df_copy['has_video']].groupby('author').size().sort_values(ascending=False),
        'audio': df_copy.loc[df_copy['has_audio']].groupby('author').size().sort_values(ascending=False),
        'documents': df_copy.loc[df_copy['has_document']].groupby('author').size().sort_values(ascending=False)
    }
    
    return {
        'media_counts': media_counts,
        'media_per_user': media_per_user
    }
