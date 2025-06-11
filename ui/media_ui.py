"""
Media Analysis tab UI component for WhatsApp Chat Analyzer
"""
import streamlit as st
import plotly.express as px
import pandas as pd
from modules.media import analyze_media

def render_media_tab(df, qualified_users):
    """
    Render the media analysis tab UI
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    qualified_users (list): List of users with minimum message count
    """
    st.header("Media Analysis")
    
    # Calculate media statistics
    with st.spinner("Analyzing media content..."):
        media_stats = analyze_media(df)
    
    if media_stats and sum(media_stats['media_counts'].values()) > 0:
        # Overview of media
        st.subheader("Media Overview")
        
        # Convert to DataFrame for plotting
        media_df = pd.DataFrame({
            'Type': list(media_stats['media_counts'].keys()),
            'Count': list(media_stats['media_counts'].values())
        })

        fig = px.bar(
            media_df,
            x='Type',
            y='Count',
            title="Media Types Shared",
            color='Type',
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Media by user
        st.subheader("Media Shared by User")

        # Create a DataFrame with users as rows and media types as columns
        media_by_user = {}
        for user in qualified_users:
            media_by_user[user] = {}
            for media_type in media_stats['media_counts'].keys():
                try:
                    media_by_user[user][media_type] = media_stats['media_per_user'][media_type].get(user, 0)
                except:
                    media_by_user[user][media_type] = 0

        media_user_df = pd.DataFrame(media_by_user).T

        # Plot only if there's data
        if media_user_df.sum().sum() > 0:
            fig = px.bar(
                media_user_df,
                x=media_user_df.index,
                y=media_user_df.columns,
                title="Media Types by User",
                labels={'x': 'User', 'value': 'Count', 'variable': 'Media Type'},
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No media content found for these users.")
    else:
        st.info("No media content detected in this chat.")
