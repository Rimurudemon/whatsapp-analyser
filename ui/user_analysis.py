"""
User Analysis tab UI component for WhatsApp Chat Analyzer
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import config
from modules.emoji_analysis import analyze_emojis
from modules.sentiment import analyze_sentiment

def render_user_analysis_tab(df, qualified_users):
    """
    Render the user analysis tab UI
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    qualified_users (list): List of users with minimum message count
    """
    st.header("User Analysis")
    
    if not qualified_users:
        st.warning("No users with enough messages to analyze.")
        return
    
    # User selector
    selected_user = st.selectbox(
        "Select a user to analyze",
        options=qualified_users
    )
    
    # Get user data
    user_data = df[df['author'] == selected_user].copy()
    
    # Calculate message length
    user_data.loc[:, 'message_length'] = user_data['message'].str.len()
    
    # Basic user metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", f"{len(user_data):,}")
        pct_of_total = len(user_data) / len(df) * 100 if len(df) > 0 else 0
        st.metric("% of All Messages", f"{pct_of_total:.1f}%")
    
    with col2:
        avg_length = user_data['message_length'].mean()
        st.metric("Avg. Message Length", f"{avg_length:.1f} chars")
        
        total_chars = user_data['message_length'].sum()
        st.metric("Total Characters", f"{total_chars:,}")
    
    with col3:
        user_data.loc[:, 'hour'] = user_data['date'].dt.hour
        most_active_hour = user_data.groupby('hour').size().idxmax()
        st.metric("Most Active Hour", f"{most_active_hour}:00")
        
        user_data.loc[:, 'day_of_week'] = user_data['date'].dt.day_name()
        most_active_day = user_data.groupby('day_of_week').size().idxmax()
        st.metric("Most Active Day", most_active_day)
    
    # Message distribution by hour
    st.subheader(f"Hourly Activity Pattern for {selected_user}")
    hourly_counts = user_data.groupby('hour').size()
    fig = px.bar(
        x=hourly_counts.index,
        y=hourly_counts.values,
        title=f"Messages by Hour of Day for {selected_user}",
        labels={'x': 'Hour', 'y': 'Number of Messages'},
        color_discrete_sequence=[config.CHART_COLORS['green']]
    )
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Message length distribution
    st.subheader(f"Message Length Distribution for {selected_user}")
    fig = px.histogram(
        user_data,
        x='message_length',
        nbins=30,
        title=f"Message Length Distribution for {selected_user}",
        labels={'message_length': 'Message Length (characters)', 'count': 'Frequency'},
        color_discrete_sequence=[config.CHART_COLORS['purple']]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick sentiment analysis for user
    user_sentiment = analyze_sentiment(user_data)
    if user_sentiment:
        st.subheader(f"Sentiment Analysis for {selected_user}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_sentiment = user_sentiment['average_sentiment']
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        
        # Calculate percentages
        total_msgs = sum(user_sentiment['sentiment_distribution'].values())
        if total_msgs > 0:
            with col2:
                positive_pct = user_sentiment['sentiment_distribution'].get('Positive', 0) / total_msgs * 100
                st.metric("Positive Messages", f"{positive_pct:.1f}%")
            
            with col3:
                negative_pct = user_sentiment['sentiment_distribution'].get('Negative', 0) / total_msgs * 100
                st.metric("Negative Messages", f"{negative_pct:.1f}%")
        
        # Show common emotions if available
        if user_sentiment.get('emotion_distribution'):
            top_emotions = sorted(user_sentiment['emotion_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
            if top_emotions:
                emotion_df = pd.DataFrame(top_emotions, columns=['Emotion', 'Count'])
                fig = px.pie(
                    emotion_df,
                    values='Count',
                    names='Emotion',
                    title=f"Top Emotions for {selected_user}",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Emoji analysis for user
    emoji_stats = analyze_emojis(user_data)
    if emoji_stats and emoji_stats.get('total_emojis', 0) > 0:
        st.subheader(f"Emoji Usage for {selected_user}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Emojis Used", emoji_stats['total_emojis'])
        with col2:
            st.metric("Unique Emojis Used", emoji_stats['unique_emojis'])
        
        if emoji_stats['most_common_emojis']:
            emoji_df = pd.DataFrame(
                emoji_stats['most_common_emojis'],
                columns=['Emoji', 'Count']
            )
            fig = px.bar(
                emoji_df,
                x='Emoji',
                y='Count',
                title=f"Top Emojis Used by {selected_user}",
                color_discrete_sequence=[config.CHART_COLORS['orange']]
            )
            st.plotly_chart(fig, use_container_width=True)
