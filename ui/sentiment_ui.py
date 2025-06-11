"""
Sentiment Analysis tab UI component for WhatsApp Chat Analyzer
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import config
from modules.sentiment import analyze_sentiment, sentiment_time_frames
from modules.enhanced_sentiment import analyze_sentiment_enhanced, sentiment_time_frames_enhanced

def render_sentiment_tab(df):
    """
    Render the sentiment analysis tab UI
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    """
    st.header("Sentiment Analysis")
    
    # Sentiment analysis method selector
    sentiment_method = st.selectbox(
        "Select sentiment analysis method",
        options=config.SENTIMENT_METHOD_OPTIONS,
        index=0
    )
    
    # Map the selection to method parameter
    selected_method = config.SENTIMENT_METHOD_MAP[sentiment_method]
    
    # Calculate sentiment statistics
    with st.spinner("Analyzing sentiment in messages..."):
        if selected_method in ["enhanced_vader", "enhanced_transformer"]:
            # Use enhanced sentiment analysis with Hinglish support
            use_transformer = (selected_method == "enhanced_transformer")
            method = "transformer" if use_transformer else "vader"
            sentiment_stats, enhanced_df = analyze_sentiment_enhanced(df, method=method)
            sentiment_frames = sentiment_time_frames_enhanced(enhanced_df, method=method)
        else:
            # Use standard sentiment analysis
            sentiment_stats = analyze_sentiment(df, method=selected_method)
            sentiment_frames = sentiment_time_frames(df, method=selected_method)
    
    if not sentiment_stats:
        st.warning("Not enough data for sentiment analysis.")
        return
    
    # Overall sentiment metrics
    st.subheader("Overall Chat Sentiment")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_sentiment = sentiment_stats['average_sentiment']
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    
    with col2:
        # Positive percentage
        total_msgs = sum(sentiment_stats['sentiment_distribution'].values())
        positive_pct = (sentiment_stats['sentiment_distribution'].get('Positive', 0) / 
                      total_msgs * 100) if total_msgs > 0 else 0
        st.metric("Positive Messages", f"{positive_pct:.1f}%")
    
    with col3:
        # Negative percentage
        negative_pct = (sentiment_stats['sentiment_distribution'].get('Negative', 0) / 
                      total_msgs * 100) if total_msgs > 0 else 0
        st.metric("Negative Messages", f"{negative_pct:.1f}%")
    
    # Language distribution information if available
    if selected_method in ["enhanced_vader", "enhanced_transformer"] and 'language_distribution' in sentiment_stats:
        st.subheader("Language Distribution")
        
        total_msgs = sentiment_stats['language_distribution']['english'] + sentiment_stats['language_distribution']['hinglish']
        english_pct = (sentiment_stats['language_distribution']['english'] / total_msgs * 100) if total_msgs > 0 else 0
        hinglish_pct = (sentiment_stats['language_distribution']['hinglish'] / total_msgs * 100) if total_msgs > 0 else 0
        
        lang_col1, lang_col2 = st.columns(2)
        with lang_col1:
            st.metric("English", f"{english_pct:.1f}%")
        with lang_col2:
            st.metric("Hinglish/Hindi", f"{hinglish_pct:.1f}%")
        
        # Language distribution chart
        language_dist_df = pd.DataFrame({
            'Language': ['English', 'Hinglish/Hindi'],
            'Percentage': [english_pct, hinglish_pct]
        })
        
        fig = px.pie(
            language_dist_df,
            values='Percentage',
            names='Language',
            title="Language Distribution in Messages",
            color_discrete_sequence=["#636EFA", "#EF553B"]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment distribution chart
    st.subheader("Sentiment Distribution")
    
    # Convert to DataFrame for plotting
    sentiment_dist_df = pd.DataFrame({
        'Category': list(sentiment_stats['sentiment_distribution'].keys()),
        'Count': list(sentiment_stats['sentiment_distribution'].values())
    })
    
    fig = px.pie(
        sentiment_dist_df,
        values='Count',
        names='Category',
        title="Message Sentiment Distribution",
        color='Category',
        color_discrete_map=config.SENTIMENT_COLORS
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Emotion distribution chart if available
    if sentiment_stats.get('emotion_distribution'):
        st.subheader("Emotion Distribution")
        
        # Convert to DataFrame for plotting
        emotion_dist_df = pd.DataFrame({
            'Emotion': list(sentiment_stats['emotion_distribution'].keys()),
            'Count': list(sentiment_stats['emotion_distribution'].values())
        }).sort_values('Count', ascending=False)
        
        fig = px.bar(
            emotion_dist_df,
            x='Emotion',
            y='Count',
            title="Detected Emotions in Messages",
            color='Emotion'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Over Time
    st.subheader("Sentiment Trends Over Time")
    
    # Sentiment category counts over time
    if not sentiment_frames['category_counts'].empty:
        fig = px.area(
            sentiment_frames['category_counts'],
            title="Sentiment Categories Over Time",
            labels={'date': 'Date', 'value': 'Number of Messages', 'variable': 'Sentiment'},
            color_discrete_map=config.SENTIMENT_COLORS
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Language usage over time for enhanced sentiment analysis
        if selected_method in ["enhanced_vader", "enhanced_transformer"] and 'language_usage' in sentiment_frames:
            language_usage = sentiment_frames['language_usage']
            if not language_usage.empty:
                fig = px.line(
                    language_usage,
                    x='date', 
                    y='is_hinglish',
                    title="Hinglish Usage Over Time",
                    labels={'date': 'Date', 'is_hinglish': 'Hinglish Messages (%)'},
                    markers=True
                )
                
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
    
    # Average sentiment over time
    if not sentiment_frames['average_by_date'].empty:
        fig = px.line(
            sentiment_frames['average_by_date'],
            x='date',
            y='sentiment',
            title="Average Sentiment Score Over Time",
            labels={'date': 'Date', 'sentiment': 'Sentiment Score (-1 to 1)'}
        )
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=sentiment_frames['average_by_date']['date'].min(),
            y0=0,
            x1=sentiment_frames['average_by_date']['date'].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    # User Sentiment Comparison
    st.subheader("User Sentiment Comparison")
    
    # Create DataFrame for user sentiment comparison
    if selected_method in ["enhanced_vader", "enhanced_transformer"]:
        # Enhanced version with Hinglish percentage
        user_sentiment_df = pd.DataFrame({
            user: {
                'Average Score': data['average'],
                'Positive %': data['positive'],
                'Negative %': data['negative'],
                'Neutral %': data['neutral'],
                'Most Common Emotion': data['most_common_emotion'],
                'Hinglish %': data.get('hinglish_percentage', 0)
            }
            for user, data in sentiment_stats['user_sentiments'].items()
        }).T
    else:
        # Standard version
        user_sentiment_df = pd.DataFrame({
            user: {
                'Average Score': data['average'],
                'Positive %': data['positive'],
                'Negative %': data['negative'],
                'Neutral %': data['neutral'],
                'Most Common Emotion': data['most_common_emotion']
            }
            for user, data in sentiment_stats['user_sentiments'].items()
        }).T
    
    # Allow user to select which users to compare
    compare_sentiment_users = st.multiselect(
        "Select users to compare (max 5)",
        options=list(sentiment_stats['user_sentiments'].keys()),
        default=list(sentiment_stats['user_sentiments'].keys())[:min(3, len(sentiment_stats['user_sentiments']))]
    )
    
    if len(compare_sentiment_users) > 5:
        st.warning("Please select up to 5 users for comparison.")
        compare_sentiment_users = compare_sentiment_users[:5]
    
    if len(compare_sentiment_users) >= 1:
        # Filter the DataFrame for selected users
        filtered_user_sentiment = user_sentiment_df.loc[compare_sentiment_users]
        
        # Average sentiment score comparison
        fig = px.bar(
            filtered_user_sentiment,
            y=filtered_user_sentiment.index,
            x='Average Score',
            title="Average Sentiment Score by User",
            labels={'y': 'User', 'x': 'Sentiment Score (-1 to 1)'},
            color='Average Score',
            color_continuous_scale=[(0, "red"), (0.5, "gray"), (1, "green")],
            range_color=[-1, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment category percentage comparison
        sentiment_pct_df = pd.DataFrame({
            'User': np.repeat(filtered_user_sentiment.index, 3),
            'Category': np.tile(['Positive', 'Neutral', 'Negative'], len(filtered_user_sentiment)),
            'Percentage': np.concatenate([
                filtered_user_sentiment['Positive %'].values,
                filtered_user_sentiment['Neutral %'].values,
                filtered_user_sentiment['Negative %'].values
            ])
        })
        
        fig = px.bar(
            sentiment_pct_df,
            x='User',
            y='Percentage',
            color='Category',
            barmode='stack',
            title="Sentiment Category Distribution by User",
            labels={'User': 'User', 'Percentage': 'Percentage of Messages'},
            color_discrete_map=config.SENTIMENT_COLORS
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Most common emotion by user
        fig = px.bar(
            filtered_user_sentiment,
            y=filtered_user_sentiment.index,
            x=[1] * len(filtered_user_sentiment),  # All bars same length
            title="Most Common Emotion by User",
            labels={'y': 'User'},
            color='Most Common Emotion',
            orientation='h',
            text='Most Common Emotion'
        )
        fig.update_traces(textposition='inside', textangle=0)
        fig.update_layout(showlegend=True)
        fig.update_xaxes(showticklabels=False)  # Hide x axis labels
        st.plotly_chart(fig, use_container_width=True)
        
        # Hinglish percentage if available
        if selected_method in ["enhanced_vader", "enhanced_transformer"] and 'Hinglish %' in filtered_user_sentiment.columns:
            fig = px.bar(
                filtered_user_sentiment,
                y=filtered_user_sentiment.index,
                x='Hinglish %',
                title="Hinglish Usage by User",
                labels={'y': 'User', 'x': 'Hinglish Messages (%)'},
                color='Hinglish %',
                color_continuous_scale=[(0, "#636EFA"), (1, "#EF553B")],
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one user for sentiment comparison.")

    # Add an explanation of the sentiment analysis methodology
    with st.expander("About Sentiment Analysis Methodology"):
        st.markdown("""
        ### Sentiment Analysis Methods
        
        This analyzer offers five sentiment analysis methods:
        
        1. **Combined (TextBlob + VADER)**: Uses both TextBlob and VADER sentiment analyzers with weighted averaging for more reliable results.
        2. **TextBlob**: Uses TextBlob's pattern-based sentiment analysis which excels at grammatical text.
        3. **VADER**: (Valence Aware Dictionary and sEntiment Reasoner) is specifically tuned for social media content and handles emojis, slang, and informal language better.
        4. **Enhanced (Hinglish Support)**: Uses VADER sentiment with Hinglish language detection and translation, providing better analysis for multilingual chats.
        5. **Enhanced + Transformer**: Combines Hinglish support with transformer-based sentiment analysis for higher accuracy.
        
        #### Sentiment Categories
        - **Positive**: Score > 0.15
        - **Negative**: Score < -0.1
        - **Neutral**: Score between -0.1 and 0.15
        
        #### Emotion Detection
        Emotions are detected using a combination of keyword matching and sentiment analysis.
        
        #### Hinglish Support
        The enhanced sentiment analysis detects Hinglish (Hindi written in Roman script) and uses:
        - Language detection based on common Hindi words
        - Transliteration from Roman script to Devanagari
        - Translation from Hindi to English before sentiment analysis
        - Batch processing for handling large datasets efficiently
        """)
