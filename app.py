"""
WhatsApp Chat Analyzer - Main Application

This application analyzes WhatsApp chat exports and provides visualizations and statistics.
"""
import streamlit as st

# Import configuration
import config

# Set page configuration first
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT
)

import pandas as pd
from datetime import timedelta

# Import backend modules
from modules.utils import download_nltk_resources
from modules.parser import parse_chat
from modules.stats import get_basic_stats

# Import UI components
from ui.overview import render_overview_tab
from ui.user_analysis import render_user_analysis_tab
from ui.time_analysis import render_time_analysis_tab
from ui.comparison import render_comparison_tab
from ui.wordcloud import render_wordcloud_tab
from ui.media_ui import render_media_tab
from ui.sentiment_ui import render_sentiment_tab
from ui.chat_patterns import render_chat_patterns_tab

# Download NLTK resources
download_nltk_resources()

def main():
    """Main application function"""
    
    # Display title and description
    st.title(config.APP_TITLE)
    st.write(config.APP_DESCRIPTION)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=['txt'])
        
        # Only show these options if a file is uploaded
        min_messages = config.DEFAULT_MIN_MESSAGES
        time_freq = config.DEFAULT_TIME_FREQ
        days_to_analyze = config.DEFAULT_DAYS_TO_ANALYZE
        
        if uploaded_file:
            min_messages = st.number_input(
                "Minimum messages per user",
                min_value=1,
                value=config.DEFAULT_MIN_MESSAGES
            )
            
            st.header("Time Series Settings")
            time_freq = st.selectbox(
                "Time series frequency",
                options=config.TIME_FREQ_OPTIONS,
                format_func=lambda x: x[0]
            )[1]
            
            days_to_analyze = st.number_input(
                "Days to analyze (most recent)",
                min_value=config.MIN_DAYS_TO_ANALYZE,
                max_value=config.MAX_DAYS_TO_ANALYZE,
                value=config.DEFAULT_DAYS_TO_ANALYZE,
            )
    
    # Main content
    if uploaded_file:
        # Parse the chat file
        file_content = uploaded_file.getvalue().decode("utf-8")
        
        with st.spinner("Parsing chat data..."):
            df, qualified_users = parse_chat(file_content, min_messages)
        
        if len(df) == 0:
            st.error("Could not parse the chat file. Please make sure it's a valid WhatsApp export.")
            return
        
        # Filter by date range if specified
        if days_to_analyze:
            end_date = df['date'].max()
            start_date = end_date - timedelta(days=days_to_analyze)
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        else:
            df_filtered = df.copy()
        
        # Calculate basic statistics
        stats = {}
        with st.spinner("Calculating statistics..."):
            stats = get_basic_stats(df_filtered)
        
        if not stats:
            st.warning("No data to display after filtering.")
            return
            
        # Create tabs for different analyses
        tabs = st.tabs([
            "Overview",
            "User Analysis",
            "Time Series",
            "User Comparison",
            "Word Clouds",
            "Media Analysis",
            "Sentiment Analysis",
            "Chat Patterns"
        ])
        
        # Render each tab
        with tabs[0]:
            render_overview_tab(df_filtered, stats)
        
        with tabs[1]:
            render_user_analysis_tab(df_filtered, qualified_users)
        
        with tabs[2]:
            render_time_analysis_tab(df_filtered, qualified_users, time_freq)
        
        with tabs[3]:
            render_comparison_tab(df_filtered, qualified_users, stats, time_freq)
        
        with tabs[4]:
            render_wordcloud_tab(df_filtered, qualified_users)
        
        with tabs[5]:
            render_media_tab(df_filtered, qualified_users)
        
        with tabs[6]:
            render_sentiment_tab(df_filtered)
        
        with tabs[7]:
            render_chat_patterns_tab(df_filtered, qualified_users)
    
    else:
        # No file uploaded yet, show instructions
        st.info("👈 Please upload a WhatsApp chat export file from the sidebar to begin analysis")

        st.markdown("""
        ### How to export your WhatsApp chat:

        1. Open the WhatsApp chat you want to analyze
        2. Tap the three dots in the top right corner
        3. Select "More" > "Export chat"
        4. Choose "Without Media"
        5. Send the export to yourself via email or save it
        6. Upload the .txt file using the uploader on the left

        ### Features:
        - View basic chat statistics and patterns
        - Analyze individual user behavior
        - Compare message patterns between users
        - Visualize activity over time
        - Generate word clouds from chat content
        - Analyze media sharing patterns
        - Detect sentiment and emotions in messages
        - Explore conversation dynamics

        ### Privacy Note:
        All analysis happens in your browser. No data is sent to any server.
        """)

if __name__ == "__main__":
    main()
