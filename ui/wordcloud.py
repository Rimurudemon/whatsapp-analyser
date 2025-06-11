"""
Word Cloud tab UI component for WhatsApp Chat Analyzer
"""
import streamlit as st
from modules.visualization import generate_word_cloud, plot_word_cloud

def render_wordcloud_tab(df, qualified_users):
    """
    Render the word cloud tab UI
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    qualified_users (list): List of users with minimum message count
    """
    st.header("Word Clouds")
    
    # User selector for word cloud
    wc_options = ["All Users"] + qualified_users
    selected_wc_user = st.selectbox(
        "Select a user for word cloud",
        options=wc_options
    )
    
    # Generate word cloud
    with st.spinner("Generating word cloud..."):
        wordcloud = generate_word_cloud(df, selected_wc_user)
    
    if wordcloud:
        # Display word cloud
        fig = plot_word_cloud(wordcloud)
        st.pyplot(fig)
    else:
        st.warning("Not enough text data to generate a word cloud.")
