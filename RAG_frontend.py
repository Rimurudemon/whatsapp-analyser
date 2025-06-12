import os
import re
import json
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime, timedelta
import plotly.express as px
from chat_ui_enhancer import enhance_message_display, display_context_chunks, format_message_chunk
import plotly.graph_objects as go
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel, configure
from langchain_core.documents import Document
from dotenv import load_dotenv
import time
import base64
from PIL import Image

# Import functions from test_raq_qna.py
from test_raq_qna import (
    parse_whatsapp_chat,
    format_messages_for_rag,
    build_faiss_db,
    load_faiss_db,
    query_db,
    qa_with_gemini,
    generate_chat_insights,
    load_chat_insights,
    setup_gemini
)

# Load environment variables from .env file
load_dotenv()

# Configuration
DB_PATH = "faiss_chat_enhanced"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 10
CONTEXT_SIZE = 3
RETRIEVER_K = 10

# Check if Google API key is set
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        st.error("GOOGLE_API_KEY environment variable is not set properly. Please set it in your .env file.")
except KeyError:
    st.error("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
    GOOGLE_API_KEY = ""

# Set page configuration
st.set_page_config(
    page_title="WhatsApp Chat RAG Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimalist dark theme with improved contrast
st.markdown("""
<style>
    /* Basic app styling */
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    /* Override Streamlit's default styles for dark theme */
    .st-emotion-cache-fblp2m {
        color: #ffffff;
    }
    .st-emotion-cache-1gulkj5 {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    /* Adjust all text for dark theme */
    p, h1, h2, h3, h4, h5, h6, span {
        color: #ffffff;
    }
    /* Style for links */
    a {
        color: #00acc1;
    }
    /* Style for code blocks */
    code {
        background-color: #2a2a2a;
        color: #00acc1;
    }
    .chat-container {
        background-color: #1e1e1e;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #333333;
    }
    .chat-message {
        background-color: #2a2a2a;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 5px;
        border: 1px solid #333333;
    }
    .chat-author {
        font-weight: bold;
        color: #00acc1;
    }
    .chat-date {
        font-size: 0.8em;
        color: #aaaaaa;
    }
    .insight-card {
        background-color: #2a2a2a;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 18px;
        border: 1px solid #333333;
    }
    .insight-card h3 {
        color: #006064;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .insight-card h2 {
        font-size: 1.8em;
        margin: 10px 0;
        color: #ffffff;
    }
    .answer-box {
        background-color: #1e1e1e;
        border-left: 5px solid #006064;
        padding: 16px;
        border-radius: 4px;
        margin-top: 12px;
        border: 1px solid #333333;
    }
    .question-box {
        background-color: #2a2a2a;
        border-left: 5px solid #00838f;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 18px;
        border: 1px solid #333333;
    }
    .chat-header {
        background-color: #006064;
        color: white;
        padding: 16px 22px;
        border-radius: 4px;
        margin-bottom: 25px;
        text-align: center;
    }
    .text-input {
        background-color: #2a2a2a !important;
        border: 1px solid #444444 !important;
        border-radius: 4px !important;
        padding: 10px 16px !important;
        color: #ffffff !important;
    }
    .text-input:focus {
        border-color: #006064 !important;
        outline: none !important;
    }
    .messages-container {
        background-color: #1e1e1e; 
        border-radius: 8px;
        padding: 20px;
        height: 520px;
        overflow-y: auto;
        margin-bottom: 20px;
        border: 1px solid #333333;
    }
    .context-display {
        background-color: #1e1e1e;
        border-left: 4px solid #006064;
        border-radius: 4px;
        padding: 16px;
        font-family: 'Roboto', sans-serif;
        font-size: 0.94em;
        line-height: 1.5;
        color: #dddddd;
        white-space: pre-wrap;
        margin-bottom: 14px;
        border: 1px solid #333333;
    }
    .context-display h4 {
        margin-top: 0;
        color: #00acc1;
        font-size: 1.05em;
        font-weight: 500;
    }
    .sidebar .stButton>button {
        width: 100%;
        margin-top: 12px;
        border-radius: 4px;
        background-color: #006064;
        color: white;
        font-weight: 500;
        padding: 10px 16px;
    }
    .sidebar .stButton>button:hover {
        background-color: #00838f;
    }
    /* Style for Ask and Clear buttons in chat */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 4px !important;
        padding: 8px 20px !important;
    }
    div[data-testid="stHorizontalBlock"] button:first-child {
        background-color: #006064 !important;
    }
    div[data-testid="stHorizontalBlock"] button:first-child:hover {
        background-color: #00838f !important;
    }
    /* Message styling - enhanced */
    .message-container {
        display: flex;
        margin-bottom: 15px;
    }
    .message-bubble {
        max-width: 80%;
        padding: 12px 18px;
        border-radius: 8px;
        position: relative;
        margin-bottom: 16px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .message-sent {
        background-color: #006064;
        color: #ffffff;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }
    .message-sent:after {
        content: "";
        position: absolute;
        right: -8px;
        bottom: 0;
        width: 8px;
        height: 13px;
        background-color: #006064;
        border-bottom-left-radius: 10px;
    }
    .message-received {
        background-color: #2a2a2a;
        color: #ffffff;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        border: 1px solid #333333;
    }
    .message-received:after {
        content: "";
        position: absolute;
        left: -8px;
        bottom: 0;
        width: 8px;
        height: 13px;
        background-color: #2a2a2a;
        border-bottom-right-radius: 10px;
        border-left: 1px solid #333333;
    }
    .message-info {
        font-size: 0.75em;
        text-align: right;
        color: #aaaaaa;
        margin-top: 5px;
    }
    .message-text {
        margin-bottom: 5px;
        line-height: 1.4;
    }
    /* Improved styling for the query input box */
    .query-input-container {
        background-color: #1e1e1e;
        border-radius: 4px;
        padding: 8px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        border: 1px solid #333333;
    }
    /* Custom styling for the expander */
    .st-emotion-cache-1l269u8 {
        background-color: #1e1e1e;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 15px;
        border: 1px solid #333333;
    }
    
    /* Style for scrollbar in messages container - minimalist dark theme */
    .messages-container::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    .messages-container::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    .messages-container::-webkit-scrollbar-thumb {
        background: #444444;
        border-radius: 2px;
    }
    
    /* Typing indicator - minimalist version without animations */
    .typing-bubble {
        background-color: #2a2a2a;
        border-radius: 4px;
        padding: 12px 16px;
        display: inline-block;
        margin-right: auto;
        max-width: 60%;
        border: 1px solid #444444;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
    }
    
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin: 0 2px;
        background-color: #006064;
        border-radius: 50%;
    }
    
    /* Custom styling for example question buttons */
    [data-testid="stButton"] button:not(.sidebar button) {
        border-radius: 4px !important;
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }
    [data-testid="stButton"] button:hover {
        background-color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Define helper functions

def load_image(image_path):
    """Load and return an image as base64 for displaying in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

def display_logo():
    """Display the app logo"""
    # Create a minimalist logo without animations
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="display: inline-block; padding: 16px; background-color: #006064; border-radius: 4px; margin-bottom: 15px;">
            <span style="font-size: 40px;">💬</span>
        </div>
        <h1 style="color: #ffffff; margin-top: 10px; font-weight: 600;">
            WhatsApp Chat Analyzer
        </h1>
        <p style="font-size: 16px; color: #cccccc; max-width: 600px; margin: 10px auto;">
            Powerful RAG-based analysis for your WhatsApp conversations
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_message_sample(df, num_messages=5):
    """Display a sample of messages in WhatsApp-like format"""
    if len(df) == 0:
        return
    
    # Sort by date and take the latest messages
    sample_df = df.sort_values('date').tail(num_messages)
    
    st.markdown("<h4>Sample Messages</h4>", unsafe_allow_html=True)
    
    for _, row in sample_df.iterrows():
        author = row['author']
        message = row['message']
        date = row['date'].strftime('%Y-%m-%d %H:%M')
        
        # Simple algorithm to determine if it's a sent or received message
        # Just alternating for visual appeal in the sample
        message_type = "message-sent" if _ % 2 == 0 else "message-received"
        
        st.markdown(f"""
        <div class="message-container">
            <div class="message-bubble {message_type}">
                <div class="message-text">{message}</div>
                <div class="message-info">
                    <span>{author} • {date}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_chat_insights(insights):
    """Display chat insights in a visually appealing way"""
    if not insights:
        return
    
    # Create three columns for the insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-card">
            <h3 style="color: #128c7e;">Messages</h3>
            <h2>{:,}</h2>
            <p>Total messages in chat</p>
        </div>
        """.format(insights['total_messages']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <h3 style="color: #128c7e;">Participants</h3>
            <h2>{:,}</h2>
            <p>Active chat members</p>
        </div>
        """.format(insights['author_stats']['total_authors']), unsafe_allow_html=True)
    
    with col3:
        # Calculate date range in days
        try:
            start_date = datetime.strptime(insights['date_range']['start'], '%Y-%m-%d')
            end_date = datetime.strptime(insights['date_range']['end'], '%Y-%m-%d')
            date_range_days = (end_date - start_date).days
        except (ValueError, KeyError):
            date_range_days = 0
            
        st.markdown("""
        <div class="insight-card">
            <h3 style="color: #128c7e;">Time Span</h3>
            <h2>{:,} days</h2>
            <p>{} to {}</p>
        </div>
        """.format(
            date_range_days,
            insights['date_range']['start'],
            insights['date_range']['end']
        ), unsafe_allow_html=True)

def plot_top_authors(insights):
    """Plot a bar chart of top message senders"""
    if not insights or 'author_stats' not in insights:
        return
    
    # Get author statistics
    author_counts = insights['author_stats']['counts']
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame({
        'Author': list(author_counts.keys()),
        'Messages': list(author_counts.values())
    }).sort_values('Messages', ascending=False)
    
    # Limit to top 5 authors
    df = df.head(5)
    
    # Create the plot
    fig = px.bar(
        df,
        x='Author',
        y='Messages',
        title='Top Message Senders',
        color='Messages',
        color_continuous_scale='Viridis',
        labels={'Author': 'Person', 'Messages': 'Number of Messages'}
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Message Count',
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_hourly_activity(insights):
    """Plot hourly activity as a heatmap"""
    if not insights or 'time_stats' not in insights or 'hourly_activity' not in insights['time_stats']:
        return
    
    # Get hourly activity
    hourly_activity = insights['time_stats']['hourly_activity']
    
    # Create list of all hours (0-23)
    all_hours = list(range(24))
    
    # Create a new dictionary to ensure proper type handling
    hour_counts = {}
    
    # Process the existing hourly activity data
    for hour_key, count in hourly_activity.items():
        # Convert string keys to integers
        if isinstance(hour_key, str):
            try:
                hour_int = int(hour_key)
                hour_counts[hour_int] = count
            except ValueError:
                continue  # Skip invalid hour strings
        else:
            hour_counts[hour_key] = count
    
    # Ensure all hours are in the data (with zeros for missing hours)
    for hour in all_hours:
        if hour not in hour_counts:
            hour_counts[hour] = 0
    
    # Convert to DataFrame for plotting - safely handling all types
    df = pd.DataFrame({
        'Hour': [f"{h:02d}:00" for h in sorted(hour_counts.keys())],
        'Count': [hour_counts[h] for h in sorted(hour_counts.keys())]
    })
    
    # Create the plot
    fig = px.bar(
        df,
        x='Hour',
        y='Count',
        title='Message Activity by Hour of Day',
        labels={'Hour': 'Time of Day', 'Count': 'Number of Messages'},
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Message Count',
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        if GOOGLE_API_KEY:
            st.session_state['model'] = setup_gemini(GOOGLE_API_KEY)
        else:
            st.session_state['model'] = None
    
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    
    if 'db' not in st.session_state:
        st.session_state['db'] = None
        
    if 'insights' not in st.session_state:
        st.session_state['insights'] = None
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        
    if 'file_path' not in st.session_state:
        st.session_state['file_path'] = None
        
    if 'show_context' not in st.session_state:
        st.session_state['show_context'] = False
        
    if 'using_saved_db' not in st.session_state:
        st.session_state['using_saved_db'] = False
        
    if 'custom_db_path' not in st.session_state:
        st.session_state['custom_db_path'] = None
        
    if 'last_loaded_db' not in st.session_state:
        st.session_state['last_loaded_db'] = None
        
    if 'clear_input' not in st.session_state:
        st.session_state['clear_input'] = False
        
    if 'enter_pressed' not in st.session_state:
        st.session_state['enter_pressed'] = False

def display_chat_history():
    """Display the chat history between user and AI in a chatbot-like interface"""
    # Create a modern chat container
    # st.markdown("<div class='messages-container' id='messages-container'>", unsafe_allow_html=True)
    st.markdown("<div>", unsafe_allow_html=True)
    
    for i, entry in enumerate(st.session_state['chat_history']):
        question = entry['question']
        answer = entry.get('answer')
        timestamp = entry.get('timestamp', datetime.now().strftime('%H:%M'))
        
        # Display the question (user message)
        st.markdown(f"""
        <div class="message-container">
            <div class="message-bubble message-sent">
                <div class="message-text">{question}</div>
                <div class="message-info">
                    <span>You • {timestamp}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Only display the answer if it exists (may be None while processing)
        if answer:
            st.markdown(f"""
            <div class="message-container">
                <div class="message-bubble message-received">
                    <div class="message-text">{answer}</div>
                    <div class="message-info">
                        <span>AI Assistant • {timestamp}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Optionally display the context if show_context is True
        if st.session_state['show_context'] and ('context' in entry or 'top_chunks' in entry):
            with st.expander(f"📋 View Context Sources (Entry #{i+1})"):
                # Use our enhanced context display
                display_context_chunks(entry, max_chunks=st.session_state.get('max_context_chunks', 5))
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # # Add JavaScript for basic auto-scrolling without animations
    # st.markdown("""
    # <script>
    #     // Function to scroll messages container to bottom
    #     function scrollToBottom() {
    #         const messagesContainer = document.getElementById('messages-container');
    #         if (messagesContainer) {
    #             messagesContainer.scrollTop = messagesContainer.scrollHeight;
    #         }
    #     }
        
    #     // Call function after page loads
    #     window.addEventListener('load', scrollToBottom);
    #     const observer = new MutationObserver(scrollToBottom);
    #     const messagesContainer = document.getElementById('messages-container');
    #     if (messagesContainer) {
    #         observer.observe(messagesContainer, { childList: true });
    #     }
    # </script>
    # """, unsafe_allow_html=True)

def display_typing_indicator():
    """Display a minimalist typing indicator without animation"""
    st.markdown("""
    <div class="message-container">
        <div class="typing-bubble">
            <div class="message-text">Processing query...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_example_questions():
    """Display example questions for the user"""
    example_questions = [
        "Who sends the most messages in this chat?",
        "What topics do people talk about most often?",
        "When are people most active in this chat?",
        "Summarize the conversation from last week",
        "What kind of media is shared in this chat?",
        "Who uses the most emojis?",
        "What was the conversation about yesterday?"
    ]
    
    return example_questions

# Main application function
def main():
    # Initialize session state
    initialize_session_state()
    
    # Display logo
    display_logo()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Initialize process_clicked to False by default
        process_clicked = False
        
        # Option to use existing database
        use_saved_db = st.checkbox("Use existing vector database", 
                                  value=st.session_state.get('using_saved_db', False),
                                  help="Use this option if you have already processed a chat file previously and just want to query the existing database")
        st.session_state['using_saved_db'] = use_saved_db
        
        if use_saved_db:
            # Find available vector databases
            available_db_paths = []
            default_db_exists = False
            
            # Check if default database exists
            if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH) and os.path.exists(os.path.join(DB_PATH, "index.faiss")):
                available_db_paths.append(DB_PATH)
                default_db_exists = True
            
            # Look for other potential vector databases (folders with index.faiss file)
            for item in os.listdir():
                if item != DB_PATH and os.path.isdir(item) and os.path.exists(os.path.join(item, "index.faiss")):
                    available_db_paths.append(item)
            
            if available_db_paths:
                # Allow selection of database if multiple are available
                selected_db_path = st.selectbox(
                    "Select vector database to load:", 
                    available_db_paths,
                    index=0 if default_db_exists else 0
                )
                
                auto_load = st.checkbox("Auto-load database", value=True)
                
                if st.button("Load Selected Database") or auto_load:
                    with st.spinner(f"Loading vector database from {selected_db_path}..."):
                        # Load the selected database
                        st.session_state['db'] = load_faiss_db(selected_db_path)
                        st.session_state['custom_db_path'] = selected_db_path
                        st.session_state['last_loaded_db'] = selected_db_path
                        
                        # Load chat insights
                        if os.path.exists("chat_insights.json"):
                            with open("chat_insights.json", "r") as f:
                                st.session_state['insights'] = json.load(f)
                                st.success(f"Loaded chat insights with {st.session_state['insights']['total_messages']} messages")
                        else:
                            # Try to find insights in the same directory as the database
                            insights_path = os.path.join(os.path.dirname(selected_db_path), "chat_insights.json")
                            if os.path.exists(insights_path):
                                with open(insights_path, "r") as f:
                                    st.session_state['insights'] = json.load(f)
                                    st.success(f"Loaded chat insights with {st.session_state['insights']['total_messages']} messages")
                            else:
                                st.warning("No chat insights found. Some features might be limited.")
                                st.session_state['insights'] = None
                            
                        st.success(f"Successfully loaded vector database from {selected_db_path}")
                        
                        # We don't need the actual dataframe for just querying
                        st.session_state['df'] = pd.DataFrame() if st.session_state['df'] is None else st.session_state['df']
                        
                        # Show database info if available
                        try:
                            db_size = len(os.listdir(selected_db_path))
                            db_created = datetime.fromtimestamp(os.path.getctime(os.path.join(selected_db_path, "index.faiss")))
                            st.info(f"Database contains {db_size} files. Created on {db_created.strftime('%Y-%m-%d')}. Ready for querying!")
                        except Exception:
                            pass
                        
                        # Set process_clicked to True to ensure analytics are displayed
                        process_clicked = True
            else:
                st.error("No vector databases found. Please process a chat file first.")
                st.session_state['using_saved_db'] = False
                use_saved_db = False
        
        if not use_saved_db:
            # File upload or selection
            st.subheader("1. Select Chat File")
            uploaded_file = st.file_uploader("Upload WhatsApp chat file", type=["txt"])
            
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                temp_file_path = f"temp_chat_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                st.session_state['file_path'] = temp_file_path
                st.success(f"Uploaded: {uploaded_file.name}")
            
            # Or use existing file
            st.markdown("**OR**")
            existing_chat_files = [f for f in os.listdir() if f.endswith('.txt') and os.path.isfile(f)]
            if existing_chat_files:
                selected_file = st.selectbox("Select existing chat file", [""] + existing_chat_files)
                if selected_file:
                    st.session_state['file_path'] = selected_file
            
            # Process button
            process_clicked = st.button("Process Chat File", key="process_chat")
        
        # Advanced options
        with st.expander("Advanced Options"):
            context_size = st.slider("Context Size (messages before/after)", min_value=1, max_value=10, value=CONTEXT_SIZE)
            retriever_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=20, value=RETRIEVER_K)
            
            # Context display options
            st.subheader("Context Display")
            st.checkbox("Show retrieved context", key="show_context", value=st.session_state['show_context'], 
                       help="Show the context messages retrieved from your chat history")
                       
            # Initialize max_context_chunks if not already in session_state
            if 'max_context_chunks' not in st.session_state:
                st.session_state['max_context_chunks'] = 5
                
            # Slider for number of context chunks to display
            if st.session_state['show_context']:
                st.session_state['max_context_chunks'] = st.slider(
                    "Number of context chunks to show", 
                    min_value=1, 
                    max_value=10, 
                    value=st.session_state['max_context_chunks'],
                    help="How many message chunks to display in the context viewer"
                )
            
            # Add database debug option
            if st.session_state['db'] is not None and st.checkbox("Show database debug info"):
                st.markdown("### Vector Database Information")
                
                try:
                    # Get information about the database
                    db_size = st.session_state['db'].index.ntotal if hasattr(st.session_state['db'], 'index') else "Unknown"
                    db_path = DB_PATH if not st.session_state.get('custom_db_path') else st.session_state.get('custom_db_path')
                    embedding_size = st.session_state['db'].index.d if hasattr(st.session_state['db'], 'index') else "Unknown"
                    
                    st.code(f"""
Database path: {db_path}
Vector count: {db_size}
Embedding dimensions: {embedding_size}
Retriever k-value: {retriever_k}
Context size: {context_size}
                    """)
                    
                    # Show last update time if available
                    try:
                        if os.path.exists(os.path.join(db_path, "index.faiss")):
                            mod_time = os.path.getmtime(os.path.join(db_path, "index.faiss"))
                            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                            st.info(f"Database last updated: {mod_time_str}")
                    except Exception:
                        pass
                    
                    # Add a clear database button
                    if st.button("Clear Loaded Database", key="clear_db"):
                        st.session_state['db'] = None
                        st.session_state['custom_db_path'] = None
                        st.success("Database cleared from memory")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error getting database info: {str(e)}")
            
            if st.button("Rebuild Vector Database"):
                if st.session_state['df'] is not None:
                    with st.spinner("Rebuilding vector database..."):
                        structured_messages = format_messages_for_rag(st.session_state['df'], context_size=context_size)
                        st.session_state['db'] = build_faiss_db(structured_messages, db_path=DB_PATH)
                    st.success("Vector database rebuilt successfully!")
                else:
                    st.warning("Please process a chat file first.")
        
        # API Key management
        with st.expander("API Keys"):
            api_key = st.text_input("Google API Key", value=GOOGLE_API_KEY, type="password")
            if st.button("Update API Key"):
                os.environ['GOOGLE_API_KEY'] = api_key
                st.session_state['model'] = setup_gemini(api_key)
                st.success("API key updated!")
        
        # Database management expander
        with st.expander("Database Management"):
            if st.session_state['db'] is not None:
                st.success(f"Database loaded: {st.session_state.get('custom_db_path', DB_PATH)}")
                
                if st.button("Export Database Info"):
                    # Create a summary of the current database
                    db_info = {
                        "path": st.session_state.get('custom_db_path', DB_PATH),
                        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "retriever_k": retriever_k,
                        "context_size": context_size
                    }
                    
                    # Add insights information if available
                    if st.session_state['insights']:
                        db_info["message_count"] = st.session_state['insights'].get('total_messages')
                        db_info["author_count"] = st.session_state['insights'].get('author_stats', {}).get('total_authors')
                        
                    # Export as JSON
                    db_info_path = "db_info.json"
                    with open(db_info_path, "w") as f:
                        json.dump(db_info, f, indent=2)
                    
                    st.download_button(
                        label="Download DB Info",
                        data=json.dumps(db_info, indent=2),
                        file_name="whatsapp_db_info.json",
                        mime="application/json",
                    )
            
            # List found databases
            st.markdown("### Found Vector Databases")
            found_dbs = [d for d in os.listdir() if os.path.isdir(d) and os.path.exists(os.path.join(d, "index.faiss"))]
            
            if found_dbs:
                for db in found_dbs:
                    st.code(f"{db} - {len(os.listdir(db))} files")
            else:
                st.info("No vector databases found in the current directory")
        
        # About section
        st.markdown("---")
        st.markdown("""
        ### About
        This app uses Retrieval-Augmented Generation (RAG) to analyze WhatsApp chats.
        
        Created with:
        - Streamlit
        - LangChain
        - Gemini AI
        - FAISS Vector Database
        """)
    
    # Main content area
    # Process a new chat file if requested and not using saved DB
    if process_clicked and not st.session_state['using_saved_db'] and st.session_state.get('file_path'):
        with st.spinner("Processing chat file..."):
            # Parse the chat
            df = parse_whatsapp_chat(st.session_state['file_path'])
            if len(df) == 0:
                st.error("No messages found in the chat file.")
                return
            
            st.session_state['df'] = df
            
            # Format messages for RAG
            structured_messages = format_messages_for_rag(df, context_size=context_size)
            
            # Generate or load insights
            insights = generate_chat_insights(df)
            st.session_state['insights'] = insights
            
            # Save insights to the same location as the database will be
            try:
                # Save insights to standard location
                with open("chat_insights.json", "w") as f:
                    json.dump(insights, f, indent=2)
                
                # Also save in the target DB folder if it's a different location
                if DB_PATH != "chat_insights.json" and DB_PATH != ".":
                    # Make sure the directory exists
                    os.makedirs(DB_PATH, exist_ok=True)
                    insights_path = os.path.join(DB_PATH, "chat_insights.json")
                    with open(insights_path, "w") as f:
                        json.dump(insights, f, indent=2)
                    st.success(f"Chat insights saved to both default and database locations")
            except Exception as e:
                st.warning(f"Failed to save insights file: {str(e)}")
            
            # Build or load the vector database
            if not os.path.exists(DB_PATH) or process_clicked:
                st.session_state['db'] = build_faiss_db(structured_messages, db_path=DB_PATH)
            else:
                st.session_state['db'] = load_faiss_db(DB_PATH)
    
    # Display content if database is loaded (either processed or using saved DB)
    if st.session_state['db'] is not None:
        # Display chat insights if available
        if st.session_state['insights'] is not None:
            try:
                st.markdown("<div class='chat-header'><h2>Chat Overview</h2></div>", unsafe_allow_html=True)
                display_chat_insights(st.session_state['insights'])
                
                # Display visualizations if we have insights
                col1, col2 = st.columns(2)
                with col1:
                    plot_top_authors(st.session_state['insights'])
                with col2:
                    plot_hourly_activity(st.session_state['insights'])
            except Exception as e:
                st.error(f"Error displaying insights: {str(e)}")
                st.info("The insights might be in an incompatible format. You can still query the database.")
                # Continue with the rest of the app even if the insights visualization fails
        
        # Display message sample if we have a dataframe and it's not empty
        if st.session_state['df'] is not None and len(st.session_state['df']) > 0:
            st.markdown("<div class='chat-header'><h2>Message Samples</h2></div>", unsafe_allow_html=True)
            display_message_sample(st.session_state['df'], num_messages=5)
        
        # Interactive analysis section - always show this when we have a DB
        st.markdown("<div class='chat-header'><h2>Ask Questions About Your Chat</h2></div>", unsafe_allow_html=True)
        
        # Warning if using just the DB without insights
        if st.session_state['insights'] is None and st.session_state['using_saved_db']:
            st.warning("Note: You are using a saved database without chat insights. Answers may have limited statistical context.")
        
        # Example questions as clickable buttons
        st.markdown("### Example Questions")
        example_cols = st.columns(2)
        examples = display_example_questions()
        
        for i, example in enumerate(examples):
            col = example_cols[i % 2]
            if col.button(example, key=f"example_{i}"):
                # Use the example as the question
                with st.spinner("Searching for relevant messages..."):
                    context = query_db(example, st.session_state['db'])
                    
                with st.spinner("Generating answer..."):
                    if st.session_state['model']:
                        answer = qa_with_gemini(context, example, st.session_state['model'])
                        
                        # Add to chat history
                        st.session_state['chat_history'].append({
                            'question': example,
                            'answer': answer,
                            'context': context
                        })
                        
                        # Force a rerun to display the new answer
                        st.rerun()
                    else:
                        st.error("Gemini model not initialized. Please check your API key.")
        
        # Create a modern chatbot-style interface
        st.markdown("<div class='chat-header'><h2>Chat with Your WhatsApp Data</h2></div>", unsafe_allow_html=True)
        
        # Display chat history first
        if st.session_state['chat_history']:
            display_chat_history()
            
            # Add option to clear chat history
            if st.button("🗑️ Clear Chat History", key="clear_history"):
                st.session_state['chat_history'] = []
                st.rerun()
        
        # Custom question input with better styling
        st.markdown("<div class='query-input-container'>", unsafe_allow_html=True)
        
        # Check if we need to clear the input (from previous clear button click)
        input_value = "" if st.session_state.get('clear_input', False) else st.session_state.get("chat_input", "")
        
        # Define a callback for when enter is pressed
        def handle_enter():
            if st.session_state.get("chat_input", "").strip():
                # This will be processed in the main flow as if submit_button was clicked
                st.session_state["enter_pressed"] = True
        
        question = st.text_input("", placeholder="Type your question about the chat...", 
                               key="chat_input", value=input_value, on_change=handle_enter)
        
        # Reset the clear flag after using it
        if st.session_state.get('clear_input', False):
            st.session_state['clear_input'] = False
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([4, 1, 1])
        with col2:
            clear_button = st.button("🔄 Clear", use_container_width=True)
        
        with col3:
            submit_button = st.button("💬 Ask", use_container_width=True)
            
        # Handle the clear button by using a special flag in session state
        if clear_button:
            # Set a flag to clear the input on next rerun
            st.session_state["clear_input"] = True
            st.rerun()
        
        # Check if Enter was pressed or submit button was clicked
        enter_pressed = st.session_state.get('enter_pressed', False)
        
        if (submit_button or enter_pressed) and question:
            # Reset the enter_pressed flag for future use
            st.session_state['enter_pressed'] = False
            
            # Add the question to chat history immediately to show it
            temp_entry = {'question': question, 'answer': None}
            st.session_state['chat_history'].append(temp_entry)
        
            # Show the updated chat with the new question
            display_chat_history()
            
            # Search for relevant messages
            context = query_db(question, st.session_state['db'])
            
            # Extract top chunks for display based on user settings
            max_chunks = st.session_state.get('max_context_chunks', 5)
            context_chunks = context.split("\n\n")
            top_chunks = context_chunks[:max_chunks] if len(context_chunks) > max_chunks else context_chunks
            
            # Use our enhanced message display
            if st.session_state['model']:
                # Show enhanced retrieval and generate answer
                answer, metadata = enhance_message_display(
                    top_chunks,
                    question,
                    context,
                    st.session_state['model']
                )
                
                # Update the last entry with the answer and all metadata
                st.session_state['chat_history'][-1] = {
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'top_chunks': top_chunks,
                    'metadata': metadata,
                    'timestamp': metadata['timestamp']
                }
                
                # We'll use a different approach for clearing the input
                # Instead of directly modifying session state, we'll use a rerun
                # The input will be cleared because we're not preserving it between reruns
                st.rerun()
            else:
                st.error("Gemini model not initialized. Please check your API key.")
        
        # We already display chat history in the chat interface above, so we don't need to display it again here
    else:
        # Display welcome message if no data is loaded
        st.markdown("""
        <div class="insight-card" style="text-align: center; padding: 30px;">
            <h1>Welcome to WhatsApp Chat Analyzer</h1>
            <p style="font-size: 16px; margin-top: 20px; color: #cccccc;">
                Analyze your WhatsApp conversations with AI-powered insights.
            </p>
            <p style="font-size: 15px; margin-top: 20px; color: #cccccc;">
                To begin:
                <ul style="text-align: left; max-width: 500px; margin: 10px auto; color: #cccccc;">
                    <li>Upload a WhatsApp chat export file from the sidebar on the left, or</li>
                    <li>If you've analyzed a chat previously, tick the "Use existing vector database" option in the sidebar</li>
                </ul>
            </p>
            <div style="margin-top: 30px; border-top: 1px solid #333333; padding-top: 20px;">
                <h3>How to export a WhatsApp chat:</h3>
                <ol style="text-align: left; max-width: 500px; margin: 0 auto; color: #cccccc;">
                    <li>Open a WhatsApp conversation</li>
                    <li>Tap on the three dots (⋮) in the top right</li>
                    <li>Select "More" > "Export chat"</li>
                    <li>Choose "Without media"</li>
                    <li>Save or share the .txt file</li>
                    <li>Upload it here</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
