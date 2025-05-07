import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import emoji
from collections import Counter

# Set page configuration
st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬", layout="wide")

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

# Function to parse WhatsApp chat
@st.cache_data
def parse_chat(file_content, min_messages=10):
    """
    Parse WhatsApp chat and return a DataFrame with messages
    
    Parameters:
    file_content (str): Content of WhatsApp chat export file
    min_messages (int): Minimum number of messages for a user to be included
    
    Returns:
    DataFrame with parsed messages
    """
    # Define regex patterns
    # Define regex patterns for the format: "23/04/2025, 02:17 - Arka: message"
    date_regex = re.compile(r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?')
    author_regex = re.compile(r' - ([^:]+):')
    
    messages = []
    lines = file_content.split('\n')
    
    current_date = None
    current_author = None
    current_message = None
    
    for i, line in enumerate(lines):
        date_match = date_regex.search(line)
        if date_match:
            # If we have previous message data, save it
            if current_date and current_author and current_message:
                messages.append({
                    'date': current_date,
                    'author': current_author,
                    'message': current_message
                })
            
            # Extract date and time
            date_str = date_match.group()
            
            # Try different date formats
            date_formats = [
                '%d/%m/%Y, %H:%M',  # 01/12/2023, 14:30
                '%d/%m/%Y, %H:%M:%S',  # 01/12/2023, 14:30:45
                '%m/%d/%Y, %H:%M',  # 12/01/2023, 14:30
                '%m/%d/%Y, %H:%M:%S',  # 12/01/2023, 14:30:45
                '%d/%m/%y, %H:%M',  # 01/12/23, 14:30
                '%d/%m/%y, %H:%M:%S',  # 01/12/23, 14:30:45
                '%m/%d/%y, %H:%M',  # 12/01/23, 14:30
                '%m/%d/%y, %H:%M:%S',  # 12/01/23, 14:30:45
                '%d/%m/%Y, %I:%M %p',  # 01/12/2023, 2:30 PM
                '%m/%d/%Y, %I:%M %p',  # 12/01/2023, 2:30 PM
                '%d/%m/%y, %I:%M %p',  # 01/12/23, 2:30 PM
                '%m/%d/%y, %I:%M %p',  # 12/01/23, 2:30 PM
            ]
            
            for fmt in date_formats:
                try:
                    current_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if not current_date:
                continue  # Skip if date parsing failed
            
            # Extract author and message
            author_match = author_regex.search(line)
            if author_match:
                current_author = author_match.group(1).strip()
                message_start = author_match.end() + 1
                current_message = line[message_start:].strip()
            else:
                # System message or other format
                current_author = "System"
                current_message = line[date_match.end():].strip()
                if current_message.startswith("- "):
                    current_message = current_message[2:].strip()
        else:
            # Continuation of previous message
            if current_message is not None:
                current_message += " " + line.strip()
    
    # Add the last message
    if current_date and current_author and current_message:
        messages.append({
            'date': current_date,
            'author': current_author,
            'message': current_message
        })
    
    # Create DataFrame
    df = pd.DataFrame(messages)
    
    if len(df) == 0:
        return df, []
    
    # Filter users with minimum message count
    user_counts = df['author'].value_counts()
    qualified_users = user_counts[user_counts >= min_messages].index.tolist()
    
    return df, qualified_users

# Function for basic statistics
def get_basic_stats(df):
    """Calculate basic statistics from the chat DataFrame"""
    if len(df) == 0:
        return {}
    
    # Total messages
    total_messages = len(df)
    
    # Messages per user
    messages_per_user = df['author'].value_counts()
    
    # Characters per user
    df['message_length'] = df['message'].apply(len)
    chars_per_user = df.groupby('author')['message_length'].sum().sort_values(ascending=False)
    
    # Average message length per user
    avg_length_per_user = df.groupby('author')['message_length'].mean().sort_values(ascending=False)
    
    # Date range
    date_range = (df['date'].max() - df['date'].min()).days + 1
    
    # Messages per day
    df['day'] = df['date'].dt.date
    messages_per_day = df.groupby('day').size()
    avg_messages_per_day = messages_per_day.mean()
    
    # Most active day
    most_active_day = messages_per_day.idxmax()
    most_active_day_count = messages_per_day.max()
    
    # Time analysis
    df['hour'] = df['date'].dt.hour
    hourly_activity = df.groupby('hour').size()
    peak_hour = hourly_activity.idxmax()
    
    # Most active user
    most_active_user = messages_per_user.idxmax()
    most_active_user_count = messages_per_user.max()
    
    # Most verbose user (most characters)
    most_verbose_user = chars_per_user.idxmax()
    most_verbose_chars = chars_per_user.max()
    
    # Day of week analysis
    df['day_of_week'] = df['date'].dt.day_name()
    day_of_week_counts = df.groupby('day_of_week').size()
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

# Function to analyze media content
def analyze_media(df):
    """Analyze media content from messages"""
    if len(df) == 0:
        return {}
    
    # Count image, video, audio, and document messages
    image_pattern = r'<Media omitted>|image omitted|\.jpg|\.jpeg|\.png|\.gif'
    video_pattern = r'video omitted|\.mp4|\.mov|\.avi'
    audio_pattern = r'audio omitted|\.mp3|\.ogg|\.m4a'
    document_pattern = r'document omitted|\.pdf|\.doc|\.docx|\.xls|\.xlsx|\.ppt|\.pptx'
    
    df['has_image'] = df['message'].str.contains(image_pattern, case=False)
    df['has_video'] = df['message'].str.contains(video_pattern, case=False)
    df['has_audio'] = df['message'].str.contains(audio_pattern, case=False)
    df['has_document'] = df['message'].str.contains(document_pattern, case=False)
    
    media_counts = {
        'images': df['has_image'].sum(),
        'videos': df['has_video'].sum(),
        'audio': df['has_audio'].sum(),
        'documents': df['has_document'].sum()
    }
    
    # Media per user
    media_per_user = {
        'images': df[df['has_image']].groupby('author').size().sort_values(ascending=False),
        'videos': df[df['has_video']].groupby('author').size().sort_values(ascending=False),
        'audio': df[df['has_audio']].groupby('author').size().sort_values(ascending=False),
        'documents': df[df['has_document']].groupby('author').size().sort_values(ascending=False)
    }
    
    return {
        'media_counts': media_counts,
        'media_per_user': media_per_user
    }

# Function to analyze emoji usage
def analyze_emojis(df):
    """Analyze emoji usage in messages"""
    if len(df) == 0:
        return {}
    
    # Extract emojis from messages
    def extract_emojis(text):
        if not isinstance(text, str):
            return []
        return [c for c in text if c in emoji.EMOJI_DATA]
    
    df['emojis'] = df['message'].apply(extract_emojis)
    df['emoji_count'] = df['emojis'].apply(len)
    
    # Count emojis
    all_emojis = []
    for emoji_list in df['emojis']:
        all_emojis.extend(emoji_list)
    
    emoji_counts = Counter(all_emojis)
    
    # Emojis per user
    user_emoji_counts = df.groupby('author')['emoji_count'].sum().sort_values(ascending=False)
    
    # Most common emoji by user
    user_most_common_emoji = {}
    for author in df['author'].unique():
        author_emojis = []
        for emoji_list in df[df['author'] == author]['emojis']:
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

# Function to generate time series for message counts
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
    
    # Filter by users if specified
    if users:
        df = df[df['author'].isin(users)]
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Set date as index for resampling
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Create a DataFrame with time series for each user
    time_series = {}
    
    # Overall time series
    all_messages = df_copy.set_index('date').resample(freq).size()
    time_series['All'] = all_messages
    
    # Per-user time series
    for user in df_copy['author'].unique():
        user_df = df_copy[df_copy['author'] == user]
        user_messages = user_df.set_index('date').resample(freq).size()
        time_series[user] = user_messages
    
    # Combine all time series
    result = pd.DataFrame(time_series)
    result = result.fillna(0)
    
    return result

# Function to create word clouds
def generate_word_cloud(df, user=None):
    """Generate word cloud for a specific user or all users"""
    if len(df) == 0:
        return None
    
    # Filter by user if specified
    if user:
        messages = df[df['author'] == user]['message'].str.cat(sep=' ')
    else:
        messages = df['message'].str.cat(sep=' ')
    
    # Remove URLs and media references
    messages = re.sub(r'https?://\S+|www\.\S+|<Media omitted>', '', messages)
    
    # Remove emojis and special characters
    messages = re.sub(r'[^\w\s]', '', messages)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=stop_words,
        min_font_size=10,
        max_words=100
    ).generate(messages)
    
    return wordcloud

# Main application
def main():
    st.title("WhatsApp Chat Analyzer 📊")
    st.write("Upload your WhatsApp chat export to analyze conversation patterns and statistics")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=['txt'])
        
        # Only show these options if a file is uploaded
        if uploaded_file:
            min_messages = st.number_input("Minimum messages per user", min_value=1, value=10)
            
            st.header("Time Series Settings")
            time_freq = st.selectbox(
                "Time series frequency",
                options=[
                    ("Daily", "D"),
                    ("Weekly", "W"),
                    ("Monthly", "M")
                ],
                format_func=lambda x: x[0]
            )[1]
            
            days_to_analyze = st.slider(
                "Days to analyze (most recent)",
                min_value=7,
                max_value=365,
                value=30
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
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        else:
            df_filtered = df
        
        # Create tabs for different analyses
        tabs = st.tabs([
            "Overview",
            "User Analysis",
            "Time Series",
            "User Comparison",
            "Word Clouds",
            "Media Analysis",
            "Chat Patterns"  # Add Chat Patterns to the initial tabs list
        ])
        
        # 1. Overview Tab
        with tabs[0]:
            st.header("Chat Overview")
            
            # Calculate basic statistics
            with st.spinner("Calculating statistics..."):
                stats = get_basic_stats(df_filtered)
            
            if not stats:
                st.warning("No data to display.")
                return
            
            # Display basic information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", f"{stats['total_messages']:,}")
                st.metric("Total Days", f"{stats['date_range']:,}")
            
            with col2:
                st.metric("Avg. Messages/Day", f"{stats['avg_messages_per_day']:.1f}")
                st.metric("Peak Hour", f"{stats['peak_hour']}:00")
            
            with col3:
                st.metric("Most Active User", stats['most_active_user'])
                st.metric("Messages", f"{stats['most_active_user_count']:,}")
            
            # Message distribution by user
            st.subheader("Message Distribution by User")
            fig = px.pie(
                values=stats['messages_per_user'].values,
                names=stats['messages_per_user'].index,
                title="Messages per User",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Message activity by hour of day
            st.subheader("Activity by Hour of Day")
            fig = px.line(
                x=stats['hourly_activity'].index,
                y=stats['hourly_activity'].values,
                markers=True,
                title="Messages by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Messages'}
            )
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Message activity by day of week
            st.subheader("Activity by Day of Week")
            fig = px.bar(
                x=stats['day_of_week_counts'].index,
                y=stats['day_of_week_counts'].values,
                title="Messages by Day of Week",
                labels={'x': 'Day', 'y': 'Number of Messages'},
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 2. User Analysis Tab
        with tabs[1]:
            st.header("User Analysis")
            
            # User selector
            selected_user = st.selectbox(
                "Select a user to analyze",
                options=qualified_users
            )
            
            # Get user data
            user_data = df_filtered[df_filtered['author'] == selected_user]
            
            # Basic user metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", f"{len(user_data):,}")
                pct_of_total = len(user_data) / len(df_filtered) * 100
                st.metric("% of All Messages", f"{pct_of_total:.1f}%")
            
            with col2:
                avg_length = user_data['message'].str.len().mean()
                st.metric("Avg. Message Length", f"{avg_length:.1f} chars")
                
                total_chars = user_data['message'].str.len().sum()
                st.metric("Total Characters", f"{total_chars:,}")
            
            with col3:
                user_data['hour'] = user_data['date'].dt.hour
                most_active_hour = user_data.groupby('hour').size().idxmax()
                st.metric("Most Active Hour", f"{most_active_hour}:00")
                
                user_data['day_of_week'] = user_data['date'].dt.day_name()
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
                color_discrete_sequence=['#00CC96']
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
                color_discrete_sequence=['#AB63FA']
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
                        color_discrete_sequence=['#FFA15A']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 3. Time Series Tab
        with tabs[2]:
            st.header("Time Series Analysis")
            
            # User selector for time series (multiselect)
            selected_users_ts = st.multiselect(
                "Select users to include (leave empty for all)",
                options=qualified_users,
                default=[qualified_users[0]] if qualified_users else []
            )
            
            # Generate time series
            with st.spinner("Generating time series..."):
                if selected_users_ts:
                    time_series_df = generate_time_series(df_filtered, selected_users_ts, time_freq)
                else:
                    time_series_df = generate_time_series(df_filtered, freq=time_freq)
            
            # Plotting the time series
            st.subheader("Message Activity Over Time")
            
            # Convert frequency to human-readable label
            freq_label = {
                'D': 'Daily',
                'W': 'Weekly',
                'M': 'Monthly'
            }.get(time_freq, 'Daily')
            
            fig = px.line(
                time_series_df,
                x=time_series_df.index,
                y=time_series_df.columns,
                title=f"{freq_label} Message Activity",
                labels={'x': 'Date', 'y': 'Number of Messages', 'variable': 'User'}
            )
            fig.update_layout(legend_title_text='User')
            st.plotly_chart(fig, use_container_width=True)
            
            # Rolling average
            st.subheader("Activity Trend (Rolling Average)")
            window_size = {
                'D': 7,   # 7-day average for daily data
                'W': 4,   # 4-week average for weekly data
                'M': 3,   # 3-month average for monthly data
            }.get(time_freq, 7)
            
            if len(time_series_df) > window_size:
                rolling_df = time_series_df.rolling(window=window_size).mean()
                
                fig = px.line(
                    rolling_df,
                    x=rolling_df.index,
                    y=rolling_df.columns,
                    title=f"{window_size}-{freq_label.lower()} Rolling Average",
                    labels={'x': 'Date', 'y': 'Number of Messages', 'variable': 'User'}
                )
                fig.update_layout(legend_title_text='User')
                st.plotly_chart(fig, use_container_width=True)
        
        # 4. User Comparison Tab
        with tabs[3]:
            st.header("User Comparison")
            
            # Allow selecting up to 5 users to compare
            compare_users = st.multiselect(
                "Select up to 5 users to compare",
                options=qualified_users,
                default=qualified_users[:min(2, len(qualified_users))]
            )
            
            if len(compare_users) > 5:
                st.warning("Please select up to 5 users for comparison.")
                compare_users = compare_users[:5]
            
            if len(compare_users) >= 2:
                # Message count comparison
                st.subheader("Message Count Comparison")
                user_counts = stats['messages_per_user'][stats['messages_per_user'].index.isin(compare_users)]
                fig = px.bar(
                    x=user_counts.index,
                    y=user_counts.values,
                    title="Message Count by User",
                    labels={'x': 'User', 'y': 'Number of Messages'},
                    color=user_counts.index,
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Message length comparison
                st.subheader("Average Message Length Comparison")
                user_avg_lengths = stats['avg_length_per_user'][stats['avg_length_per_user'].index.isin(compare_users)]
                fig = px.bar(
                    x=user_avg_lengths.index,
                    y=user_avg_lengths.values,
                    title="Average Message Length by User",
                    labels={'x': 'User', 'y': 'Average Characters per Message'},
                    color=user_avg_lengths.index,
                    text_auto='.1f'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Side-by-side hourly activity comparison
                st.subheader("Hourly Activity Pattern Comparison")
                fig = make_subplots(rows=1, cols=len(compare_users), 
                                   subplot_titles=[f"{user}" for user in compare_users],
                                   shared_yaxes=True)
                
                for i, user in enumerate(compare_users):
                    user_data = df_filtered[df_filtered['author'] == user]
                    user_data['hour'] = user_data['date'].dt.hour
                    hourly_counts = user_data.groupby('hour').size().reindex(range(24), fill_value=0)
                    
                    fig.add_trace(
                        go.Bar(
                            x=hourly_counts.index,
                            y=hourly_counts.values,
                            name=user,
                            marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                        ),
                        row=1, col=i+1
                    )
                
                fig.update_layout(
                    height=400,
                    title_text="Hourly Activity Comparison",
                    showlegend=False
                )
                
                for i in range(len(compare_users)):
                    fig.update_xaxes(title_text="Hour", row=1, col=i+1)
                    if i == 0:
                        fig.update_yaxes(title_text="Number of Messages", row=1, col=i+1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series comparison
                st.subheader("Activity Over Time Comparison")
                ts_comparison = generate_time_series(df_filtered, compare_users, time_freq)
                
                fig = px.line(
                    ts_comparison,
                    x=ts_comparison.index,
                    y=ts_comparison.columns,
                    title="Message Activity Comparison",
                    labels={'x': 'Date', 'y': 'Number of Messages', 'variable': 'User'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least 2 users to compare.")
        
        # 5. Word Clouds Tab
        with tabs[4]:
            st.header("Word Clouds")
            
            # User selector for word cloud
            wc_options = ["All Users"] + qualified_users
            selected_wc_user = st.selectbox(
                "Select a user for word cloud",
                options=wc_options
            )
            
            # Generate word cloud
            with st.spinner("Generating word cloud..."):
                if selected_wc_user == "All Users":
                    wordcloud = generate_word_cloud(df_filtered)
                else:
                    wordcloud = generate_word_cloud(df_filtered, selected_wc_user)
            
            if wordcloud:
                # Display word cloud
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(plt)
            else:
                st.warning("Not enough text data to generate a word cloud.")
        
        # 6. Media Analysis Tab
        with tabs[5]:
            st.header("Media Analysis")
            
            # Calculate media statistics
            with st.spinner("Analyzing media content..."):
                media_stats = analyze_media(df_filtered)
            
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

        # 7. Chat Patterns Tab - replace the tabs.append code with a proper tab access
        with tabs[6]:  # Use index 6 to access the Chat Patterns tab
            st.header("Chat Patterns Analysis")

            # Response time analysis
            st.subheader("Response Time Analysis")

            # Select users for response time analysis
            response_users = st.multiselect(
                "Select users to analyze response times (max 3)",
                options=qualified_users,
                default=qualified_users[:min(2, len(qualified_users))]
            )

            if len(response_users) >= 2 and len(response_users) <= 3:
                # Calculate response times between users
                response_analysis = {}

                for i, user1 in enumerate(response_users):
                    for user2 in response_users[i+1:]:
                        # Filter messages by these two users
                        user_messages = df_filtered[df_filtered['author'].isin([user1, user2])].sort_values('date')
                        
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

                if response_analysis:
                    # Calculate average response times
                    avg_response_times = {k: np.mean(v) for k, v in response_analysis.items()}

                    # Create a DataFrame for plotting
                    response_df = pd.DataFrame({
                        'Direction': list(avg_response_times.keys()),
                        'Avg Response Time (min)': list(avg_response_times.values())
                    })

                    fig = px.bar(
                        response_df,
                        x='Direction',
                        y='Avg Response Time (min)',
                        title="Average Response Time (minutes)",
                        color='Direction',
                        text_auto='.1f'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Distribution of response times
                    st.subheader("Response Time Distribution")

                    # Create figure with 1 row and up to 3 columns
                    num_pairs = len(response_analysis)
                    fig = make_subplots(rows=1, cols=num_pairs, 
                                       subplot_titles=list(response_analysis.keys()))

                    for i, (direction, times) in enumerate(response_analysis.items()):
                        # Filter out extreme values (>95th percentile) for better visualization
                        upper_limit = np.percentile(times, 95)
                        filtered_times = [t for t in times if t <= upper_limit]
                        
                        fig.add_trace(
                            go.Histogram(
                                x=filtered_times,
                                name=direction,
                                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                            ),
                            row=1, col=i+1
                        )

                    fig.update_layout(
                        height=400,
                        title_text="Response Time Distribution (minutes)",
                        showlegend=False
                    )

                    for i in range(num_pairs):
                        fig.update_xaxes(title_text="Minutes", row=1, col=i+1)
                        if i == 0:
                            fig.update_yaxes(title_text="Frequency", row=1, col=i+1)

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough conversation data between these users to analyze response times.")
            elif len(response_users) > 3:
                st.warning("Please select at most 3 users for response time analysis.")
            else:
                st.info("Please select at least 2 users to analyze response patterns.")

            # Conversation Initiator Analysis
            st.subheader("Conversation Initiator Analysis")

            # Define a conversation gap (e.g., 3 hours)
            conversation_gap_hours = st.slider("Hours of inactivity to define a new conversation", 1, 12, 3)
            conversation_gap = timedelta(hours=conversation_gap_hours)

            # Find conversation initiators
            df_sorted = df_filtered.sort_values('date')

            # Add a column to identify new conversations
            df_sorted['time_diff'] = df_sorted['date'].diff()
            df_sorted['new_conversation'] = df_sorted['time_diff'] > conversation_gap

            # Count initiators
            initiators = df_sorted[df_sorted['new_conversation']]['author'].value_counts()
            total_conversations = len(initiators)

            if total_conversations > 0:
                # Convert to DataFrame for plotting
                initiator_df = pd.DataFrame({
                    'User': initiators.index,
                    'Count': initiators.values,
                    'Percentage': (initiators.values / initiators.sum() * 100).round(1)
                })

                # Create the plot
                fig = px.pie(
                    initiator_df,
                    values='Count',
                    names='User',
                    title=f"Conversation Initiators (Total: {total_conversations} conversations)",
                    hover_data=['Percentage'],
                    labels={'Percentage': '% of conversations'}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to analyze conversation initiators.")

            # Activity Heatmap
            st.subheader("Activity Heatmap")

            # Create day of week and hour columns
            heatmap_df = df_filtered.copy()
            heatmap_df['day_of_week'] = heatmap_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
            heatmap_df['hour'] = heatmap_df['date'].dt.hour

            # Create a heatmap of activity
            day_hour_counts = heatmap_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')

            # Create a pivot table for the heatmap
            pivot_data = day_hour_counts.pivot(index='day_of_week', columns='hour', values='count').fillna(0)

            # Reindex to ensure all days and hours appear
            pivot_data = pivot_data.reindex(range(7), fill_value=0)
            pivot_data = pivot_data.reindex(columns=range(24), fill_value=0)

            # Replace day numbers with names
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_data.index = day_names

            # Create the heatmap
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Hour of Day", y="Day of Week", color="Message Count"),
                x=[f"{i:02d}:00" for i in range(24)],
                y=day_names,
                color_continuous_scale="Viridis",
                title="Activity Heatmap (Day of Week vs Hour of Day)"
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

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

        ### Privacy Note:
        All analysis happens in your browser. No data is sent to any server.
        """)

if __name__ == "__main__":
    main()