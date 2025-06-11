"""
Chat parser module for WhatsApp Chat Analyzer
"""
import re
import pandas as pd
from datetime import datetime
import streamlit as st

@st.cache_data
def parse_chat(file_content, min_messages=10):
    """
    Parse WhatsApp chat and return a DataFrame with messages
    
    Parameters:
    file_content (str): Content of WhatsApp chat export file
    min_messages (int): Minimum number of messages for a user to be included
    
    Returns:
    DataFrame with parsed messages and list of qualified users
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
