import streamlit as st
import re
from datetime import datetime

def enhance_message_display(context_chunks, question, context, model):
    """
    Enhance message display by showing message retrieval process.
    
    Args:
        context_chunks (list): List of retrieved context chunks
        question (str): The user's question
        context (str): The full context string
        model: The Gemini model
    
    Returns:
        tuple: (answer, metadata)
    """
    # Count non-empty chunks
    non_empty_chunks = [c for c in context_chunks if c.strip()]
    total_chunks = len(non_empty_chunks)
    
    # Display retrieval information
    st.success(f"✅ Found {total_chunks} relevant messages in your chat")
    
    # Show typing indicator with a more specific message
    st.info("🤔 Analyzing messages and generating answer...")
    
    # Get chat history excluding the current question
    if 'chat_history' in st.session_state:
        history = st.session_state['chat_history'][:-1] 
    else:
        history = []
    
    # Simulate typing with a progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        # Update progress bar
        progress_bar.progress(i + 1)
        
        # No need to sleep, the progress bar adds enough visual effect
        if i == 30:
            st.info("💭 Analyzing context and generating response...")
        elif i == 70:
            st.info("✍️ Crafting your answer...")
    
    # Now call the QA function - import dynamically to avoid circular imports
    from test_raq_qna import qa_with_gemini
    answer = qa_with_gemini(context, question, model, history)
    
    # Remove progress bar when done
    progress_bar.empty()
    
    # Create metadata for retrieval
    metadata = {
        'timestamp': datetime.now().strftime('%H:%M'),
        'found_chunks': total_chunks,
        'top_chunks': context_chunks,
        'context': context
    }
    
    return answer, metadata

def format_message_chunk(chunk_text, index):
    """
    Format a chunk of WhatsApp message in a more readable way.
    
    Args:
        chunk_text (str): The message chunk text
        index (int): The chunk index
        
    Returns:
        str: Formatted HTML for the chunk
    """
    # Clean up the chunk text
    chunk_text = chunk_text.strip()
    
    # Check for conversation section header format used by query_db
    conversation_header_match = re.search(r'^---\s+Conversation\s+(\d+)\s+\(centered around ([^,]+),\s*([^)]+)\)\s+---', chunk_text)
    
    # If this is a conversation header, format it specially
    if conversation_header_match:
        conv_num = conversation_header_match.group(1)
        conv_date = conversation_header_match.group(2)
        conv_author = conversation_header_match.group(3)
        
        # Extract conversation content (everything after the header line)
        content_start = chunk_text.find('\n') + 1
        content = chunk_text[content_start:] if content_start > 0 else ""
        
        # Format message lines for better readability
        formatted_content = ""
        for line in content.split('\n'):
            # Try to identify message pattern with ">>" marker
            is_central = ">>" in line
            line_class = "central-message" if is_central else "context-message"
            line_style = "background-color: #dcf8c6; font-weight: bold;" if is_central else ""
            
            formatted_content += f'<div style="{line_style}" class="{line_class}">{line}</div>'
        
        html = """
        <div class="context-display">
            <div style="background-color: #128c7e; color: white; padding: 8px; border-radius: 5px; margin-bottom: 8px;">
                <span style="font-weight: bold;">Conversation {conv_num}</span>
                <span style="float: right;">{conv_date}, {conv_author}</span>
            </div>
            <div style="margin-left: 5px; border-left: 3px solid #efefef; padding-left: 8px; white-space: pre-wrap;">
                {formatted_content}
            </div>
            <div style="text-align: right; font-size: 0.75em; color: #667781; margin-top: 5px;">
                Source #{index+1}
            </div>
        </div>
        """.format(
            conv_num=conv_num,
            conv_date=conv_date,
            conv_author=conv_author,
            formatted_content=formatted_content,
            index=index
        )
    else:
        # Try to extract date, time, and author using common WhatsApp patterns
        # Common formats: [DD/MM/YY, HH:MM:SS] Author: Message
        # or [DD/MM/YYYY, HH:MM:SS] Author: Message
        date_author_match = re.search(r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{1,2}(?::\d{1,2})?)\]?\s*([^:]+?):', chunk_text)
        
        if date_author_match:
            date = date_author_match.group(1)
            time = date_author_match.group(2)
            author = date_author_match.group(3).strip()
            
            # Extract the message content (everything after the author colon)
            message_start = chunk_text.find(':', chunk_text.find(author)) + 1
            message_content = chunk_text[message_start:].strip() if message_start > 0 else chunk_text
            
            # Format in a WhatsApp-like style
            html = """
            <div class="context-display">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: #128c7e;">{author}</span>
                    <span style="color: #667781; font-size: 0.85em;">{date}, {time}</span>
                </div>
                <div style="margin-left: 5px; border-left: 3px solid #efefef; padding-left: 8px; white-space: pre-wrap;">
                    {message_content}
                </div>
                <div style="text-align: right; font-size: 0.75em; color: #667781; margin-top: 5px;">
                    Source #{index+1}
                </div>
            </div>
            """.format(
                author=author,
                date=date,
                time=time,
                message_content=message_content.replace('\n', '<br>'),
                index=index
            )
        else:
            # Fallback for chunks that don't match the pattern
            author_match = re.search(r'^\[?([^:]+?)(?:\]|\:)', chunk_text)
            author = author_match.group(1).strip() if author_match else "Unknown"
            
            # Use proper string formatting instead of f-string in a regular string
            author_display = "from {}".format(author) if author != "Unknown" else ""
            
            html = """
            <div class="context-display">
                <span style="color: #128c7e; font-weight: bold;">Source #{} {}:</span><br>
                <div style="margin-left: 5px; border-left: 3px solid #efefef; padding-left: 8px; white-space: pre-wrap;">
                    {}
                </div>
            </div>
            """.format(
                index+1,
                author_display,
                chunk_text.replace('\n', '<br>')
            )
    
    return html

def display_context_chunks(entry, max_chunks=5):
    """
    Display context chunks in a more readable format.
    
    Args:
        entry (dict): Chat history entry containing context
        max_chunks (int): Maximum number of chunks to display
    """
    # Get context chunks depending on what's available
    if 'top_chunks' in entry and isinstance(entry['top_chunks'], list):
        chunks = entry['top_chunks'][:max_chunks]
    elif 'context' in entry and isinstance(entry['context'], str):
        # Split context by the conversation headers to maintain context grouping
        chunks = []
        parts = entry['context'].split("--- Conversation")
        
        if len(parts) > 1:
            # First part might be empty or just have newlines
            for i, part in enumerate(parts[1:], 1):  # Skip the first empty part
                chunks.append("--- Conversation" + part)
                if i >= max_chunks:
                    break
        else:
            # Fall back to simple splitting if no conversation headers
            chunks = entry['context'].split("\n\n")[:max_chunks]
    else:
        chunks = []
    
    # Count non-empty chunks
    non_empty_chunks = [c for c in chunks if c and c.strip()]
    
    if non_empty_chunks:
        for j, chunk in enumerate(non_empty_chunks):
            if chunk.strip():
                html = format_message_chunk(chunk, j)
                st.markdown(html, unsafe_allow_html=True)
        
        # Calculate total chunks based on what's available
        if 'context' in entry and isinstance(entry['context'], str):
            # Try to count conversation sections first
            conversation_sections = entry['context'].count("--- Conversation")
            if conversation_sections > 0:
                total_chunks = conversation_sections
            else:
                # Fall back to paragraph counting
                total_chunks = len([p for p in entry['context'].split("\n\n") if p.strip()])
        else:
            total_chunks = len(non_empty_chunks)
        
        # Show info about additional context
        if total_chunks > len(non_empty_chunks):
            st.info(f"{total_chunks - len(non_empty_chunks)} more sources were used but not shown")
    else:
        st.info("No specific message contexts were found for this query.")
