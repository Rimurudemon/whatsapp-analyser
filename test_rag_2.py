import os
import re
import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel, configure
from langchain_core.documents import Document
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# ---------------------- Configuration ----------------------
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
DB_PATH = "whatsapp_analyser/whatsapp-analyser/faiss_chat_enhanced"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 10
CONTEXT_SIZE = 3  # Number of messages before and after to include as conversation context
RETRIEVER_K = 10   # Number of conversation chunks to retrieve

# ---------------------- Google Gemini Setup ----------------------
def setup_gemini(api_key):
    configure(api_key=api_key)
    return GenerativeModel('gemini-2.5-flash-preview-05-20')

# ---------------------- Enhanced WhatsApp Chat Parser ----------------------
def parse_whatsapp_chat(file_path: str) -> pd.DataFrame:
    """
    Parse WhatsApp chat export file into a structured DataFrame
    
    Parameters:
    file_path (str): Path to WhatsApp chat export file
    
    Returns:
    DataFrame: Pandas DataFrame with parsed messages
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            file_content = file.read()
    except FileNotFoundError:
        print(f"Error: Chat file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading chat file: {str(e)}")
        return pd.DataFrame()
    
    # Define regex patterns
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
        print("No messages parsed from chat file")
        return df
        
    print(f"Successfully parsed {len(df)} messages from {file_path}")
    return df

def format_messages_for_rag(df: pd.DataFrame, context_size=3) -> List[Dict[str, Any]]:
    """
    Format parsed messages into a structure suitable for RAG with conversation context
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    context_size (int): Number of messages before and after to include as context
    
    Returns:
    List[Dict]: List of structured message dictionaries
    """
    # Sort DataFrame by date to ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)
    structured_messages = []
    
    total_messages = len(df)
    
    for i, row in df.iterrows():
        # Format current message date as a string
        date_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Get context messages (before and after current message)
        start_idx = max(0, i - context_size)
        end_idx = min(total_messages, i + context_size + 1)
        
        # Get the context messages
        context_messages = []
        for j in range(start_idx, end_idx):
            if j == i:  # Add marker for the current message
                prefix = ">> "  # Mark the current message
            else:
                prefix = "   "
                
            ctx_date_str = df.iloc[j]['date'].strftime('%Y-%m-%d %H:%M:%S')
            ctx_msg = f"{prefix}[{ctx_date_str}] {df.iloc[j]['author']}: {df.iloc[j]['message']}"
            context_messages.append(ctx_msg)
        
        # Join context messages
        conversation_context = "\n".join(context_messages)
        
        # Structure the message with metadata and conversation context
        message_dict = {
            'date': date_str,
            'author': row['author'],
            'message': row['message'],
            'text': f"[{date_str}] {row['author']}: {row['message']}",
            'conversation_context': conversation_context
        }
        
        structured_messages.append(message_dict)
    
    return structured_messages

# ---------------------- Build & Save Enhanced FAISS DB ----------------------
def build_faiss_db(structured_messages: List[Dict[str, Any]], db_path=DB_PATH):
    """
    Build and save FAISS vector database from structured messages
    
    Parameters:
    structured_messages (List[Dict]): List of message dictionaries with metadata
    db_path (str): Path to save the FAISS database
    
    Returns:
    FAISS: FAISS vector database instance
    """
    # Create documents with metadata and conversation context
    docs = []
    for msg in structured_messages:
        # Create document with conversation context and metadata
        doc = Document(
            page_content=msg['conversation_context'],  # Using conversation context instead of single message
            metadata={
                'date': msg['date'],
                'author': msg['author'],
                'message': msg['message'],
                'primary_message': msg['text']
            }
        )
        docs.append(doc)
    
    # Create text chunks with context-aware splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = splitter.split_documents(docs)
    print(f"Created {len(chunked_docs)} chunks from {len(docs)} messages")
    
    # Initialize embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    
    # Create and save FAISS database
    db = FAISS.from_documents(chunked_docs, embedder)
    db.save_local(db_path)
    print(f"FAISS DB saved at {db_path}")

    return db

# ---------------------- Load FAISS DB ----------------------
def load_faiss_db(db_path=DB_PATH):
    """
    Load FAISS vector database from disk
    
    Parameters:
    db_path (str): Path to the saved FAISS database
    
    Returns:
    FAISS: FAISS vector database instance
    """
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)

# ---------------------- Query FAISS DB ----------------------
def query_db(question: str, db) -> str:
    """
    Query the FAISS database for relevant message chunks
    
    Parameters:
    question (str): User's question about the chat
    db (FAISS): FAISS vector database instance
    
    Returns:
    str: Formatted context from relevant chunks
    """
    # Search with metadata filter options
    retriever = db.as_retriever(
        search_kwargs={"k": RETRIEVER_K}  # Retrieve conversation chunks based on config
    )
    
    # Get documents with high relevance to the query
    # Using invoke() instead of get_relevant_documents() to fix deprecation warning
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} relevant chunks for query: {question}")
    
    # Format the context with clear delimiters and metadata
    context_pieces = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        
        # Extract conversation date for the section header
        conversation_date = metadata.get('date', 'Unknown Date').split(' ')[0]
        primary_author = metadata.get('author', 'Unknown')
        
        formatted_chunk = (
            f"--- Conversation {i+1} (centered around {conversation_date}, {primary_author}) ---\n"
            f"{doc.page_content}\n"
        )
        context_pieces.append(formatted_chunk)
    
    # Join all conversation pieces with clear separation
    return "\n\n" + "\n\n".join(context_pieces) + "\n\n"

# ---------------------- Enhanced Gemini Q&A ----------------------
def qa_with_gemini(context: str, question: str, model, chat_history=None) -> str:
    """
    Generate an answer to the user's question using Gemini AI
    
    Parameters:
    context (str): Retrieved context from relevant messages
    question (str): User's question
    model: Gemini model instance
    chat_history (list): Optional previous Q&A exchanges
    
    Returns:
    str: Gemini's answer based on the context
    """
    # Load chat insights to enhance context
    insights = load_chat_insights()
    
    # Create a summary of insights
    insights_summary = ""
    if insights:
        insights_summary = f"""
        Chat Overview:
        - Total messages: {insights.get('total_messages', 'Unknown')}
        - Time period: {insights.get('date_range', {}).get('start', 'Unknown')} to {insights.get('date_range', {}).get('end', 'Unknown')}
        - Number of participants: {insights.get('author_stats', {}).get('total_authors', 'Unknown')}
        - Most active hour of day: {insights.get('time_stats', {}).get('most_active_hour', 'Unknown')}:00
        """
    
    system_prompt = """
    You are an AI assistant that helps analyze WhatsApp chat conversations.
    You will be given context from a WhatsApp chat that includes conversation threads (not just individual messages), 
    overall chat statistics, and a question about it. You may also have access to previous questions and answers
    in the conversation.
    
    Follow these guidelines:
    1. Base your answers primarily on the provided conversation contexts
    2. Pay attention to the conversation flow and back-and-forth exchanges between participants
    3. Consider the messages marked with ">>" as the central messages in each conversation segment
    4. Use the chat statistics for general information about the conversation
    5. If the context doesn't contain enough information, say so
    6. Include specific examples and quotes from the messages when relevant
    7. Format dates and times consistently
    8. When referring to conversations, mention both the date and participants involved
    9. Consider previous questions and answers when responding to follow-up questions
    10. Keep your responses concise and focused on answering the specific question
    
    Context is formatted with multiple conversation segments, each containing several messages 
    that provide context around a central message.

    Your response must not only quote the context but also synthesize it into a coherent answer.
    """
    
    # Format previous conversation history if available
    conversation_history = ""
    if chat_history and len(chat_history) > 0:
        conversation_history = "Previous conversation:\n"
        # Include up to 3 previous exchanges for context (excluding the current one)
        for i, entry in enumerate(chat_history[-3:]):
            conversation_history += f"Question {i+1}: {entry['question']}\n"
            conversation_history += f"Answer {i+1}: {entry['answer']}\n\n"
    
    user_prompt = f"""
    {insights_summary}
    
    {conversation_history}
    
    Context from WhatsApp Chat:
    {context}
    
    Question: {question}
    """
    
    # Use system prompt for better control of response
    response = model.generate_content([
        {"role": "user", "parts": [system_prompt + user_prompt]}
    ])
    
    return response.text

# ---------------------- Main Pipeline ----------------------
def main():
    """
    Main execution pipeline for WhatsApp Chat QA system
    """
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    model = setup_gemini(GOOGLE_API_KEY)
    
    print("Welcome to WhatsApp Chat Analyzer Q&A System!")
    print("---------------------------------------------")

    # Default chat file path
    default_chat_path = "chat.txt"
    
    # Allow custom chat file path
    chat_path = input(f"Enter chat file path (default: {default_chat_path}): ").strip()
    if not chat_path:
        chat_path = default_chat_path
    
    # Parse the chat and prepare for RAG
    df = parse_whatsapp_chat(chat_path)
    if len(df) == 0:
        print("No messages found. Exiting.")
        return
    
    # Format messages for RAG with conversation context
    structured_messages = format_messages_for_rag(df, context_size=CONTEXT_SIZE)
    print(f"Formatted messages with {CONTEXT_SIZE} messages of conversation context")
    
    # Check if we need to build or load the database
    force_rebuild = False
    if os.path.exists(DB_PATH):
        rebuild = input("Vector database already exists. Rebuild? (y/n): ").lower()
        force_rebuild = rebuild == 'y'
    
    # Build or load the database
    if not os.path.exists(DB_PATH) or force_rebuild:
        print(f"Building vector database at {DB_PATH}...")
        db = build_faiss_db(structured_messages)
    else:
        print(f"Loading existing vector database from {DB_PATH}...")
        db = load_faiss_db()
    
    # Generate or load chat insights
    print("\nGenerating chat insights...")
    insights = generate_chat_insights(df)
    
    # Print chat statistics from insights
    print(f"\nChat Statistics:")
    print(f"- Total messages: {insights['total_messages']}")
    print(f"- Participants: {insights['author_stats']['total_authors']}")
    print(f"- Date range: {insights['date_range']['start']} to {insights['date_range']['end']}")
    
    # Get top 3 authors
    top_authors = sorted(
        insights['author_stats']['counts'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    top_authors_str = ", ".join([f"{name} ({count})" for name, count in top_authors])
    print(f"- Top authors: {top_authors_str}")
    print(f"- Most active hour: {insights['time_stats']['most_active_hour']}:00")
    print(f"- Average message length: {insights['message_stats']['avg_length']:.1f} characters")
    
    # Interactive Q&A loop
    print("\nYou can now ask questions about your WhatsApp chat!")
    print("Type 'exit' to quit or 'help' for example questions")
    
    example_questions = [
        "Who sends the most messages in this chat?",
        "What topics do people talk about most often?",
        "When are people most active in this chat?",
        "Summarize the conversation from last week",
        "What kind of media is shared in this chat?",
        "Who uses the most emojis?",
        "What was the conversation about yesterday?"
    ]
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() == 'exit':
            print("Thank you for using WhatsApp Chat Analyzer Q&A System!")
            break
            
        if question.lower() == 'help':
            print("\nExample questions you can ask:")
            for i, q in enumerate(example_questions):
                print(f"{i+1}. {q}")
            continue
        
        if not question:
            continue
            
        # Get context and generate answer
        print("\nSearching for relevant messages...")
        context = query_db(question, db)
        
        # # Show truncated context for user understanding
        # print("\n--- Excerpt of Retrieved Context ---")
        # print(context[:500] + "\n...\n")
        
        print("Generating answer...")
        answer = qa_with_gemini(context, question, model)
        
        print("\n--- Answer ---")
        print(answer)

# ---------------------- Chat Insights ----------------------
def generate_chat_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate insights from the chat data to enhance RAG responses
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    
    Returns:
    Dict: Dictionary of chat insights
    """
    insights = {}
    
    # Basic statistics
    insights["total_messages"] = int(len(df))
    insights["date_range"] = {
        "start": df['date'].min().strftime('%Y-%m-%d'),
        "end": df['date'].max().strftime('%Y-%m-%d'),
    }
    
    # Author statistics
    # Convert numpy int64 to Python int for JSON serialization
    author_counts = {k: int(v) for k, v in df['author'].value_counts().to_dict().items()}
    insights["author_stats"] = {
        "counts": author_counts,
        "total_authors": len(author_counts),
    }
    
    # Time statistics
    df['hour'] = df['date'].dt.hour
    # Convert numpy int64 to Python int for JSON serialization
    hourly_activity = {int(k): int(v) for k, v in df.groupby('hour').size().to_dict().items()}
    insights["time_stats"] = {
        "hourly_activity": hourly_activity,
        "most_active_hour": int(max(hourly_activity.items(), key=lambda x: x[1])[0])
    }
    
    # Message length statistics
    df['message_length'] = df['message'].str.len()
    insights["message_stats"] = {
        "avg_length": float(df['message_length'].mean()),
        "max_length": int(df['message_length'].max()),
    }
    
    # Save insights to file
    with open("chat_insights.json", "w") as f:
        json.dump(insights, f, indent=2)
    
    print("Chat insights generated and saved to chat_insights.json")
    return insights

def load_chat_insights() -> Dict[str, Any]:
    """
    Load previously generated chat insights
    
    Returns:
    Dict: Dictionary of chat insights or empty dict if not found
    """
    try:
        with open("chat_insights.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error loading chat insights: {str(e)}")
        return {}

if __name__ == "__main__":
    main()
