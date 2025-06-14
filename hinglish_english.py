#hinglish to english translation
import re
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datetime import datetime


tokenizer = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator")
model = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator")

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


def translate_hinglish_to_english(df):
    """
    Translate Hinglish messages to English using a pre-trained model.
    
    Parameters:
    df (DataFrame): DataFrame containing messages with a 'message' column
    
    Returns:
    DataFrame: DataFrame with an additional 'translated_message' column
    """
    if 'message' not in df.columns:
        print("DataFrame does not contain 'message' column")
        return df
    
    translated_messages = []
    
    for message in df['message']:
        inputs = tokenizer(message, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=512)
        translated_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_messages.append(translated_message)
    df['translated_message'] = translated_messages

    return df


#write on a file

def main(file_path: str):
    df = parse_whatsapp_chat(file_path)
    df = translate_hinglish_to_english(df)
    df.to_csv("translated_chat.csv", index=False)