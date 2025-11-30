"""
Step 2: Extract JSON to message-level data
===========================================

This script:
1. Loads the cleaned gpt_threads_clean.xlsx file
2. Parses the thread_json column (contains OpenAI API response)
3. Extracts each message with its metadata
4. Outputs a message-level file: chats.xlsx

Input:  gpt_threads_clean.xlsx
Output: chats.xlsx

Each row in output represents one message with:
- thread_id: the conversation thread
- role: 'user' or 'assistant'
- assistant_id: the AI assistant ID (null for user messages)
- assistant_name: mapped name (normalGPT, coordinator, creator, coach)
- content: the message text
- created_at: timestamp of message
"""

import pandas as pd
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'gpt_threads_clean.xlsx'
OUTPUT_FILE = 'chats.xlsx'

# Assistant ID to name mapping
ASSISTANT_MAP = {
    'asst_KAQF4Rb3jQxQWip1id7nWJ1J': 'normalGPT',
    'asst_kOnqT3stmx9b74wdsIFHkQBK': 'coordinator',
    'asst_jk7NGXtujfF2aM9tYOZyu1RA': 'creator',
    'asst_9ATKOZ8G3hXlAwct2bqS7uLv': 'coach',
    None: 'user'
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def parse_thread_json(thread_id, json_str):
    """
    Parse a thread's JSON and extract all messages.
    
    Parameters
    ----------
    thread_id : str
        The thread identifier
    json_str : str
        The JSON string containing messages
        
    Returns
    -------
    list
        List of dictionaries, one per message
    """
    messages = []
    
    try:
        data = json.loads(json_str)
        
        # The messages are in data['data']
        for msg in data.get('data', []):
            # Extract content text
            content = None
            if msg.get('content') and len(msg['content']) > 0:
                content = msg['content'][0].get('text', {}).get('value', '')
            
            # Get assistant ID and map to name
            assistant_id = msg.get('assistant_id')
            assistant_name = ASSISTANT_MAP.get(assistant_id, 'unknown')
            
            messages.append({
                'thread_id': thread_id,
                'msg_id': msg.get('id'),
                'role': msg.get('role'),
                'assistant_id': assistant_id,
                'assistant_name': assistant_name,
                'created_at': msg.get('created_at'),
                'content': content
            })
            
    except json.JSONDecodeError as e:
        print(f"  JSON parse error for thread {thread_id}: {e}")
    except Exception as e:
        print(f"  Error processing thread {thread_id}: {e}")
    
    return messages


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 2: JSON EXTRACTION")
    print("=" * 60)
    
    # Load cleaned data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Loaded {len(df)} threads")
    
    # Extract all messages
    print("\nExtracting messages from JSON...")
    all_messages = []
    
    for idx, row in df.iterrows():
        thread_messages = parse_thread_json(row['thread_id'], row['thread_json'])
        all_messages.extend(thread_messages)
        
        # Progress indicator
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1} threads...")
    
    # Create message dataframe
    msg_df = pd.DataFrame(all_messages)
    print(f"\n  Total messages extracted: {len(msg_df)}")
    
    # Summary stats
    print("\n" + "-" * 40)
    print("MESSAGE SUMMARY")
    print("-" * 40)
    print(f"\nUnique threads: {msg_df['thread_id'].nunique()}")
    print(f"\nRole distribution:")
    print(msg_df['role'].value_counts())
    print(f"\nAssistant name distribution:")
    print(msg_df['assistant_name'].value_counts())
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    msg_df.to_excel(OUTPUT_FILE, index=False)
    print(f"  Saved {len(msg_df)} messages")
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Total messages: {len(msg_df)}")
    print(f"User messages: {(msg_df['role'] == 'user').sum()}")
    print(f"Assistant messages: {(msg_df['role'] == 'assistant').sum()}")


if __name__ == "__main__":
    main()

