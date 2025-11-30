"""
Step 3: Extract REAL user messages only
========================================

This script identifies and extracts genuine user messages, filtering out
system/orchestration messages.

The pattern:
- General AI (normalGPT only): User messages are straightforward (1:1 with assistant)
- Agentic AI (coordinator/creator/coach): 
  - REAL user messages have routing suffix: "Please check if this is a task for a creator or a coach"
  - System messages like "_input_1..." are NOT real user messages
  - We strip the suffix to get the actual user input

Input:  chats.xlsx
Output: user_messages.xlsx

"""

import pandas as pd
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'chats.xlsx'
OUTPUT_FILE = 'user_messages.xlsx'

# Routing suffix pattern (appended to user messages in Agentic AI condition)
ROUTING_SUFFIX_PATTERN = r'\s*Please check if this is a task for a creator or a coach\s*$'

# System message patterns to exclude
SYSTEM_MESSAGE_STARTS = ['_input_1', '_input_2', '_input']

# =============================================================================
# FUNCTIONS
# =============================================================================

def classify_thread(assistant_names):
    """
    Classify a thread based on which assistants it contains.
    
    Parameters
    ----------
    assistant_names : set
        Set of assistant_name values in the thread
        
    Returns
    -------
    str
        'General AI' or 'Agentic AI'
    """
    agentic_agents = {'coordinator', 'creator', 'coach'}
    
    # Remove 'user' from consideration
    assistants = assistant_names - {'user'}
    
    if assistants & agentic_agents:
        return 'Agentic AI'
    elif 'normalGPT' in assistants:
        return 'General AI'
    else:
        return 'Unknown'


def is_system_message(content):
    """
    Check if a message is a system/orchestration message.
    
    Parameters
    ----------
    content : str
        Message content
        
    Returns
    -------
    bool
        True if system message, False if real user message
    """
    if pd.isna(content):
        return True
    
    content_str = str(content).strip()
    
    # Check if starts with system patterns
    for pattern in SYSTEM_MESSAGE_STARTS:
        if content_str.startswith(pattern):
            return True
    
    return False


def has_routing_suffix(content):
    """
    Check if message has the routing suffix.
    
    Parameters
    ----------
    content : str
        Message content
        
    Returns
    -------
    bool
        True if has routing suffix
    """
    if pd.isna(content):
        return False
    
    return bool(re.search(ROUTING_SUFFIX_PATTERN, str(content)))


def strip_routing_suffix(content):
    """
    Remove routing suffix from message content.
    
    Parameters
    ----------
    content : str
        Message content (may have routing suffix)
        
    Returns
    -------
    str
        Clean message content
    """
    if pd.isna(content):
        return content
    
    return re.sub(ROUTING_SUFFIX_PATTERN, '', str(content)).strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 3: EXTRACT REAL USER MESSAGES")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Total messages: {len(df)}")
    print(f"  Unique threads: {df['thread_id'].nunique()}")
    
    # Classify each thread
    print("\nClassifying threads by condition...")
    thread_assistants = df.groupby('thread_id')['assistant_name'].apply(set)
    thread_condition = thread_assistants.apply(classify_thread)
    
    print("  Thread conditions:")
    print(thread_condition.value_counts())
    
    # Map condition to dataframe
    df['condition'] = df['thread_id'].map(thread_condition.to_dict())
    
    # Get user messages only
    print("\nFiltering to user messages...")
    user_msgs = df[df['role'] == 'user'].copy()
    print(f"  Total user messages: {len(user_msgs)}")
    
    # Analyze patterns
    print("\nAnalyzing message patterns...")
    user_msgs['is_system'] = user_msgs['content'].apply(is_system_message)
    user_msgs['has_suffix'] = user_msgs['content'].apply(has_routing_suffix)
    
    print(f"  System messages (_input_*): {user_msgs['is_system'].sum()}")
    print(f"  Messages with routing suffix: {user_msgs['has_suffix'].sum()}")
    
    # Filter to REAL user messages
    print("\nExtracting real user messages...")
    
    # For General AI: all user messages are real (no system messages in these threads)
    general_ai_msgs = user_msgs[
        (user_msgs['condition'] == 'General AI') & 
        (~user_msgs['is_system'])
    ].copy()
    print(f"  General AI real messages: {len(general_ai_msgs)}")
    
    # For Agentic AI: only messages with routing suffix are real
    agentic_ai_msgs = user_msgs[
        (user_msgs['condition'] == 'Agentic AI') & 
        (user_msgs['has_suffix']) &
        (~user_msgs['is_system'])
    ].copy()
    print(f"  Agentic AI real messages: {len(agentic_ai_msgs)}")
    
    # Combine
    real_msgs = pd.concat([general_ai_msgs, agentic_ai_msgs], ignore_index=True)
    print(f"  Total real user messages: {len(real_msgs)}")
    
    # Strip routing suffix to get clean content
    print("\nCleaning message content...")
    real_msgs['content_clean'] = real_msgs['content'].apply(strip_routing_suffix)
    
    # Add word count
    real_msgs['word_count'] = real_msgs['content_clean'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    
    # Select output columns
    output_columns = [
        'thread_id',
        'condition',
        'msg_id',
        'content_clean',
        'word_count',
        'content',  # original for reference
        'created_at'
    ]
    
    output_df = real_msgs[output_columns].copy()
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    output_df.to_excel(OUTPUT_FILE, index=False)
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"\nReal user messages by condition:")
    print(output_df['condition'].value_counts())
    print(f"\nTotal: {len(output_df)} real user messages")
    print(f"\nWord count stats:")
    print(f"  Mean: {output_df['word_count'].mean():.1f}")
    print(f"  Median: {output_df['word_count'].median():.1f}")
    print(f"  Min: {output_df['word_count'].min()}")
    print(f"  Max: {output_df['word_count'].max()}")


if __name__ == "__main__":
    main()

