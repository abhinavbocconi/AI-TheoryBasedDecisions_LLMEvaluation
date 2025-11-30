"""
Merge Full Dataset to User Messages
====================================

This script merges participant-level data from the full experimental dataset
to the message-level user_messages.xlsx file.

WHAT THIS DOES:
---------------
1. Loads user_messages.xlsx (3,028 messages from 1,411 threads)
2. Loads finalData_SS_981_withThreadID.xlsx (participant data with thread IDs)
3. Filters to only Post=1 rows (when AI was used)
4. Expands any semicolon-separated thread_ids (some participants have multiple threads)
5. Merges participant data to each message based on thread_id
6. REMOVES messages from thread_ids NOT in our experimental sample
   (we only want messages from the 613 participants in our study)

WHY THIS MATTERS:
-----------------
- We want to analyze user messages alongside their experimental condition (aristotle)
- We need demographics, psychological measures, and outcomes for each message
- Filtering ensures we only analyze messages from verified study participants

INPUT FILES:
------------
- user_messages.xlsx: Message-level data with columns:
    [thread_id, condition, msg_id, content_clean, word_count, content_value, created_at]
    
- FullData/finalData_SS_981_withThreadID.xlsx: Participant data with columns:
    [uniqueID, aristotle, value, confidence, algoAversion, etc. + thread_id]

OUTPUT FILE:
------------
- user_messages_fulldata.xlsx: Message-level data with ALL participant columns merged

LINKING LOGIC:
--------------
user_messages.thread_id  <-->  finalData.thread_id (where Post=1)

Note: Some participants have multiple thread_ids (semicolon-separated).
We split these to match individual threads in user_messages.

Author: Generated for ChatAnalysis project
"""

import pandas as pd


def merge_messages_with_fulldata():
    print("=" * 70)
    print("MERGE USER MESSAGES WITH FULL EXPERIMENTAL DATA")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Load the datasets
    # =========================================================================
    print("\n[STEP 1] Loading datasets...")
    
    df_messages = pd.read_excel('user_messages.xlsx')
    print(f"  user_messages.xlsx:")
    print(f"    - {len(df_messages)} messages")
    print(f"    - {df_messages['thread_id'].nunique()} unique threads")
    
    df_full = pd.read_excel('FullData/finalData_SS_981_withThreadID.xlsx')
    print(f"  finalData_SS_981_withThreadID.xlsx:")
    print(f"    - {len(df_full)} total rows")
    print(f"    - {df_full['uniqueID'].nunique()} unique participants")
    
    # =========================================================================
    # STEP 2: Filter to Post=1 only (when AI was used)
    # =========================================================================
    print("\n[STEP 2] Filtering to Post=1 (AI condition)...")
    
    df_post = df_full[df_full['Post'] == 1].copy()
    df_post_with_threads = df_post[df_post['thread_id'].notna()].copy()
    
    print(f"  Post=1 rows: {len(df_post)}")
    print(f"  Post=1 with thread_id: {len(df_post_with_threads)}")
    
    # Check distribution by aristotle condition
    print(f"\n  By aristotle condition:")
    for aristotle_val in sorted(df_post_with_threads['aristotle'].unique()):
        count = len(df_post_with_threads[df_post_with_threads['aristotle'] == aristotle_val])
        cond_name = {0: 'Human Only', 1: 'General AI', 2: 'Agentic AI'}[aristotle_val]
        print(f"    aristotle={aristotle_val} ({cond_name}): {count} participants")
    
    # =========================================================================
    # STEP 3: Expand semicolon-separated thread_ids
    # =========================================================================
    print("\n[STEP 3] Expanding multi-thread entries...")
    
    # Check for semicolon-separated thread_ids
    has_semicolon = df_post_with_threads['thread_id'].str.contains(';', na=False)
    n_multi = has_semicolon.sum()
    print(f"  Participants with multiple threads: {n_multi}")
    
    if n_multi > 0:
        # Split semicolon-separated thread_ids into separate rows
        # This ensures each thread_id gets its own row for matching
        df_expanded = df_post_with_threads.copy()
        df_expanded['thread_id'] = df_expanded['thread_id'].str.split(';')
        df_expanded = df_expanded.explode('thread_id')
        df_expanded['thread_id'] = df_expanded['thread_id'].str.strip()  # Remove whitespace
        print(f"  After expansion: {len(df_expanded)} rows (from {len(df_post_with_threads)})")
    else:
        df_expanded = df_post_with_threads.copy()
        print(f"  No expansion needed: {len(df_expanded)} rows")
    
    # =========================================================================
    # STEP 4: Identify valid thread_ids (in our sample)
    # =========================================================================
    print("\n[STEP 4] Identifying valid thread_ids in sample...")
    
    valid_thread_ids = set(df_expanded['thread_id'].unique())
    message_thread_ids = set(df_messages['thread_id'].unique())
    
    threads_in_both = valid_thread_ids & message_thread_ids
    threads_only_in_messages = message_thread_ids - valid_thread_ids
    threads_only_in_fulldata = valid_thread_ids - message_thread_ids
    
    print(f"  Thread_ids in fulldata: {len(valid_thread_ids)}")
    print(f"  Thread_ids in user_messages: {len(message_thread_ids)}")
    print(f"  Thread_ids in BOTH: {len(threads_in_both)}")
    print(f"  Thread_ids only in messages (will be REMOVED): {len(threads_only_in_messages)}")
    print(f"  Thread_ids only in fulldata (no messages): {len(threads_only_in_fulldata)}")
    
    # =========================================================================
    # STEP 5: Filter messages to only those in our sample
    # =========================================================================
    print("\n[STEP 5] Filtering messages to sample participants only...")
    
    messages_before = len(df_messages)
    df_messages_filtered = df_messages[df_messages['thread_id'].isin(valid_thread_ids)].copy()
    messages_after = len(df_messages_filtered)
    messages_removed = messages_before - messages_after
    
    print(f"  Messages before: {messages_before}")
    print(f"  Messages after: {messages_after}")
    print(f"  Messages REMOVED (not in sample): {messages_removed}")
    
    # =========================================================================
    # STEP 6: Merge participant data to messages
    # =========================================================================
    print("\n[STEP 6] Merging participant data to messages...")
    
    # Select columns to merge (exclude thread_id since it's the key)
    # Also exclude columns that already exist in messages
    existing_cols = set(df_messages_filtered.columns)
    merge_cols = ['thread_id'] + [col for col in df_expanded.columns 
                                   if col != 'thread_id' and col not in existing_cols]
    
    df_to_merge = df_expanded[merge_cols].copy()
    
    # Remove duplicates (in case same thread appears multiple times)
    df_to_merge = df_to_merge.drop_duplicates(subset=['thread_id'])
    
    print(f"  Columns being added: {len(merge_cols) - 1}")  # -1 for thread_id
    
    # Perform the merge
    df_merged = df_messages_filtered.merge(
        df_to_merge,
        on='thread_id',
        how='left'
    )
    
    # =========================================================================
    # STEP 7: Validate the merge
    # =========================================================================
    print("\n[STEP 7] Validating merge...")
    
    unmatched = df_merged['uniqueID'].isna().sum()
    print(f"  Merged rows: {len(df_merged)}")
    print(f"  Unmatched rows: {unmatched}")
    
    if unmatched > 0:
        print(f"  WARNING: {unmatched} messages could not be matched!")
    else:
        print(f"  ✓ All messages matched successfully!")
    
    # =========================================================================
    # STEP 8: Summary statistics
    # =========================================================================
    print("\n" + "-" * 70)
    print("FINAL DATASET SUMMARY")
    print("-" * 70)
    
    print(f"\n  Total messages: {len(df_merged)}")
    print(f"  Unique participants: {df_merged['uniqueID'].nunique()}")
    print(f"  Unique threads: {df_merged['thread_id'].nunique()}")
    
    print(f"\n  Messages by aristotle condition:")
    for aristotle_val in sorted(df_merged['aristotle'].dropna().unique()):
        count = len(df_merged[df_merged['aristotle'] == aristotle_val])
        n_users = df_merged[df_merged['aristotle'] == aristotle_val]['uniqueID'].nunique()
        cond_name = {0: 'Human Only', 1: 'General AI', 2: 'Agentic AI'}[int(aristotle_val)]
        print(f"    aristotle={int(aristotle_val)} ({cond_name}): {count} messages from {n_users} participants")
    
    print(f"\n  Columns in output: {len(df_merged.columns)}")
    print(f"    - Original message columns: {len(df_messages.columns)}")
    print(f"    - Added participant columns: {len(df_merged.columns) - len(df_messages.columns)}")
    
    # =========================================================================
    # STEP 9: Save the merged dataset
    # =========================================================================
    output_file = 'user_messages_fulldata.xlsx'
    print(f"\n[STEP 9] Saving to {output_file}...")
    df_merged.to_excel(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Output: {output_file}")
    print(f"Rows: {len(df_merged)} | Columns: {len(df_merged.columns)}")
    
    return df_merged


if __name__ == "__main__":
    merge_messages_with_fulldata()

