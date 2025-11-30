"""
Merge Thread IDs and Prompt Count to Full Dataset
==================================================

This script merges thread_id and prompt_count from threads_cleaned.xlsx
to finalData_SS_981 (1).xlsx using uniqueID as the linking key.

Input:
    - finalData_SS_981 (1).xlsx: Main experimental data (981 participants × 2 timepoints)
    - threads_cleaned.xlsx: Thread-level data with thread_id and prompt_count

Output:
    - finalData_SS_981_withThreadID.xlsx: Merged dataset with new columns:
        - thread_id: OpenAI thread ID(s) for the participant
        - prompt_count: Number of prompts sent by the participant
"""

import pandas as pd


def merge_thread_data():
    print("=" * 60)
    print("MERGING THREAD DATA TO FULL DATASET")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    df_full = pd.read_excel('finalData_SS_981 (1).xlsx')
    df_threads = pd.read_excel('threads_cleaned.xlsx')
    
    print(f"  finalData: {len(df_full)} rows, {df_full['uniqueID'].nunique()} unique participants")
    print(f"  threads_cleaned: {len(df_threads)} rows, {df_threads['uniqueID'].nunique()} unique participants")
    
    # Check overlap
    full_ids = set(df_full['uniqueID'].unique())
    thread_ids = set(df_threads['uniqueID'].unique())
    overlap = len(full_ids & thread_ids)
    print(f"\n  Overlap: {overlap} participants have thread data")
    print(f"  Only in finalData (no threads): {len(full_ids - thread_ids)}")
    print(f"  Only in threads (not in finalData): {len(thread_ids - full_ids)}")
    
    # Select columns to merge from threads
    threads_to_merge = df_threads[['uniqueID', 'thread_id', 'prompt_count']].copy()
    
    # Handle duplicates in threads (if any participant has multiple rows)
    # Group by uniqueID and aggregate: concatenate thread_ids, sum prompt_counts
    duplicates = threads_to_merge['uniqueID'].duplicated().sum()
    if duplicates > 0:
        print(f"\n  Note: {duplicates} duplicate uniqueIDs in threads - aggregating...")
        threads_to_merge = threads_to_merge.groupby('uniqueID').agg({
            'thread_id': lambda x: ';'.join(x.dropna().astype(str)),
            'prompt_count': 'sum'
        }).reset_index()
    
    # Merge datasets - ONLY for Post=1 (AI was used only in post condition)
    print("\nMerging datasets (only for Post=1)...")
    
    # Split data into Pre (Post=0) and Post (Post=1)
    df_pre = df_full[df_full['Post'] == 0].copy()
    df_post = df_full[df_full['Post'] == 1].copy()
    
    print(f"  Pre (Post=0): {len(df_pre)} rows")
    print(f"  Post (Post=1): {len(df_post)} rows")
    
    # Merge thread data only to Post rows
    df_post_merged = df_post.merge(
        threads_to_merge,
        on='uniqueID',
        how='left'
    )
    
    # Add empty columns to Pre rows
    df_pre['thread_id'] = None
    df_pre['prompt_count'] = None
    
    # Combine back together
    df_merged = pd.concat([df_pre, df_post_merged], ignore_index=True)
    
    # Sort by uniqueID and Post to restore original order
    df_merged = df_merged.sort_values(['uniqueID', 'Post']).reset_index(drop=True)
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("MERGE SUMMARY")
    print("-" * 40)
    print(f"  Original finalData rows: {len(df_full)}")
    print(f"  Merged dataset rows: {len(df_merged)}")
    print(f"  Rows with thread_id: {df_merged['thread_id'].notna().sum()}")
    print(f"  Rows without thread_id: {df_merged['thread_id'].isna().sum()}")
    
    # Check by condition (only Post=1 rows should have threads)
    print("\n  Thread data by aristotle condition (Post=1 only):")
    df_post_only = df_merged[df_merged['Post'] == 1]
    thread_by_condition = df_post_only.groupby('aristotle')['thread_id'].apply(
        lambda x: f"{x.notna().sum()}/{len(x)} rows"
    )
    for cond, counts in thread_by_condition.items():
        cond_name = {0: 'Human Only', 1: 'General AI', 2: 'Agentic AI'}[cond]
        print(f"    aristotle={cond} ({cond_name}): {counts}")
    
    # Prompt count stats for those with threads
    with_threads = df_merged[df_merged['prompt_count'].notna()]
    print(f"\n  Prompt count stats (for participants with threads):")
    print(f"    Mean: {with_threads['prompt_count'].mean():.1f}")
    print(f"    Median: {with_threads['prompt_count'].median():.1f}")
    print(f"    Min: {with_threads['prompt_count'].min():.0f}")
    print(f"    Max: {with_threads['prompt_count'].max():.0f}")
    
    # Save merged dataset
    output_file = 'finalData_SS_981_withThreadID.xlsx'
    print(f"\nSaving to {output_file}...")
    df_merged.to_excel(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Output: {output_file}")
    print(f"Total columns: {len(df_merged.columns)} (added: thread_id, prompt_count)")
    
    return df_merged


if __name__ == "__main__":
    merge_thread_data()

