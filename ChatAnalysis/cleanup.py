"""
Step 1: Clean up raw thread data
=================================

This script:
1. Loads the raw gpt_threads.xlsx file
2. Removes rows where thread_json is NULL (no message data)
3. Keeps only thread_id and thread_json columns
4. Outputs a cleaned file: gpt_threads_clean.xlsx

Input:  gpt_threads.xlsx
Output: gpt_threads_clean.xlsx
"""

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'gpt_threads.xlsx'
OUTPUT_FILE = 'gpt_threads_clean.xlsx'

# Columns to keep
COLUMNS_TO_KEEP = ['thread_id', 'thread_json']

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 1: CLEANUP")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check for thread_json column
    if 'thread_json' not in df.columns:
        print("ERROR: 'thread_json' column not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Count NULLs before
    null_count = df['thread_json'].isna().sum()
    print(f"\n  Rows with NULL thread_json: {null_count}")
    print(f"  Rows with data: {len(df) - null_count}")
    
    # Remove NULL rows
    print("\nRemoving rows with NULL thread_json...")
    df_clean = df[df['thread_json'].notna()].copy()
    print(f"  Remaining rows: {len(df_clean)}")
    
    # Keep only required columns
    print(f"\nKeeping only columns: {COLUMNS_TO_KEEP}")
    df_clean = df_clean[COLUMNS_TO_KEEP]
    
    # Remove error JSONs (optional - check for error in JSON)
    error_count = df_clean['thread_json'].str.contains('"error"', na=False).sum()
    print(f"  Rows with error JSON: {error_count}")
    
    df_clean = df_clean[~df_clean['thread_json'].str.contains('"error"', na=False)]
    print(f"  After removing errors: {len(df_clean)} rows")
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    df_clean.to_excel(OUTPUT_FILE, index=False)
    print(f"  Saved {len(df_clean)} rows")
    
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Threads with message data: {len(df_clean)}")


if __name__ == "__main__":
    main()

