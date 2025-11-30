"""
STEP 7: COMBINE GPT AND CLAUDE CLASSIFICATIONS (20-VOTE MAJORITY)

This script combines the 10 classification votes from GPT-5.1 and 10 from 
Claude Sonnet 4.5 into a single 20-vote majority classification.

Logic:
------
1. Each message was classified 10 times by GPT and 10 times by Claude
2. We combine all 20 votes and take the mode (most frequent classification)
3. This gives a more robust final classification than either model alone

Input:  user_messages_fulldata_classified.xlsx (with gpt_all_classifications, claude_all_classifications)
Output: user_messages_fulldata_classified.xlsx (updated with combined classification columns)

New columns added:
- category_combined: The final classification (mode of all 20 votes)
- combined_mode_count: How many of 20 votes matched the mode
- combined_total_votes: Total valid votes (should be 20 if no errors)
- combined_confidence: Mode count / total votes (0.0 to 1.0)

Author: Auto-generated for ChatAnalysis project
"""

import pandas as pd
import ast
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'user_messages_fulldata_classified.xlsx'
OUTPUT_FILE = 'user_messages_fulldata_classified.xlsx'  # Overwrite with new columns

# Confidence threshold: 11/20 means majority (>50%)
HIGH_CONFIDENCE_THRESHOLD = 11


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_classification_list(s):
    """
    Parse string representation of classification list back to Python list.
    
    The gpt_all_classifications and claude_all_classifications columns are stored
    as string representations of lists (e.g., "['Task Delegation', 'Other', ...]").
    
    Parameters
    ----------
    s : str or None
        String representation of a list
        
    Returns
    -------
    list
        Parsed Python list, or empty list if parsing fails
    """
    try:
        if pd.isna(s):
            return []
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


def get_combined_majority(gpt_votes, claude_votes):
    """
    Combine votes from both models and return the majority classification.
    
    Parameters
    ----------
    gpt_votes : list
        List of 10 classifications from GPT-5.1
    claude_votes : list
        List of 10 classifications from Claude Sonnet 4.5
        
    Returns
    -------
    tuple
        (mode_category, mode_count, total_votes)
        - mode_category: The most frequent classification
        - mode_count: How many times the mode appeared
        - total_votes: Total number of valid votes
    """
    combined = gpt_votes + claude_votes
    
    if not combined:
        return ('Other', 0, 0)
    
    counter = Counter(combined)
    mode_category, mode_count = counter.most_common(1)[0]
    
    return (mode_category, mode_count, len(combined))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 7: COMBINE CLASSIFICATIONS (20-VOTE MAJORITY)")
    print("=" * 60)
    
    # Load classified data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Total messages: {len(df)}")
    
    # Check required columns exist
    required_cols = ['gpt_all_classifications', 'claude_all_classifications', 
                     'category_gpt', 'category_claude']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print("Please run classify_messages.py first.")
        return
    
    # Parse the stored classification lists
    print("\nParsing classification votes...")
    df['gpt_votes'] = df['gpt_all_classifications'].apply(parse_classification_list)
    df['claude_votes'] = df['claude_all_classifications'].apply(parse_classification_list)
    
    # Verify vote counts
    gpt_vote_counts = df['gpt_votes'].apply(len)
    claude_vote_counts = df['claude_votes'].apply(len)
    print(f"  GPT votes per message: {gpt_vote_counts.min()}-{gpt_vote_counts.max()} (expected: 10)")
    print(f"  Claude votes per message: {claude_vote_counts.min()}-{claude_vote_counts.max()} (expected: 10)")
    
    # Combine votes and get majority
    print("\nComputing 20-vote majority...")
    results = df.apply(
        lambda row: get_combined_majority(row['gpt_votes'], row['claude_votes']), 
        axis=1
    )
    
    df['category_combined'] = [r[0] for r in results]
    df['combined_mode_count'] = [r[1] for r in results]
    df['combined_total_votes'] = [r[2] for r in results]
    df['combined_confidence'] = df['combined_mode_count'] / df['combined_total_votes']
    
    # Clean up temporary columns
    df = df.drop(columns=['gpt_votes', 'claude_votes'])
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("COMBINED CLASSIFICATION RESULTS")
    print("-" * 40)
    
    print("\nCategory distribution (20-vote majority):")
    print(df['category_combined'].value_counts().to_string())
    
    print(f"\nConfidence metrics:")
    print(f"  Mean confidence: {df['combined_confidence'].mean():.1%}")
    print(f"  Median confidence: {df['combined_confidence'].median():.1%}")
    print(f"  High confidence (>={HIGH_CONFIDENCE_THRESHOLD}/20): "
          f"{(df['combined_mode_count'] >= HIGH_CONFIDENCE_THRESHOLD).sum()} "
          f"({(df['combined_mode_count'] >= HIGH_CONFIDENCE_THRESHOLD).mean():.1%})")
    
    # Compare with individual model classifications
    print("\nAgreement with individual models:")
    agree_gpt = (df['category_combined'] == df['category_gpt']).sum()
    agree_claude = (df['category_combined'] == df['category_claude']).sum()
    both_agree = ((df['category_combined'] == df['category_gpt']) & 
                  (df['category_combined'] == df['category_claude'])).sum()
    
    print(f"  Combined matches GPT: {agree_gpt}/{len(df)} ({agree_gpt/len(df)*100:.1f}%)")
    print(f"  Combined matches Claude: {agree_claude}/{len(df)} ({agree_claude/len(df)*100:.1f}%)")
    print(f"  All three agree: {both_agree}/{len(df)} ({both_agree/len(df)*100:.1f}%)")
    
    # Cases where combined differs from both
    differs_from_both = ((df['category_combined'] != df['category_gpt']) & 
                         (df['category_combined'] != df['category_claude'])).sum()
    print(f"  Combined differs from both: {differs_from_both}/{len(df)} ({differs_from_both/len(df)*100:.1f}%)")
    
    # Save updated file
    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"  Saved {len(df)} rows | {len(df.columns)} columns")
    
    print("\n" + "=" * 60)
    print("COMBINATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_FILE}")
    print("\nNew columns added:")
    print("  - category_combined: Final 20-vote majority classification")
    print("  - combined_mode_count: Votes for winning category (out of 20)")
    print("  - combined_total_votes: Total valid votes")
    print("  - combined_confidence: Mode count / total votes")
    print("\nOriginal columns preserved:")
    print("  - category_gpt, category_claude (mode of 10 repeats each)")
    print("  - gpt_all_classifications, claude_all_classifications (all 10 votes each)")


if __name__ == "__main__":
    main()
