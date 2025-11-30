"""
STEP 8: ANALYZE CHAT BEHAVIOR PATTERNS

This script performs comprehensive analysis of user chat behavior to support
the reviewer response about how participants used AI in different conditions.

Analyses performed:
1. Add message_order within each thread (1st, 2nd, 3rd... query)
2. Query count distribution analysis
3. Classification distribution (overall, by condition, by round)
4. Subsample analysis (experienced managers, PhD holders)
5. Extract common example queries for each classification type

Output:
- Updated user_messages_fulldata_classified.xlsx (with message_order, round_group)
- Console output with all statistics for the paragraph
- Example queries for inclusion in the paper

Author: Auto-generated for ChatAnalysis project
"""

import pandas as pd
import numpy as np
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'user_messages_fulldata_classified.xlsx'
OUTPUT_FILE = 'user_messages_fulldata_classified.xlsx'  # Update in place

# Subsample definitions (based on variable coding)
EXPERIENCED_MANAGER_THRESHOLD = 5  # managerialExperience >= 5 means >=10 years
PHD_THRESHOLD = 5  # educationLevel >= 5 means PhD

# Round grouping
ROUND_GROUPS = {1: '1st Query', 2: '2nd Query'}  # 3+ becomes '3rd+ Query'


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("STEP 8: ANALYZE CHAT BEHAVIOR PATTERNS")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Total messages: {len(df)}")
    print(f"  Unique threads: {df['thread_id'].nunique()}")
    print(f"  Unique participants: {df['uniqueID'].nunique()}")
    
    # -------------------------------------------------------------------------
    # 1. ADD MESSAGE ORDER WITHIN EACH THREAD
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. DERIVING MESSAGE ORDER WITHIN THREADS")
    print("-" * 70)
    
    # Sort by thread and timestamp
    df = df.sort_values(['thread_id', 'created_at']).reset_index(drop=True)
    
    # Add message_order (1, 2, 3, ...) within each thread
    df['message_order'] = df.groupby('thread_id').cumcount() + 1
    
    # Add round_group for analysis (1st, 2nd, 3rd+)
    def assign_round_group(order):
        if order == 1:
            return '1st Query'
        elif order == 2:
            return '2nd Query'
        else:
            return '3rd+ Query'
    
    df['round_group'] = df['message_order'].apply(assign_round_group)
    
    # Summary of message order
    print(f"\n  Message order distribution:")
    order_dist = df['message_order'].value_counts().sort_index()
    for order, count in order_dist.items():
        if order <= 5:
            print(f"    Query #{order}: {count} messages")
    if order_dist.index.max() > 5:
        print(f"    Query #6+: {order_dist[order_dist.index > 5].sum()} messages")
    
    print(f"\n  Round group distribution:")
    print(df['round_group'].value_counts().to_string())
    
    # -------------------------------------------------------------------------
    # 2. QUERY COUNT DISTRIBUTION (PER USER)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. QUERY COUNT DISTRIBUTION (PER USER/THREAD)")
    print("-" * 70)
    
    # Get queries per thread
    queries_per_thread = df.groupby('thread_id').size()
    
    print(f"\n  Statistics:")
    print(f"    Mean: {queries_per_thread.mean():.2f}")
    print(f"    Median: {queries_per_thread.median():.1f}")
    print(f"    Std: {queries_per_thread.std():.2f}")
    print(f"    Min: {queries_per_thread.min()}")
    print(f"    Max: {queries_per_thread.max()}")
    
    # Distribution buckets
    single_query = (queries_per_thread == 1).sum()
    two_queries = (queries_per_thread == 2).sum()
    three_plus = (queries_per_thread >= 3).sum()
    total_threads = len(queries_per_thread)
    
    print(f"\n  Engagement buckets:")
    print(f"    Single query (autopilot): {single_query} ({single_query/total_threads*100:.1f}%)")
    print(f"    Two queries: {two_queries} ({two_queries/total_threads*100:.1f}%)")
    print(f"    Three+ queries (copilot): {three_plus} ({three_plus/total_threads*100:.1f}%)")
    
    # By condition
    print(f"\n  Query counts by condition:")
    for cond_val, cond_name in [(1, 'General AI'), (2, 'Agentic AI')]:
        cond_threads = df[df['aristotle'] == cond_val]['thread_id'].unique()
        cond_queries = queries_per_thread[queries_per_thread.index.isin(cond_threads)]
        print(f"    {cond_name}: mean={cond_queries.mean():.2f}, median={cond_queries.median():.1f}, n={len(cond_queries)}")
    
    # -------------------------------------------------------------------------
    # 3. CLASSIFICATION DISTRIBUTION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. CLASSIFICATION DISTRIBUTION (COMBINED 20-VOTE)")
    print("-" * 70)
    
    # Overall distribution
    print("\n  Overall distribution:")
    overall_dist = df['category_combined'].value_counts()
    overall_pct = df['category_combined'].value_counts(normalize=True) * 100
    for cat in overall_dist.index:
        print(f"    {cat}: {overall_dist[cat]} ({overall_pct[cat]:.1f}%)")
    
    # By condition
    print("\n  Distribution by condition:")
    for cond_val, cond_name in [(1, 'General AI'), (2, 'Agentic AI')]:
        cond_df = df[df['aristotle'] == cond_val]
        cond_pct = cond_df['category_combined'].value_counts(normalize=True) * 100
        print(f"\n    {cond_name} (n={len(cond_df)}):")
        for cat in overall_dist.index:
            pct = cond_pct.get(cat, 0)
            print(f"      {cat}: {pct:.1f}%")
    
    # By round
    print("\n  Distribution by query round:")
    for round_name in ['1st Query', '2nd Query', '3rd+ Query']:
        round_df = df[df['round_group'] == round_name]
        round_pct = round_df['category_combined'].value_counts(normalize=True) * 100
        print(f"\n    {round_name} (n={len(round_df)}):")
        for cat in ['Task Delegation', 'Refinement Request', 'Information Seeking', 
                    'Evaluation Seeking', 'Clarification', 'Acknowledgment', 'Other']:
            pct = round_pct.get(cat, 0)
            print(f"      {cat}: {pct:.1f}%")
    
    # -------------------------------------------------------------------------
    # 4. SUBSAMPLE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. SUBSAMPLE ANALYSIS")
    print("-" * 70)
    
    # Experienced managers (managerialExperience >= 5)
    exp_managers = df[df['managerialExperience'] >= EXPERIENCED_MANAGER_THRESHOLD]
    non_exp = df[df['managerialExperience'] < EXPERIENCED_MANAGER_THRESHOLD]
    
    print(f"\n  EXPERIENCED MANAGERS (managerialExperience >= {EXPERIENCED_MANAGER_THRESHOLD}, i.e., >=10 years)")
    print(f"    Messages: {len(exp_managers)}")
    print(f"    Unique participants: {exp_managers['uniqueID'].nunique()}")
    
    # Query count for experienced managers
    exp_threads = exp_managers['thread_id'].unique()
    exp_queries = queries_per_thread[queries_per_thread.index.isin(exp_threads)]
    print(f"    Mean queries: {exp_queries.mean():.2f} (vs {queries_per_thread.mean():.2f} overall)")
    
    # Classification distribution
    exp_pct = exp_managers['category_combined'].value_counts(normalize=True) * 100
    print(f"    Classification distribution:")
    for cat in ['Task Delegation', 'Refinement Request', 'Information Seeking', 'Evaluation Seeking']:
        pct = exp_pct.get(cat, 0)
        overall = overall_pct.get(cat, 0)
        diff = pct - overall
        print(f"      {cat}: {pct:.1f}% ({'+' if diff > 0 else ''}{diff:.1f}% vs overall)")
    
    # Non-experienced managers
    print(f"\n  LESS EXPERIENCED MANAGERS (managerialExperience < {EXPERIENCED_MANAGER_THRESHOLD})")
    print(f"    Messages: {len(non_exp)}")
    print(f"    Unique participants: {non_exp['uniqueID'].nunique()}")
    non_exp_threads = non_exp['thread_id'].unique()
    non_exp_queries = queries_per_thread[queries_per_thread.index.isin(non_exp_threads)]
    print(f"    Mean queries: {non_exp_queries.mean():.2f}")
    non_exp_pct = non_exp['category_combined'].value_counts(normalize=True) * 100
    print(f"    Task Delegation: {non_exp_pct.get('Task Delegation', 0):.1f}%")
    
    # PhD holders
    phd_holders = df[df['educationLevel'] >= PHD_THRESHOLD]
    non_phd = df[df['educationLevel'] < PHD_THRESHOLD]
    
    print(f"\n  PHD HOLDERS (educationLevel >= {PHD_THRESHOLD})")
    print(f"    Messages: {len(phd_holders)}")
    print(f"    Unique participants: {phd_holders['uniqueID'].nunique()}")
    
    phd_threads = phd_holders['thread_id'].unique()
    phd_queries = queries_per_thread[queries_per_thread.index.isin(phd_threads)]
    print(f"    Mean queries: {phd_queries.mean():.2f} (vs {queries_per_thread.mean():.2f} overall)")
    
    phd_pct = phd_holders['category_combined'].value_counts(normalize=True) * 100
    print(f"    Classification distribution:")
    for cat in ['Task Delegation', 'Refinement Request', 'Information Seeking', 'Evaluation Seeking']:
        pct = phd_pct.get(cat, 0)
        overall = overall_pct.get(cat, 0)
        diff = pct - overall
        print(f"      {cat}: {pct:.1f}% ({'+' if diff > 0 else ''}{diff:.1f}% vs overall)")
    
    # -------------------------------------------------------------------------
    # 5. EXAMPLE QUERIES
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. EXAMPLE QUERIES BY CLASSIFICATION TYPE")
    print("-" * 70)
    
    # Get examples for key categories (prioritize shorter, clear examples)
    key_categories = ['Task Delegation', 'Information Seeking', 'Evaluation Seeking', 'Refinement Request']
    
    for cat in key_categories:
        cat_df = df[df['category_combined'] == cat].copy()
        cat_df['word_count'] = cat_df['content_clean'].apply(lambda x: len(str(x).split()))
        
        # Get examples of different lengths
        print(f"\n  {cat.upper()} (n={len(cat_df)}):")
        
        # Short example (under 30 words)
        short_examples = cat_df[cat_df['word_count'] <= 30].nsmallest(3, 'word_count')
        if not short_examples.empty:
            print(f"\n    Short example:")
            ex = short_examples.iloc[0]['content_clean']
            print(f'    "{ex[:200]}..."' if len(ex) > 200 else f'    "{ex}"')
        
        # Medium example (30-60 words)
        med_examples = cat_df[(cat_df['word_count'] > 30) & (cat_df['word_count'] <= 60)]
        if not med_examples.empty:
            print(f"\n    Medium example:")
            ex = med_examples.iloc[0]['content_clean']
            print(f'    "{ex[:300]}..."' if len(ex) > 300 else f'    "{ex}"')
    
    # -------------------------------------------------------------------------
    # 6. FIRST QUERY ANALYSIS (AUTOPILOT INDICATOR)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("6. FIRST QUERY ANALYSIS (KEY FOR AUTOPILOT NARRATIVE)")
    print("-" * 70)
    
    first_queries = df[df['message_order'] == 1]
    
    print(f"\n  First queries: {len(first_queries)}")
    first_pct = first_queries['category_combined'].value_counts(normalize=True) * 100
    print(f"\n  First query classification:")
    for cat in first_pct.index:
        print(f"    {cat}: {first_pct[cat]:.1f}%")
    
    # First queries by condition
    print(f"\n  First query: Task Delegation % by condition:")
    for cond_val, cond_name in [(1, 'General AI'), (2, 'Agentic AI')]:
        cond_first = first_queries[first_queries['aristotle'] == cond_val]
        task_del_pct = (cond_first['category_combined'] == 'Task Delegation').mean() * 100
        print(f"    {cond_name}: {task_del_pct:.1f}%")
    
    # -------------------------------------------------------------------------
    # 7. CONDITION COMPARISON SUMMARY (FOR VISUAL)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("7. CONDITION x ROUND COMPARISON (FOR VISUAL)")
    print("-" * 70)
    
    print("\n  Cross-tabulation: Round x Condition x Category")
    print("  (Values are percentages within each round-condition group)\n")
    
    for round_name in ['1st Query', '2nd Query', '3rd+ Query']:
        print(f"  {round_name}:")
        for cond_val, cond_name in [(1, 'General AI'), (2, 'Agentic AI')]:
            subset = df[(df['round_group'] == round_name) & (df['aristotle'] == cond_val)]
            n = len(subset)
            if n > 0:
                pcts = subset['category_combined'].value_counts(normalize=True) * 100
                top3 = pcts.head(3)
                top3_str = ", ".join([f"{cat}: {pct:.0f}%" for cat, pct in top3.items()])
                print(f"    {cond_name} (n={n}): {top3_str}")
        print()
    
    # -------------------------------------------------------------------------
    # SAVE UPDATED DATA
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("SAVING UPDATED DATA")
    print("-" * 70)
    
    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"  Added columns: message_order, round_group")
    print(f"  Saved {len(df)} rows")
    
    # -------------------------------------------------------------------------
    # SUMMARY FOR PARAGRAPH
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER PARAGRAPH")
    print("=" * 70)
    
    print(f"""
KEY STATISTICS FOR THE PARAGRAPH:

1. QUERY DISTRIBUTION:
   - Participants averaged {queries_per_thread.mean():.1f} queries (range: {queries_per_thread.min()}-{queries_per_thread.max()})
   - {single_query/total_threads*100:.0f}% asked only 1 query (autopilot)
   - {three_plus/total_threads*100:.0f}% asked 3+ queries (copilot engagement)

2. FIRST QUERY BEHAVIOR:
   - {first_pct.get('Task Delegation', 0):.0f}% of first queries were Task Delegation
   - This represents pure autopilot: delegating the entire task to AI

3. EVOLUTION ACROSS ROUNDS:
   - 1st Query: Task Delegation {df[df['round_group']=='1st Query']['category_combined'].value_counts(normalize=True).get('Task Delegation', 0)*100:.0f}%
   - 2nd Query: Task Delegation {df[df['round_group']=='2nd Query']['category_combined'].value_counts(normalize=True).get('Task Delegation', 0)*100:.0f}%
   - 3rd+ Query: Task Delegation {df[df['round_group']=='3rd+ Query']['category_combined'].value_counts(normalize=True).get('Task Delegation', 0)*100:.0f}%
   
   - 1st Query: Refinement Request {df[df['round_group']=='1st Query']['category_combined'].value_counts(normalize=True).get('Refinement Request', 0)*100:.0f}%
   - 2nd Query: Refinement Request {df[df['round_group']=='2nd Query']['category_combined'].value_counts(normalize=True).get('Refinement Request', 0)*100:.0f}%
   - 3rd+ Query: Refinement Request {df[df['round_group']=='3rd+ Query']['category_combined'].value_counts(normalize=True).get('Refinement Request', 0)*100:.0f}%

4. EXPERIENCED MANAGERS (>=10 years):
   - Mean queries: {exp_queries.mean():.2f} vs {non_exp_queries.mean():.2f} for less experienced
   - Task Delegation: {exp_pct.get('Task Delegation', 0):.1f}% vs {non_exp_pct.get('Task Delegation', 0):.1f}%
   
5. PHD HOLDERS:
   - Mean queries: {phd_queries.mean():.2f}
   - Task Delegation: {phd_pct.get('Task Delegation', 0):.1f}%
""")
    
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
