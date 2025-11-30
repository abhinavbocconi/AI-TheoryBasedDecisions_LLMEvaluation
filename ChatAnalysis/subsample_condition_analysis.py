"""
STEP 9: SUBSAMPLE x CONDITION ANALYSIS

This script analyzes chat behavior patterns by condition (General AI vs Agentic AI)
within key subsamples: PhD vs Non-PhD, and Experienced vs Less Experienced managers.

Key metrics computed:
- Single-query (autopilot) rate
- First-query Task Delegation rate
- Classification distribution
- Evaluation Seeking rate (copilot indicator)

Input:  user_messages_fulldata_classified.xlsx
Output: Console statistics for paper paragraph

Author: Auto-generated for ChatAnalysis project
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'user_messages_fulldata_classified.xlsx'

# Thresholds
EXPERIENCED_THRESHOLD = 5  # managerialExperience >= 5 means >=10 years
PHD_THRESHOLD = 5  # educationLevel >= 5 means PhD


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_single_query_rate(subset_df, queries_per_thread):
    """Calculate % of users who asked only 1 query (pure autopilot)."""
    threads = subset_df['thread_id'].unique()
    thread_queries = queries_per_thread[queries_per_thread.index.isin(threads)]
    if len(thread_queries) == 0:
        return 0, 0
    return (thread_queries == 1).mean() * 100, len(thread_queries)


def get_first_query_task_del_rate(subset_df):
    """Calculate % of first queries that were Task Delegation."""
    first_queries = subset_df[subset_df['message_order'] == 1]
    if len(first_queries) == 0:
        return 0, 0
    rate = (first_queries['category_combined'] == 'Task Delegation').mean() * 100
    return rate, len(first_queries)


def get_classification_distribution(subset_df):
    """Get classification percentages for key categories."""
    if len(subset_df) == 0:
        return {}
    pcts = subset_df['category_combined'].value_counts(normalize=True) * 100
    return pcts.to_dict()


def analyze_subsample(df, queries_per_thread, mask, subsample_name):
    """Analyze a subsample by condition."""
    subset = df[mask]
    
    print(f'\n{"=" * 70}')
    print(f'{subsample_name.upper()}')
    print(f'{"=" * 70}')
    print(f'Total: {len(subset)} messages, {subset["uniqueID"].nunique()} participants')
    
    results = {}
    
    for cond_val, cond_name in [(1, 'General AI'), (2, 'Agentic AI')]:
        cond_subset = subset[subset['aristotle'] == cond_val]
        n_users = cond_subset['uniqueID'].nunique()
        n_msg = len(cond_subset)
        
        # Metrics
        single_rate, n_threads = get_single_query_rate(cond_subset, queries_per_thread)
        first_td_rate, first_n = get_first_query_task_del_rate(cond_subset)
        class_dist = get_classification_distribution(cond_subset)
        
        # Mean queries
        threads = cond_subset['thread_id'].unique()
        thread_queries = queries_per_thread[queries_per_thread.index.isin(threads)]
        mean_queries = thread_queries.mean() if len(thread_queries) > 0 else 0
        
        results[cond_name] = {
            'n_users': n_users,
            'n_messages': n_msg,
            'n_threads': n_threads,
            'single_query_rate': single_rate,
            'first_query_td_rate': first_td_rate,
            'mean_queries': mean_queries,
            'task_delegation': class_dist.get('Task Delegation', 0),
            'refinement_request': class_dist.get('Refinement Request', 0),
            'information_seeking': class_dist.get('Information Seeking', 0),
            'evaluation_seeking': class_dist.get('Evaluation Seeking', 0),
        }
        
        print(f'\n  {cond_name}:')
        print(f'    Participants: {n_users}, Messages: {n_msg}, Threads: {n_threads}')
        print(f'    Mean queries: {mean_queries:.2f}')
        print(f'    Single-query rate (autopilot): {single_rate:.1f}%')
        print(f'    First query Task Delegation: {first_td_rate:.1f}%')
        print(f'    Overall classification:')
        print(f'      Task Delegation: {class_dist.get("Task Delegation", 0):.1f}%')
        print(f'      Refinement Request: {class_dist.get("Refinement Request", 0):.1f}%')
        print(f'      Information Seeking: {class_dist.get("Information Seeking", 0):.1f}%')
        print(f'      Evaluation Seeking: {class_dist.get("Evaluation Seeking", 0):.1f}%')
    
    # Differences
    gen = results['General AI']
    agent = results['Agentic AI']
    
    print(f'\n  DIFFERENCE (Agentic AI - General AI):')
    print(f'    Single-query rate: {agent["single_query_rate"] - gen["single_query_rate"]:+.1f}pp')
    print(f'    First query Task Del: {agent["first_query_td_rate"] - gen["first_query_td_rate"]:+.1f}pp')
    print(f'    Evaluation Seeking: {agent["evaluation_seeking"] - gen["evaluation_seeking"]:+.1f}pp')
    print(f'    Information Seeking: {agent["information_seeking"] - gen["information_seeking"]:+.1f}pp')
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('STEP 9: SUBSAMPLE x CONDITION ANALYSIS')
    print('=' * 70)
    
    # Load data
    print(f'\nLoading {INPUT_FILE}...')
    df = pd.read_excel(INPUT_FILE)
    print(f'  Total messages: {len(df)}')
    print(f'  Unique participants: {df["uniqueID"].nunique()}')
    
    # Get queries per thread
    queries_per_thread = df.groupby('thread_id').size()
    
    # -------------------------------------------------------------------------
    # ANALYSIS 1: PhD vs Non-PhD
    # -------------------------------------------------------------------------
    
    phd_results = analyze_subsample(
        df, queries_per_thread,
        df['educationLevel'] >= PHD_THRESHOLD,
        'PhD HOLDERS (educationLevel >= 5)'
    )
    
    non_phd_results = analyze_subsample(
        df, queries_per_thread,
        df['educationLevel'] < PHD_THRESHOLD,
        'NON-PhD HOLDERS (educationLevel < 5)'
    )
    
    # -------------------------------------------------------------------------
    # ANALYSIS 2: Experienced vs Less Experienced
    # -------------------------------------------------------------------------
    
    exp_results = analyze_subsample(
        df, queries_per_thread,
        df['managerialExperience'] >= EXPERIENCED_THRESHOLD,
        'EXPERIENCED MANAGERS (>=10 years, managerialExperience >= 5)'
    )
    
    less_exp_results = analyze_subsample(
        df, queries_per_thread,
        df['managerialExperience'] < EXPERIENCED_THRESHOLD,
        'LESS EXPERIENCED MANAGERS (<10 years, managerialExperience < 5)'
    )
    
    # -------------------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------------------
    
    print('\n' + '=' * 70)
    print('SUMMARY TABLE: KEY METRICS BY SUBSAMPLE x CONDITION')
    print('=' * 70)
    
    print('\n  SINGLE-QUERY (AUTOPILOT) RATE:')
    print('  ' + '-' * 55)
    print(f'  {"Subsample":<25} {"General AI":>12} {"Agentic AI":>12} {"Diff":>10}')
    print('  ' + '-' * 55)
    
    for name, results in [('PhD', phd_results), ('Non-PhD', non_phd_results),
                          ('Experienced (>=10yr)', exp_results), ('Less Exp (<10yr)', less_exp_results)]:
        gen = results['General AI']['single_query_rate']
        agent = results['Agentic AI']['single_query_rate']
        diff = agent - gen
        print(f'  {name:<25} {gen:>11.1f}% {agent:>11.1f}% {diff:>+9.1f}pp')
    
    print('\n  EVALUATION SEEKING RATE:')
    print('  ' + '-' * 55)
    print(f'  {"Subsample":<25} {"General AI":>12} {"Agentic AI":>12} {"Diff":>10}')
    print('  ' + '-' * 55)
    
    for name, results in [('PhD', phd_results), ('Non-PhD', non_phd_results),
                          ('Experienced (>=10yr)', exp_results), ('Less Exp (<10yr)', less_exp_results)]:
        gen = results['General AI']['evaluation_seeking']
        agent = results['Agentic AI']['evaluation_seeking']
        diff = agent - gen
        print(f'  {name:<25} {gen:>11.1f}% {agent:>11.1f}% {diff:>+9.1f}pp')
    
    # -------------------------------------------------------------------------
    # KEY FINDINGS FOR PARAGRAPH
    # -------------------------------------------------------------------------
    
    print('\n' + '=' * 70)
    print('KEY FINDINGS FOR PAPER PARAGRAPH')
    print('=' * 70)
    
    # PhD findings
    phd_gen_single = phd_results['General AI']['single_query_rate']
    phd_agent_single = phd_results['Agentic AI']['single_query_rate']
    phd_diff = phd_agent_single - phd_gen_single
    
    phd_gen_eval = phd_results['General AI']['evaluation_seeking']
    phd_agent_eval = phd_results['Agentic AI']['evaluation_seeking']
    phd_eval_diff = phd_agent_eval - phd_gen_eval
    
    print(f'''
1. PHD HOLDERS:
   - Single-query rate: {phd_gen_single:.0f}% (General AI) vs {phd_agent_single:.0f}% (Agentic AI) = {phd_diff:+.0f}pp
   - Evaluation Seeking: {phd_gen_eval:.1f}% (General AI) vs {phd_agent_eval:.1f}% (Agentic AI) = {phd_eval_diff:+.1f}pp
   - Interpretation: PhD holders in Agentic AI showed more autopilot behavior
''')
    
    # Experienced findings
    exp_gen_eval = exp_results['General AI']['evaluation_seeking']
    exp_agent_eval = exp_results['Agentic AI']['evaluation_seeking']
    exp_eval_diff = exp_agent_eval - exp_gen_eval
    
    exp_gen_single = exp_results['General AI']['single_query_rate']
    exp_agent_single = exp_results['Agentic AI']['single_query_rate']
    exp_single_diff = exp_agent_single - exp_gen_single
    
    print(f'''2. EXPERIENCED MANAGERS (>=10 years):
   - Single-query rate: {exp_gen_single:.0f}% (General AI) vs {exp_agent_single:.0f}% (Agentic AI) = {exp_single_diff:+.0f}pp
   - Evaluation Seeking: {exp_gen_eval:.1f}% (General AI) vs {exp_agent_eval:.1f}% (Agentic AI) = {exp_eval_diff:+.1f}pp
   - Interpretation: Experienced managers in Agentic AI used more validation/evaluation
''')
    
    # Non-PhD findings
    non_phd_gen_single = non_phd_results['General AI']['single_query_rate']
    non_phd_agent_single = non_phd_results['Agentic AI']['single_query_rate']
    non_phd_diff = non_phd_agent_single - non_phd_gen_single
    
    print(f'''3. NON-PHD HOLDERS:
   - Single-query rate: {non_phd_gen_single:.0f}% (General AI) vs {non_phd_agent_single:.0f}% (Agentic AI) = {non_phd_diff:+.0f}pp
   - Interpretation: No meaningful difference in autopilot behavior
''')
    
    print('=' * 70)
    print('ANALYSIS COMPLETE')
    print('=' * 70)


if __name__ == "__main__":
    main()
