"""
STEP 11: SINGLE-QUERY USER ANALYSIS

This script analyzes the intent distribution of single-query users (autopilot)
versus multi-query users, providing evidence for different engagement orientations.

Key insight: Single-query users showed higher task delegation and lower evaluation
seeking compared to multi-query users, suggesting autopilot intent from the outset.

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


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('STEP 11: SINGLE-QUERY USER ANALYSIS')
    print('=' * 70)
    
    # Load data
    print(f'\nLoading {INPUT_FILE}...')
    df = pd.read_excel(INPUT_FILE)
    
    # Get queries per thread
    queries_per_thread = df.groupby('thread_id').size()
    total_threads = len(queries_per_thread)
    
    # Identify single-query and multi-query threads
    single_query_threads = queries_per_thread[queries_per_thread == 1].index
    multi_query_threads = queries_per_thread[queries_per_thread > 1].index
    
    n_single = len(single_query_threads)
    n_multi = len(multi_query_threads)
    
    print(f'\n  Total threads: {total_threads}')
    print(f'  Single-query users: {n_single} ({n_single/total_threads*100:.0f}%)')
    print(f'  Multi-query users: {n_multi} ({n_multi/total_threads*100:.0f}%)')
    
    # =========================================================================
    # SINGLE-QUERY USER INTENT DISTRIBUTION
    # =========================================================================
    print('\n' + '-' * 70)
    print('1. SINGLE-QUERY USER INTENT DISTRIBUTION')
    print('-' * 70)
    
    single_df = df[df['thread_id'].isin(single_query_threads)]
    single_pcts = single_df['category_combined'].value_counts(normalize=True) * 100
    
    print(f'\n  Single-query users (n={n_single}):')
    for cat in ['Task Delegation', 'Information Seeking', 'Other', 
                'Refinement Request', 'Evaluation Seeking', 'Acknowledgment']:
        pct = single_pcts.get(cat, 0)
        print(f'    {cat}: {pct:.1f}%')
    
    # =========================================================================
    # COMPARISON: SINGLE vs MULTI-QUERY USERS
    # =========================================================================
    print('\n' + '-' * 70)
    print('2. COMPARISON: SINGLE vs MULTI-QUERY USERS (First Query Only)')
    print('-' * 70)
    
    multi_df = df[df['thread_id'].isin(multi_query_threads)]
    multi_first = multi_df[multi_df['message_order'] == 1]
    multi_first_pcts = multi_first['category_combined'].value_counts(normalize=True) * 100
    
    print(f'\n  {"Category":<25} {"Single-Query":>15} {"Multi (1st)":>15} {"Diff":>10}')
    print('  ' + '-' * 65)
    
    key_categories = ['Task Delegation', 'Information Seeking', 'Evaluation Seeking', 
                      'Refinement Request', 'Other']
    
    for cat in key_categories:
        single_pct = single_pcts.get(cat, 0)
        multi_pct = multi_first_pcts.get(cat, 0)
        diff = single_pct - multi_pct
        print(f'  {cat:<25} {single_pct:>14.1f}% {multi_pct:>14.1f}% {diff:>+9.1f}pp')
    
    # =========================================================================
    # SINGLE-QUERY USERS BY CONDITION
    # =========================================================================
    print('\n' + '-' * 70)
    print('3. SINGLE-QUERY USERS BY CONDITION')
    print('-' * 70)
    
    for cond_val, cond_name in [(1, 'General AI'), (2, 'Agentic AI')]:
        subset = single_df[single_df['aristotle'] == cond_val]
        n = len(subset)
        pcts = subset['category_combined'].value_counts(normalize=True) * 100
        
        print(f'\n  {cond_name} (n={n}):')
        for cat in ['Task Delegation', 'Information Seeking', 'Evaluation Seeking']:
            print(f'    {cat}: {pcts.get(cat, 0):.1f}%')
    
    # =========================================================================
    # SINGLE-QUERY USERS BY SUBSAMPLE
    # =========================================================================
    print('\n' + '-' * 70)
    print('4. SINGLE-QUERY USERS BY SUBSAMPLE')
    print('-' * 70)
    
    # PhD vs Non-PhD single-query users
    phd_single = single_df[single_df['educationLevel'] >= 5]
    non_phd_single = single_df[single_df['educationLevel'] < 5]
    
    print(f'\n  PhD single-query users (n={len(phd_single)}):')
    phd_pcts = phd_single['category_combined'].value_counts(normalize=True) * 100
    for cat in ['Task Delegation', 'Information Seeking', 'Evaluation Seeking']:
        print(f'    {cat}: {phd_pcts.get(cat, 0):.1f}%')
    
    print(f'\n  Non-PhD single-query users (n={len(non_phd_single)}):')
    non_phd_pcts = non_phd_single['category_combined'].value_counts(normalize=True) * 100
    for cat in ['Task Delegation', 'Information Seeking', 'Evaluation Seeking']:
        print(f'    {cat}: {non_phd_pcts.get(cat, 0):.1f}%')
    
    # Experienced vs Less Experienced
    exp_single = single_df[single_df['managerialExperience'] >= 5]
    less_exp_single = single_df[single_df['managerialExperience'] < 5]
    
    print(f'\n  Experienced single-query users (n={len(exp_single)}):')
    exp_pcts = exp_single['category_combined'].value_counts(normalize=True) * 100
    for cat in ['Task Delegation', 'Information Seeking', 'Evaluation Seeking']:
        print(f'    {cat}: {exp_pcts.get(cat, 0):.1f}%')
    
    print(f'\n  Less Experienced single-query users (n={len(less_exp_single)}):')
    less_exp_pcts = less_exp_single['category_combined'].value_counts(normalize=True) * 100
    for cat in ['Task Delegation', 'Information Seeking', 'Evaluation Seeking']:
        print(f'    {cat}: {less_exp_pcts.get(cat, 0):.1f}%')
    
    # =========================================================================
    # KEY FINDINGS SUMMARY
    # =========================================================================
    print('\n' + '=' * 70)
    print('KEY FINDINGS FOR PARAGRAPH')
    print('=' * 70)
    
    task_del_single = single_pcts.get('Task Delegation', 0)
    task_del_multi = multi_first_pcts.get('Task Delegation', 0)
    eval_seek_single = single_pcts.get('Evaluation Seeking', 0)
    eval_seek_multi = multi_first_pcts.get('Evaluation Seeking', 0)
    
    print(f'''
SINGLE-QUERY USERS ({n_single} users, {n_single/total_threads*100:.0f}% of participants):

1. Intent Distribution:
   - Task Delegation: {task_del_single:.0f}%
   - Information Seeking: {single_pcts.get('Information Seeking', 0):.0f}%
   - Evaluation Seeking: {eval_seek_single:.0f}%

2. Comparison with Multi-Query Users (first query):
   - Task Delegation: {task_del_single:.0f}% vs {task_del_multi:.0f}% ({task_del_single - task_del_multi:+.0f}pp)
   - Evaluation Seeking: {eval_seek_single:.0f}% vs {eval_seek_multi:.0f}% ({eval_seek_single - eval_seek_multi:+.0f}pp)

3. Interpretation:
   Single-query users showed {task_del_single - task_del_multi:.0f}pp MORE task delegation 
   and {eval_seek_multi - eval_seek_single:.0f}pp LESS evaluation seeking compared to 
   multi-query users, indicating different engagement orientations from the outset.
''')
    
    # =========================================================================
    # SUGGESTED PARAGRAPH TEXT
    # =========================================================================
    print('=' * 70)
    print('SUGGESTED PARAGRAPH TEXT')
    print('=' * 70)
    
    paragraph = f"""Among the {n_single} single-query users ({n_single/total_threads*100:.0f}% of participants), 
task delegation was the dominant intent ({task_del_single:.0f}%), with relatively low 
evaluation seeking ({eval_seek_single:.0f}%); in contrast, users who continued beyond 
the first query showed lower initial task delegation ({task_del_multi:.0f}%) and higher 
evaluation seeking ({eval_seek_multi:.0f}%), suggesting different engagement orientations 
from the outset."""
    
    print('\n' + ' '.join(paragraph.split()))
    
    print('\n' + '=' * 70)
    print('ANALYSIS COMPLETE')
    print('=' * 70)


if __name__ == "__main__":
    main()
