"""
STEP 10: GENERATE ANALYSIS FIGURES

Generates figures for Section 6.4: Human-AI Integration Analysis

Metrics:
- Autopilot Behavior = Task Delegation rate (intent-based)
- Copilot Behavior = Evaluation Seeking + Refinement Request rate (intent-based)

Output:
- figure_intent_by_round.png: User intent distribution across query rounds
- figure_subsample_condition.png: Autopilot/Copilot behavior by subsample and condition

Configuration:
- Colors: SDA Bocconi Blue (#002855) + Yellow (#F2A900)
- Font: Times New Roman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'user_messages_fulldata_classified.xlsx'
FIGURE1_OUTPUT = 'figure_intent_by_round.png'
FIGURE2_OUTPUT = 'figure_subsample_condition.png'

# SDA Bocconi Colors
BOCCONI_BLUE = '#002855'    # Institutional dark blue (Agentic AI)
BOCCONI_YELLOW = '#F2A900'  # Energy/creativity yellow (General AI)

CONDITION_COLORS = {
    'General AI': BOCCONI_YELLOW,
    'Agentic AI': BOCCONI_BLUE,
}

# Category colors for Figure 1
COLORS = {
    'Task Delegation': '#002855',      # Bocconi Blue (Autopilot)
    'Refinement Request': '#E07B00',   # Darker orange
    'Information Seeking': '#4A90A4',  # Teal
    'Evaluation Seeking': '#8B4513',   # Brown
    'Other': '#808080',                # Gray
    'Acknowledgment': '#228B22',       # Forest Green
}

# Font settings - Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_autopilot_rate(subset_df):
    """Autopilot = Task Delegation rate"""
    if len(subset_df) == 0:
        return 0
    return (subset_df['category_combined'] == 'Task Delegation').mean() * 100


def compute_copilot_rate(subset_df):
    """Copilot = Evaluation Seeking + Refinement Request rate"""
    if len(subset_df) == 0:
        return 0
    copilot_count = ((subset_df['category_combined'] == 'Evaluation Seeking') | 
                     (subset_df['category_combined'] == 'Refinement Request')).sum()
    return copilot_count / len(subset_df) * 100


def get_single_query_rate(subset_df, queries_per_thread):
    """Calculate % of users who asked only 1 query"""
    threads = subset_df['thread_id'].unique()
    thread_queries = queries_per_thread[queries_per_thread.index.isin(threads)]
    if len(thread_queries) == 0:
        return 0
    return (thread_queries == 1).mean() * 100


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('STEP 10: GENERATE ANALYSIS FIGURES')
    print('=' * 70)
    print('\nConfiguration:')
    print('  - Autopilot = Task Delegation rate (intent-based)')
    print('  - Copilot = Evaluation + Refinement rate (intent-based)')
    print('  - Colors: Bocconi Blue + Yellow')
    print('  - Font: Times New Roman')
    
    # Load data
    df = pd.read_excel(INPUT_FILE)
    queries_per_thread = df.groupby('thread_id').size()
    
    # =========================================================================
    # COMPUTE METRICS
    # =========================================================================
    print('\n' + '-' * 70)
    print('COMPUTING METRICS')
    print('-' * 70)
    
    # Overall
    overall_autopilot = compute_autopilot_rate(df)
    overall_copilot = compute_copilot_rate(df)
    
    print(f'\n  Overall:')
    print(f'    Autopilot (Task Delegation): {overall_autopilot:.1f}%')
    print(f'    Copilot (Eval + Refine): {overall_copilot:.1f}%')
    
    # By round
    print(f'\n  By Query Round:')
    for round_name in ['1st Query', '2nd Query', '3rd+ Query']:
        round_df = df[df['round_group'] == round_name]
        auto = compute_autopilot_rate(round_df)
        copilot = compute_copilot_rate(round_df)
        print(f'    {round_name}: Autopilot={auto:.1f}%, Copilot={copilot:.1f}%')
    
    # =========================================================================
    # SUBSAMPLE x CONDITION ANALYSIS
    # =========================================================================
    print('\n' + '-' * 70)
    print('SUBSAMPLE x CONDITION: INTENT-BASED METRICS')
    print('-' * 70)
    
    # Define subsamples
    phd_df = df[df['educationLevel'] >= 5]
    non_phd_df = df[df['educationLevel'] < 5]
    exp_df = df[df['managerialExperience'] >= 5]
    less_exp_df = df[df['managerialExperience'] < 5]
    
    results = {}
    
    for name, subset in [('PhD', phd_df), ('Non-PhD', non_phd_df), 
                         ('Experienced', exp_df), ('Less Exp', less_exp_df)]:
        results[name] = {}
        for cond_val, cond_name in [(1, 'General AI'), (2, 'Agentic AI')]:
            cond_subset = subset[subset['aristotle'] == cond_val]
            
            results[name][cond_name] = {
                'autopilot': compute_autopilot_rate(cond_subset),
                'copilot': compute_copilot_rate(cond_subset),
                'single_query': get_single_query_rate(cond_subset, queries_per_thread),
                'n': len(cond_subset)
            }
    
    # Print results
    print('\n  AUTOPILOT RATE (Task Delegation):')
    print(f'  {"Subsample":<15} {"General AI":>12} {"Agentic AI":>12} {"Diff":>10}')
    print('  ' + '-' * 50)
    for name in ['PhD', 'Non-PhD', 'Experienced', 'Less Exp']:
        gen = results[name]['General AI']['autopilot']
        agent = results[name]['Agentic AI']['autopilot']
        diff = agent - gen
        print(f'  {name:<15} {gen:>11.1f}% {agent:>11.1f}% {diff:>+9.1f}pp')
    
    print('\n  COPILOT RATE (Eval + Refine):')
    print(f'  {"Subsample":<15} {"General AI":>12} {"Agentic AI":>12} {"Diff":>10}')
    print('  ' + '-' * 50)
    for name in ['PhD', 'Non-PhD', 'Experienced', 'Less Exp']:
        gen = results[name]['General AI']['copilot']
        agent = results[name]['Agentic AI']['copilot']
        diff = agent - gen
        print(f'  {name:<15} {gen:>11.1f}% {agent:>11.1f}% {diff:>+9.1f}pp')
    
    # =========================================================================
    # FIGURE 1: Intent Distribution by Query Round
    # =========================================================================
    print('\n' + '-' * 70)
    print('CREATING FIGURE 1: Intent by Query Round')
    print('-' * 70)
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    rounds = ['1st Query', '2nd Query', '3rd+ Query']
    categories = ['Task Delegation', 'Refinement Request', 'Information Seeking', 
                  'Evaluation Seeking', 'Other', 'Acknowledgment']
    
    round_data = {}
    for round_name in rounds:
        round_df = df[df['round_group'] == round_name]
        pcts = round_df['category_combined'].value_counts(normalize=True) * 100
        round_data[round_name] = [pcts.get(cat, 0) for cat in categories]
    
    x = np.arange(len(rounds))
    width = 0.12
    
    for i, cat in enumerate(categories):
        values = [round_data[r][i] for r in rounds]
        offset = (i - len(categories)/2 + 0.5) * width
        ax1.bar(x + offset, values, width, label=cat, color=COLORS[cat], 
                edgecolor='white', linewidth=0.5)
    
    ax1.set_ylabel('Percentage of Messages', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Query Round', fontweight='bold', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds, fontsize=10)
    ax1.set_ylim(0, 40)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('User Intent Distribution by Query Round', fontweight='bold', 
                  fontsize=12, pad=15)
    
    for i, round_name in enumerate(rounds):
        n = len(df[df['round_group'] == round_name])
        ax1.annotate(f'n={n}', xy=(i, -3), ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(FIGURE1_OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {FIGURE1_OUTPUT}')
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Subsample x Condition (Intent-Based)
    # =========================================================================
    print('\n' + '-' * 70)
    print('CREATING FIGURE 2: Subsample x Condition (Intent-Based)')
    print('-' * 70)
    
    fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    subsamples = ['PhD\nHolders', 'Non-PhD', 'Experienced\n(≥10 yrs)', 'Less\nExperienced']
    subsample_keys = ['PhD', 'Non-PhD', 'Experienced', 'Less Exp']
    
    # Left panel: Autopilot (Task Delegation rate)
    ax2a = axes[0]
    
    gen_autopilot = [results[k]['General AI']['autopilot'] for k in subsample_keys]
    agent_autopilot = [results[k]['Agentic AI']['autopilot'] for k in subsample_keys]
    
    x = np.arange(len(subsamples))
    width = 0.35
    
    ax2a.bar(x - width/2, gen_autopilot, width, label='General AI', 
             color=BOCCONI_YELLOW, edgecolor='white', linewidth=0.5)
    ax2a.bar(x + width/2, agent_autopilot, width, label='Agentic AI',
             color=BOCCONI_BLUE, edgecolor='white', linewidth=0.5)
    
    ax2a.set_ylabel('Task Delegation Rate (%)', fontweight='bold', fontsize=11)
    ax2a.set_xticks(x)
    ax2a.set_xticklabels(subsamples, fontsize=9)
    ax2a.set_ylim(0, 45)
    ax2a.legend(loc='upper right', fontsize=9)
    ax2a.spines['top'].set_visible(False)
    ax2a.spines['right'].set_visible(False)
    ax2a.set_title('Autopilot Behavior', fontweight='bold', fontsize=12, pad=10)
    
    # Add difference annotations
    for i, key in enumerate(subsample_keys):
        diff = results[key]['Agentic AI']['autopilot'] - results[key]['General AI']['autopilot']
        if abs(diff) >= 5:
            y_pos = max(gen_autopilot[i], agent_autopilot[i]) + 2
            color = BOCCONI_BLUE if diff > 0 else BOCCONI_YELLOW
            ax2a.annotate(f'{diff:+.0f}pp', xy=(i, y_pos), ha='center', 
                         fontsize=9, fontweight='bold', color=color)
    
    # Right panel: Copilot (Eval + Refine rate)
    ax2b = axes[1]
    
    gen_copilot = [results[k]['General AI']['copilot'] for k in subsample_keys]
    agent_copilot = [results[k]['Agentic AI']['copilot'] for k in subsample_keys]
    
    ax2b.bar(x - width/2, gen_copilot, width, label='General AI',
             color=BOCCONI_YELLOW, edgecolor='white', linewidth=0.5)
    ax2b.bar(x + width/2, agent_copilot, width, label='Agentic AI',
             color=BOCCONI_BLUE, edgecolor='white', linewidth=0.5)
    
    ax2b.set_ylabel('Copilot Rate (%)', fontweight='bold', fontsize=11)
    ax2b.set_xticks(x)
    ax2b.set_xticklabels(subsamples, fontsize=9)
    ax2b.set_ylim(0, 55)
    ax2b.legend(loc='upper right', fontsize=9)
    ax2b.spines['top'].set_visible(False)
    ax2b.spines['right'].set_visible(False)
    ax2b.set_title('Copilot Behavior', fontweight='bold', fontsize=12, pad=10)
    
    ax2b.annotate('(Evaluation + Refinement)', xy=(0.5, -0.12), xycoords='axes fraction',
                  ha='center', fontsize=8, style='italic', color='gray')
    
    # Add difference annotations
    for i, key in enumerate(subsample_keys):
        diff = results[key]['Agentic AI']['copilot'] - results[key]['General AI']['copilot']
        if abs(diff) >= 3:
            y_pos = max(gen_copilot[i], agent_copilot[i]) + 2
            color = BOCCONI_BLUE if diff > 0 else BOCCONI_YELLOW
            ax2b.annotate(f'{diff:+.0f}pp', xy=(i, y_pos), ha='center', 
                         fontsize=9, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(FIGURE2_OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {FIGURE2_OUTPUT}')
    plt.close()
    
    # =========================================================================
    # KEY STATISTICS SUMMARY
    # =========================================================================
    print('\n' + '=' * 70)
    print('KEY STATISTICS SUMMARY')
    print('=' * 70)
    
    n_messages = len(df)
    n_users = df['uniqueID'].nunique()
    n_threads = df['thread_id'].nunique()
    
    single_query_threads = queries_per_thread[queries_per_thread == 1].index
    n_single = len(single_query_threads)
    single_pct = n_single / n_threads * 100
    
    print(f'\n  Dataset:')
    print(f'    Messages: {n_messages:,}')
    print(f'    Participants: {n_users}')
    print(f'    Threads: {n_threads}')
    print(f'    Mean queries/user: {queries_per_thread.mean():.1f}')
    print(f'    Single-query users: {n_single} ({single_pct:.0f}%)')
    
    print(f'\n  Intent Evolution (1st → 3rd+ Query):')
    r1_auto = compute_autopilot_rate(df[df['round_group'] == '1st Query'])
    r3_auto = compute_autopilot_rate(df[df['round_group'] == '3rd+ Query'])
    r1_copilot = compute_copilot_rate(df[df['round_group'] == '1st Query'])
    r3_copilot = compute_copilot_rate(df[df['round_group'] == '3rd+ Query'])
    print(f'    Autopilot: {r1_auto:.0f}% → {r3_auto:.0f}%')
    print(f'    Copilot: {r1_copilot:.0f}% → {r3_copilot:.0f}%')
    
    print('\n' + '=' * 70)
    print('ANALYSIS COMPLETE')
    print('=' * 70)
    print(f'\nOutput files:')
    print(f'  - {FIGURE1_OUTPUT}')
    print(f'  - {FIGURE2_OUTPUT}')


if __name__ == "__main__":
    main()
