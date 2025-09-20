#!/usr/bin/env python3
"""
Cohen's Kappa Analysis for SPS and CIT Measures
Author: Cohen's Kappa Inter-Rater Reliability Analysis
Dataset: Business Theory Evaluations (Human Experts vs LLMs)

This script calculates Cohen's kappa coefficients for:
- SPS: Subjective Probability of Success (0-100 scale)
- CIT: Confidence in Theory Assessment (1-7 scale)

Comparisons:
- Human Expert vs Claude (Anthropic)
- Human Expert vs GPT (OpenAI o4-mini)
- Human Expert vs LLM Average
- Claude vs GPT

GitHub: https://github.com/[your-repo]/CohenK
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

def validate_data_structure(df):
    """Validate that the CSV has the expected structure"""
    required_cols = [
        'expertProbability', 'expertConfidence',
        'anthropic_avg_sps_llm', 'anthropic_avg_cit_llm',
        'openai_avg_sps_llm', 'openai_avg_cit_llm',
        'cross_model_avg_sps_llm', 'cross_model_avg_cit_llm'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check if teresaID column exists (with or without space)
    id_col = None
    for potential_name in ['teresaID', 'Teresa ID', 'TeresaID']:
        if potential_name in df.columns:
            id_col = potential_name
            break

    if id_col is None:
        print("Warning: No teresaID column found")
    else:
        print(f"Found ID column: '{id_col}'")
        # Verify IDs are 1-1962
        if id_col in df.columns:
            ids = pd.to_numeric(df[id_col], errors='coerce').dropna()
            expected_range = set(range(1, 1963))
            actual_range = set(ids.astype(int))
            if actual_range != expected_range:
                missing = expected_range - actual_range
                extra = actual_range - expected_range
                if missing:
                    print(f"Warning: Missing teresaIDs: {sorted(list(missing))[:10]}...")
                if extra:
                    print(f"Warning: Unexpected teresaIDs: {sorted(list(extra))[:10]}...")
            else:
                print("✓ All teresaIDs 1-1962 present")

    return id_col

def discretize_sps_terciles(df, expert_col):
    """
    Discretize SPS scores into terciles based on expert distribution

    Logic: Use expert data to define tercile boundaries, then apply to all raters
    This ensures meaningful categories based on human judgment patterns
    """
    expert_data = df[expert_col].dropna()
    if len(expert_data) == 0:
        raise ValueError(f"No valid data in {expert_col}")

    # Calculate terciles (33rd and 67th percentiles)
    t33, t67 = expert_data.quantile([0.33, 0.67])

    bins = [-np.inf, t33, t67, np.inf]
    labels = ['Low', 'Medium', 'High']

    print(f"SPS Tercile boundaries: Low (≤{t33:.1f}), Medium ({t33:.1f}-{t67:.1f}), High (>{t67:.1f})")

    return bins, labels

def discretize_cit_categories(df):
    """
    Discretize CIT scores into logical categories

    Logic: 1-7 scale grouped as Low (1-3), Medium (4-5), High (6-7)
    Based on semantic meaning of confidence levels
    """
    bins = [0, 3, 5, 7]
    labels = ['Low', 'Medium', 'High']

    print("CIT Categories: Low (1-3), Medium (4-5), High (6-7)")

    return bins, labels

def calculate_kappa_with_validation(data1, data2, comparison_name, measure_name):
    """
    Calculate Cohen's kappa with comprehensive validation

    Validates data, handles missing values, and provides detailed output
    """
    # Convert to pandas Series if needed
    if not isinstance(data1, pd.Series):
        data1 = pd.Series(data1)
    if not isinstance(data2, pd.Series):
        data2 = pd.Series(data2)

    # Remove rows where either rater has missing values
    mask = data1.notna() & data2.notna()
    clean_data1 = data1[mask]
    clean_data2 = data2[mask]

    if len(clean_data1) == 0:
        print(f"  {comparison_name} {measure_name}: κ = N/A (no valid pairs)")
        return np.nan

    # Validate that both have the same categories
    cats1 = set(clean_data1.unique())
    cats2 = set(clean_data2.unique())
    if cats1 != cats2:
        print(f"  Warning: Different categories in {comparison_name} {measure_name}")
        print(f"    Rater 1: {sorted(cats1)}")
        print(f"    Rater 2: {sorted(cats2)}")

    # Calculate kappa
    kappa = cohen_kappa_score(clean_data1, clean_data2)
    n_pairs = len(clean_data1)

    print(f"  {comparison_name} {measure_name}: κ = {kappa:.3f} (n={n_pairs})")

    return kappa

def main():
    """Main analysis function"""

    print("="*80)
    print("COHEN'S KAPPA ANALYSIS - SPS AND CIT MEASURES")
    print("="*80)
    print("Evaluating inter-rater reliability between human experts and LLMs")
    print("on Subjective Probability of Success and Confidence in Theory measures")
    print("="*80)

    # Load and validate data
    try:
        df = pd.read_csv('/Users/anilpandey/Documents/Code/RnRLLMEvaluation/CohenK/business_evaluation_full.csv')
        print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("Error: business_evaluation_full.csv not found in current directory")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Validate data structure
    id_col = validate_data_structure(df)

    # Define columns
    sps_cols = {
        'Human': 'expertProbability',
        'Claude': 'anthropic_avg_sps_llm',
        'GPT': 'openai_avg_sps_llm',
        'LLM_Avg': 'cross_model_avg_sps_llm'
    }

    cit_cols = {
        'Human': 'expertConfidence',
        'Claude': 'anthropic_avg_cit_llm',
        'GPT': 'openai_avg_cit_llm',
        'LLM_Avg': 'cross_model_avg_cit_llm'
    }

    # Convert to numeric
    print("\nConverting data to numeric format...")
    all_cols = list(sps_cols.values()) + list(cit_cols.values())
    for col in all_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            missing = df[col].isnull().sum()
            available = len(df) - missing
            print(f"  {col}: {available}/{len(df)} valid values")
        else:
            print(f"  Warning: {col} not found in data")

    # Discretize SPS scores
    print("\n" + "="*60)
    print("SPS DISCRETIZATION")
    print("="*60)

    try:
        sps_bins, sps_labels = discretize_sps_terciles(df, sps_cols['Human'])

        sps_discretized = {}
        for rater, col in sps_cols.items():
            if col in df.columns:
                sps_discretized[rater] = pd.cut(df[col], bins=sps_bins, labels=sps_labels, include_lowest=True)

                # Show distribution
                counts = sps_discretized[rater].value_counts().sort_index()
                total = counts.sum()
                dist_str = " | ".join([f"{label}:{counts.get(label, 0)}({counts.get(label, 0)/total*100:.1f}%)"
                                     for label in sps_labels])
                print(f"  {rater}: {dist_str}")

    except Exception as e:
        print(f"Error in SPS discretization: {e}")
        return

    # Discretize CIT scores
    print("\n" + "="*60)
    print("CIT DISCRETIZATION")
    print("="*60)

    try:
        cit_bins, cit_labels = discretize_cit_categories(df)

        cit_discretized = {}
        for rater, col in cit_cols.items():
            if col in df.columns:
                cit_discretized[rater] = pd.cut(df[col], bins=cit_bins, labels=cit_labels, include_lowest=True)

                # Show distribution
                counts = cit_discretized[rater].value_counts().sort_index()
                total = counts.sum()
                dist_str = " | ".join([f"{label}:{counts.get(label, 0)}({counts.get(label, 0)/total*100:.1f}%)"
                                     for label in cit_labels])
                print(f"  {rater}: {dist_str}")

    except Exception as e:
        print(f"Error in CIT discretization: {e}")
        return

    # Calculate Cohen's Kappa
    print("\n" + "="*60)
    print("COHEN'S KAPPA RESULTS")
    print("="*60)

    comparisons = [
        ('Human vs Claude', 'Human', 'Claude'),
        ('Human vs GPT', 'Human', 'GPT'),
        ('Human vs LLM Avg', 'Human', 'LLM_Avg'),
        ('Claude vs GPT', 'Claude', 'GPT')
    ]

    kappa_results = {'SPS': {}, 'CIT': {}}

    for comp_name, rater1, rater2 in comparisons:
        print(f"\n{comp_name}:")

        # SPS Kappa
        if rater1 in sps_discretized and rater2 in sps_discretized:
            sps_kappa = calculate_kappa_with_validation(
                sps_discretized[rater1], sps_discretized[rater2],
                comp_name, "SPS"
            )
            kappa_results['SPS'][comp_name] = sps_kappa
        else:
            print(f"  {comp_name} SPS: κ = N/A (missing data)")
            kappa_results['SPS'][comp_name] = np.nan

        # CIT Kappa
        if rater1 in cit_discretized and rater2 in cit_discretized:
            cit_kappa = calculate_kappa_with_validation(
                cit_discretized[rater1], cit_discretized[rater2],
                comp_name, "CIT"
            )
            kappa_results['CIT'][comp_name] = cit_kappa
        else:
            print(f"  {comp_name} CIT: κ = N/A (missing data)")
            kappa_results['CIT'][comp_name] = np.nan

    # Generate LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE OUTPUT")
    print("="*60)

    print("\n% Cohen's Kappa for SPS and CIT measures")
    print("% Copy and paste into LaTeX:")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("\\textbf{Comparison} & \\textbf{SPS κ} & \\textbf{CIT κ} \\\\")
    print("\\hline")

    for comp_name, _, _ in comparisons:
        sps_val = kappa_results['SPS'].get(comp_name, np.nan)
        cit_val = kappa_results['CIT'].get(comp_name, np.nan)
        sps_str = f"{sps_val:.3f}" if not np.isnan(sps_val) else "N/A"
        cit_str = f"{cit_val:.3f}" if not np.isnan(cit_val) else "N/A"
        print(f"{comp_name} & {sps_str} & {cit_str} \\\\")

    print("\\hline")
    print("\\end{tabular}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    def interpret_kappa(kappa):
        if np.isnan(kappa):
            return "N/A"
        elif kappa < 0.20:
            return "Poor"
        elif kappa <= 0.40:
            return "Fair"
        elif kappa <= 0.60:
            return "Moderate"
        elif kappa <= 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"

    print("Interpretation Guide:")
    print("  κ < 0.20 = Poor | 0.21-0.40 = Fair | 0.41-0.60 = Moderate")
    print("  0.61-0.80 = Substantial | 0.80+ = Almost Perfect")

    print("\nKey Findings:")
    for measure in ['SPS', 'CIT']:
        valid_kappas = [k for k in kappa_results[measure].values() if not np.isnan(k)]
        if valid_kappas:
            avg_kappa = np.mean(valid_kappas)
            interpretation = interpret_kappa(avg_kappa)
            print(f"  {measure}: Average κ = {avg_kappa:.3f} ({interpretation})")
        else:
            print(f"  {measure}: No valid kappa values")

    print(f"\n✓ Analysis complete! Results ready for publication.")

if __name__ == "__main__":
    main()