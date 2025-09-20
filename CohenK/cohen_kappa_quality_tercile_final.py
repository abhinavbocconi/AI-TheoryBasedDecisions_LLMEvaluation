#!/usr/bin/env python3
"""
Cohen's Kappa Analysis for Quality Dimensions - Tercile Method
Author: Cohen's Kappa Inter-Rater Reliability Analysis
Dataset: Business Theory Evaluations (Human Experts vs LLMs)

This script calculates Cohen's kappa coefficients for 5 quality dimensions using
tercile categorization approach (alternative method):
- Novelty, Feasibility, Environmental Impact, Financial Impact, Overall Quality
- Method: Group ratings into Low (1-2), Medium (3), High (4-5) categories

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
    """Validate that the CSV has the expected structure for quality dimensions"""
    required_cols = [
        'expertNovelty', 'expertImplementation ', 'expertEnvironmental', 'expertFinancial', 'expertQuality',
        'anthropic_avg_novelty', 'anthropic_avg_feasibility_and_scalability', 'anthropic_avg_environmental_impact',
        'anthropic_avg_financial_impact', 'anthropic_avg_quality',
        'openai_avg_novelty', 'openai_avg_feasibility_and_scalability', 'openai_avg_environmental_impact',
        'openai_avg_financial_impact', 'openai_avg_quality',
        'cross_model_avg_novelty', 'cross_model_avg_feasibility_and_scalability', 'cross_model_avg_environmental_impact',
        'cross_model_avg_financial_impact', 'cross_model_avg_quality'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check teresaID column
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
        ids = pd.to_numeric(df[id_col], errors='coerce').dropna()
        if len(ids) == 1962 and ids.min() == 1 and ids.max() == 1962:
            print("✓ All teresaIDs 1-1962 present")
        else:
            print(f"Warning: teresaID range issues (found {len(ids)} IDs, range {ids.min()}-{ids.max()})")

    return id_col

def discretize_to_terciles(series):
    """Convert 1-5 scale to Low/Medium/High tercile categories"""
    return pd.cut(series,
                  bins=[0, 2.5, 3.5, 5],
                  labels=['Low', 'Medium', 'High'],
                  include_lowest=True)

def calculate_kappa_with_validation(data1, data2, comparison_name, dimension_name):
    """Calculate Cohen's kappa with comprehensive validation"""
    if not isinstance(data1, pd.Series):
        data1 = pd.Series(data1)
    if not isinstance(data2, pd.Series):
        data2 = pd.Series(data2)

    # Remove rows where either rater has missing values
    mask = data1.notna() & data2.notna()
    clean_data1 = data1[mask]
    clean_data2 = data2[mask]

    if len(clean_data1) == 0:
        print(f"  {dimension_name:13s}: κ = N/A (no valid pairs)")
        return np.nan

    # Calculate kappa
    kappa = cohen_kappa_score(clean_data1, clean_data2)
    n_pairs = len(clean_data1)

    print(f"  {dimension_name:13s}: κ = {kappa:.3f} (n={n_pairs})")

    return kappa

def main():
    """Main analysis function"""

    print("="*80)
    print("COHEN'S KAPPA ANALYSIS - QUALITY DIMENSIONS (TERCILE METHOD)")
    print("="*80)
    print("Alternative approach: Group ratings into Low (1-2), Medium (3), High (4-5)")
    print("Evaluating 5 dimensions: Novelty, Feasibility, Environmental, Financial, Quality")
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

    # Define dimension mappings
    dimensions = ['Novelty', 'Feasibility', 'Environmental', 'Financial', 'Quality']

    human_cols = {
        'Novelty': 'expertNovelty',
        'Feasibility': 'expertImplementation ',  # Note: includes trailing space in CSV
        'Environmental': 'expertEnvironmental',
        'Financial': 'expertFinancial',
        'Quality': 'expertQuality'
    }

    anthropic_cols = {
        'Novelty': 'anthropic_avg_novelty',
        'Feasibility': 'anthropic_avg_feasibility_and_scalability',
        'Environmental': 'anthropic_avg_environmental_impact',
        'Financial': 'anthropic_avg_financial_impact',
        'Quality': 'anthropic_avg_quality'
    }

    openai_cols = {
        'Novelty': 'openai_avg_novelty',
        'Feasibility': 'openai_avg_feasibility_and_scalability',
        'Environmental': 'openai_avg_environmental_impact',
        'Financial': 'openai_avg_financial_impact',
        'Quality': 'openai_avg_quality'
    }

    cross_model_cols = {
        'Novelty': 'cross_model_avg_novelty',
        'Feasibility': 'cross_model_avg_feasibility_and_scalability',
        'Environmental': 'cross_model_avg_environmental_impact',
        'Financial': 'cross_model_avg_financial_impact',
        'Quality': 'cross_model_avg_quality'
    }

    # Convert to numeric and validate
    print("\nConverting data to numeric format...")
    all_cols = []
    for col_dict in [human_cols, anthropic_cols, openai_cols, cross_model_cols]:
        all_cols.extend(col_dict.values())

    for col in all_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            missing = df[col].isnull().sum()
            available = len(df) - missing
            value_range = f"{df[col].min():.1f}-{df[col].max():.1f}"
            print(f"  {col}: {available}/{len(df)} valid, range {value_range}")
        else:
            print(f"  Warning: {col} not found in data")

    # Apply tercile discretization to all data
    print("\n" + "="*60)
    print("TERCILE DISCRETIZATION")
    print("="*60)
    print("Method: Low (1-2), Medium (3), High (4-5) for ALL raters")
    print("Rationale: Groups ratings into meaningful performance categories")

    discretized_data = {}

    # Process each dimension
    for dim in dimensions:
        print(f"\n{dim}:")

        # Apply tercile discretization to all raters
        for rater, col_dict in [('Human', human_cols), ('Claude', anthropic_cols),
                               ('GPT', openai_cols), ('LLM_Avg', cross_model_cols)]:
            col = col_dict[dim]
            if col in df.columns:
                discretized_data[f"{dim}_{rater}"] = discretize_to_terciles(df[col])
                counts = discretized_data[f"{dim}_{rater}"].value_counts().sort_index()
                total = counts.sum()
                dist_str = " | ".join([f"{cat}:{counts.get(cat,0)}({counts.get(cat,0)/total*100:.1f}%)"
                                     for cat in ['Low', 'Medium', 'High']])
                print(f"  {rater}: {dist_str}")
            else:
                print(f"  {rater}: Missing column {col}")

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

    kappa_results = {}

    for comp_name, rater1, rater2 in comparisons:
        print(f"\n{comp_name}:")
        kappa_results[comp_name] = {}

        for dim in dimensions:
            key1 = f"{dim}_{rater1}"
            key2 = f"{dim}_{rater2}"

            if key1 in discretized_data and key2 in discretized_data:
                kappa = calculate_kappa_with_validation(
                    discretized_data[key1], discretized_data[key2],
                    comp_name, dim
                )
                kappa_results[comp_name][dim] = kappa
            else:
                print(f"  {dim:13s}: κ = N/A (missing data)")
                kappa_results[comp_name][dim] = np.nan

    # Generate LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE OUTPUT")
    print("="*60)

    print("\n% Cohen's Kappa for Quality Dimensions - Tercile Method")
    print("% Alternative approach using tercile discretization (Low/Medium/High)")
    print("% Copy and paste into LaTeX:")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("\\textbf{Rater Comparison} & \\textbf{Novelty} & \\textbf{Feasibility} & \\textbf{Environmental} & \\textbf{Financial} & \\textbf{Quality}\\\\")
    print("\\hline")

    for comp_name, _, _ in comparisons:
        row_values = []
        for dim in dimensions:
            kappa = kappa_results[comp_name].get(dim, np.nan)
            kappa_str = f"{kappa:.3f}" if not np.isnan(kappa) else "N/A"
            row_values.append(kappa_str)
        print(f"{comp_name} & {' & '.join(row_values)} \\\\")

    print("\\hline")
    print("\\end{tabular}")

    # Comparison with integer method (provide reference values)
    print("\n" + "="*60)
    print("COMPARISON WITH INTEGER METHOD")
    print("="*60)

    # Integer method results for comparison (from previous analysis)
    integer_results = {
        'Human vs Claude': {'Novelty': 0.056, 'Feasibility': 0.108, 'Environmental': 0.054, 'Financial': 0.129, 'Quality': 0.154},
        'Human vs GPT': {'Novelty': 0.069, 'Feasibility': -0.002, 'Environmental': 0.085, 'Financial': 0.051, 'Quality': 0.064},
        'Human vs LLM Avg': {'Novelty': 0.074, 'Feasibility': 0.043, 'Environmental': 0.061, 'Financial': 0.083, 'Quality': 0.124},
        'Claude vs GPT': {'Novelty': 0.505, 'Feasibility': 0.025, 'Environmental': 0.554, 'Financial': 0.299, 'Quality': 0.070}
    }

    print("Improvement Analysis (Tercile κ - Integer κ):")
    print("Positive values indicate tercile method performs better")

    total_improvements = []
    for comp_name, _, _ in comparisons:
        print(f"\n{comp_name}:")
        comp_improvements = []
        for dim in dimensions:
            tercile_kappa = kappa_results[comp_name].get(dim, np.nan)
            integer_kappa = integer_results[comp_name].get(dim, np.nan)

            if not np.isnan(tercile_kappa) and not np.isnan(integer_kappa):
                improvement = tercile_kappa - integer_kappa
                comp_improvements.append(improvement)
                total_improvements.append(improvement)
                print(f"  {dim:13s}: {improvement:+.3f} ({tercile_kappa:.3f} vs {integer_kappa:.3f})")
            else:
                print(f"  {dim:13s}: N/A")

        if comp_improvements:
            avg_improvement = np.mean(comp_improvements)
            print(f"  Average: {avg_improvement:+.3f}")

    if total_improvements:
        overall_improvement = np.mean(total_improvements)
        positive_improvements = sum(1 for x in total_improvements if x > 0)
        total_comparisons = len(total_improvements)

        print(f"\nOVERALL IMPROVEMENT:")
        print(f"  Mean improvement: {overall_improvement:+.3f}")
        print(f"  Improvements: {positive_improvements}/{total_comparisons} ({positive_improvements/total_comparisons*100:.1f}%)")

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

    # Calculate averages by comparison type
    print("\nAverage Kappa by Comparison:")
    for comp_name, _, _ in comparisons:
        valid_kappas = [k for k in kappa_results[comp_name].values() if not np.isnan(k)]
        if valid_kappas:
            avg_kappa = np.mean(valid_kappas)
            interpretation = interpret_kappa(avg_kappa)
            print(f"  {comp_name}: κ = {avg_kappa:.3f} ({interpretation})")

    # Overall statistics
    all_kappas = []
    for comp_name in kappa_results:
        for dim in dimensions:
            kappa = kappa_results[comp_name].get(dim, np.nan)
            if not np.isnan(kappa):
                all_kappas.append(kappa)

    if all_kappas:
        print(f"\nOverall Statistics:")
        print(f"  Total valid comparisons: {len(all_kappas)}")
        print(f"  Mean κ: {np.mean(all_kappas):.3f}")
        print(f"  Range: {np.min(all_kappas):.3f} - {np.max(all_kappas):.3f}")

        # Distribution by interpretation
        interpretations = [interpret_kappa(k) for k in all_kappas]
        from collections import Counter
        interp_counts = Counter(interpretations)
        print(f"  Agreement levels:")
        for interp, count in interp_counts.most_common():
            pct = count/len(all_kappas)*100
            print(f"    {interp}: {count} ({pct:.1f}%)")

    print(f"\n✓ Tercile method analysis complete!")
    print("NOTE: This approach trades granularity for improved agreement detection")
    print("by grouping ratings into broader Low/Medium/High categories.")

if __name__ == "__main__":
    main()