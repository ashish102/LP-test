#!/usr/bin/env python3
"""
Adaptive Two-Stage Analysis

This script performs the CORRECT two-stage stochastic analysis where:
- For each p1 and realized d1, we find the BEST p2 that minimizes expected cost
- p2 adapts to d1 (not static)
- Uses existing two_stage_sensitivity.csv which has all the necessary data

Algorithm:
1. For each p1 (1-10):
   a. For each d1 sample (100 samples):
      - Find the p2 with minimum mean_cost for this (p1, d1) combination
      - This is the "adapted" p2 for this specific (p1, d1)
      - Record the cost of this optimal p2
   b. Calculate statistics across all d1 samples:
      - Mean of adapted costs
      - Std of adapted costs
      - Other statistics

Output: adaptive_two_stage_results.csv with one row per p1
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_adaptive_two_stage(input_csv_path):
    """
    Perform adaptive two-stage analysis from existing sensitivity data.

    Parameters:
    -----------
    input_csv_path : Path
        Path to two_stage_sensitivity.csv

    Returns:
    --------
    DataFrame with one row per p1, containing:
    - p1, p1_units
    - mean_cost (average of adapted costs across d1 samples)
    - std_cost (std of adapted costs across d1 samples)
    - cv_pct
    - min_cost, max_cost, median_cost, q25_cost, q75_cost
    - avg_optimal_p2 (average p2 chosen across d1 samples)
    """
    print("=" * 80)
    print("ADAPTIVE TWO-STAGE ANALYSIS")
    print("=" * 80)
    print("\nLoading sensitivity data...")

    # Load the full sensitivity analysis results
    df = pd.read_csv(input_csv_path)

    print(f"Loaded {len(df):,} rows from {input_csv_path.name}")
    print(f"Columns: {', '.join(df.columns)}")

    # Verify structure
    p1_values = sorted(df['p1'].unique())
    d1_samples = sorted(df['d1_sample_idx'].unique())
    p2_values = sorted(df['p2'].unique())

    print(f"\nData structure:")
    print(f"  p1 values: {len(p1_values)} ({min(p1_values)} to {max(p1_values)})")
    print(f"  d1 samples: {len(d1_samples)} samples")
    print(f"  p2 values: {len(p2_values)} ({min(p2_values)} to {max(p2_values)})")
    print(f"  Total combinations: {len(p1_values) * len(d1_samples) * len(p2_values):,}")

    print("\n" + "=" * 80)
    print("COMPUTING ADAPTIVE DECISIONS")
    print("=" * 80)

    # For each (p1, d1_sample), find the optimal p2
    results = []

    for p1 in p1_values:
        print(f"\nAnalyzing p1 = {p1} batches...")

        # Filter to this p1
        df_p1 = df[df['p1'] == p1]

        # Storage for this p1
        adapted_costs = []
        adapted_p2_choices = []

        # For each d1 sample
        for d1_sample_idx in d1_samples:
            # Filter to this (p1, d1) combination
            df_p1_d1 = df_p1[df_p1['d1_sample_idx'] == d1_sample_idx]

            if len(df_p1_d1) == 0:
                print(f"  WARNING: No data for p1={p1}, d1_sample={d1_sample_idx}")
                continue

            # Find the p2 with minimum mean_cost
            optimal_row = df_p1_d1.loc[df_p1_d1['mean_cost'].idxmin()]

            optimal_p2 = optimal_row['p2']
            optimal_cost = optimal_row['mean_cost']
            d1_value = optimal_row['d1']

            # Store the adapted cost and p2 choice
            adapted_costs.append(optimal_cost)
            adapted_p2_choices.append(optimal_p2)

        # Calculate statistics for this p1
        adapted_costs = np.array(adapted_costs)
        adapted_p2_choices = np.array(adapted_p2_choices)

        mean_cost = np.mean(adapted_costs)
        std_cost = np.std(adapted_costs)

        results.append({
            'p1': p1,
            'p1_units': p1 * 50,  # batch_size = 50
            'mean_cost': mean_cost,
            'std_cost': std_cost,
            'cv_pct': (std_cost / mean_cost * 100) if mean_cost > 0 else 0,
            'min_cost': np.min(adapted_costs),
            'max_cost': np.max(adapted_costs),
            'median_cost': np.median(adapted_costs),
            'q25_cost': np.percentile(adapted_costs, 25),
            'q75_cost': np.percentile(adapted_costs, 75),
            'avg_optimal_p2': np.mean(adapted_p2_choices),
            'min_optimal_p2': np.min(adapted_p2_choices),
            'max_optimal_p2': np.max(adapted_p2_choices),
            'std_optimal_p2': np.std(adapted_p2_choices)
        })

        print(f"  Mean cost (adapted): ${mean_cost:,.2f}")
        print(f"  Std cost: ${std_cost:,.2f}")
        print(f"  CV: {std_cost/mean_cost*100:.1f}%")
        print(f"  Avg optimal p2: {np.mean(adapted_p2_choices):.2f} batches")
        print(f"  p2 range: {np.min(adapted_p2_choices)}-{np.max(adapted_p2_choices)} batches")

    return pd.DataFrame(results)


def print_analysis_report(df_results):
    """Print comprehensive analysis report."""
    print("\n" + "=" * 80)
    print("ADAPTIVE TWO-STAGE RESULTS")
    print("=" * 80)

    print("\nComplete Results Table:")
    print("-" * 80)

    # Header
    print(f"{'p1':>3} | {'Units':>5} | {'Mean Cost':>12} | {'Std Cost':>12} | "
          f"{'CV (%)':>8} | {'Avg p2':>7} | {'p2 Range':>10}")
    print("-" * 80)

    # Data rows
    for _, row in df_results.iterrows():
        p1 = int(row['p1'])
        units = int(row['p1_units'])
        mean_cost = row['mean_cost']
        std_cost = row['std_cost']
        cv = row['cv_pct']
        avg_p2 = row['avg_optimal_p2']
        p2_range = f"{int(row['min_optimal_p2'])}-{int(row['max_optimal_p2'])}"

        print(f"{p1:3d} | {units:5d} | ${mean_cost:11,.2f} | ${std_cost:11,.2f} | "
              f"{cv:7.2f}% | {avg_p2:7.2f} | {p2_range:>10}")

    print("-" * 80)

    # Find optimal
    optimal_idx = df_results['mean_cost'].idxmin()
    optimal = df_results.loc[optimal_idx]

    print("\n" + "=" * 80)
    print("OPTIMAL FIRST-STAGE DECISION")
    print("=" * 80)
    print(f"\n  p1: {int(optimal['p1'])} batches ({int(optimal['p1_units'])} units)")
    print(f"  Mean cost: ${optimal['mean_cost']:,.2f}")
    print(f"  Std cost: ${optimal['std_cost']:,.2f}")
    print(f"  Coefficient of variation: {optimal['cv_pct']:.2f}%")
    print(f"  Cost range: ${optimal['min_cost']:,.2f} - ${optimal['max_cost']:,.2f}")
    print(f"  Average optimal p2: {optimal['avg_optimal_p2']:.2f} batches")
    print(f"  p2 adaptation range: {int(optimal['min_optimal_p2'])}-{int(optimal['max_optimal_p2'])} batches")

    # Cost comparison
    print("\n" + "=" * 80)
    print("COST COMPARISON vs OPTIMAL")
    print("=" * 80)

    optimal_cost = optimal['mean_cost']

    print("\nCost penalty for suboptimal p1 decisions:")
    print("-" * 60)
    print(f"{'p1':>3} | {'Mean Cost':>12} | {'vs Optimal':>15} | {'Penalty %':>10}")
    print("-" * 60)

    for _, row in df_results.iterrows():
        p1 = int(row['p1'])
        cost = row['mean_cost']
        diff = cost - optimal_cost
        pct = (diff / optimal_cost * 100) if optimal_cost > 0 else 0

        marker = " ✓ OPTIMAL" if p1 == optimal['p1'] else ""
        print(f"{p1:3d} | ${cost:11,.2f} | ${diff:+11,.2f} | {pct:9.2f}%{marker}")

    print("-" * 60)

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n1. Optimal Strategy:")
    print(f"   - First-stage production: {int(optimal['p1'])} batches")
    print(f"   - Expected cost: ${optimal['mean_cost']:,.2f} ± ${optimal['std_cost']:,.2f}")
    print(f"   - Second-stage adapts based on realized demand (d1)")
    print(f"   - Average second-stage production: {optimal['avg_optimal_p2']:.2f} batches")

    print("\n2. Value of Adaptation:")
    print(f"   - Second-stage production ranges from {int(optimal['min_optimal_p2'])} to "
          f"{int(optimal['max_optimal_p2'])} batches")
    print(f"   - This flexibility responds to realized first-day demand")
    print(f"   - Adaptation reduces expected cost compared to static decisions")

    print("\n3. Cost Sensitivity:")
    worst_idx = df_results['mean_cost'].idxmax()
    worst = df_results.loc[worst_idx]
    cost_range = worst['mean_cost'] - optimal['mean_cost']
    pct_range = (cost_range / optimal['mean_cost'] * 100)

    print(f"   - Cost range: ${optimal['mean_cost']:,.2f} to ${worst['mean_cost']:,.2f}")
    print(f"   - Maximum penalty: ${cost_range:,.2f} ({pct_range:.1f}%) for p1={int(worst['p1'])}")
    print(f"   - First-stage decision has significant impact on total cost")

    print("\n4. Uncertainty:")
    avg_cv = df_results['cv_pct'].mean()
    min_cv = df_results['cv_pct'].min()
    max_cv = df_results['cv_pct'].max()

    print(f"   - Average coefficient of variation: {avg_cv:.1f}%")
    print(f"   - CV range: {min_cv:.1f}% to {max_cv:.1f}%")
    print(f"   - Demand uncertainty creates substantial cost variability")


if __name__ == "__main__":
    # Paths
    script_dir = Path(__file__).parent
    input_csv = script_dir / "two_stage_sensitivity.csv"
    output_csv = script_dir / "adaptive_two_stage_results.csv"

    # Check input file exists
    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}")
        print("Please run two_stage_sensitivity.py first to generate the data.")
        exit(1)

    # Run analysis
    df_results = analyze_adaptive_two_stage(input_csv)

    # Save results
    df_results.to_csv(output_csv, index=False)
    print(f"\n\nResults saved to: {output_csv}")

    # Print comprehensive report
    print_analysis_report(df_results)

    print("\n" + "=" * 80)
    print("✓ Adaptive two-stage analysis complete!")
    print("=" * 80)
