#!/usr/bin/env python3
"""
Visualization script for two-stage sensitivity analysis results.

Creates various plots to understand the impact of first and second
production decisions on cost and uncertainty.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results():
    """Load CSV results."""
    csv_path = Path(__file__).parent / "two_stage_sensitivity.csv"
    agg_path = Path(__file__).parent / "two_stage_aggregated_p1_p2.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}\nRun two_stage_sensitivity.py first!")

    df = pd.read_csv(csv_path)
    agg_df = pd.read_csv(agg_path) if agg_path.exists() else None

    return df, agg_df


def plot_p1_p2_heatmap(agg_df, output_dir):
    """Create heatmap of average cost by (p1, p2)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pivot for heatmap
    pivot_mean = agg_df.pivot(index='p2', columns='p1', values='avg_mean_cost')
    pivot_std = agg_df.pivot(index='p2', columns='p1', values='avg_std_cost')

    # Plot 1: Mean cost heatmap
    ax1 = axes[0]
    sns.heatmap(pivot_mean, annot=True, fmt='.0f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Average Mean Cost ($)'},
                ax=ax1, vmin=pivot_mean.min().min(), vmax=pivot_mean.max().max())
    ax1.set_title('Average Mean Cost by (p1, p2)\nAveraged over 100 d1 samples',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('First Production (p1, batches)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Second Production (p2, batches)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    # Find and mark optimal
    min_val = pivot_mean.min().min()
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            if abs(pivot_mean.iloc[i, j] - min_val) < 1:
                ax1.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    # Plot 2: Std cost heatmap
    ax2 = axes[1]
    sns.heatmap(pivot_std, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Average Std Cost ($)'},
                ax=ax2, vmin=0)
    ax2.set_title('Average Cost Std Dev by (p1, p2)\nAveraged over 100 d1 samples',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('First Production (p1, batches)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Second Production (p2, batches)', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    plt.tight_layout()
    output_path = output_dir / "heatmap_p1_p2_cost.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap: {output_path}")
    plt.close()


def plot_p2_given_p1(agg_df, output_dir):
    """Plot cost vs p2 for different p1 values."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Mean cost
    ax1 = axes[0]
    for p1 in range(1, 11):
        data = agg_df[agg_df['p1'] == p1]
        ax1.plot(data['p2'], data['avg_mean_cost'], '-o', label=f'p1={p1}',
                markersize=6, linewidth=2)

    ax1.set_xlabel('Second Production (p2, batches)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Mean Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Impact of Second Production Decision\nfor Different First Production Levels',
                  fontsize=14, fontweight='bold')
    ax1.legend(ncol=5, loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Std cost
    ax2 = axes[1]
    for p1 in range(1, 11):
        data = agg_df[agg_df['p1'] == p1]
        ax2.plot(data['p2'], data['avg_std_cost'], '-o', label=f'p1={p1}',
                markersize=6, linewidth=2)

    ax2.set_xlabel('Second Production (p2, batches)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Cost Std Dev ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Cost Uncertainty by Second Production Decision\nfor Different First Production Levels',
                  fontsize=14, fontweight='bold')
    ax2.legend(ncol=5, loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "p2_given_p1_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved p2 analysis: {output_path}")
    plt.close()


def plot_optimal_p2_given_p1(agg_df, output_dir):
    """For each p1, find optimal p2 and plot."""
    optimal_p2 = []
    optimal_costs = []

    for p1 in range(1, 11):
        data = agg_df[agg_df['p1'] == p1]
        optimal_idx = data['avg_mean_cost'].idxmin()
        optimal_row = data.loc[optimal_idx]
        optimal_p2.append(optimal_row['p2'])
        optimal_costs.append(optimal_row['avg_mean_cost'])

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Optimal p2 vs p1
    ax1 = axes[0]
    ax1.plot(range(1, 11), optimal_p2, '-o', linewidth=3, markersize=10, color='steelblue')
    ax1.set_xlabel('First Production (p1, batches)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Optimal Second Production (p2, batches)', fontsize=12, fontweight='bold')
    ax1.set_title('Optimal Second Production Decision Given First Production',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 11))
    ax1.set_yticks(range(1, 11))

    # Add values on points
    for p1, p2 in enumerate(optimal_p2, 1):
        ax1.annotate(f'{int(p2)}', xy=(p1, p2), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontweight='bold')

    # Plot 2: Optimal cost vs p1
    ax2 = axes[1]
    ax2.plot(range(1, 11), optimal_costs, '-o', linewidth=3, markersize=10, color='darkgreen')
    ax2.set_xlabel('First Production (p1, batches)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Minimum Average Cost ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Minimum Achievable Cost by First Production\n(with optimal second production)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 11))

    # Mark global optimal
    global_optimal_idx = np.argmin(optimal_costs)
    ax2.scatter([global_optimal_idx + 1], [optimal_costs[global_optimal_idx]],
               s=300, color='red', marker='*', zorder=5,
               label=f'Global optimal: p1={global_optimal_idx+1}, p2={int(optimal_p2[global_optimal_idx])}')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "optimal_p2_given_p1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved optimal p2 analysis: {output_path}")
    plt.close()


def plot_d1_impact(df, output_dir, p1_value=2, p2_value=2):
    """Show how d1 realization affects cost for fixed (p1, p2)."""
    data = df[(df['p1'] == p1_value) & (df['p2'] == p2_value)].copy()
    data = data.sort_values('d1')

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Cost vs d1
    ax1 = axes[0]
    ax1.scatter(data['d1'], data['mean_cost'], alpha=0.6, s=30)
    ax1.plot(data['d1'].rolling(10).mean(), data['mean_cost'].rolling(10).mean(),
            'r-', linewidth=2, label='Moving average (10 samples)')
    ax1.set_xlabel('Realized Demand on Day 1 (d1)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Impact of Day-1 Demand Realization on Cost\n(p1={p1_value}, p2={p2_value})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Uncertainty vs d1
    ax2 = axes[1]
    ax2.scatter(data['d1'], data['cv_pct'], alpha=0.6, s=30, color='orange')
    ax2.plot(data['d1'].rolling(10).mean(), data['cv_pct'].rolling(10).mean(),
            'r-', linewidth=2, label='Moving average (10 samples)')
    ax2.set_xlabel('Realized Demand on Day 1 (d1)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Cost Uncertainty vs Day-1 Demand\n(p1={p1_value}, p2={p2_value})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"d1_impact_p1_{p1_value}_p2_{p2_value}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved d1 impact analysis: {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    print("=" * 80)
    print("TWO-STAGE SENSITIVITY ANALYSIS VISUALIZATION")
    print("=" * 80)

    # Load results
    print("\nLoading results...")
    df, agg_df = load_results()

    print(f"✓ Loaded {len(df):,} rows from detailed results")
    if agg_df is not None:
        print(f"✓ Loaded {len(agg_df):,} rows from aggregated results")

    # Create output directory
    output_dir = Path(__file__).parent / "two_stage_plots"
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating visualizations in: {output_dir}")
    print("-" * 80)

    # Generate plots
    if agg_df is not None:
        plot_p1_p2_heatmap(agg_df, output_dir)
        plot_p2_given_p1(agg_df, output_dir)
        plot_optimal_p2_given_p1(agg_df, output_dir)

    # Example d1 impact plots for a few (p1, p2) combinations
    for p1, p2 in [(2, 2), (2, 3), (3, 2)]:
        plot_d1_impact(df, output_dir, p1, p2)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated plots saved in: {output_dir}")
    print("\nGenerated visualizations:")
    print("  1. heatmap_p1_p2_cost.png - Cost heatmap by (p1, p2)")
    print("  2. p2_given_p1_analysis.png - How p2 affects cost for each p1")
    print("  3. optimal_p2_given_p1.png - Best p2 for each p1 value")
    print("  4. d1_impact_p1_*_p2_*.png - Impact of realized d1 on cost")
    print("=" * 80)


if __name__ == "__main__":
    main()
