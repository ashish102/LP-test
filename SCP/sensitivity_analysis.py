#!/usr/bin/env python3
"""
Sensitivity Analysis: First Production Decision Impact

This script analyzes how the total cost changes when we fix the first
production decision to different values (1 to 10 batches) and let the
optimizer choose the rest of the production plan.

This helps understand:
- How sensitive is the total cost to the first decision?
- What is the cost penalty of suboptimal first decisions?
- How "flat" is the objective function around the optimum?
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def build_scp_model_with_fixed_vars(config, fixed_production_days):
    """
    Build the full SCP model but with some production variables fixed.

    Parameters:
    -----------
    config : dict
        Configuration parameters
    fixed_production_days : dict
        Dictionary mapping day (1-indexed) to fixed production batch count

    Returns:
    --------
    Tuple of (c, constraints, bounds, integrality)
    """
    H = config['planning_horizon']
    LT = config['lead_time']
    batch_size = config['batch_size']
    demands = np.array(config['demands'])
    initial_inv = config['initial_inventory']

    # Costs
    c_supply = config['costs']['supply_per_unit']
    c_inv = config['costs']['inventory_holding_per_unit_per_day']
    c_delay = config['costs']['delay_penalty_per_unit_per_day']
    c_final = config['costs']['final_unmet_demand_penalty_per_unit']

    # Total variables: n[1..H], I[1..H], B[1..H]
    num_vars = 3 * H

    n_idx = lambda t: t - 1
    I_idx = lambda t: H + t - 1
    B_idx = lambda t: 2*H + t - 1

    # Objective function coefficients
    c = np.zeros(num_vars)

    # Production costs
    for t in range(1, H + 1):
        c[n_idx(t)] = c_supply * batch_size

    # Inventory holding costs
    for t in range(1, H + 1):
        c[I_idx(t)] = c_inv

    # Delay penalties
    for t in range(1, H + 1):
        if t < H:
            c[B_idx(t)] = c_delay
        else:
            c[B_idx(t)] = c_delay + c_final

    # Integrality constraints
    integrality = np.zeros(num_vars)
    for t in range(1, H + 1):
        integrality[n_idx(t)] = 1

    # Variable bounds
    max_batches = int(np.ceil(np.sum(demands) / batch_size)) + 5
    lb = np.zeros(num_vars)
    ub = np.array([max_batches]*H + [np.inf]*H + [np.inf]*H)

    # Fix production variables
    for day, batch_count in fixed_production_days.items():
        lb[n_idx(day)] = batch_count
        ub[n_idx(day)] = batch_count

    bounds = Bounds(lb=lb, ub=ub)

    # Constraints: Inventory balance for each day
    constraints = []

    for t in range(1, H + 1):
        A = np.zeros(num_vars)

        # I[t] coefficient: +1
        A[I_idx(t)] = 1

        # I[t-1] coefficient: -1 (if t > 1)
        if t > 1:
            A[I_idx(t-1)] = -1

        # P[t-LT] = batch_size * n[t-LT] coefficient: -batch_size
        if t > LT:
            A[n_idx(t - LT)] = -batch_size

        # B[t-1] coefficient: +1 (if t > 1)
        if t > 1:
            A[B_idx(t-1)] = 1

        # B[t] coefficient: -1
        A[B_idx(t)] = -1

        # RHS
        if t == 1:
            rhs = -demands[t-1] - initial_inv
        else:
            rhs = -demands[t-1]

        # Add equality constraint
        constraints.append(LinearConstraint(A, rhs, rhs))

    return c, constraints, bounds, integrality


def sensitivity_analysis(config, first_prod_range):
    """
    Perform sensitivity analysis by varying the first production decision.

    Parameters:
    -----------
    config : dict
        Configuration
    first_prod_range : list or range
        Range of first production values to test

    Returns:
    --------
    dict with results for each first production value
    """
    results = []

    print("=" * 80)
    print("SENSITIVITY ANALYSIS: First Production Decision Impact")
    print("=" * 80)
    print(f"\nAnalyzing cost impact for first production = {min(first_prod_range)} to {max(first_prod_range)} batches")
    print("-" * 80)

    H = config['planning_horizon']

    for first_prod in first_prod_range:
        # Fix first production to this value
        fixed_vars = {1: first_prod}

        # Build and solve model
        c, constraints, bounds, integrality = \
            build_scp_model_with_fixed_vars(config, fixed_vars)

        result = milp(c=c, constraints=constraints, bounds=bounds,
                     integrality=integrality, options={'disp': False})

        if result.success:
            production_plan = [int(result.x[t]) for t in range(H)]
            total_cost = result.fun

            # Calculate cost components
            batch_size = config['batch_size']
            c_supply = config['costs']['supply_per_unit']
            c_inv = config['costs']['inventory_holding_per_unit_per_day']
            c_delay = config['costs']['delay_penalty_per_unit_per_day']
            c_final = config['costs']['final_unmet_demand_penalty_per_unit']

            total_production = sum(production_plan) * batch_size
            inventory = [result.x[H + t] for t in range(H)]
            backlog = [result.x[2*H + t] for t in range(H)]

            supply_cost = c_supply * total_production
            inv_cost = c_inv * sum(inventory)
            delay_cost = c_delay * sum(backlog)
            final_cost = c_final * backlog[-1]

            results.append({
                'first_production': first_prod,
                'first_production_units': first_prod * batch_size,
                'total_cost': total_cost,
                'production_plan': production_plan,
                'supply_cost': supply_cost,
                'inventory_cost': inv_cost,
                'delay_cost': delay_cost,
                'final_cost': final_cost,
                'total_production': total_production,
                'final_inventory': inventory[-1],
                'final_backlog': backlog[-1],
                'success': True
            })

            print(f"First Prod: {first_prod:2d} batches ({first_prod*batch_size:3d} units) | "
                  f"Total Cost: ${total_cost:8.2f} | Plan: {production_plan}")

        else:
            results.append({
                'first_production': first_prod,
                'first_production_units': first_prod * batch_size,
                'total_cost': None,
                'success': False,
                'message': result.message
            })
            print(f"First Prod: {first_prod:2d} batches | FAILED: {result.message}")

    print("=" * 80)

    return results


def plot_sensitivity(results, config, output_path='first_production_sensitivity.png'):
    """
    Create visualization of sensitivity analysis results.
    """
    # Extract data for plotting
    first_prods = [r['first_production'] for r in results if r['success']]
    first_prod_units = [r['first_production_units'] for r in results if r['success']]
    total_costs = [r['total_cost'] for r in results if r['success']]
    supply_costs = [r['supply_cost'] for r in results if r['success']]
    inv_costs = [r['inventory_cost'] for r in results if r['success']]
    delay_costs = [r['delay_cost'] for r in results if r['success']]

    # Find optimal
    optimal_idx = np.argmin(total_costs)
    optimal_prod = first_prods[optimal_idx]
    optimal_cost = total_costs[optimal_idx]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Total Cost vs First Production
    ax1 = axes[0]
    ax1.plot(first_prods, total_costs, 'b-o', linewidth=2, markersize=8, label='Total Cost')
    ax1.axvline(x=optimal_prod, color='r', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_prod} batches)')
    ax1.axhline(y=optimal_cost, color='r', linestyle='--', alpha=0.3)
    ax1.scatter([optimal_prod], [optimal_cost], color='red', s=200, zorder=5, marker='*',
                label=f'Minimum: ${optimal_cost:.2f}')

    ax1.set_xlabel('First Production Decision (batches)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Sensitivity Analysis: Impact of First Production Decision on Total Cost',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Add cost values as annotations
    for i, (prod, cost) in enumerate(zip(first_prods, total_costs)):
        if prod == optimal_prod:
            ax1.annotate(f'${cost:.0f}', xy=(prod, cost), xytext=(0, 15),
                        textcoords='offset points', ha='center', fontweight='bold',
                        fontsize=9, color='red')
        elif i % 2 == 0:  # Annotate every other point to avoid clutter
            ax1.annotate(f'${cost:.0f}', xy=(prod, cost), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=8)

    # Plot 2: Cost Components Breakdown
    ax2 = axes[1]
    width = 0.6
    x_pos = np.array(first_prods)

    ax2.bar(x_pos, supply_costs, width, label='Supply Cost', alpha=0.8)
    ax2.bar(x_pos, inv_costs, width, bottom=supply_costs, label='Inventory Holding', alpha=0.8)
    ax2.bar(x_pos, delay_costs, width,
            bottom=np.array(supply_costs) + np.array(inv_costs),
            label='Delay Penalty', alpha=0.8)

    ax2.axvline(x=optimal_prod, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_xlabel('First Production Decision (batches)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Cost Components Breakdown', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    return fig


def print_summary(results):
    """Print summary statistics."""
    successful_results = [r for r in results if r['success']]

    if not successful_results:
        print("No successful optimizations to summarize.")
        return

    costs = [r['total_cost'] for r in successful_results]
    optimal_idx = np.argmin(costs)
    optimal = successful_results[optimal_idx]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nOptimal First Production Decision: {optimal['first_production']} batches "
          f"({optimal['first_production_units']} units)")
    print(f"Optimal Total Cost: ${optimal['total_cost']:.2f}")
    print(f"Optimal Production Plan: {optimal['production_plan']}")

    print(f"\n\nCost Range:")
    print(f"  Minimum Cost: ${min(costs):.2f} (first prod = {successful_results[np.argmin(costs)]['first_production']} batches)")
    print(f"  Maximum Cost: ${max(costs):.2f} (first prod = {successful_results[np.argmax(costs)]['first_production']} batches)")
    print(f"  Cost Range: ${max(costs) - min(costs):.2f}")
    print(f"  Percent Increase (min to max): {(max(costs) / min(costs) - 1) * 100:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)

    # Run sensitivity analysis for first production = 1 to 10 batches
    first_prod_range = range(1, 11)
    results = sensitivity_analysis(config, first_prod_range)

    # Print summary
    print_summary(results)

    # Create plot
    output_path = Path(__file__).parent / "first_production_sensitivity.png"
    plot_sensitivity(results, config, output_path)

    print("\n" + "=" * 80)
    print("✓ Sensitivity analysis complete!")
    print("=" * 80)
