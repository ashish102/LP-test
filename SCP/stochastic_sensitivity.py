#!/usr/bin/env python3
"""
Stochastic Sensitivity Analysis: First Production Decision with Uncertain Demand

This script extends the sensitivity analysis to handle stochastic demand:
- Demand values in config are treated as MEANS
- Actual demand is normally distributed: N(μ, σ²) where σ = 0.3 × μ
- Demand is truncated at 0 (no negative demand)
- For each first production decision (1-10 batches):
  * Generate 100 random demand trajectories
  * Simulate rolling horizon optimization:
    - At START of each day t, solve MIP:
      * State reflects realized demand from days 1 to t-1 (past, known)
      * Use MEAN forecast for days t to H (today and future, UNKNOWN)
      * CRITICAL: Solver does NOT know today's (day t) realized demand!
    - Execute production decision for day t
    - Observe realized demand for day t (after MIP solve)
    - Calculate actual costs based on realized outcomes
    - Update state and proceed to next day
  * Average costs across 100 scenarios
- Plot mean cost with confidence intervals

This represents realistic Model Predictive Control (MPC) under uncertainty.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import truncnorm


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def generate_demand_scenario(mean_demands, std_ratio=0.3, seed=None):
    """
    Generate a random demand scenario.

    Parameters:
    -----------
    mean_demands : array-like
        Mean demand for each period
    std_ratio : float
        Standard deviation as ratio of mean (default 0.3 = 30%)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    Array of realized demands (truncated normal, non-negative)
    """
    if seed is not None:
        np.random.seed(seed)

    realized_demands = []
    for mu in mean_demands:
        sigma = std_ratio * mu

        # Truncated normal: truncate at 0 (no negative demand)
        # truncnorm parameters: a, b are bounds in standard deviations from mean
        a = -mu / sigma if sigma > 0 else 0  # lower bound in std units
        b = np.inf  # no upper bound

        if sigma > 0:
            demand = truncnorm.rvs(a, b, loc=mu, scale=sigma)
        else:
            demand = mu

        realized_demands.append(max(0, demand))

    return np.array(realized_demands)


def build_scp_model_general(config, fixed_production_days, current_day, initial_state, future_demands):
    """
    Build SCP model for rolling horizon with custom demands.

    Parameters:
    -----------
    config : dict
        Configuration parameters
    fixed_production_days : dict
        Fixed production decisions {day: batches}
    current_day : int
        Current day (1-indexed)
    initial_state : dict
        {'inventory': float, 'backlog': float}
    future_demands : array-like
        Demand forecast for remaining periods
    """
    H = config['planning_horizon']
    LT = config['lead_time']
    batch_size = config['batch_size']

    # Use provided future demands
    demands = np.array(future_demands)

    # Costs
    c_supply = config['costs']['supply_per_unit']
    c_inv = config['costs']['inventory_holding_per_unit_per_day']
    c_delay = config['costs']['delay_penalty_per_unit_per_day']
    c_final = config['costs']['final_unmet_demand_penalty_per_unit']

    # Variables: n[current_day..H], I[current_day..H], B[current_day..H]
    remaining_days = H - current_day + 1
    num_vars = 3 * remaining_days

    # Variable indices (0-indexed within remaining periods)
    n_idx = lambda t: t - current_day
    I_idx = lambda t: remaining_days + (t - current_day)
    B_idx = lambda t: 2*remaining_days + (t - current_day)

    # Objective function
    c = np.zeros(num_vars)

    for t in range(current_day, H + 1):
        c[n_idx(t)] = c_supply * batch_size
        c[I_idx(t)] = c_inv
        if t < H:
            c[B_idx(t)] = c_delay
        else:
            c[B_idx(t)] = c_delay + c_final

    # Integrality
    integrality = np.zeros(num_vars)
    for t in range(current_day, H + 1):
        integrality[n_idx(t)] = 1

    # Bounds
    max_batches = int(np.ceil(np.sum(demands) / batch_size)) + 10
    lb = np.zeros(num_vars)
    ub = np.array([max_batches]*remaining_days + [np.inf]*remaining_days + [np.inf]*remaining_days)

    # Fix some production variables
    for day, batches in fixed_production_days.items():
        if day >= current_day:
            lb[n_idx(day)] = batches
            ub[n_idx(day)] = batches

    bounds = Bounds(lb=lb, ub=ub)

    # Constraints
    constraints = []

    for t in range(current_day, H + 1):
        A = np.zeros(num_vars)

        # I[t] - I[t-1] - batch_size*n[t-LT] + B[t-1] - B[t] = -D[t]
        A[I_idx(t)] = 1

        if t > current_day:
            A[I_idx(t-1)] = -1

        if t > LT:
            arrival_day = t - LT
            if arrival_day >= current_day:
                A[n_idx(arrival_day)] = -batch_size

        if t > current_day:
            A[B_idx(t-1)] = 1

        A[B_idx(t)] = -1

        # RHS
        rhs = -demands[t-1]

        if t == current_day:
            rhs -= initial_state['inventory']
            rhs += initial_state['backlog']

        constraints.append(LinearConstraint(A, rhs, rhs))

    return c, constraints, bounds, integrality, remaining_days


def simulate_day(inventory, backlog, production_arriving, demand):
    """
    Simulate one day of operations with realized demand.

    Returns:
    --------
    (new_inventory, new_backlog, inventory_cost, backlog_cost)
    """
    # Net position = inventory - backlog + production - demand
    net = inventory - backlog + production_arriving - demand

    if net >= 0:
        new_inventory = net
        new_backlog = 0
    else:
        new_inventory = 0
        new_backlog = -net

    return new_inventory, new_backlog


def rolling_horizon_stochastic_simulation(config, first_prod, realized_demands, verbose=False):
    """
    Simulate rolling horizon optimization with stochastic realized demand.

    At START of each day t:
    - State reflects realized demands from days 1 to t-1 (past, known)
    - Solve MIP using MEAN forecast for days t to H (today and future, unknown)
    - Execute production decision for day t
    - Observe REALIZED demand for day t (not known during MIP solve!)
    - Update state and move to next day

    Key: When solving MIP at start of day t, we do NOT know day t's demand yet!

    Returns:
    --------
    Total cost for this scenario
    """
    H = config['planning_horizon']
    LT = config['lead_time']
    batch_size = config['batch_size']
    mean_demands = np.array(config['demands'])

    # Costs
    c_supply = config['costs']['supply_per_unit']
    c_inv = config['costs']['inventory_holding_per_unit_per_day']
    c_delay = config['costs']['delay_penalty_per_unit_per_day']
    c_final = config['costs']['final_unmet_demand_penalty_per_unit']

    # Track actual state
    actual_inventory = config['initial_inventory']
    actual_backlog = 0

    # Track all production decisions
    production_decisions = np.zeros(H)

    # Track costs
    total_supply_cost = 0
    total_inv_cost = 0
    total_delay_cost = 0

    # Day 1 START: Optimize with first production fixed
    # At start of day 1, we know NOTHING about realized demands yet
    # Use MEAN forecast for ALL days 1-H (including today, day 1!)
    fixed_vars = {1: first_prod}
    opt_demands = mean_demands.copy()  # Mean forecast for days 1-H

    c, constraints, bounds, integrality, _ = build_scp_model_general(
        config, fixed_vars, 1,
        {'inventory': actual_inventory, 'backlog': actual_backlog},
        opt_demands
    )

    result = milp(c=c, constraints=constraints, bounds=bounds,
                 integrality=integrality, options={'disp': False})

    if not result.success:
        if verbose:
            print(f"  Day 1 optimization failed: {result.message}")
        return None

    # Extract production decisions for all days
    for t in range(H):
        production_decisions[t] = int(result.x[t])

    # Now simulate forward day by day with realized demands
    for day in range(1, H + 1):
        # Production arriving today (from day-LT)
        prod_arriving = 0
        if day > LT:
            prod_arriving = batch_size * production_decisions[day - LT - 1]

        # Realized demand for today
        demand = realized_demands[day - 1]

        # Simulate the day
        new_inventory, new_backlog = simulate_day(
            actual_inventory, actual_backlog, prod_arriving, demand
        )

        # Calculate costs for this day
        supply_cost_today = c_supply * batch_size * production_decisions[day - 1]
        inv_cost_today = c_inv * new_inventory
        delay_cost_today = c_delay * new_backlog

        total_supply_cost += supply_cost_today
        total_inv_cost += inv_cost_today
        total_delay_cost += delay_cost_today

        # Update state
        actual_inventory = new_inventory
        actual_backlog = new_backlog

        # Re-optimize for next day if not at horizon end
        if day < H:
            # CRITICAL: At START of day (day+1), we don't know today's demand yet!
            # - Known: Realized demands from days 1 to day (already in state)
            # - Unknown: Demands for days (day+1) to H - use MEAN forecast
            # - Today's demand (day+1) is NOT known when solving MIP!
            opt_demands = mean_demands.copy()

            # Fix production decisions for days we've already executed (1 to day)
            # Do NOT fix future days - re-optimize them with updated state
            fixed_vars = {d: production_decisions[d-1] for d in range(1, day + 1)}

            c, constraints, bounds, integrality, _ = build_scp_model_general(
                config, fixed_vars, day + 1,
                {'inventory': actual_inventory, 'backlog': actual_backlog},
                opt_demands
            )

            result = milp(c=c, constraints=constraints, bounds=bounds,
                         integrality=integrality, options={'disp': False})

            if result.success:
                # Update production decisions for future days
                for t in range(day + 1, H + 1):
                    idx = t - (day + 1)
                    production_decisions[t - 1] = int(result.x[idx])

    # Add final penalty if there's unmet demand
    final_cost = c_final * actual_backlog
    total_delay_cost += final_cost

    total_cost = total_supply_cost + total_inv_cost + total_delay_cost

    return total_cost


def stochastic_sensitivity_analysis(config, first_prod_range, num_scenarios=100):
    """
    Perform stochastic sensitivity analysis.

    For each first production value:
    - Generate num_scenarios demand scenarios
    - Run rolling horizon simulation for each
    - Calculate mean and std of costs
    """
    results = []

    print("=" * 80)
    print("STOCHASTIC SENSITIVITY ANALYSIS: First Production Decision")
    print("=" * 80)
    print(f"\nDemand Model: Normal(μ, σ²) where σ = 0.3×μ, truncated at 0")
    print(f"Number of scenarios: {num_scenarios}")
    print(f"Analyzing first production: {min(first_prod_range)} to {max(first_prod_range)} batches")
    print("-" * 80)

    mean_demands = np.array(config['demands'])

    for first_prod in first_prod_range:
        print(f"\nFirst Production = {first_prod} batches ({first_prod * config['batch_size']} units)")

        scenario_costs = []

        for scenario in range(num_scenarios):
            # Generate random demand trajectory
            realized_demands = generate_demand_scenario(mean_demands, std_ratio=0.3, seed=scenario * 100 + first_prod)

            # Run simulation
            cost = rolling_horizon_stochastic_simulation(config, first_prod, realized_demands)

            if cost is not None:
                scenario_costs.append(cost)

            if (scenario + 1) % 20 == 0:
                print(f"  Completed {scenario + 1}/{num_scenarios} scenarios...")

        if scenario_costs:
            mean_cost = np.mean(scenario_costs)
            std_cost = np.std(scenario_costs)
            min_cost = np.min(scenario_costs)
            max_cost = np.max(scenario_costs)

            results.append({
                'first_production': first_prod,
                'first_production_units': first_prod * config['batch_size'],
                'mean_cost': mean_cost,
                'std_cost': std_cost,
                'min_cost': min_cost,
                'max_cost': max_cost,
                'costs': scenario_costs,
                'num_scenarios': len(scenario_costs),
                'success': True
            })

            print(f"  Mean Cost: ${mean_cost:.2f} ± ${std_cost:.2f}")
            print(f"  Range: [${min_cost:.2f}, ${max_cost:.2f}]")
        else:
            results.append({
                'first_production': first_prod,
                'success': False
            })
            print(f"  FAILED: No successful scenarios")

    print("\n" + "=" * 80)
    return results


def plot_stochastic_sensitivity(results, output_path='stochastic_sensitivity.png'):
    """
    Create visualization comparing deterministic vs stochastic sensitivity.
    """
    successful = [r for r in results if r['success']]

    first_prods = [r['first_production'] for r in successful]
    mean_costs = [r['mean_cost'] for r in successful]
    std_costs = [r['std_cost'] for r in successful]
    min_costs = [r['min_cost'] for r in successful]
    max_costs = [r['max_cost'] for r in successful]

    # Find optimal
    optimal_idx = np.argmin(mean_costs)
    optimal_prod = first_prods[optimal_idx]
    optimal_mean = mean_costs[optimal_idx]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Mean cost with confidence bands
    ax1 = axes[0]

    # Plot mean line
    ax1.plot(first_prods, mean_costs, 'b-o', linewidth=2, markersize=8, label='Mean Cost', zorder=3)

    # Add ±1 std deviation band
    lower_band = np.array(mean_costs) - np.array(std_costs)
    upper_band = np.array(mean_costs) + np.array(std_costs)
    ax1.fill_between(first_prods, lower_band, upper_band, alpha=0.3, label='±1 Std Dev', zorder=1)

    # Add min/max range as lighter band
    ax1.fill_between(first_prods, min_costs, max_costs, alpha=0.15, color='gray',
                     label='Min-Max Range', zorder=0)

    # Highlight optimal
    ax1.axvline(x=optimal_prod, color='r', linestyle='--', alpha=0.7,
                label=f'Optimal ({optimal_prod} batches)')
    ax1.scatter([optimal_prod], [optimal_mean], color='red', s=200, zorder=5,
                marker='*', label=f'Minimum: ${optimal_mean:.2f}')

    ax1.set_xlabel('First Production Decision (batches)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Stochastic Sensitivity Analysis: First Production Decision Impact\n' +
                  'Demand ~ N(μ, σ²) where σ = 0.3μ, 100 scenarios per point',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)

    # Add annotations
    for i, (prod, cost, std) in enumerate(zip(first_prods, mean_costs, std_costs)):
        if prod == optimal_prod:
            ax1.annotate(f'${cost:.0f}\n±${std:.0f}',
                        xy=(prod, cost), xytext=(0, 20),
                        textcoords='offset points', ha='center',
                        fontweight='bold', fontsize=8, color='red')
        elif i % 2 == 0:
            ax1.annotate(f'${cost:.0f}', xy=(prod, cost), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=7)

    # Plot 2: Coefficient of Variation (relative uncertainty)
    ax2 = axes[1]

    cv = np.array(std_costs) / np.array(mean_costs) * 100  # CV as percentage

    ax2.bar(first_prods, cv, width=0.6, alpha=0.7, color='steelblue')
    ax2.axvline(x=optimal_prod, color='r', linestyle='--', alpha=0.7, linewidth=2)

    ax2.set_xlabel('First Production Decision (batches)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Cost Uncertainty by First Production Decision',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for prod, cv_val in zip(first_prods, cv):
        ax2.text(prod, cv_val + 0.1, f'{cv_val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    return fig


def print_stochastic_summary(results):
    """Print summary of stochastic analysis."""
    successful = [r for r in results if r['success']]

    if not successful:
        print("No successful results to summarize.")
        return

    mean_costs = [r['mean_cost'] for r in successful]
    optimal_idx = np.argmin(mean_costs)
    optimal = successful[optimal_idx]

    print("\n" + "=" * 80)
    print("STOCHASTIC ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nOptimal First Production (Stochastic): {optimal['first_production']} batches "
          f"({optimal['first_production_units']} units)")
    print(f"Mean Cost: ${optimal['mean_cost']:.2f} ± ${optimal['std_cost']:.2f}")
    print(f"Cost Range: [${optimal['min_cost']:.2f}, ${optimal['max_cost']:.2f}]")
    print(f"Coefficient of Variation: {optimal['std_cost']/optimal['mean_cost']*100:.1f}%")

    print(f"\n\nComparison Across First Production Values:")
    print(f"{'First Prod':>12} {'Mean Cost':>12} {'Std Dev':>12} {'CV(%)':>8} {'vs Optimal':>12}")
    print("-" * 60)

    for r in successful:
        cv = r['std_cost'] / r['mean_cost'] * 100
        diff = r['mean_cost'] - optimal['mean_cost']
        print(f"{r['first_production']:>12} ${r['mean_cost']:>11.2f} "
              f"${r['std_cost']:>11.2f} {cv:>7.1f}% "
              f"+${diff:>10.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)

    # Run stochastic sensitivity analysis
    first_prod_range = range(1, 11)
    num_scenarios = 100

    results = stochastic_sensitivity_analysis(config, first_prod_range, num_scenarios)

    # Print summary
    print_stochastic_summary(results)

    # Create plot
    output_path = Path(__file__).parent / "stochastic_sensitivity.png"
    plot_stochastic_sensitivity(results, output_path)

    print("\n" + "=" * 80)
    print("✓ Stochastic sensitivity analysis complete!")
    print("=" * 80)
