#!/usr/bin/env python3
"""
Two-Stage Stochastic Sensitivity Analysis

Investigates the impact of BOTH first and second production decisions on total cost.

For each combination of:
- p1: First production decision (1-10 batches)
- d1: Realized demand on day 1 (100 samples)
- p2: Second production decision (1-10 batches)

Run 100 demand scenarios for days 2-10 and calculate:
- Mean cost across scenarios
- Standard deviation of cost
- Other statistics

Process:
1. Fix first production p1
2. Sample day-1 demand d1 (100 samples per p1)
3. For each (p1, d1) pair, try all p2 values (1-10)
4. For each (p1, d1, p2), run 100 demand scenarios (days 2-10)
5. Simulation logic for each scenario:
   - Day 1: Execute p1, observe d1 (from sample)
   - Day 2: Execute p2, observe d2 (from scenario)
   - Days 3-10: Use MIP optimization with mean forecasts
6. Save results to CSV (10 × 100 × 10 = 10,000 rows)

Total simulations: 10 × 100 × 10 × 100 = 1,000,000

Output: two_stage_sensitivity.csv with columns:
- p1, p1_units, d1_sample_idx, d1, p2, p2_units
- mean_cost, std_cost, cv_pct, min_cost, max_cost, median_cost
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import truncnorm
from tqdm import tqdm


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def generate_demand_sample(mu, std_ratio=0.3, random_state=None):
    """Generate a single demand sample from truncated normal."""
    sigma = std_ratio * mu
    if sigma > 0:
        a = -mu / sigma  # Lower bound in std units
        rng = np.random.RandomState(random_state)
        demand = truncnorm.rvs(a, np.inf, loc=mu, scale=sigma, random_state=rng)
    else:
        demand = mu
    return max(0, demand)


def build_scp_model_general(config, fixed_production_days, current_day, initial_state, future_demands):
    """Build SCP model for rolling horizon with custom demands."""
    H = config['planning_horizon']
    LT = config['lead_time']
    batch_size = config['batch_size']
    demands = np.array(future_demands)

    c_supply = config['costs']['supply_per_unit']
    c_inv = config['costs']['inventory_holding_per_unit_per_day']
    c_delay = config['costs']['delay_penalty_per_unit_per_day']
    c_final = config['costs']['final_unmet_demand_penalty_per_unit']

    remaining_days = H - current_day + 1
    num_vars = 3 * remaining_days

    n_idx = lambda t: t - current_day
    I_idx = lambda t: remaining_days + (t - current_day)
    B_idx = lambda t: 2*remaining_days + (t - current_day)

    c = np.zeros(num_vars)
    for t in range(current_day, H + 1):
        c[n_idx(t)] = c_supply * batch_size
        c[I_idx(t)] = c_inv
        c[B_idx(t)] = c_delay if t < H else c_delay + c_final

    integrality = np.zeros(num_vars)
    for t in range(current_day, H + 1):
        integrality[n_idx(t)] = 1

    max_batches = int(np.ceil(np.sum(demands) / batch_size)) + 10
    lb = np.zeros(num_vars)
    ub = np.array([max_batches]*remaining_days + [np.inf]*remaining_days + [np.inf]*remaining_days)

    for day, batches in fixed_production_days.items():
        if day >= current_day:
            lb[n_idx(day)] = batches
            ub[n_idx(day)] = batches

    bounds = Bounds(lb=lb, ub=ub)

    constraints = []
    for t in range(current_day, H + 1):
        A = np.zeros(num_vars)
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

        rhs = -demands[t-1]
        if t == current_day:
            rhs -= initial_state['inventory']
            rhs += initial_state['backlog']

        constraints.append(LinearConstraint(A, rhs, rhs))

    return c, constraints, bounds, integrality


def simulate_day(inventory, backlog, production_arriving, demand):
    """Simulate one day of operations."""
    net = inventory - backlog + production_arriving - demand
    if net >= 0:
        return net, 0
    else:
        return 0, -net


def two_stage_simulation(config, p1, d1, p2, demands_2_to_10):
    """
    Simulate supply chain with fixed p1, p2 and realized demands.

    Parameters:
    -----------
    config : dict
        Configuration
    p1 : int
        First production decision (batches)
    d1 : float
        Realized demand on day 1
    p2 : int
        Second production decision (batches)
    demands_2_to_10 : array
        Realized demands for days 2-10 (length 9)

    Returns:
    --------
    Total cost for this scenario
    """
    H = config['planning_horizon']
    LT = config['lead_time']
    batch_size = config['batch_size']
    mean_demands = np.array(config['demands'])

    c_supply = config['costs']['supply_per_unit']
    c_inv = config['costs']['inventory_holding_per_unit_per_day']
    c_delay = config['costs']['delay_penalty_per_unit_per_day']
    c_final = config['costs']['final_unmet_demand_penalty_per_unit']

    # Combine all realized demands
    all_demands = np.concatenate([[d1], demands_2_to_10])

    # State
    inventory = config['initial_inventory']
    backlog = 0

    # Production decisions
    production_decisions = np.zeros(H)
    production_decisions[0] = p1
    production_decisions[1] = p2

    # Total costs
    total_supply_cost = 0
    total_inv_cost = 0
    total_delay_cost = 0

    # Day 1: Execute p1, observe d1
    prod_arriving_1 = 0  # No production arrives on day 1 (LT=2)
    inventory, backlog = simulate_day(inventory, backlog, prod_arriving_1, d1)

    total_supply_cost += c_supply * batch_size * p1
    total_inv_cost += c_inv * inventory
    total_delay_cost += c_delay * backlog

    # Day 2: Execute p2, observe d2
    prod_arriving_2 = 0  # No production arrives on day 2 (LT=2, production from day 0 doesn't exist)
    inventory, backlog = simulate_day(inventory, backlog, prod_arriving_2, demands_2_to_10[0])

    total_supply_cost += c_supply * batch_size * p2
    total_inv_cost += c_inv * inventory
    total_delay_cost += c_delay * backlog

    # Days 3-10: Use MIP optimization
    for day in range(3, H + 1):
        # Production arriving today
        prod_arriving = 0
        if day > LT:
            prod_arriving = batch_size * production_decisions[day - LT - 1]

        # Realized demand
        demand = all_demands[day - 1]

        # Simulate the day
        inventory, backlog = simulate_day(inventory, backlog, prod_arriving, demand)

        # Costs
        total_supply_cost += c_supply * batch_size * production_decisions[day - 1]
        total_inv_cost += c_inv * inventory
        total_delay_cost += c_delay * backlog

        # Re-optimize for future days
        if day < H:
            # At start of day (day+1), use mean forecast for days (day+1) to H
            opt_demands = mean_demands.copy()

            # Fix production for days 1 to day (already executed)
            fixed_vars = {d: production_decisions[d-1] for d in range(1, day + 1)}

            c, constraints, bounds, integrality = build_scp_model_general(
                config, fixed_vars, day + 1,
                {'inventory': inventory, 'backlog': backlog},
                opt_demands
            )

            result = milp(c=c, constraints=constraints, bounds=bounds,
                         integrality=integrality, options={'disp': False})

            if result.success:
                # Update future production decisions
                for t in range(day + 1, H + 1):
                    idx = t - (day + 1)
                    production_decisions[t - 1] = int(result.x[idx])

    # Add final penalty
    total_delay_cost += c_final * backlog

    return total_supply_cost + total_inv_cost + total_delay_cost


def run_two_stage_analysis(config, num_d1_samples=100, num_scenarios=100):
    """
    Run complete two-stage sensitivity analysis.

    For each p1 (1-10):
        Sample d1 100 times
        For each (p1, d1):
            For each p2 (1-10):
                Run 100 scenarios with different demands[2-10]
                Calculate statistics

    Returns DataFrame with 10,000 rows (10 × 100 × 10)
    """
    mean_demands = np.array(config['demands'])
    batch_size = config['batch_size']

    results = []
    total_iterations = 10 * num_d1_samples * 10

    print("=" * 80)
    print("TWO-STAGE STOCHASTIC SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  First production (p1): 1-10 batches")
    print(f"  Day-1 demand samples: {num_d1_samples}")
    print(f"  Second production (p2): 1-10 batches")
    print(f"  Scenarios per (p1,d1,p2): {num_scenarios}")
    print(f"\nTotal combinations: {total_iterations}")
    print(f"Total simulations: {total_iterations * num_scenarios:,}")
    print("=" * 80)

    with tqdm(total=total_iterations, desc="Progress") as pbar:
        for p1 in range(1, 11):
            # Sample day-1 demand 100 times
            for d1_sample_idx in range(num_d1_samples):
                # Generate realized demand for day 1
                seed_d1 = 1000 * p1 + d1_sample_idx
                d1 = generate_demand_sample(mean_demands[0], std_ratio=0.3, random_state=seed_d1)

                for p2 in range(1, 11):
                    # For this (p1, d1, p2), run num_scenarios demand scenarios
                    scenario_costs = []

                    for scenario_idx in range(num_scenarios):
                        # Generate demands for days 2-10
                        seed_scenario = 100000 * p1 + 1000 * d1_sample_idx + 10 * p2 + scenario_idx
                        rng = np.random.RandomState(seed_scenario)

                        demands_2_to_10 = []
                        for day_idx in range(1, 10):  # Days 2-10
                            mu = mean_demands[day_idx]
                            sigma = 0.3 * mu
                            if sigma > 0:
                                a = -mu / sigma
                                demand = truncnorm.rvs(a, np.inf, loc=mu, scale=sigma, random_state=rng)
                            else:
                                demand = mu
                            demands_2_to_10.append(max(0, demand))

                        # Run simulation
                        cost = two_stage_simulation(config, p1, d1, p2, demands_2_to_10)
                        scenario_costs.append(cost)

                    # Calculate statistics
                    scenario_costs = np.array(scenario_costs)
                    mean_cost = np.mean(scenario_costs)
                    std_cost = np.std(scenario_costs)

                    results.append({
                        'p1': p1,
                        'p1_units': p1 * batch_size,
                        'd1_sample_idx': d1_sample_idx,
                        'd1': d1,
                        'p2': p2,
                        'p2_units': p2 * batch_size,
                        'mean_cost': mean_cost,
                        'std_cost': std_cost,
                        'cv_pct': (std_cost / mean_cost * 100) if mean_cost > 0 else 0,
                        'min_cost': np.min(scenario_costs),
                        'max_cost': np.max(scenario_costs),
                        'median_cost': np.median(scenario_costs),
                        'q25_cost': np.percentile(scenario_costs, 25),
                        'q75_cost': np.percentile(scenario_costs, 75)
                    })

                    pbar.update(1)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)

    print("\nStarting two-stage sensitivity analysis...")
    print("This will take a while (approximately 5-10 minutes)...")
    print()

    # Run analysis
    df = run_two_stage_analysis(config, num_d1_samples=100, num_scenarios=100)

    # Save to CSV
    output_path = Path(__file__).parent / "two_stage_sensitivity.csv"
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"CSV size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nOverall cost statistics:")
    print(f"  Mean of means: ${df['mean_cost'].mean():,.2f}")
    print(f"  Std of means: ${df['mean_cost'].std():,.2f}")
    print(f"  Min mean cost: ${df['mean_cost'].min():,.2f}")
    print(f"  Max mean cost: ${df['mean_cost'].max():,.2f}")

    # Find optimal combination
    optimal_idx = df['mean_cost'].idxmin()
    optimal_row = df.loc[optimal_idx]

    print(f"\nOptimal combination:")
    print(f"  p1: {optimal_row['p1']:.0f} batches ({optimal_row['p1_units']:.0f} units)")
    print(f"  d1 sample: #{optimal_row['d1_sample_idx']:.0f} (d1={optimal_row['d1']:.1f})")
    print(f"  p2: {optimal_row['p2']:.0f} batches ({optimal_row['p2_units']:.0f} units)")
    print(f"  Mean cost: ${optimal_row['mean_cost']:,.2f}")
    print(f"  Std cost: ${optimal_row['std_cost']:,.2f}")
    print(f"  CV: {optimal_row['cv_pct']:.1f}%")

    # Aggregate by (p1, p2) - average over d1 samples
    print("\n" + "=" * 80)
    print("AGGREGATED BY (p1, p2) - Average over d1 samples")
    print("=" * 80)

    agg_by_p1_p2 = df.groupby(['p1', 'p2']).agg({
        'mean_cost': 'mean',
        'std_cost': 'mean'
    }).reset_index()

    agg_by_p1_p2.columns = ['p1', 'p2', 'avg_mean_cost', 'avg_std_cost']

    # Save aggregated results
    agg_output_path = Path(__file__).parent / "two_stage_aggregated_p1_p2.csv"
    agg_by_p1_p2.to_csv(agg_output_path, index=False)

    print(f"\nAggregated results saved to: {agg_output_path}")
    print(f"Shape: {agg_by_p1_p2.shape}")

    # Find optimal (p1, p2)
    optimal_agg_idx = agg_by_p1_p2['avg_mean_cost'].idxmin()
    optimal_agg = agg_by_p1_p2.loc[optimal_agg_idx]

    print(f"\nOptimal (p1, p2) combination:")
    print(f"  p1: {optimal_agg['p1']:.0f} batches")
    print(f"  p2: {optimal_agg['p2']:.0f} batches")
    print(f"  Average mean cost: ${optimal_agg['avg_mean_cost']:,.2f}")
    print(f"  Average std cost: ${optimal_agg['avg_std_cost']:,.2f}")

    print("\n" + "=" * 80)
    print("✓ Two-stage sensitivity analysis complete!")
    print("=" * 80)
    print("\nTo analyze results:")
    print("  1. Load two_stage_sensitivity.csv in your preferred tool")
    print("  2. Use two_stage_aggregated_p1_p2.csv for (p1,p2) heatmaps")
    print("  3. Import into Apache Superset for interactive visualization")
    print("=" * 80)
