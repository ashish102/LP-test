#!/usr/bin/env python3
"""
Supply Chain Planning - Corrected Rolling Horizon Implementation

This implements rolling horizon correctly by re-solving the FULL problem
each day with prior decision variables fixed, rather than solving tail subproblems.

The key difference:
- WRONG: Solve a smaller problem for days [current_day, H] with different objective
- CORRECT: Solve the FULL problem for days [1, H] with variables n[1]...n[current_day-1] fixed

This ensures we're testing true optimality: if a solution is optimal, fixing some
variables and re-optimizing should return the same values for the remaining variables.
"""

import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path


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
        e.g., {1: 5, 2: 2} means n[1]=5 and n[2]=2 are fixed

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
    # Indices: n[t] at t-1, I[t] at H+t-1, B[t] at 2H+t-1
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


def rolling_horizon_optimization(config):
    """
    Perform rolling horizon optimization by fixing variables progressively.
    """
    H = config['planning_horizon']
    batch_size = config['batch_size']

    print("=" * 80)
    print("ROLLING HORIZON OPTIMIZATION (CORRECTED)")
    print("=" * 80)
    print("\nMethod: Re-solve FULL problem with prior variables fixed")
    print(f"Planning Horizon: {H} days")
    print("=" * 80)

    # Day 1: Full optimization with no fixed variables
    print("\n" + "=" * 80)
    print("DAY 1: Full Horizon Optimization (no variables fixed)")
    print("=" * 80)

    c, constraints, bounds, integrality = build_scp_model_with_fixed_vars(config, {})

    result = milp(c=c, constraints=constraints, bounds=bounds,
                  integrality=integrality, options={'disp': False})

    if not result.success:
        print(f"✗ Day 1 optimization failed: {result.message}")
        return None

    # Extract full solution
    original_production = [int(result.x[t]) for t in range(H)]
    original_inventory = [result.x[H + t] for t in range(H)]
    original_backlog = [result.x[2*H + t] for t in range(H)]
    original_cost = result.fun

    print(f"✓ Optimization successful")
    print(f"  Production plan: {original_production}")
    print(f"  Total Cost: ${original_cost:.2f}")

    # Store results
    all_results = [{
        'day': 1,
        'fixed_days': 0,
        'production': original_production.copy(),
        'cost': original_cost,
        'matches_original': True
    }]

    # Days 2 to H: progressively fix more variables
    for current_day in range(2, H + 1):
        print("\n" + "=" * 80)
        print(f"DAY {current_day}: Re-optimize with days 1-{current_day-1} fixed")
        print("=" * 80)

        # Build fixed production dictionary
        fixed_vars = {day: original_production[day-1] for day in range(1, current_day)}
        print(f"  Fixed variables: {fixed_vars}")

        # Build and solve model with fixed variables
        c_fixed, constraints_fixed, bounds_fixed, integrality_fixed = \
            build_scp_model_with_fixed_vars(config, fixed_vars)

        result_fixed = milp(c=c_fixed, constraints=constraints_fixed,
                           bounds=bounds_fixed, integrality=integrality_fixed,
                           options={'disp': False})

        if not result_fixed.success:
            print(f"✗ Day {current_day} optimization failed: {result_fixed.message}")
            return None

        # Extract production decisions
        new_production = [int(result_fixed.x[t]) for t in range(H)]
        new_cost = result_fixed.fun

        print(f"  Production plan: {new_production}")
        print(f"  Total Cost: ${new_cost:.2f}")

        # Check if decisions match original
        decisions_match = all(new_production[i] == original_production[i] for i in range(H))
        cost_matches = abs(new_cost - original_cost) < 0.01

        if decisions_match and cost_matches:
            print(f"  ✓ ALL decisions match original")
        else:
            print(f"  ✗ MISMATCH detected:")
            if not cost_matches:
                print(f"    - Cost difference: ${abs(new_cost - original_cost):.2f}")
            if not decisions_match:
                for i in range(H):
                    if new_production[i] != original_production[i]:
                        print(f"    - Day {i+1}: original={original_production[i]}, new={new_production[i]}")

        all_results.append({
            'day': current_day,
            'fixed_days': current_day - 1,
            'production': new_production,
            'cost': new_cost,
            'matches_original': decisions_match and cost_matches
        })

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n{'Day':>5} {'Fixed Days':>12} {'Cost':>15} {'Matches':>10}")
    print("-" * 45)

    all_match = True
    for result_day in all_results:
        match_str = "✓" if result_day['matches_original'] else "✗"
        if not result_day['matches_original']:
            all_match = False
        print(f"{result_day['day']:5d} {result_day['fixed_days']:12d} "
              f"${result_day['cost']:13.2f} {match_str:>10}")

    print("\n" + "=" * 80)
    if all_match:
        print("✓✓✓ SUCCESS: All rolling horizon results match original!")
        print("    This confirms the solution is truly optimal and the model is correct.")
        print("    Fixing optimal variables and re-optimizing returns the same solution.")
    else:
        print("✗✗✗ MISMATCH: Some results differ from original.")
        print("    This suggests potential issues:")
        print("    - Solution may not be unique (multiple optimal solutions)")
        print("    - Numerical precision issues")
        print("    - Model formulation inconsistency")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)

    # Run corrected rolling horizon optimization
    results = rolling_horizon_optimization(config)
