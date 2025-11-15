#!/usr/bin/env python3
"""
Supply Chain Planning - Rolling Horizon Implementation

This script implements a rolling horizon optimization where the model is
re-solved each day with prior decisions fixed as input.

The expectation is that results should match the original full-horizon
optimization since we're just progressively fixing optimal decisions.

Process:
- Day 1: Optimize for entire horizon H, get production plan
- Day 2: Fix day 1 production, re-optimize days 2 to H with updated state
- Day 3: Fix days 1-2 production, re-optimize days 3 to H
- ...continue until horizon end

This tests the consistency and correctness of the optimization model.
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


def build_rolling_scp_model(config, current_day, fixed_production, initial_state):
    """
    Build the Supply Chain Planning MIP model for rolling horizon.

    Parameters:
    -----------
    config : dict
        Configuration parameters
    current_day : int
        Current day (1-indexed), the day we're optimizing FROM
    fixed_production : list
        Production decisions already made for days 1 to current_day-1
    initial_state : dict
        Initial inventory and backlog for the current day
        {'inventory': float, 'backlog': float}

    Returns:
    --------
    Tuple of (c, constraints, bounds, integrality, remaining_days)
    """
    H = config['planning_horizon']
    LT = config['lead_time']
    batch_size = config['batch_size']
    demands = np.array(config['demands'])

    # Costs
    c_supply = config['costs']['supply_per_unit']
    c_inv = config['costs']['inventory_holding_per_unit_per_day']
    c_delay = config['costs']['delay_penalty_per_unit_per_day']
    c_final = config['costs']['final_unmet_demand_penalty_per_unit']

    # Days remaining to optimize (from current_day to H)
    remaining_days = H - current_day + 1

    # Variables for remaining days: n[current_day..H], I[current_day..H], B[current_day..H]
    num_vars = 3 * remaining_days

    # Variable indices (relative to current_day)
    n_idx = lambda t: t - current_day  # n[t] at index t-current_day
    I_idx = lambda t: remaining_days + (t - current_day)
    B_idx = lambda t: 2*remaining_days + (t - current_day)

    # Objective function coefficients
    c = np.zeros(num_vars)

    # Production costs
    for t in range(current_day, H + 1):
        c[n_idx(t)] = c_supply * batch_size

    # Inventory holding costs
    for t in range(current_day, H + 1):
        c[I_idx(t)] = c_inv

    # Delay penalties
    for t in range(current_day, H + 1):
        if t < H:
            c[B_idx(t)] = c_delay
        else:
            c[B_idx(t)] = c_delay + c_final

    # Integrality constraints
    integrality = np.zeros(num_vars)
    for t in range(current_day, H + 1):
        integrality[n_idx(t)] = 1

    # Variable bounds
    max_batches = int(np.ceil(np.sum(demands) / batch_size)) + 5
    bounds = Bounds(
        lb=np.zeros(num_vars),
        ub=np.array([max_batches]*remaining_days + [np.inf]*remaining_days + [np.inf]*remaining_days)
    )

    # Constraints
    constraints = []

    # Inventory balance constraints for each day from current_day to H
    for t in range(current_day, H + 1):
        A = np.zeros(num_vars)

        # I[t] coefficient: +1
        A[I_idx(t)] = 1

        # I[t-1] coefficient: -1 (if t > current_day, otherwise use initial state)
        if t > current_day:
            A[I_idx(t-1)] = -1

        # P[t-LT] coefficient: -batch_size (production arriving)
        # Need to check if t-LT is in our decision window or already fixed
        if t > LT:
            arrival_day = t - LT
            if arrival_day >= current_day:
                # This production is in our decision window
                A[n_idx(arrival_day)] = -batch_size

        # B[t-1] coefficient: +1 (if t > current_day, otherwise use initial state)
        if t > current_day:
            A[B_idx(t-1)] = 1

        # B[t] coefficient: -1
        A[B_idx(t)] = -1

        # RHS calculation
        # Start with -D[t] (demand at day t, 0-indexed in array)
        rhs = -demands[t-1]

        # Account for initial state at current_day
        if t == current_day:
            rhs -= initial_state['inventory']
            rhs += initial_state['backlog']

        # Account for fixed production that arrives at day t
        if t > LT:
            arrival_day = t - LT
            if arrival_day < current_day and arrival_day <= len(fixed_production):
                # Production was already decided
                rhs += batch_size * fixed_production[arrival_day - 1]

        # Add equality constraint
        constraints.append(LinearConstraint(A, rhs, rhs))

    return c, constraints, bounds, integrality, remaining_days


def simulate_state(config, production_decisions, up_to_day):
    """
    Simulate inventory and backlog state up to a given day.

    Parameters:
    -----------
    config : dict
        Configuration
    production_decisions : list
        Production decisions (batches) for each day
    up_to_day : int
        Day to simulate up to (1-indexed)

    Returns:
    --------
    dict with 'inventory' and 'backlog' at end of up_to_day
    """
    H = config['planning_horizon']
    LT = config['lead_time']
    batch_size = config['batch_size']
    demands = config['demands']
    initial_inv = config['initial_inventory']

    inventory = initial_inv
    backlog = 0

    for t in range(1, up_to_day + 1):
        # Production arriving today (from day t-LT)
        production_arriving = 0
        if t > LT and t - LT <= len(production_decisions):
            production_arriving = batch_size * production_decisions[t - LT - 1]

        # Demand today
        demand = demands[t - 1]

        # Update inventory and backlog
        # Available = previous inventory + production arriving + previous backlog
        # We need: demand
        available = inventory + production_arriving
        net_position = available + backlog - demand

        if net_position >= 0:
            # Can meet all current and past demand
            inventory = net_position
            backlog = 0
        else:
            # Cannot meet all demand
            inventory = 0
            backlog = -net_position

    return {'inventory': inventory, 'backlog': backlog}


def rolling_horizon_optimization(config):
    """
    Perform rolling horizon optimization.

    At each day, re-optimize with prior decisions fixed.
    """
    H = config['planning_horizon']
    batch_size = config['batch_size']

    print("=" * 80)
    print("ROLLING HORIZON OPTIMIZATION")
    print("=" * 80)
    print(f"\nPlanning Horizon: {H} days")
    print(f"Running optimization {H} times, once for each day")
    print("=" * 80)

    # First, run full horizon optimization (Day 1)
    print("\n" + "=" * 80)
    print("DAY 1: Full Horizon Optimization")
    print("=" * 80)

    from scp_model import build_scp_model
    c, constraints, bounds, integrality = build_scp_model(config)

    result = milp(c=c, constraints=constraints, bounds=bounds,
                  integrality=integrality, options={'disp': False})

    if not result.success:
        print(f"✗ Day 1 optimization failed: {result.message}")
        return None

    # Extract full production plan
    full_production = [int(result.x[t]) for t in range(H)]
    full_inventory = [result.x[H + t] for t in range(H)]
    full_backlog = [result.x[2*H + t] for t in range(H)]
    full_cost = result.fun

    print(f"✓ Day 1 Complete")
    print(f"  Production decisions: {full_production}")
    print(f"  Total Cost: ${full_cost:.2f}")

    # Store results from each day
    daily_results = [{
        'day': 1,
        'production': full_production.copy(),
        'inventory': full_inventory.copy(),
        'backlog': full_backlog.copy(),
        'cost': full_cost,
        'new_decisions': full_production.copy()
    }]

    # Now run rolling horizon for days 2 to H
    for current_day in range(2, H + 1):
        print("\n" + "=" * 80)
        print(f"DAY {current_day}: Rolling Horizon Re-optimization")
        print("=" * 80)

        # Fixed production from previous optimization (days 1 to current_day-1)
        fixed_production = full_production[:current_day-1]
        print(f"  Fixed production (days 1-{current_day-1}): {fixed_production}")

        # Calculate initial state for current day
        initial_state = simulate_state(config, full_production, current_day - 1)
        print(f"  Initial state: Inventory={initial_state['inventory']:.2f}, "
              f"Backlog={initial_state['backlog']:.2f}")

        # Build and solve model for remaining days
        c_roll, constraints_roll, bounds_roll, integrality_roll, remaining = \
            build_rolling_scp_model(config, current_day, fixed_production, initial_state)

        print(f"  Optimizing for days {current_day} to {H} ({remaining} days remaining)")

        result_roll = milp(c=c_roll, constraints=constraints_roll,
                          bounds=bounds_roll, integrality=integrality_roll,
                          options={'disp': False})

        if not result_roll.success:
            print(f"✗ Day {current_day} optimization failed: {result_roll.message}")
            return None

        # Extract production decisions for remaining days
        new_production = [int(result_roll.x[t]) for t in range(remaining)]
        new_inventory = [result_roll.x[remaining + t] for t in range(remaining)]
        new_backlog = [result_roll.x[2*remaining + t] for t in range(remaining)]

        # Reconstruct full production plan
        full_plan_this_day = fixed_production + new_production

        print(f"  New production decisions (days {current_day}-{H}): {new_production}")
        print(f"  Full production plan: {full_plan_this_day}")
        print(f"  Objective value: ${result_roll.fun:.2f}")

        # Compare with original decision for current_day
        original_decision = full_production[current_day - 1]
        new_decision = new_production[0]

        if original_decision == new_decision:
            print(f"  ✓ Day {current_day} decision matches original: {new_decision} batches")
        else:
            print(f"  ✗ MISMATCH: Original={original_decision}, New={new_decision}")

        daily_results.append({
            'day': current_day,
            'fixed_production': fixed_production,
            'production': full_plan_this_day,
            'inventory': new_inventory,
            'backlog': new_backlog,
            'cost': result_roll.fun,
            'new_decisions': new_production
        })

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(f"\n{'Day':>5} {'Original':>12} {'Rolling':>12} {'Match':>8}")
    print("-" * 42)

    all_match = True
    for day_result in daily_results:
        day = day_result['day']
        original = full_production[day - 1]
        rolling = day_result['new_decisions'][0] if day_result['new_decisions'] else original
        match = "✓" if original == rolling else "✗"
        if original != rolling:
            all_match = False
        print(f"{day:5d} {original:12d} {rolling:12d} {match:>8}")

    print("\n" + "=" * 80)
    if all_match:
        print("✓✓✓ SUCCESS: All rolling horizon decisions match original optimization!")
        print("    This confirms the model is consistent and deterministic.")
    else:
        print("✗✗✗ MISMATCH: Some rolling horizon decisions differ from original.")
        print("    This may indicate model inconsistency or numerical issues.")
    print("=" * 80)

    return daily_results


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)

    # Run rolling horizon optimization
    results = rolling_horizon_optimization(config)
