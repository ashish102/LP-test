#!/usr/bin/env python3
"""
Supply Chain Planning - Single Node MIP Model

This model optimizes production planning for a single-node supply chain with:
- Production process with lead time
- Batch size constraints on production
- Time-varying demand
- Inventory holding costs
- Delay penalties for unmet demand
- High penalty for final unmet demand at horizon

Decision Variables:
- Production quantities at each time period (batch-constrained)
- Inventory levels
- Backlog (delayed demand)

Objective: Minimize total cost (production + inventory + delay penalties)
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


def build_scp_model(config):
    """
    Build the Supply Chain Planning MIP model.

    Variables (for each time period t = 1 to H):
    - n[t]: Number of batches to produce (integer) - indices 0 to H-1
    - I[t]: Inventory at end of day t (continuous) - indices H to 2H-1
    - B[t]: Backlog at end of day t (continuous) - indices 2H to 3H-1

    Total variables: 3*H (H integers, 2H continuous)
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

    # Total variables: n[0..H-1], I[0..H-1], B[0..H-1]
    num_vars = 3 * H

    # Variable indices
    n_idx = lambda t: t  # n[t] at index t
    I_idx = lambda t: H + t  # I[t] at index H+t
    B_idx = lambda t: 2*H + t  # B[t] at index 2H+t

    # Objective function coefficients
    # Minimize: sum(c_supply * batch_size * n[t]) + sum(c_inv * I[t]) + sum(c_delay * B[t]) + c_final * B[H-1]
    c = np.zeros(num_vars)

    # Production costs
    for t in range(H):
        c[n_idx(t)] = c_supply * batch_size

    # Inventory holding costs
    for t in range(H):
        c[I_idx(t)] = c_inv

    # Delay penalties (backlog cost per day)
    for t in range(H):
        if t < H - 1:
            c[B_idx(t)] = c_delay
        else:
            c[B_idx(t)] = c_delay + c_final  # Final period includes high penalty

    # Integrality constraints (1 = integer, 0 = continuous)
    integrality = np.zeros(num_vars)
    for t in range(H):
        integrality[n_idx(t)] = 1  # n[t] are integers

    # Variable bounds
    # n[t] >= 0, I[t] >= 0, B[t] >= 0
    # Set reasonable upper bounds
    max_batches = int(np.ceil(np.sum(demands) / batch_size)) + 5
    bounds = Bounds(
        lb=np.zeros(num_vars),
        ub=np.array([max_batches]*H + [np.inf]*H + [np.inf]*H)
    )

    # Constraints
    constraints = []

    # Inventory balance constraints for each time period t
    # I[t] = I[t-1] + P[t-LT] - D[t] + B[t-1] - B[t]
    # Rearranged: I[t] - I[t-1] - batch_size*n[t-LT] - B[t] + B[t-1] = -D[t]
    # Or: I[t] - I[t-1] - batch_size*n[t-LT] + B[t-1] - B[t] + D[t] = 0

    for t in range(H):
        A = np.zeros(num_vars)

        # I[t] coefficient: +1
        A[I_idx(t)] = 1

        # I[t-1] coefficient: -1 (if t > 0)
        if t > 0:
            A[I_idx(t-1)] = -1

        # P[t-LT] = batch_size * n[t-LT] coefficient: -batch_size (if production arrives)
        if t >= LT:
            A[n_idx(t - LT)] = -batch_size

        # B[t-1] coefficient: +1 (if t > 0)
        if t > 0:
            A[B_idx(t-1)] = 1

        # B[t] coefficient: -1
        A[B_idx(t)] = -1

        # RHS: -D[t] for equality, but we account for initial inventory at t=0
        if t == 0:
            rhs = -demands[t] - initial_inv
        else:
            rhs = -demands[t]

        # Add equality constraint
        constraints.append(LinearConstraint(A, rhs, rhs))

    return c, constraints, bounds, integrality


def solve_scp_model(config):
    """Solve the Supply Chain Planning MIP model."""

    print("=" * 80)
    print("SUPPLY CHAIN PLANNING - SINGLE NODE OPTIMIZATION")
    print("=" * 80)
    print("\nProblem Configuration:")
    print("-" * 80)
    print(f"Planning Horizon: {config['planning_horizon']} days")
    print(f"Lead Time: {config['lead_time']} days")
    print(f"Batch Size: {config['batch_size']} units")
    print(f"Initial Inventory: {config['initial_inventory']} units")
    print(f"\nDemand Pattern:")
    for i, d in enumerate(config['demands'], 1):
        print(f"  Day {i:2d}: {d:3d} units")
    print(f"  Total: {sum(config['demands'])} units")

    print(f"\nCost Structure:")
    print(f"  Supply cost: ${config['costs']['supply_per_unit']}/unit")
    print(f"  Inventory holding: ${config['costs']['inventory_holding_per_unit_per_day']}/unit/day")
    print(f"  Delay penalty: ${config['costs']['delay_penalty_per_unit_per_day']}/unit/day")
    print(f"  Final unmet demand penalty: ${config['costs']['final_unmet_demand_penalty_per_unit']}/unit")
    print("-" * 80)

    # Build model
    print("\nBuilding MIP model...")
    c, constraints, bounds, integrality = build_scp_model(config)

    H = config['planning_horizon']
    batch_size = config['batch_size']

    print(f"Model size: {3*H} variables ({H} integer, {2*H} continuous), {H} constraints")

    print("\nSolving with HiGHS backend...")
    print("-" * 80)

    # Solve
    result = milp(
        c=c,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
        options={'disp': True}
    )

    print("\n" + "=" * 80)
    print("SOLUTION RESULTS")
    print("=" * 80)

    if result.success:
        print("\n✓ Optimization successful!")
        print("-" * 80)

        # Extract solution
        n_idx = lambda t: t
        I_idx = lambda t: H + t
        B_idx = lambda t: 2*H + t

        production = np.array([result.x[n_idx(t)] * batch_size for t in range(H)])
        inventory = np.array([result.x[I_idx(t)] for t in range(H)])
        backlog = np.array([result.x[B_idx(t)] for t in range(H)])

        print("\nProduction Plan (units):")
        print(f"{'Day':>5} {'Batches':>8} {'Production':>12} {'Arrives':>10}")
        print("-" * 40)
        for t in range(H):
            batches = int(result.x[n_idx(t)])
            prod = production[t]
            arrives_day = t + config['lead_time']
            if batches > 0:
                if arrives_day < H:
                    print(f"{t+1:5d} {batches:8d} {prod:12.0f} {'Day ' + str(arrives_day+1):>10}")
                else:
                    print(f"{t+1:5d} {batches:8d} {prod:12.0f} {'(after H)':>10}")

        total_production = np.sum(production)
        print(f"\nTotal Production: {total_production:.0f} units")

        print("\n\nInventory and Backlog Profile:")
        print(f"{'Day':>5} {'Demand':>8} {'Inventory':>12} {'Backlog':>10} {'Service Level':>15}")
        print("-" * 55)

        demands = config['demands']
        for t in range(H):
            service_level = 100 * (1 - backlog[t] / max(1, sum(demands[:t+1])))
            print(f"{t+1:5d} {demands[t]:8.0f} {inventory[t]:12.2f} {backlog[t]:10.2f} {service_level:14.1f}%")

        print("\n" + "-" * 80)
        print("Cost Breakdown:")
        print("-" * 80)

        # Calculate costs
        supply_cost = config['costs']['supply_per_unit'] * total_production
        inv_cost = config['costs']['inventory_holding_per_unit_per_day'] * np.sum(inventory)
        delay_cost = config['costs']['delay_penalty_per_unit_per_day'] * np.sum(backlog)
        final_cost = config['costs']['final_unmet_demand_penalty_per_unit'] * backlog[-1]

        print(f"Supply Cost:              ${supply_cost:12.2f}")
        print(f"Inventory Holding Cost:   ${inv_cost:12.2f}")
        print(f"Delay Penalty Cost:       ${delay_cost:12.2f}")
        print(f"Final Unmet Demand Cost:  ${final_cost:12.2f}")
        print("-" * 80)
        print(f"Total Cost:               ${result.fun:12.2f}")
        print("=" * 80)

        print("\nKey Metrics:")
        print(f"  Total Demand: {sum(demands):.0f} units")
        print(f"  Total Production: {total_production:.0f} units")
        print(f"  Final Inventory: {inventory[-1]:.2f} units")
        print(f"  Final Backlog: {backlog[-1]:.2f} units")
        print(f"  Average Inventory: {np.mean(inventory):.2f} units")
        print(f"  Average Backlog: {np.mean(backlog):.2f} units")
        print(f"  Max Inventory: {np.max(inventory):.2f} units")
        print(f"  Max Backlog: {np.max(backlog):.2f} units")

        fill_rate = 100 * (sum(demands) - backlog[-1]) / sum(demands)
        print(f"  Fill Rate: {fill_rate:.1f}%")

        print("\n" + "=" * 80)
        print("Solver Details:")
        print(f"  Method: HiGHS (MILP)")
        print(f"  Status: {result.message}")
        print("=" * 80)

        return result, {
            'production': production,
            'inventory': inventory,
            'backlog': backlog
        }
    else:
        print(f"\n✗ Optimization failed!")
        print(f"Status: {result.message}")
        print("=" * 80)
        return result, None


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)

    # Solve model
    result, solution = solve_scp_model(config)
