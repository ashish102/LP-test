#!/usr/bin/env python3
"""
Debug script to understand the rolling horizon mismatch.

This compares the states calculated by the original optimization
vs. the simulated states in rolling horizon.
"""

import json
import numpy as np
from pathlib import Path
from scp_model import build_scp_model, load_config
from scipy.optimize import milp


def extract_full_solution(config):
    """Run full horizon optimization and extract complete solution."""
    c, constraints, bounds, integrality = build_scp_model(config)
    result = milp(c=c, constraints=constraints, bounds=bounds,
                  integrality=integrality, options={'disp': False})

    H = config['planning_horizon']

    production = [int(result.x[t]) for t in range(H)]
    inventory = [result.x[H + t] for t in range(H)]
    backlog = [result.x[2*H + t] for t in range(H)]

    return {
        'production': production,
        'inventory': inventory,
        'backlog': backlog,
        'cost': result.fun
    }


def simulate_step_by_step(config, production_plan):
    """
    Simulate the inventory and backlog evolution day by day.
    """
    H = config['planning_horizon']
    LT = config['lead_time']
    batch_size = config['batch_size']
    demands = config['demands']
    initial_inv = config['initial_inventory']

    states = []

    inventory = initial_inv
    backlog = 0

    for t in range(1, H + 1):
        # Production arriving today
        prod_arriving = 0
        if t > LT:
            prod_arriving = batch_size * production_plan[t - LT - 1]

        # Demand today
        demand = demands[t - 1]

        # Calculate net inventory position change
        # Net position = inventory - backlog
        net_start = inventory - backlog
        net_end = net_start + prod_arriving - demand

        # Split into inventory and backlog
        if net_end >= 0:
            inventory = net_end
            backlog = 0
        else:
            inventory = 0
            backlog = -net_end

        states.append({
            'day': t,
            'prod_arriving': prod_arriving,
            'demand': demand,
            'inventory': inventory,
            'backlog': backlog,
            'net_position': inventory - backlog
        })

    return states


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)

    # Get full optimization solution
    print("=" * 80)
    print("RUNNING FULL HORIZON OPTIMIZATION")
    print("=" * 80)

    solution = extract_full_solution(config)

    print(f"\nProduction plan: {solution['production']}")
    print(f"Total cost: ${solution['cost']:.2f}")

    # Simulate step by step
    print("\n" + "=" * 80)
    print("SIMULATING STATE EVOLUTION")
    print("=" * 80)

    sim_states = simulate_step_by_step(config, solution['production'])

    print(f"\n{'Day':>4} {'Prod':>6} {'Demand':>8} {'Sim Inv':>10} {'Opt Inv':>10} {'Match':>7} | "
          f"{'Sim BL':>10} {'Opt BL':>10} {'Match':>7}")
    print("-" * 90)

    all_match_inv = True
    all_match_bl = True

    for i, state in enumerate(sim_states):
        opt_inv = solution['inventory'][i]
        opt_bl = solution['backlog'][i]
        sim_inv = state['inventory']
        sim_bl = state['backlog']

        match_inv = "✓" if abs(sim_inv - opt_inv) < 0.01 else "✗"
        match_bl = "✓" if abs(sim_bl - opt_bl) < 0.01 else "✗"

        if abs(sim_inv - opt_inv) >= 0.01:
            all_match_inv = False
        if abs(sim_bl - opt_bl) >= 0.01:
            all_match_bl = False

        print(f"{state['day']:4d} {state['prod_arriving']:6.0f} {state['demand']:8.0f} "
              f"{sim_inv:10.2f} {opt_inv:10.2f} {match_inv:>7} | "
              f"{sim_bl:10.2f} {opt_bl:10.2f} {match_bl:>7}")

    print("\n" + "=" * 80)
    if all_match_inv and all_match_bl:
        print("✓ Simulation matches optimization perfectly!")
    else:
        print("✗ Mismatch between simulation and optimization")
        print("  This indicates an issue with either:")
        print("  1. The simulation logic")
        print("  2. The constraint formulation")
        print("  3. Numerical precision")
    print("=" * 80)
