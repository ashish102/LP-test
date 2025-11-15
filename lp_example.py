#!/usr/bin/env python3
"""
Linear Programming Example: Factory Production Optimization

This example demonstrates solving a linear programming problem using SciPy
with the HiGHS backend to optimize factory production of two products.

Problem Statement:
-----------------
A factory produces two products: A and B
- Product A generates $40 profit per unit
- Product B generates $30 profit per unit

Constraints:
- Labor hours: Product A requires 2 hours, Product B requires 1 hour (max 100 hours)
- Materials: Each product requires 1 unit of material (max 80 units)
- Machine time: Product A requires 1 hour, Product B requires 2 hours (max 90 hours)

Objective: Maximize total profit
"""

import numpy as np
from scipy.optimize import linprog

def solve_factory_optimization():
    """
    Solve the factory production optimization problem using linear programming.
    """
    print("=" * 80)
    print("LINEAR PROGRAMMING: Factory Production Optimization")
    print("=" * 80)
    print("\nProblem Setup:")
    print("-" * 80)
    print("Products:")
    print("  - Product A: $40 profit/unit")
    print("  - Product B: $30 profit/unit")
    print("\nConstraints:")
    print("  - Labor hours:  2*A + 1*B ≤ 100 hours")
    print("  - Materials:    1*A + 1*B ≤ 80 units")
    print("  - Machine time: 1*A + 2*B ≤ 90 hours")
    print("  - Non-negativity: A ≥ 0, B ≥ 0")
    print("-" * 80)

    # Objective function coefficients (we minimize, so negate for maximization)
    # Maximize: 40*A + 30*B  =>  Minimize: -40*A - 30*B
    c = np.array([-40, -30])

    # Inequality constraint matrix (A_ub * x <= b_ub)
    # 2*A + 1*B <= 100  (labor)
    # 1*A + 1*B <= 80   (materials)
    # 1*A + 2*B <= 90   (machine time)
    A_ub = np.array([
        [2, 1],  # Labor constraint
        [1, 1],  # Materials constraint
        [1, 2]   # Machine time constraint
    ])

    b_ub = np.array([100, 80, 90])

    # Bounds for variables (non-negativity)
    # Product A: [0, infinity)
    # Product B: [0, infinity)
    bounds = [(0, None), (0, None)]

    print("\nSolving with HiGHS backend...")
    print("-" * 80)

    # Solve using HiGHS method
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs',
        options={'disp': True}
    )

    print("\n" + "=" * 80)
    print("SOLUTION RESULTS")
    print("=" * 80)

    if result.success:
        print("\n✓ Optimization successful!")
        print("-" * 80)
        print(f"\nOptimal Production Plan:")
        print(f"  - Product A: {result.x[0]:.2f} units")
        print(f"  - Product B: {result.x[1]:.2f} units")
        print(f"\nMaximum Profit: ${-result.fun:.2f}")

        # Calculate resource utilization
        print(f"\nResource Utilization:")
        labor_used = 2 * result.x[0] + 1 * result.x[1]
        materials_used = 1 * result.x[0] + 1 * result.x[1]
        machine_used = 1 * result.x[0] + 2 * result.x[1]

        print(f"  - Labor hours:  {labor_used:.2f} / 100 hours ({labor_used/100*100:.1f}%)")
        print(f"  - Materials:    {materials_used:.2f} / 80 units ({materials_used/80*100:.1f}%)")
        print(f"  - Machine time: {machine_used:.2f} / 90 hours ({machine_used/90*100:.1f}%)")

        # Identify binding constraints
        print(f"\nBinding Constraints (fully utilized):")
        if abs(labor_used - 100) < 0.01:
            print("  ✓ Labor hours")
        if abs(materials_used - 80) < 0.01:
            print("  ✓ Materials")
        if abs(machine_used - 90) < 0.01:
            print("  ✓ Machine time")

        print("\n" + "=" * 80)
        print(f"\nSolver Details:")
        print(f"  - Method: HiGHS")
        print(f"  - Iterations: {result.nit}")
        print(f"  - Status: {result.message}")
        print("=" * 80)

    else:
        print(f"\n✗ Optimization failed!")
        print(f"Status: {result.message}")
        print("=" * 80)

    return result


if __name__ == "__main__":
    result = solve_factory_optimization()
