#!/usr/bin/env python3
"""
Mixed Integer Programming Example: Project Selection Optimization

This example demonstrates solving a mixed integer programming problem using SciPy
with the HiGHS backend to optimize project selection and resource allocation.

Problem Statement:
-----------------
A company must decide which projects to undertake and how many hours to allocate.

Projects Available:
- Project A: $50k base revenue, 100-200 hours required, $10k setup cost
- Project B: $40k base revenue, 80-150 hours required, $8k setup cost
- Project C: $60k base revenue, 120-250 hours required, $12k setup cost

Additional revenue: $300 per hour worked on any project

Constraints:
- Total available hours: 500
- Total budget for setup costs: $25,000
- Each project has minimum and maximum hour requirements if selected
- Projects are either selected (1) or not selected (0) - binary decision

Objective: Maximize total revenue (base + hourly)
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def solve_project_selection():
    """
    Solve the project selection optimization problem using mixed integer programming.
    """
    print("=" * 80)
    print("MIXED INTEGER PROGRAMMING: Project Selection Optimization")
    print("=" * 80)
    print("\nProblem Setup:")
    print("-" * 80)
    print("Projects:")
    print("  - Project A: $50,000 base + $300/hour, 100-200h range, $10,000 setup")
    print("  - Project B: $40,000 base + $300/hour, 80-150h range, $8,000 setup")
    print("  - Project C: $60,000 base + $300/hour, 120-250h range, $12,000 setup")
    print("\nConstraints:")
    print("  - Total hours available: 500 hours")
    print("  - Total budget for setup: $25,000")
    print("  - Each project must be within its hour range if selected")
    print("-" * 80)

    # Variables:
    # x[0] = binary: select project A (1) or not (0)
    # x[1] = binary: select project B (1) or not (0)
    # x[2] = binary: select project C (1) or not (0)
    # x[3] = continuous: hours allocated to project A
    # x[4] = continuous: hours allocated to project B
    # x[5] = continuous: hours allocated to project C

    # Objective function: Maximize total revenue
    # Revenue = 50000*x[0] + 300*x[3] + 40000*x[1] + 300*x[4] + 60000*x[2] + 300*x[5]
    # For minimization, negate the coefficients
    c = np.array([-50000, -40000, -60000, -300, -300, -300])

    # Integrality constraints (1 = integer/binary, 0 = continuous)
    integrality = np.array([1, 1, 1, 0, 0, 0])

    # Bounds for variables
    # Binary variables: [0, 1]
    # Hour variables: [0, max_hours] (will be further constrained)
    bounds = Bounds(
        lb=[0, 0, 0, 0, 0, 0],  # Lower bounds
        ub=[1, 1, 1, 200, 150, 250]  # Upper bounds
    )

    # Linear constraints
    constraints = []

    # Constraint 1: Total hours <= 500
    # x[3] + x[4] + x[5] <= 500
    A_hours = np.array([[0, 0, 0, 1, 1, 1]])
    constraints.append(LinearConstraint(A_hours, -np.inf, 500))

    # Constraint 2: Total setup cost <= 25000
    # 10000*x[0] + 8000*x[1] + 12000*x[2] <= 25000
    A_budget = np.array([[10000, 8000, 12000, 0, 0, 0]])
    constraints.append(LinearConstraint(A_budget, -np.inf, 25000))

    # Constraint 3: Minimum hours for Project A if selected
    # x[3] >= 100*x[0]  =>  -x[3] + 100*x[0] <= 0
    A_minA = np.array([[100, 0, 0, -1, 0, 0]])
    constraints.append(LinearConstraint(A_minA, -np.inf, 0))

    # Constraint 4: Maximum hours for Project A if selected
    # x[3] <= 200*x[0]  =>  x[3] - 200*x[0] <= 0
    A_maxA = np.array([[-200, 0, 0, 1, 0, 0]])
    constraints.append(LinearConstraint(A_maxA, -np.inf, 0))

    # Constraint 5: Minimum hours for Project B if selected
    # x[4] >= 80*x[1]  =>  -x[4] + 80*x[1] <= 0
    A_minB = np.array([[0, 80, 0, 0, -1, 0]])
    constraints.append(LinearConstraint(A_minB, -np.inf, 0))

    # Constraint 6: Maximum hours for Project B if selected
    # x[4] <= 150*x[1]  =>  x[4] - 150*x[1] <= 0
    A_maxB = np.array([[0, -150, 0, 0, 1, 0]])
    constraints.append(LinearConstraint(A_maxB, -np.inf, 0))

    # Constraint 7: Minimum hours for Project C if selected
    # x[5] >= 120*x[2]  =>  -x[5] + 120*x[2] <= 0
    A_minC = np.array([[0, 0, 120, 0, 0, -1]])
    constraints.append(LinearConstraint(A_minC, -np.inf, 0))

    # Constraint 8: Maximum hours for Project C if selected
    # x[5] <= 250*x[2]  =>  x[5] - 250*x[2] <= 0
    A_maxC = np.array([[0, 0, -250, 0, 0, 1]])
    constraints.append(LinearConstraint(A_maxC, -np.inf, 0))

    print("\nSolving with HiGHS backend...")
    print("-" * 80)

    # Solve using MILP with HiGHS
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
        select_A = result.x[0]
        select_B = result.x[1]
        select_C = result.x[2]
        hours_A = result.x[3]
        hours_B = result.x[4]
        hours_C = result.x[5]

        print(f"\nOptimal Project Selection:")
        print(f"  - Project A: {'SELECTED' if select_A > 0.5 else 'NOT SELECTED'}")
        if select_A > 0.5:
            print(f"    → Hours allocated: {hours_A:.2f}")
            print(f"    → Revenue: ${50000 + 300*hours_A:,.2f} (${50000:,} base + ${300*hours_A:,.2f} hourly)")
            print(f"    → Setup cost: $10,000")

        print(f"  - Project B: {'SELECTED' if select_B > 0.5 else 'NOT SELECTED'}")
        if select_B > 0.5:
            print(f"    → Hours allocated: {hours_B:.2f}")
            print(f"    → Revenue: ${40000 + 300*hours_B:,.2f} (${40000:,} base + ${300*hours_B:,.2f} hourly)")
            print(f"    → Setup cost: $8,000")

        print(f"  - Project C: {'SELECTED' if select_C > 0.5 else 'NOT SELECTED'}")
        if select_C > 0.5:
            print(f"    → Hours allocated: {hours_C:.2f}")
            print(f"    → Revenue: ${60000 + 300*hours_C:,.2f} (${60000:,} base + ${300*hours_C:,.2f} hourly)")
            print(f"    → Setup cost: $12,000")

        # Calculate totals
        total_hours = hours_A + hours_B + hours_C
        total_setup = 10000*select_A + 8000*select_B + 12000*select_C
        total_revenue = -result.fun  # Negate because we minimized

        print(f"\nResource Utilization:")
        print(f"  - Total hours used: {total_hours:.2f} / 500 hours ({total_hours/500*100:.1f}%)")
        print(f"  - Total setup cost: ${total_setup:,.2f} / $25,000 ({total_setup/25000*100:.1f}%)")

        print(f"\nFinancial Summary:")
        print(f"  - Total Revenue: ${total_revenue:,.2f}")
        print(f"  - Total Setup Costs: ${total_setup:,.2f}")
        print(f"  - Net Profit: ${total_revenue - total_setup:,.2f}")

        print("\n" + "=" * 80)
        print(f"\nSolver Details:")
        print(f"  - Method: HiGHS (MILP)")
        print(f"  - Status: {result.message}")
        print("=" * 80)

    else:
        print(f"\n✗ Optimization failed!")
        print(f"Status: {result.message}")
        print("=" * 80)

    return result


if __name__ == "__main__":
    result = solve_project_selection()
