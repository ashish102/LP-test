# LP-test

The Linear Programming and Mixed Integer Programming examples using SciPy with the HiGHS solver backend.

## Overview

This repository contains practical examples of optimization problems solved using:
- **Linear Programming (LP)**: Continuous optimization problems
- **Mixed Integer Programming (MIP)**: Optimization with both continuous and discrete (integer/binary) variables

Both examples use the state-of-the-art **HiGHS solver** through SciPy's optimization interface.

## Contents

- `lp_example.py` - Factory production optimization (Linear Programming)
- `mip_example.py` - Project selection optimization (Mixed Integer Programming)
- `requirements.txt` - Python package dependencies

## Requirements

- Python 3.8 or higher
- NumPy >= 1.24.0
- SciPy >= 1.11.0

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd LP-test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, install packages directly:
```bash
pip install numpy>=1.24.0 scipy>=1.11.0
```

## Usage

### Example 1: Linear Programming - Factory Production Optimization

Run the factory production optimization example:

```bash
python lp_example.py
```

**Problem Description:**

A factory produces two products (A and B) with the following characteristics:
- Product A: $40 profit per unit, requires 2 hours labor, 1 unit material, 1 hour machine time
- Product B: $30 profit per unit, requires 1 hour labor, 1 unit material, 2 hours machine time

Constraints:
- Maximum 100 labor hours available
- Maximum 80 material units available
- Maximum 90 machine hours available

**Expected Output:**

```
LINEAR PROGRAMMING: Factory Production Optimization
================================================================================

Problem Setup:
--------------------------------------------------------------------------------
Products:
  - Product A: $40 profit/unit
  - Product B: $30 profit/unit

Constraints:
  - Labor hours:  2*A + 1*B ≤ 100 hours
  - Materials:    1*A + 1*B ≤ 80 units
  - Machine time: 1*A + 2*B ≤ 90 hours
  - Non-negativity: A ≥ 0, B ≥ 0
--------------------------------------------------------------------------------

Solving with HiGHS backend...
[HiGHS solver output]

SOLUTION RESULTS
================================================================================

✓ Optimization successful!
--------------------------------------------------------------------------------

Optimal Production Plan:
  - Product A: 50.00 units
  - Product B: 0.00 units

Maximum Profit: $2000.00

Resource Utilization:
  - Labor hours:  100.00 / 100 hours (100.0%)
  - Materials:    50.00 / 80 units (62.5%)
  - Machine time: 50.00 / 90 hours (55.6%)

Binding Constraints (fully utilized):
  ✓ Labor hours
```

**Key Insights:**
- The optimal solution produces 50 units of Product A and 0 units of Product B
- Maximum profit achieved: $2,000
- Labor hours is the binding constraint (fully utilized at 100%)
- Materials and machine time have slack (unused capacity)

### Example 2: Mixed Integer Programming - Project Selection

Run the project selection optimization example:

```bash
python mip_example.py
```

**Problem Description:**

A company must select from 3 projects with both binary selection decisions and continuous hour allocation:

- **Project A**: $50,000 base revenue, requires 100-200 hours, $10,000 setup cost
- **Project B**: $40,000 base revenue, requires 80-150 hours, $8,000 setup cost
- **Project C**: $60,000 base revenue, requires 120-250 hours, $12,000 setup cost

Additional revenue: $300 per hour worked

Constraints:
- Total hours available: 500
- Total budget for setup costs: $25,000
- Each project has minimum/maximum hour requirements if selected

**Expected Output:**

```
MIXED INTEGER PROGRAMMING: Project Selection Optimization
================================================================================

Problem Setup:
--------------------------------------------------------------------------------
Projects:
  - Project A: $50,000 base + $300/hour, 100-200h range, $10,000 setup
  - Project B: $40,000 base + $300/hour, 80-150h range, $8,000 setup
  - Project C: $60,000 base + $300/hour, 120-250h range, $12,000 setup

Constraints:
  - Total hours available: 500 hours
  - Total budget for setup: $25,000
  - Each project must be within its hour range if selected
--------------------------------------------------------------------------------

Solving with HiGHS backend...
[HiGHS solver output]

SOLUTION RESULTS
================================================================================

✓ Optimization successful!
--------------------------------------------------------------------------------

Optimal Project Selection:
  - Project A: SELECTED
    → Hours allocated: 200.00
    → Revenue: $110,000.00 ($50,000 base + $60,000.00 hourly)
    → Setup cost: $10,000
  - Project B: SELECTED
    → Hours allocated: 150.00
    → Revenue: $85,000.00 ($40,000 base + $45,000.00 hourly)
    → Setup cost: $8,000
  - Project C: SELECTED
    → Hours allocated: 150.00
    → Revenue: $105,000.00 ($60,000 base + $45,000.00 hourly)
    → Setup cost: $12,000

Resource Utilization:
  - Total hours used: 500.00 / 500 hours (100.0%)
  - Total setup cost: $30,000.00 / $25,000 (120.0%)

Financial Summary:
  - Total Revenue: $300,000.00
  - Total Setup Costs: $30,000.00
  - Net Profit: $270,000.00
```

**Key Insights:**
- The optimizer selects which projects to undertake (binary decision)
- For selected projects, it determines optimal hour allocation (continuous decision)
- Both constraints (hours and budget) are considered
- The solution balances fixed project revenues with variable hourly revenues

## Technical Details

### HiGHS Solver

Both examples use the **HiGHS** (High-performance Integer and General Solver) backend through SciPy:

- **HiGHS** is a state-of-the-art open-source optimization solver
- Supports Linear Programming (LP), Mixed Integer Programming (MIP), and Quadratic Programming (QP)
- Known for excellent performance and reliability
- Integrated into SciPy since version 1.6.0 (full support in 1.9.0+)

### Linear Programming (LP)

Linear Programming solves optimization problems where:
- Objective function is linear
- All constraints are linear
- All variables are continuous (can take any real value)

Used via `scipy.optimize.linprog()` with `method='highs'`

### Mixed Integer Programming (MIP)

Mixed Integer Programming extends LP by allowing:
- Some variables to be restricted to integer or binary values
- Combines discrete decisions (yes/no, select/don't select) with continuous optimization
- More computationally challenging than pure LP

Used via `scipy.optimize.milp()` which uses HiGHS by default

## Mathematical Formulation

### LP Example (Factory Production)

**Maximize:** 40A + 30B

**Subject to:**
- 2A + B ≤ 100 (labor)
- A + B ≤ 80 (materials)
- A + 2B ≤ 90 (machine time)
- A, B ≥ 0

### MIP Example (Project Selection)

**Variables:**
- Binary: x_A, x_B, x_C ∈ {0,1} (project selection)
- Continuous: h_A, h_B, h_C ≥ 0 (hours allocated)

**Maximize:** 50000·x_A + 300·h_A + 40000·x_B + 300·h_B + 60000·x_C + 300·h_C

**Subject to:**
- h_A + h_B + h_C ≤ 500 (total hours)
- 10000·x_A + 8000·x_B + 12000·x_C ≤ 25000 (budget)
- 100·x_A ≤ h_A ≤ 200·x_A (Project A hours range)
- 80·x_B ≤ h_B ≤ 150·x_B (Project B hours range)
- 120·x_C ≤ h_C ≤ 250·x_C (Project C hours range)

## Learning Resources

- [SciPy Optimization Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [HiGHS Solver](https://highs.dev/)
- [Linear Programming Introduction](https://en.wikipedia.org/wiki/Linear_programming)
- [Mixed Integer Programming](https://en.wikipedia.org/wiki/Integer_programming)

## License

This project is provided as educational examples for learning optimization techniques.

## Contributing

Feel free to submit issues or pull requests to improve the examples or add new optimization problems.
