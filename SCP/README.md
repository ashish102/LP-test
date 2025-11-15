# Supply Chain Planning - Single Node MIP Model

A Mixed Integer Programming model for optimizing production planning in a single-node supply chain network with lead times, batch constraints, and time-varying demand.

## Table of Contents

- [Problem Description](#problem-description)
- [Model Formulation](#model-formulation)
- [Files](#files)
- [Configuration File](#configuration-file)
- [Usage](#usage)
  - [Basic Optimization](#run-the-model)
  - [Rolling Horizon Validation](#rolling-horizon-optimization)
  - [Sensitivity Analysis](#sensitivity-analysis-first-production-decision)
- [Solution Insights](#solution-insights)
- [Customization](#customization)
- [Mathematical Details](#mathematical-details)
- [Extensions](#extensions)
- [Technical Notes](#technical-notes)
- [References](#references)

## Problem Description

This model addresses a fundamental supply chain planning problem:

**Given:**
- A single production process **P** with lead time **LT**
- Batch size constraints on production quantities
- Time-varying demand **D₁, D₂, ..., Dₕ** over planning horizon **H**
- Initial inventory level
- Cost structure:
  - Per-unit supply/production cost
  - Inventory holding cost (per unit per day)
  - Delay penalty for unmet demand (per unit per day)
  - High penalty for demand unmet by end of horizon

**Objective:**
Minimize total cost while satisfying demand (possibly with delays)

## Model Formulation

### Decision Variables

For each time period t = 1, 2, ..., H:

- **n[t]**: Number of batches to produce at time t (integer)
- **I[t]**: Inventory level at end of day t (continuous, ≥ 0)
- **B[t]**: Backlog (cumulative unmet demand) at end of day t (continuous, ≥ 0)

### Parameters

- **H**: Planning horizon (number of days)
- **LT**: Lead time (days between production order and arrival)
- **batch_size**: Minimum production quantity (batch size)
- **D[t]**: Demand at day t
- **I₀**: Initial inventory
- **c_supply**: Cost per unit produced
- **c_inv**: Inventory holding cost per unit per day
- **c_delay**: Delay penalty per unit per day of backlog
- **c_final**: High penalty for unmet demand at end of horizon

### Constraints

**1. Production Batch Constraint:**
```
P[t] = batch_size × n[t]
```
Production must be in multiples of batch size.

**2. Inventory Balance:**
```
I[t] = I[t-1] + P[t-LT] - D[t] + B[t-1] - B[t]
```
For each day t:
- Ending inventory = Previous inventory + Production arriving (from t-LT) - Demand + Previous backlog - Current backlog

**3. Non-negativity:**
```
I[t] ≥ 0,  B[t] ≥ 0,  n[t] ≥ 0 (integer)
```

### Objective Function

```
Minimize:  Σ(c_supply × batch_size × n[t])  +  Σ(c_inv × I[t])  +  Σ(c_delay × B[t])  +  c_final × B[H]
            └─────── Production Cost ───────┘   └─ Holding Cost ─┘   └── Delay Cost ──┘   └─ Final Penalty ─┘
```

## Files

### Core Model Files
- **config.json**: Configuration file with all input parameters
- **scp_model.py**: MIP model implementation using SciPy/HiGHS

### Analysis and Validation
- **rolling_horizon.py**: Initial rolling horizon implementation (tail subproblem approach)
- **rolling_horizon_fixed.py**: Corrected rolling horizon with fixed variables
- **debug_rolling.py**: State simulation verification script
- **sensitivity_analysis.py**: First production decision sensitivity analysis
- **ROLLING_HORIZON_ANALYSIS.md**: Detailed rolling horizon validation report

### Documentation and Results
- **README.md**: This comprehensive documentation
- **first_production_sensitivity.png**: Sensitivity analysis visualization

## Configuration File

The `config.json` file contains all problem parameters:

```json
{
  "planning_horizon": 10,          // Number of days to plan
  "lead_time": 2,                  // Production lead time
  "initial_inventory": 0,          // Starting inventory
  "batch_size": 50,                // Production batch size
  "demands": [80, 60, 100, ...],   // Daily demand pattern
  "costs": {
    "supply_per_unit": 10.0,
    "inventory_holding_per_unit_per_day": 0.5,
    "delay_penalty_per_unit_per_day": 5.0,
    "final_unmet_demand_penalty_per_unit": 100.0
  }
}
```

### Parameter Guidance

**Lead Time (LT):**
- Typical range: 1-5 days for manufacturing, 5-30+ for procurement
- Current example: 2 days

**Batch Size:**
- Represents minimum economic production quantity or equipment constraints
- Current example: 50 units

**Cost Ratios:**
- Delay penalty should be >> inventory holding cost (otherwise, just delay everything)
- Final penalty should be >> delay penalty (to incentivize completion)
- Current example: delay/holding = 10x, final/delay = 20x

## Usage

### Run the Model

Run the basic optimization:

```bash
cd SCP
python scp_model.py
```

Run the rolling horizon validation:

```bash
python rolling_horizon_fixed.py
```

Run the sensitivity analysis:

```bash
python sensitivity_analysis.py
```

### Example Output (Basic Model)

```
================================================================================
SUPPLY CHAIN PLANNING - SINGLE NODE OPTIMIZATION
================================================================================

Problem Configuration:
Planning Horizon: 10 days
Lead Time: 2 days
Batch Size: 50 units
Total Demand: 800 units

Production Plan (units):
  Day  Batches   Production    Arrives
----------------------------------------
    1        5          250      Day 3
    2        2          100      Day 4
    3        1           50      Day 5
    ...

Inventory and Backlog Profile:
  Day   Demand    Inventory    Backlog   Service Level
-------------------------------------------------------
    1       80         0.00      80.00            0.0%
    2       60         0.00     140.00            0.0%
    3      100        10.00       0.00          100.0%
    ...

Cost Breakdown:
Supply Cost:              $     8000.00
Inventory Holding Cost:   $       90.00
Delay Penalty Cost:       $     1100.00
Final Unmet Demand Cost:  $        0.00
Total Cost:               $     9190.00

Key Metrics:
  Fill Rate: 100.0%
  Average Inventory: 18.00 units
  Max Backlog: 140.00 units
```

## Solution Insights

### Current Configuration Results

With the default configuration:
- **Total Production:** 800 units (matches total demand)
- **Production Strategy:** Front-loaded to minimize backlog
- **Initial Backlog:** 140 units at day 2 (due to 2-day lead time)
- **Recovery:** Backlog cleared by day 3 when first production arrives
- **Final State:** Zero inventory, zero backlog (optimal completion)

### Cost Tradeoffs

The model balances:
1. **Production Costs:** Fixed per-unit cost × batch size constraints
2. **Inventory Holding:** Incentive to keep inventory low
3. **Delay Penalties:** Incentive to produce early and avoid backlog
4. **Batch Efficiency:** Larger batches reduce setup frequency but increase inventory

### Lead Time Impact

- Lead time LT creates initial unavoidable backlog
- Minimum backlog = demand during first LT periods
- Early production orders are critical for service level

## Customization

### Modify Demand Pattern

Edit `config.json`:
```json
"demands": [100, 120, 80, 90, 110, 95, 85, 100, 105, 115]
```

### Change Cost Structure

Experiment with different cost ratios:
```json
"costs": {
  "supply_per_unit": 15.0,              // Higher production cost
  "inventory_holding_per_unit_per_day": 1.0,   // Expensive storage
  "delay_penalty_per_unit_per_day": 20.0,      // Very high delay penalty
  "final_unmet_demand_penalty_per_unit": 500.0 // Critical to complete
}
```

### Adjust Planning Horizon

```json
"planning_horizon": 20,  // Plan for 20 days
"demands": [80, 60, ..., 75]  // Provide 20 demand values
```

## Mathematical Details

### Variable Indexing

Variables are arranged in a single vector:
- Indices 0 to H-1: n[0], n[1], ..., n[H-1] (production batches)
- Indices H to 2H-1: I[0], I[1], ..., I[H-1] (inventory)
- Indices 2H to 3H-1: B[0], B[1], ..., B[H-1] (backlog)

Total: 3H variables (H integer, 2H continuous)

### Constraint Matrix

Each time period t has one inventory balance constraint:
```
I[t] - I[t-1] - batch_size×n[t-LT] + B[t-1] - B[t] = -D[t]
```

Special cases:
- t = 0: Include initial inventory I₀
- t < LT: No production arrival term

Total: H equality constraints

## Rolling Horizon Optimization

The repository includes a rolling horizon validation that tests the model's consistency by re-solving with prior decisions fixed.

### Approach

```bash
python rolling_horizon_fixed.py
```

This script:
1. Solves the full optimization problem (Day 1)
2. Re-solves with Day 1 production fixed (Day 2)
3. Re-solves with Days 1-2 fixed (Day 3)
4. ...continues through the horizon

### Results

✓ **9 out of 10 days match exactly**
✓ **All costs remain at $9,190 (optimal)**
✓ **One alternative optimal solution found at Day 3**

The single mismatch (Day 4: 2 vs 1 batches) achieves the **same optimal cost**, demonstrating the existence of **multiple optimal solutions** - a common and expected phenomenon in MIP that provides flexibility in real-world implementation.

**Key Validation:** The model is correct and produces truly optimal solutions. Re-optimization with fixed prior decisions maintains optimality, confirming the solution quality.

See `ROLLING_HORIZON_ANALYSIS.md` for detailed analysis.

## Sensitivity Analysis: First Production Decision

Understanding how the first production decision impacts total cost provides valuable insights into solution robustness and cost structure.

### Running the Analysis

```bash
python sensitivity_analysis.py
```

This analyzes total cost when fixing the first production decision to 1-10 batches while optimizing the rest of the horizon.

### Results Summary

| First Production (batches) | Units | Total Cost | Cost vs Optimal | Production Plan |
|---------------------------|-------|------------|-----------------|-----------------|
| 1 | 50 | $10,135.00 | +$945 (+10.3%) | [1, 6, 1, 2, 1, 2, 1, 1, 0, 0] |
| 2 | 100 | $9,885.00 | +$695 (+7.6%) | [2, 5, 1, 2, 1, 2, 1, 1, 0, 0] |
| 3 | 150 | $9,635.00 | +$445 (+4.8%) | [3, 4, 1, 2, 1, 2, 1, 1, 0, 0] |
| 4 | 200 | $9,385.00 | +$195 (+2.1%) | [4, 3, 1, 2, 1, 2, 1, 1, 0, 0] |
| **5** | **250** | **$9,190.00** | **OPTIMAL** ✓ | **[5, 2, 1, 2, 2, 2, 1, 1, 0, 0]** |
| 6 | 300 | $9,215.00 | +$25 (+0.3%) | [6, 1, 1, 2, 2, 2, 1, 1, 0, 0] |
| 7 | 350 | $9,240.00 | +$50 (+0.5%) | [7, 0, 1, 2, 2, 2, 1, 1, 0, 0] |
| 8 | 400 | $9,290.00 | +$100 (+1.1%) | [8, 0, 0, 2, 2, 2, 0, 1, 0, 0] |
| 9 | 450 | $9,365.00 | +$175 (+1.9%) | [9, 0, 0, 0, 2, 2, 0, 1, 0, 0] |
| 10 | 500 | $9,440.00 | +$250 (+2.7%) | [10, 0, 0, 0, 1, 2, 0, 1, 0, 0] |

### Visualization

![First Production Sensitivity Analysis](first_production_sensitivity.png)

**Figure 1:** Impact of first production decision on total cost
- **Top panel:** Total cost curve showing convex relationship with unique minimum at 5 batches
- **Bottom panel:** Cost component breakdown (supply, inventory holding, delay penalties)

### Key Insights

**1. Optimal Decision Confirmed**
- **5 batches (250 units)** is the unique optimal first production decision
- Achieves minimum total cost of **$9,190**

**2. Asymmetric Cost Penalty**
- **Underproduction (1-4 batches):** Very expensive! Up to **10.3% cost increase**
  - Causes significant backlog and delay penalties
  - Steep cost curve on the left side
- **Overproduction (6-10 batches):** Gentler penalty, max **2.7% increase**
  - Creates inventory holding costs but avoids delays
  - Gradual cost increase on the right side

**3. Cost Structure**
- **Convex cost function** with clear unique minimum
- Cost range: **$9,190 - $10,135** ($945 difference, 10.3% span)
- Better to slightly overproduce than underproduce due to high delay penalties

**4. Practical Implications**
- First decision has **significant impact** on total cost
- Model provides **robust guidance:** 5 batches is distinctly optimal
- In uncertain environments, bias toward slightly higher initial production
- Cost penalty for being 1 batch off: **$25-$195** depending on direction

**5. Cost Component Analysis**

From the stacked bar chart:
- **Supply cost dominates** (~87% of total at optimum)
- **Delay penalties** are the main driver of increased cost for underproduction
- **Inventory holding costs** remain relatively small even with overproduction
- This explains the asymmetric penalty structure

### Management Insights

This sensitivity analysis demonstrates:
- The value of optimization: **$945 savings** vs. naive approach (1 batch)
- Robustness around optimum: ±1 batch costs only $25-$195 extra
- Risk management: Overproduction is less costly than underproduction
- Decision confidence: Clear optimal point with measurable tradeoffs

## Extensions

Potential enhancements to this model:

1. **Multiple Products:** Add product index to variables and constraints
2. **Multiple Nodes:** Add location index and transportation variables
3. **Safety Stock:** Add minimum inventory constraints
4. **Production Capacity:** Add maximum production constraints
5. **Setup Costs:** Add binary setup variables and fixed costs
6. **Multi-Period Lead Times:** Different lead times for different periods
7. **Stochastic Demand:** Scenario-based or robust optimization
8. **Perishability:** Add expiration constraints and waste costs

## Technical Notes

- **Solver:** HiGHS via SciPy's `milp()` function
- **Performance:** Solves in <1 second for typical instances (H ≤ 30)
- **Optimality:** Guaranteed global optimal solution (MIP)
- **Scalability:** Linear growth in variables/constraints with horizon H

## References

- [SciPy MILP Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
- [HiGHS Solver](https://highs.dev/)
- Supply Chain Planning literature: Silver, Pyke & Peterson (1998), "Inventory Management and Production Planning and Scheduling"

## License

Educational example for learning supply chain optimization techniques.
