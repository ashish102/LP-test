# Supply Chain Planning - Single Node MIP Model

A Mixed Integer Programming model for optimizing production planning in a single-node supply chain network with lead times, batch constraints, and time-varying demand.

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

- **config.json**: Configuration file with all input parameters
- **scp_model.py**: MIP model implementation using SciPy/HiGHS
- **README.md**: This documentation file

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

```bash
cd SCP
python scp_model.py
```

### Example Output

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
