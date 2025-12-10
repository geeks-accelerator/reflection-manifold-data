# Analysis Data

Processed analysis outputs for verifying paper findings. Total size: ~24 MB.

## Directory Structure

```
data/
├── bootstrap/      # Bootstrap confidence intervals
├── permanova/      # PERMANOVA variance decomposition
├── variance/       # Variance ratio by loop
├── trajectory/     # Trajectory convergence metrics
└── manifold/       # UMAP coordinates and attractor basins
```

---

## Data Dictionary

### bootstrap/

Bootstrap validation results for key observations (10,000 iterations each).

#### bootstrap_obs2_results.json

Observation 2: Style variance exceeds provider variance.

| Field | Type | Description |
|-------|------|-------------|
| `bootstrap_mean` | float | Mean R² across bootstrap samples (0-1) |
| `confidence_interval_lower` | float | 2.5th percentile (95% CI lower bound) |
| `confidence_interval_upper` | float | 97.5th percentile (95% CI upper bound) |
| `n_samples` | int | Bootstrap iterations (10000) |
| `observation` | string | "obs2" |
| `hypothesis` | string | Hypothesis being tested |

#### bootstrap_obs5_results.json

Observation 5: Shift magnitude declines with loop.

| Field | Type | Description |
|-------|------|-------------|
| `bootstrap_mean` | float | Mean decline rate |
| `confidence_interval_lower` | float | 2.5th percentile |
| `confidence_interval_upper` | float | 97.5th percentile |
| `n_samples` | int | Bootstrap iterations (10000) |
| `loop_coefficients` | array | Per-loop regression coefficients |

#### bootstrap_obs8_results.json

Observation 8: Shape-scale correlations.

| Field | Type | Description |
|-------|------|-------------|
| `correlation` | float | Pearson correlation coefficient |
| `p_value` | float | Statistical significance |
| `confidence_interval` | array | [lower, upper] bounds |

---

### permanova/

PERMANOVA (Permutational Multivariate Analysis of Variance) results.

#### permanova_results_fixed.csv

Overall variance decomposition.

| Column | Type | Description |
|--------|------|-------------|
| `factor` | string | "style" \| "provider" \| "category" |
| `R2` | float | Variance explained (0-1) |
| `F_statistic` | float | F-test statistic |
| `p_value` | float | Permutation p-value |

#### permanova_results_fixed_loop7.csv

Loop 7 (final iteration) variance decomposition.

Same schema as above, restricted to loop 7 data.

---

### variance/

#### variance_ratio_by_loop.csv

How variance explained evolves across reflection loops.

| Column | Type | Description |
|--------|------|-------------|
| `loop` | int | Reflection loop number (0-7) |
| `style_r2` | float | Style variance explained (R²) |
| `provider_r2` | float | Provider variance explained (R²) |
| `category_r2` | float | Category variance explained (R²) |

**Key finding**: Style R² increases from ~10% (loop 0) to ~17% (loop 7), while provider R² remains stable at ~10%.

---

### trajectory/

#### trajectory_convergence.csv

Trajectory convergence metrics per experiment.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Unique experiment identifier |
| `provider` | string | Model provider (anthropic, openai, google, xai, deepseek) |
| `style` | string | Reflection style |
| `convergence_loop` | int | Loop where trajectory stabilized |
| `final_radius` | float | Final attractor basin radius |

#### curvature_evolution.csv

Trajectory curvature across loops.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Unique experiment identifier |
| `loop` | int | Reflection loop (0-7) |
| `curvature` | float | Local trajectory curvature |
| `direction_change` | float | Angular change from previous loop |

---

### manifold/

#### umap_coordinates.csv

2D UMAP projection coordinates for visualization.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Unique experiment identifier |
| `loop` | int | Reflection loop (0-7) |
| `umap_x` | float | UMAP dimension 1 |
| `umap_y` | float | UMAP dimension 2 |
| `provider` | string | Model provider |
| `style` | string | Reflection style |

#### attractor_basins.csv

Identified attractor basin centers and radii.

| Column | Type | Description |
|--------|------|-------------|
| `basin_id` | int | Attractor basin identifier |
| `center_x` | float | Basin center (UMAP dim 1) |
| `center_y` | float | Basin center (UMAP dim 2) |
| `radius` | float | Basin radius |
| `dominant_style` | string | Most common style in basin |
| `n_trajectories` | int | Trajectories converging to basin |

---

## Relationship to Paper

| Paper Section | Data File(s) |
|---------------|--------------|
| Table 1: Variance decomposition | `permanova/*.csv`, `variance/*.csv` |
| Figure 5: UMAP visualization | `manifold/umap_coordinates.csv` |
| Figure 6: Variance bars | `variance/variance_ratio_by_loop.csv` |
| Figure 7: Bootstrap CIs | `bootstrap/*.json` |

---

## Usage Example

```python
import pandas as pd
import json

# Load variance data
variance = pd.read_csv('variance/variance_ratio_by_loop.csv')
print(variance[['loop', 'style_r2', 'provider_r2']])

# Load bootstrap results
with open('bootstrap/bootstrap_obs2_results.json') as f:
    obs2 = json.load(f)
print(f"Style R² = {obs2['bootstrap_mean']:.3f} [{obs2['confidence_interval_lower']:.3f}, {obs2['confidence_interval_upper']:.3f}]")
```

---

*See [scripts/](../scripts/) for analysis code.*
