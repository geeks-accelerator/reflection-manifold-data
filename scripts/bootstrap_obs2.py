#!/usr/bin/env python3
"""
Bootstrap validation for Observation 2: Manifold Style Topology

Tests whether style clustering patterns persist across 1000 bootstrap resamples:
- H1: Strong variance in pairwise distances (68% range)
- H2: Style variance > provider variance (style clustering stronger)

METHODOLOGY FIX (2025-11-29):
- Original bootstrap used curvature 3D space → 19% variance (WRONG)
- Corrected bootstrap uses UMAP 2D space → should match original 68% (CORRECT)
- Now validates exact same metric as original observation

Bootstrap methodology:
1. Load UMAP coordinates (1 row per experiment, not per loop)
2. Resample all experiments with replacement (1000 iterations)
3. For each sample: compute style-level pairwise distances in UMAP 2D space
4. Calculate 95% confidence intervals
5. Test: Do patterns persist across bootstrap samples?
"""

import argparse
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import json
from pathlib import Path
from tqdm import tqdm

# Bootstrap parameters
N_BOOTSTRAP = 1000  # Restored from 100 (now fast with 2D distances)
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95

# Output path (fixed - validation data goes to data/validation/)
OUTPUT_PATH = Path("data/validation/bootstrap_obs2_results.json")

# Original findings (from Obs 2)
ORIGINAL_VARIANCE_PCT = 68  # (6.04 - 3.59) / 3.59 = 0.68
ORIGINAL_STYLE_VARIANCE = 68
ORIGINAL_PROVIDER_VARIANCE = 40  # from original Obs 2 analysis

def load_data(batch_id: str):
    """Load UMAP coordinates and aggregate to per-experiment means."""
    umap_path = Path(f"analysis/figures/{batch_id}/umap_coordinates.csv")
    print(f"Loading UMAP coordinates from {umap_path}...")
    df = pd.read_csv(umap_path)
    n_rows = len(df)
    n_experiments = len(df['run_id'].unique())
    print(f"Loaded {n_rows} rows ({n_experiments} experiments × ~{n_rows/n_experiments:.1f} loops)")

    # Verify expected columns
    required_cols = ['run_id', 'provider', 'reflection_style', 'category', 'umap_x', 'umap_y']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Aggregate to per-experiment means (fixes pseudo-replication issue)
    print(f"Aggregating to per-experiment means...")
    agg_df = df.groupby(['run_id', 'provider', 'reflection_style', 'category']).agg({
        'umap_x': 'mean',
        'umap_y': 'mean'
    }).reset_index()
    print(f"  Result: {len(agg_df)} experiments (1 row per experiment)")

    return agg_df

def compute_pairwise_distances_umap(df, groupby_col, subsample_size=None):
    """Compute mean pairwise distances using UMAP 2D coordinates."""
    distances = {}
    for group_name in df[groupby_col].unique():
        group_data = df[df[groupby_col] == group_name]

        # UMAP 2D coordinates (not 3D curvature metrics)
        coords = group_data[['umap_x', 'umap_y']].values

        # Subsample if requested (for speed)
        if subsample_size and len(coords) > subsample_size:
            indices = np.random.choice(len(coords), size=subsample_size, replace=False)
            coords = coords[indices]

        # Vectorized pairwise distances
        if len(coords) > 1:
            dists = pdist(coords, metric='euclidean')
            distances[group_name] = np.mean(dists)
        else:
            distances[group_name] = 0.0

    return distances

def compute_style_metrics(df):
    """Compute style clustering metrics in UMAP space."""
    # Compute pairwise distances for styles
    style_distances = compute_pairwise_distances_umap(df, 'reflection_style')

    # Compute variance (range / min)
    dist_values = list(style_distances.values())
    if len(dist_values) < 2:
        return None

    min_dist = np.min(dist_values)
    max_dist = np.max(dist_values)

    # Avoid division by zero
    if min_dist == 0:
        variance_pct = 0.0
    else:
        variance_pct = ((max_dist - min_dist) / min_dist) * 100

    # Compute provider variance for comparison
    provider_distances = compute_pairwise_distances_umap(df, 'provider')
    provider_dist_values = list(provider_distances.values())
    provider_min = np.min(provider_dist_values)
    provider_max = np.max(provider_dist_values)

    if provider_min == 0:
        provider_variance_pct = 0.0
    else:
        provider_variance_pct = ((provider_max - provider_min) / provider_min) * 100

    return {
        'style_variance_pct': variance_pct,
        'provider_variance_pct': provider_variance_pct,
        'style_distances': style_distances,
        'provider_distances': provider_distances,
    }

def bootstrap_resample_experiments(df, random_state):
    """Resample experiments with replacement (1 row per experiment)."""
    n_experiments = len(df)
    # Sample experiment indices with replacement
    resampled_indices = random_state.choice(n_experiments, size=n_experiments, replace=True)
    return df.iloc[resampled_indices].reset_index(drop=True)

def run_bootstrap(batch_id: str):
    """Run bootstrap validation."""
    # Load data
    df = load_data(batch_id)

    # Compute original metrics (sanity check)
    print("\nOriginal metrics (sanity check):")
    original = compute_style_metrics(df)
    if original is None:
        raise ValueError("Failed to compute original metrics")

    print(f"  Style variance:           {original['style_variance_pct']:.1f}%")
    print(f"  Provider variance:        {original['provider_variance_pct']:.1f}%")
    print(f"  Style > Provider:         {original['style_variance_pct'] > original['provider_variance_pct']}")
    print(f"  Expected style variance:  {ORIGINAL_VARIANCE_PCT}%")

    # Initialize random state
    random_state = np.random.RandomState(RANDOM_SEED)

    # Bootstrap iterations
    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations...")
    bootstrap_results = []

    for i in tqdm(range(N_BOOTSTRAP)):
        # Resample experiments
        resampled_df = bootstrap_resample_experiments(df, random_state)

        # Compute metrics
        metrics = compute_style_metrics(resampled_df)
        if metrics is not None:
            bootstrap_results.append(metrics)

    if len(bootstrap_results) == 0:
        raise ValueError("No valid bootstrap samples")

    # Extract distributions
    style_variance_dist = np.array([r['style_variance_pct'] for r in bootstrap_results])
    provider_variance_dist = np.array([r['provider_variance_pct'] for r in bootstrap_results])

    # Calculate 95% confidence intervals
    alpha = 1 - CONFIDENCE_LEVEL
    style_variance_ci = (
        np.percentile(style_variance_dist, alpha/2 * 100),
        np.percentile(style_variance_dist, (1 - alpha/2) * 100)
    )
    provider_variance_ci = (
        np.percentile(provider_variance_dist, alpha/2 * 100),
        np.percentile(provider_variance_dist, (1 - alpha/2) * 100)
    )

    # Test acceptance criteria
    # H1: Style variance ≥ 60% (strong separation)
    h1_samples_pass = np.sum(style_variance_dist >= 60)
    h1_pass_rate = h1_samples_pass / len(bootstrap_results)
    h1_pass = h1_pass_rate >= 0.95

    # H2: Style variance > provider variance
    h2_samples_pass = np.sum(style_variance_dist > provider_variance_dist)
    h2_pass_rate = h2_samples_pass / len(bootstrap_results)
    h2_pass = h2_pass_rate >= 0.95

    # Effect size stability
    style_variance_within_20pct = np.sum(
        (style_variance_dist >= ORIGINAL_VARIANCE_PCT * 0.8) &
        (style_variance_dist <= ORIGINAL_VARIANCE_PCT * 1.2)
    ) / len(bootstrap_results)

    # Prepare results
    results = {
        'observation': 'Obs 2: Manifold Style Topology',
        'n_bootstrap': len(bootstrap_results),
        'confidence_level': CONFIDENCE_LEVEL,
        'note': 'CORRECTED: Uses UMAP 2D coordinates (exact same metric as original observation)',
        'original_findings': {
            'style_variance_pct': ORIGINAL_VARIANCE_PCT,
            'provider_variance_pct': ORIGINAL_PROVIDER_VARIANCE,
        },
        'original_computed': {
            'style_variance_pct': float(original['style_variance_pct']),
            'provider_variance_pct': float(original['provider_variance_pct']),
        },
        'bootstrap_distributions': {
            'style_variance_pct': {
                'mean': float(np.mean(style_variance_dist)),
                'std': float(np.std(style_variance_dist)),
                'ci_lower': float(style_variance_ci[0]),
                'ci_upper': float(style_variance_ci[1]),
                'min': float(np.min(style_variance_dist)),
                'max': float(np.max(style_variance_dist)),
            },
            'provider_variance_pct': {
                'mean': float(np.mean(provider_variance_dist)),
                'std': float(np.std(provider_variance_dist)),
                'ci_lower': float(provider_variance_ci[0]),
                'ci_upper': float(provider_variance_ci[1]),
            },
        },
        'hypothesis_tests': {
            'h1_style_variance_gte_60pct': {
                'threshold': 60,
                'samples_pass': int(h1_samples_pass),
                'pass_rate': float(h1_pass_rate),
                'result': 'PASS' if h1_pass else 'FAIL',
            },
            'h2_style_gt_provider_variance': {
                'samples_pass': int(h2_samples_pass),
                'pass_rate': float(h2_pass_rate),
                'result': 'PASS' if h2_pass else 'FAIL',
            },
        },
        'effect_size_stability': {
            'style_variance_within_20pct': float(style_variance_within_20pct),
        },
        'n3_validation': {
            'all_hypotheses_pass': bool(h1_pass and h2_pass),
            'effect_sizes_stable': bool(style_variance_within_20pct >= 0.95),
            'promoted_to_n3': bool((h1_pass and h2_pass) and (style_variance_within_20pct >= 0.95)),
        }
    }

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("BOOTSTRAP VALIDATION RESULTS - Observation 2 (CORRECTED)")
    print(f"{'='*80}")

    print(f"\n[H1] Style variance ≥ 60% (strong separation):")
    print(f"  Original:           {ORIGINAL_VARIANCE_PCT}%")
    print(f"  Bootstrap mean:     {results['bootstrap_distributions']['style_variance_pct']['mean']:.1f}%")
    print(f"  95% CI:             [{style_variance_ci[0]:.1f}%, {style_variance_ci[1]:.1f}%]")
    print(f"  Samples ≥60%:       {h1_samples_pass}/{len(bootstrap_results)} ({h1_pass_rate*100:.1f}%)")
    print(f"  Result:             {'✅ PASS' if h1_pass else '❌ FAIL'}")

    print(f"\n[H2] Style > Provider variance:")
    print(f"  Style mean:         {results['bootstrap_distributions']['style_variance_pct']['mean']:.1f}%")
    print(f"  Provider mean:      {results['bootstrap_distributions']['provider_variance_pct']['mean']:.1f}%")
    print(f"  Samples style>prov: {h2_samples_pass}/{len(bootstrap_results)} ({h2_pass_rate*100:.1f}%)")
    print(f"  Result:             {'✅ PASS' if h2_pass else '❌ FAIL'}")

    print(f"\n[Effect Size Stability]")
    print(f"  Style variance within ±20%:  {style_variance_within_20pct*100:.1f}%")

    print(f"\n{'='*80}")
    print(f"N≥3 VALIDATION: {'✅ PROMOTED' if results['n3_validation']['promoted_to_n3'] else '❌ NOT PROMOTED'}")
    print(f"{'='*80}")

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for Observation 2: Manifold Style Topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 scripts/bootstrap_obs2.py --batch-id full-scale-2025-11-20-v2
        """
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="Batch ID (e.g., full-scale-2025-11-20-v2)"
    )
    args = parser.parse_args()

    results = run_bootstrap(args.batch_id)
