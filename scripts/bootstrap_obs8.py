#!/usr/bin/env python3
"""
Bootstrap validation for Observation 8: Curvature Metric Correlations

Includes pseudo-replication fix (aggregates to per-experiment means before bootstrap)
and performance optimizations (vectorized resampling, pre-extracted numpy arrays).

Tests whether correlation patterns persist across 1000 bootstrap resamples:
- H1: Shift ↔ Effort correlation r ≥ 0.8 (original: r=0.862)
- H2: Variance ↔ Distance correlation r ≥ 0.8 (original: r=0.860)
- H3: Variance ↔ Shift near-zero |r| ≤ 0.2 (independence claim)
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import json
from pathlib import Path
from tqdm import tqdm

# Bootstrap parameters
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95

# Output path (fixed - validation data goes to data/validation/)
OUTPUT_PATH = Path("data/validation/bootstrap_obs8_results.json")

# Original findings (from Obs 8)
ORIGINAL_SHIFT_EFFORT = 0.862
ORIGINAL_VARIANCE_DISTANCE = 0.860
ORIGINAL_VARIANCE_SHIFT = 0.14  # Near-zero (independence)

def load_data(batch_id: str):
    """Load curvature evolution data and aggregate to per-experiment means."""
    data_path = Path(f"analysis/figures/{batch_id}/curvature_evolution.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    n_rows = len(df)
    n_experiments = len(df['run_id'].unique())
    print(f"Loaded {n_rows} rows ({n_experiments} experiments × ~{n_rows/n_experiments:.1f} loops)")

    # CRITICAL FIX: Aggregate to per-experiment means (not per-loop)
    # This fixes pseudo-replication issue (Issue #9 from Codex review)
    print(f"Aggregating {n_rows} rows to {n_experiments} per-experiment means...")
    experiment_df = df.groupby('run_id').agg({
        'cosine_shift': 'mean',
        'cognitive_effort': 'mean',
        'semantic_variance': 'mean',
        'distance_from_origin': 'mean'
    }).reset_index()

    print(f"  Result: {len(experiment_df)} experiments (correct sample size for correlations)")

    # Extract metric columns as numpy arrays
    metrics = {
        'run_ids': experiment_df['run_id'].values,
        'shift': experiment_df['cosine_shift'].values,
        'effort': experiment_df['cognitive_effort'].values,
        'variance': experiment_df['semantic_variance'].values,
        'distance': experiment_df['distance_from_origin'].values,
    }

    # Experiment IDs for bootstrap resampling
    unique_ids = experiment_df['run_id'].values

    return metrics, unique_ids

def compute_correlations_from_arrays(shift, effort, variance, distance):
    """Compute correlations from numpy arrays."""
    shift_effort_r, shift_effort_p = pearsonr(shift, effort)
    variance_distance_r, variance_distance_p = pearsonr(variance, distance)
    variance_shift_r, variance_shift_p = pearsonr(variance, shift)

    return {
        'shift_effort_r': shift_effort_r,
        'shift_effort_p': shift_effort_p,
        'variance_distance_r': variance_distance_r,
        'variance_distance_p': variance_distance_p,
        'variance_shift_r': variance_shift_r,
        'variance_shift_p': variance_shift_p,
    }

def bootstrap_resample_indices(unique_ids, random_state):
    """Resample experiment indices with replacement."""
    # Sample experiments (now we have 1 row per experiment, so just sample row indices)
    n_experiments = len(unique_ids)
    resampled_indices = random_state.choice(n_experiments, size=n_experiments, replace=True)

    return resampled_indices

def run_bootstrap(batch_id: str):
    """Run bootstrap validation."""
    # Load data (now aggregated to per-experiment)
    metrics, unique_ids = load_data(batch_id)

    # Compute original correlations (sanity check)
    print("\nOriginal correlations (sanity check):")
    original = compute_correlations_from_arrays(
        metrics['shift'], metrics['effort'],
        metrics['variance'], metrics['distance']
    )
    print(f"  Shift ↔ Effort:       r={original['shift_effort_r']:.4f}, p={original['shift_effort_p']:.4e}")
    print(f"  Variance ↔ Distance:  r={original['variance_distance_r']:.4f}, p={original['variance_distance_p']:.4e}")
    print(f"  Variance ↔ Shift:     r={original['variance_shift_r']:.4f}, p={original['variance_shift_p']:.4e}")

    # Initialize random state
    random_state = np.random.RandomState(RANDOM_SEED)

    # Bootstrap iterations
    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations...")
    bootstrap_results = []

    for i in tqdm(range(N_BOOTSTRAP)):
        # Resample indices (experiment-level, not loop-level)
        indices = bootstrap_resample_indices(unique_ids, random_state)

        # Extract resampled arrays
        shift_resampled = metrics['shift'][indices]
        effort_resampled = metrics['effort'][indices]
        variance_resampled = metrics['variance'][indices]
        distance_resampled = metrics['distance'][indices]

        # Compute correlations
        correlations = compute_correlations_from_arrays(
            shift_resampled, effort_resampled,
            variance_resampled, distance_resampled
        )
        bootstrap_results.append(correlations)

    # Convert to arrays for analysis
    shift_effort_dist = np.array([r['shift_effort_r'] for r in bootstrap_results])
    variance_distance_dist = np.array([r['variance_distance_r'] for r in bootstrap_results])
    variance_shift_dist = np.array([r['variance_shift_r'] for r in bootstrap_results])

    # Calculate 95% confidence intervals
    alpha = 1 - CONFIDENCE_LEVEL
    shift_effort_ci = (
        np.percentile(shift_effort_dist, alpha/2 * 100),
        np.percentile(shift_effort_dist, (1 - alpha/2) * 100)
    )
    variance_distance_ci = (
        np.percentile(variance_distance_dist, alpha/2 * 100),
        np.percentile(variance_distance_dist, (1 - alpha/2) * 100)
    )
    variance_shift_ci = (
        np.percentile(variance_shift_dist, alpha/2 * 100),
        np.percentile(variance_shift_dist, (1 - alpha/2) * 100)
    )

    # Test acceptance criteria
    # H1: Shift ↔ Effort r ≥ 0.8
    h1_samples_pass = np.sum(shift_effort_dist >= 0.8)
    h1_pass_rate = h1_samples_pass / N_BOOTSTRAP
    h1_pass = h1_pass_rate >= 0.95

    # H2: Variance ↔ Distance r ≥ 0.8
    h2_samples_pass = np.sum(variance_distance_dist >= 0.8)
    h2_pass_rate = h2_samples_pass / N_BOOTSTRAP
    h2_pass = h2_pass_rate >= 0.95

    # H3: Variance ↔ Shift independence |r| ≤ 0.2
    h3_samples_pass = np.sum(np.abs(variance_shift_dist) <= 0.2)
    h3_pass_rate = h3_samples_pass / N_BOOTSTRAP
    h3_pass = h3_pass_rate >= 0.95

    # Effect size stability (±20% of original)
    shift_effort_within_20pct = np.sum(
        (shift_effort_dist >= ORIGINAL_SHIFT_EFFORT * 0.8) &
        (shift_effort_dist <= ORIGINAL_SHIFT_EFFORT * 1.2)
    ) / N_BOOTSTRAP

    variance_distance_within_20pct = np.sum(
        (variance_distance_dist >= ORIGINAL_VARIANCE_DISTANCE * 0.8) &
        (variance_distance_dist <= ORIGINAL_VARIANCE_DISTANCE * 1.2)
    ) / N_BOOTSTRAP

    # Prepare results
    results = {
        'observation': 'Obs 8: Curvature Metric Correlations',
        'n_bootstrap': N_BOOTSTRAP,
        'confidence_level': CONFIDENCE_LEVEL,
        'original_findings': {
            'shift_effort_r': ORIGINAL_SHIFT_EFFORT,
            'variance_distance_r': ORIGINAL_VARIANCE_DISTANCE,
            'variance_shift_r': ORIGINAL_VARIANCE_SHIFT,
        },
        'original_computed': original,
        'bootstrap_distributions': {
            'shift_effort': {
                'mean': float(np.mean(shift_effort_dist)),
                'std': float(np.std(shift_effort_dist)),
                'ci_lower': float(shift_effort_ci[0]),
                'ci_upper': float(shift_effort_ci[1]),
                'min': float(np.min(shift_effort_dist)),
                'max': float(np.max(shift_effort_dist)),
            },
            'variance_distance': {
                'mean': float(np.mean(variance_distance_dist)),
                'std': float(np.std(variance_distance_dist)),
                'ci_lower': float(variance_distance_ci[0]),
                'ci_upper': float(variance_distance_ci[1]),
                'min': float(np.min(variance_distance_dist)),
                'max': float(np.max(variance_distance_dist)),
            },
            'variance_shift': {
                'mean': float(np.mean(variance_shift_dist)),
                'std': float(np.std(variance_shift_dist)),
                'ci_lower': float(variance_shift_ci[0]),
                'ci_upper': float(variance_shift_ci[1]),
                'min': float(np.min(variance_shift_dist)),
                'max': float(np.max(variance_shift_dist)),
            },
        },
        'hypothesis_tests': {
            'h1_shift_effort_r_gte_0.8': {
                'threshold': 0.8,
                'samples_pass': int(h1_samples_pass),
                'pass_rate': float(h1_pass_rate),
                'result': 'PASS' if h1_pass else 'FAIL',
            },
            'h2_variance_distance_r_gte_0.8': {
                'threshold': 0.8,
                'samples_pass': int(h2_samples_pass),
                'pass_rate': float(h2_pass_rate),
                'result': 'PASS' if h2_pass else 'FAIL',
            },
            'h3_variance_shift_independence_abs_r_lte_0.2': {
                'threshold': 0.2,
                'samples_pass': int(h3_samples_pass),
                'pass_rate': float(h3_pass_rate),
                'result': 'PASS' if h3_pass else 'FAIL',
            },
        },
        'effect_size_stability': {
            'shift_effort_within_20pct': float(shift_effort_within_20pct),
            'variance_distance_within_20pct': float(variance_distance_within_20pct),
        },
        'n3_validation': {
            'all_hypotheses_pass': bool(h1_pass and h2_pass and h3_pass),
            'effect_sizes_stable': bool((shift_effort_within_20pct >= 0.95) and (variance_distance_within_20pct >= 0.95)),
            'promoted_to_n3': bool((h1_pass and h2_pass and h3_pass) and
                            ((shift_effort_within_20pct >= 0.95) and (variance_distance_within_20pct >= 0.95))),
        }
    }

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("BOOTSTRAP VALIDATION RESULTS - Observation 8")
    print(f"{'='*80}")

    print(f"\n[H1] Shift ↔ Effort correlation r ≥ 0.8:")
    print(f"  Bootstrap mean:     r={results['bootstrap_distributions']['shift_effort']['mean']:.4f}")
    print(f"  95% CI:             [{shift_effort_ci[0]:.4f}, {shift_effort_ci[1]:.4f}]")
    print(f"  Samples ≥0.8:       {h1_samples_pass}/{N_BOOTSTRAP} ({h1_pass_rate*100:.1f}%)")
    print(f"  Result:             {'✅ PASS' if h1_pass else '❌ FAIL'}")

    print(f"\n[H2] Variance ↔ Distance correlation r ≥ 0.8:")
    print(f"  Bootstrap mean:     r={results['bootstrap_distributions']['variance_distance']['mean']:.4f}")
    print(f"  95% CI:             [{variance_distance_ci[0]:.4f}, {variance_distance_ci[1]:.4f}]")
    print(f"  Samples ≥0.8:       {h2_samples_pass}/{N_BOOTSTRAP} ({h2_pass_rate*100:.1f}%)")
    print(f"  Result:             {'✅ PASS' if h2_pass else '❌ FAIL'}")

    print(f"\n[H3] Variance ↔ Shift independence |r| ≤ 0.2:")
    print(f"  Bootstrap mean:     r={results['bootstrap_distributions']['variance_shift']['mean']:.4f}")
    print(f"  95% CI:             [{variance_shift_ci[0]:.4f}, {variance_shift_ci[1]:.4f}]")
    print(f"  Samples |r|≤0.2:    {h3_samples_pass}/{N_BOOTSTRAP} ({h3_pass_rate*100:.1f}%)")
    print(f"  Result:             {'✅ PASS' if h3_pass else '❌ FAIL'}")

    print(f"\n[Effect Size Stability]")
    print(f"  Shift↔Effort within ±20%:      {shift_effort_within_20pct*100:.1f}%")
    print(f"  Variance↔Distance within ±20%: {variance_distance_within_20pct*100:.1f}%")

    print(f"\n{'='*80}")
    print(f"N≥3 VALIDATION: {'✅ PROMOTED' if results['n3_validation']['promoted_to_n3'] else '❌ NOT PROMOTED'}")
    print(f"{'='*80}")

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for Observation 8: Curvature Metric Correlations (FAST VERSION)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 scripts/bootstrap_obs8_fast.py --batch-id full-scale-2025-11-20-v2
        """
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="Batch ID (e.g., full-scale-2025-11-20-v2)"
    )
    args = parser.parse_args()

    results = run_bootstrap(args.batch_id)
