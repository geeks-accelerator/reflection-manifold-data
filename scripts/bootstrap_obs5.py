#!/usr/bin/env python3
"""
Bootstrap validation for Observation 5: Reflection Stabilization Dynamics

Tests whether stabilization patterns persist across 1000 bootstrap resamples:
- H1: Two-phase dynamics (Loop 0→1 spike, Loop 1→7 decline in shift/effort)
- H2: Variance growth (monotonic increase 0.02→0.11, NOT convergence)
- H3: Provider ranking stability (DeepSeek fastest, Moonshot slowest)

Bootstrap methodology:
1. Resample all experiments with replacement (1000 iterations)
2. For each sample: compute provider-level loop means
3. Calculate 95% confidence intervals
4. Test: Do patterns persist across bootstrap samples?

Acceptance criteria (N≥3 promotion):
- ✅ Pattern persists across ≥95% of bootstrap samples (p < 0.05)
- ✅ Effect size remains similar (±20% of original estimate)
- ✅ Provider rankings stable (no flips in bootstrap samples)
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json
from pathlib import Path
from tqdm import tqdm

# Bootstrap parameters
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95

# Output path (fixed - validation data goes to data/validation/)
OUTPUT_PATH = Path("data/validation/bootstrap_obs5_results.json")

# Original findings (from Obs 5)
# Note: Provider names from CSV are lowercase, 'moonshot' = 'moonshotai'
ORIGINAL_PROVIDER_RANKING = ['deepseek', 'xai', 'google', 'anthropic', 'openai', 'moonshotai']
ORIGINAL_STABILIZATION_LOOPS = {
    'deepseek': 2.4,
    'xai': 3.6,
    'google': 3.5,
    'anthropic': 4.6,
    'openai': 5.1,
    'moonshotai': 5.3,
}

def load_data(batch_id: str):
    """Load curvature evolution data."""
    data_path = Path(f"analysis/figures/{batch_id}/curvature_evolution.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    n_experiments = len(df) / 8
    print(f"Loaded {len(df)} rows ({n_experiments:.0f} experiments × 8 loops)")

    # Normalize provider names to lowercase for consistency
    df['provider'] = df['provider'].str.lower()

    return df

def compute_stabilization_patterns(df):
    """Compute key patterns from Obs 5.

    ISSUE #5 FIX: H1/H2 now test per-provider patterns (not aggregated).
    Original observation claimed "per provider" dynamics, so we count how many
    providers show the pattern, rather than averaging across all providers.
    """

    # Group by provider and loop
    provider_loop_means = df.groupby(['provider', 'loop_number']).agg({
        'cosine_shift': 'mean',
        'cognitive_effort': 'mean',
        'semantic_variance': 'mean',
        'distance_from_origin': 'mean',
    }).reset_index()

    # H1/H2: Compute per-provider ratios (FIX #5: not aggregated)
    provider_shift_ratios = {}
    provider_variance_ratios = {}

    for provider in df['provider'].unique():
        provider_data = provider_loop_means[provider_loop_means['provider'] == provider]

        # H1: Shift decline (Loop 1 → Loop 7)
        loop1_shift = provider_data[provider_data['loop_number'] == 1]['cosine_shift'].values
        loop7_shift = provider_data[provider_data['loop_number'] == 7]['cosine_shift'].values

        if len(loop1_shift) > 0 and len(loop7_shift) > 0:
            provider_shift_ratios[provider] = loop7_shift[0] / loop1_shift[0]
        else:
            provider_shift_ratios[provider] = np.nan

        # H2: Variance growth (Loop 1 → Loop 7)
        loop1_variance = provider_data[provider_data['loop_number'] == 1]['semantic_variance'].values
        loop7_variance = provider_data[provider_data['loop_number'] == 7]['semantic_variance'].values

        if len(loop1_variance) > 0 and len(loop7_variance) > 0:
            provider_variance_ratios[provider] = loop7_variance[0] / loop1_variance[0]
        else:
            provider_variance_ratios[provider] = np.nan

    # H1: How many providers show shift decline (ratio < 1.0)?
    providers_with_decline = sum(1 for r in provider_shift_ratios.values() if not np.isnan(r) and r < 1.0)

    # H2: How many providers show variance growth (ratio > 1.0)?
    providers_with_growth = sum(1 for r in provider_variance_ratios.values() if not np.isnan(r) and r > 1.0)

    # For backward compatibility, also compute aggregated means
    loop1_shift_mean = provider_loop_means[provider_loop_means['loop_number'] == 1]['cosine_shift'].mean()
    loop7_shift_mean = provider_loop_means[provider_loop_means['loop_number'] == 7]['cosine_shift'].mean()
    shift_decline_ratio_mean = loop7_shift_mean / loop1_shift_mean

    loop1_variance_mean = provider_loop_means[provider_loop_means['loop_number'] == 1]['semantic_variance'].mean()
    loop7_variance_mean = provider_loop_means[provider_loop_means['loop_number'] == 7]['semantic_variance'].mean()
    variance_growth_ratio_mean = loop7_variance_mean / loop1_variance_mean

    # H3: Provider ranking (stabilization loop, approximated by mean shift Loops 2-4)
    # Lower mean shift = earlier stabilization
    provider_stabilization = {}
    for provider in provider_loop_means['provider'].unique():
        provider_data = provider_loop_means[
            (provider_loop_means['provider'] == provider) &
            (provider_loop_means['loop_number'].between(2, 4))
        ]
        mean_shift = provider_data['cosine_shift'].mean()
        provider_stabilization[provider] = mean_shift

    # Sort providers by stabilization speed (lower shift = faster)
    provider_ranking = sorted(provider_stabilization.keys(),
                            key=lambda p: provider_stabilization[p])

    # Compute Spearman correlation with original ranking
    try:
        original_indices = [ORIGINAL_PROVIDER_RANKING.index(p) for p in provider_ranking]
        bootstrap_indices = list(range(len(provider_ranking)))
        rank_correlation, rank_p = spearmanr(original_indices, bootstrap_indices)
    except ValueError:
        # Handle case where provider set doesn't match original
        rank_correlation = np.nan
        rank_p = np.nan

    return {
        # New per-provider metrics (FIX #5)
        'providers_with_decline': providers_with_decline,
        'providers_with_growth': providers_with_growth,
        'provider_shift_ratios': provider_shift_ratios,
        'provider_variance_ratios': provider_variance_ratios,
        # Backward compatibility (aggregated means)
        'shift_decline_ratio': shift_decline_ratio_mean,
        'variance_growth_ratio': variance_growth_ratio_mean,
        'loop1_shift': loop1_shift_mean,
        'loop7_shift': loop7_shift_mean,
        'loop1_variance': loop1_variance_mean,
        'loop7_variance': loop7_variance_mean,
        # H3 provider ranking (unchanged)
        'provider_ranking': provider_ranking,
        'provider_stabilization': provider_stabilization,
        'rank_correlation': rank_correlation,
        'rank_p': rank_p,
    }

def bootstrap_resample_indices(experiment_indices, unique_ids, random_state):
    """Resample experiment indices with replacement."""
    n_experiments = len(unique_ids)
    resampled_exp_ids = random_state.choice(unique_ids, size=n_experiments, replace=True)

    all_indices = []
    for exp_id in resampled_exp_ids:
        all_indices.extend(experiment_indices[exp_id])

    return np.array(all_indices)

def run_bootstrap(batch_id: str):
    """Run bootstrap validation."""
    # Load data
    df = load_data(batch_id)

    # Precompute experiment indices
    print("\nPrecomputing experiment groups...")
    unique_ids = df['run_id'].unique()
    experiment_indices = {}
    for exp_id in unique_ids:
        experiment_indices[exp_id] = df[df['run_id'] == exp_id].index.tolist()
    print(f"  Indexed {len(experiment_indices)} experiments")

    # Compute original patterns (sanity check)
    print("\nOriginal patterns (sanity check):")
    original = compute_stabilization_patterns(df)
    print(f"  Shift decline (Loop 1→7):      {original['loop1_shift']:.4f} → {original['loop7_shift']:.4f} (ratio: {original['shift_decline_ratio']:.4f})")
    print(f"  Providers with decline (ratio<1): {original['providers_with_decline']}/6")
    print(f"  Variance growth (Loop 1→7):    {original['loop1_variance']:.4f} → {original['loop7_variance']:.4f} (ratio: {original['variance_growth_ratio']:.4f})")
    print(f"  Providers with growth (ratio>1):  {original['providers_with_growth']}/6")
    print(f"  Provider ranking:              {original['provider_ranking']}")
    print(f"  Rank correlation with original: ρ={original['rank_correlation']:.4f}, p={original['rank_p']:.4e}")

    # Initialize random state
    random_state = np.random.RandomState(RANDOM_SEED)

    # Bootstrap iterations
    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations...")
    bootstrap_results = []

    for i in tqdm(range(N_BOOTSTRAP)):
        # Resample indices
        indices = bootstrap_resample_indices(experiment_indices, unique_ids, random_state)

        # Extract resampled dataframe
        resampled_df = df.iloc[indices].copy()

        # Compute patterns
        patterns = compute_stabilization_patterns(resampled_df)
        bootstrap_results.append(patterns)

    # Extract distributions
    shift_decline_dist = np.array([r['shift_decline_ratio'] for r in bootstrap_results])
    variance_growth_dist = np.array([r['variance_growth_ratio'] for r in bootstrap_results])
    # FIX #4: Keep NaN values (they count as failures)
    rank_correlation_dist = np.array([r['rank_correlation'] for r in bootstrap_results])
    # Extract per-provider counts (FIX #5)
    providers_decline_dist = np.array([r['providers_with_decline'] for r in bootstrap_results])
    providers_growth_dist = np.array([r['providers_with_growth'] for r in bootstrap_results])

    # Calculate 95% confidence intervals
    alpha = 1 - CONFIDENCE_LEVEL
    shift_decline_ci = (
        np.percentile(shift_decline_dist, alpha/2 * 100),
        np.percentile(shift_decline_dist, (1 - alpha/2) * 100)
    )
    variance_growth_ci = (
        np.percentile(variance_growth_dist, alpha/2 * 100),
        np.percentile(variance_growth_dist, (1 - alpha/2) * 100)
    )
    rank_correlation_ci = (
        np.percentile(rank_correlation_dist, alpha/2 * 100),
        np.percentile(rank_correlation_dist, (1 - alpha/2) * 100)
    )

    # Test acceptance criteria (FIXED #4 and #5)

    # H1: Shift decline per-provider (FIX #5: majority of providers show decline)
    # Require ≥5/6 providers to show ratio < 1.0
    h1_samples_pass = np.sum(providers_decline_dist >= 5)
    h1_pass_rate = h1_samples_pass / N_BOOTSTRAP
    h1_pass = h1_pass_rate >= 0.95

    # H2: Variance growth per-provider (FIX #5: majority of providers show growth)
    # Require ≥5/6 providers to show ratio > 1.0
    h2_samples_pass = np.sum(providers_growth_dist >= 5)
    h2_pass_rate = h2_samples_pass / N_BOOTSTRAP
    h2_pass = h2_pass_rate >= 0.95

    # H3: Provider ranking stability (FIX #4: NaN counts as failure)
    # rank_correlation_dist includes NaN, and NaN >= 0.8 evaluates to False
    h3_samples_pass = np.sum(rank_correlation_dist >= 0.8)
    h3_pass_rate = h3_samples_pass / N_BOOTSTRAP  # FIX #4: Use N_BOOTSTRAP, not filtered length
    h3_pass = h3_pass_rate >= 0.95

    # Effect size stability
    shift_decline_within_20pct = np.sum(
        (shift_decline_dist >= original['shift_decline_ratio'] * 0.8) &
        (shift_decline_dist <= original['shift_decline_ratio'] * 1.2)
    ) / N_BOOTSTRAP

    variance_growth_within_20pct = np.sum(
        (variance_growth_dist >= original['variance_growth_ratio'] * 0.8) &
        (variance_growth_dist <= original['variance_growth_ratio'] * 1.2)
    ) / N_BOOTSTRAP

    # Prepare results
    results = {
        'observation': 'Obs 5: Reflection Stabilization Dynamics',
        'n_bootstrap': N_BOOTSTRAP,
        'confidence_level': CONFIDENCE_LEVEL,
        'original_findings': {
            'shift_decline_ratio': float(original['shift_decline_ratio']),
            'variance_growth_ratio': float(original['variance_growth_ratio']),
            'provider_ranking': original['provider_ranking'],
            'expected_ranking': ORIGINAL_PROVIDER_RANKING,
        },
        'original_computed': {
            'loop1_shift': float(original['loop1_shift']),
            'loop7_shift': float(original['loop7_shift']),
            'loop1_variance': float(original['loop1_variance']),
            'loop7_variance': float(original['loop7_variance']),
            'rank_correlation': float(original['rank_correlation']) if not np.isnan(original['rank_correlation']) else None,
        },
        'bootstrap_distributions': {
            'shift_decline_ratio': {
                'mean': float(np.mean(shift_decline_dist)),
                'std': float(np.std(shift_decline_dist)),
                'ci_lower': float(shift_decline_ci[0]),
                'ci_upper': float(shift_decline_ci[1]),
                'min': float(np.min(shift_decline_dist)),
                'max': float(np.max(shift_decline_dist)),
            },
            'variance_growth_ratio': {
                'mean': float(np.mean(variance_growth_dist)),
                'std': float(np.std(variance_growth_dist)),
                'ci_lower': float(variance_growth_ci[0]),
                'ci_upper': float(variance_growth_ci[1]),
                'min': float(np.min(variance_growth_dist)),
                'max': float(np.max(variance_growth_dist)),
            },
            'rank_correlation': {
                'mean': float(np.mean(rank_correlation_dist)),
                'std': float(np.std(rank_correlation_dist)),
                'ci_lower': float(rank_correlation_ci[0]),
                'ci_upper': float(rank_correlation_ci[1]),
                'min': float(np.min(rank_correlation_dist)),
                'max': float(np.max(rank_correlation_dist)),
            },
        },
        'hypothesis_tests': {
            'h1_shift_decline_per_provider': {
                'description': 'Majority of providers (≥5/6) show shift decline (ratio < 1.0)',
                'threshold_providers': 5,
                'total_providers': 6,
                'samples_pass': int(h1_samples_pass),
                'pass_rate': float(h1_pass_rate),
                'result': 'PASS' if h1_pass else 'FAIL',
                'note': 'FIX #5: Changed from aggregated mean to per-provider majority test',
            },
            'h2_variance_growth_per_provider': {
                'description': 'Majority of providers (≥5/6) show variance growth (ratio > 1.0)',
                'threshold_providers': 5,
                'total_providers': 6,
                'samples_pass': int(h2_samples_pass),
                'pass_rate': float(h2_pass_rate),
                'result': 'PASS' if h2_pass else 'FAIL',
                'note': 'FIX #5: Changed from aggregated mean to per-provider majority test',
            },
            'h3_provider_ranking_correlation_gte_0.8': {
                'threshold': 0.8,
                'direction': 'greater_than_or_equal',
                'samples_pass': int(h3_samples_pass),
                'pass_rate': float(h3_pass_rate),
                'result': 'PASS' if h3_pass else 'FAIL',
                'note': 'FIX #4: NaN values now count as failures (not excluded from denominator)',
            },
        },
        'effect_size_stability': {
            'shift_decline_within_20pct': float(shift_decline_within_20pct),
            'variance_growth_within_20pct': float(variance_growth_within_20pct),
        },
        'n3_validation': {
            'all_hypotheses_pass': bool(h1_pass and h2_pass and h3_pass),
            'effect_sizes_stable': bool((shift_decline_within_20pct >= 0.95) and (variance_growth_within_20pct >= 0.95)),
            'promoted_to_n3': bool((h1_pass and h2_pass and h3_pass) and
                                ((shift_decline_within_20pct >= 0.95) and (variance_growth_within_20pct >= 0.95))),
        }
    }

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("BOOTSTRAP VALIDATION RESULTS - Observation 5")
    print(f"{'='*80}")

    print(f"\n[H1] Two-phase dynamics (shift decline Loop 1→7) - PER-PROVIDER TEST (FIX #5):")
    print(f"  Original:                 {original['loop1_shift']:.4f} → {original['loop7_shift']:.4f} (mean ratio: {original['shift_decline_ratio']:.4f})")
    print(f"  Original providers:       {original['providers_with_decline']}/6 show decline")
    print(f"  Bootstrap mean ratio:     {results['bootstrap_distributions']['shift_decline_ratio']['mean']:.4f}")
    print(f"  95% CI (mean ratio):      [{shift_decline_ci[0]:.4f}, {shift_decline_ci[1]:.4f}]")
    print(f"  Samples ≥5/6 providers:   {h1_samples_pass}/{N_BOOTSTRAP} ({h1_pass_rate*100:.1f}%)")
    print(f"  Result:                   {'✅ PASS' if h1_pass else '❌ FAIL'}")

    print(f"\n[H2] Variance growth (NOT convergence, Loop 1→7) - PER-PROVIDER TEST (FIX #5):")
    print(f"  Original:                 {original['loop1_variance']:.4f} → {original['loop7_variance']:.4f} (mean ratio: {original['variance_growth_ratio']:.4f})")
    print(f"  Original providers:       {original['providers_with_growth']}/6 show growth")
    print(f"  Bootstrap mean ratio:     {results['bootstrap_distributions']['variance_growth_ratio']['mean']:.4f}")
    print(f"  95% CI (mean ratio):      [{variance_growth_ci[0]:.4f}, {variance_growth_ci[1]:.4f}]")
    print(f"  Samples ≥5/6 providers:   {h2_samples_pass}/{N_BOOTSTRAP} ({h2_pass_rate*100:.1f}%)")
    print(f"  Result:                   {'✅ PASS' if h2_pass else '❌ FAIL'}")

    print(f"\n[H3] Provider ranking stability (FIX #4: NaN counts as failure):")
    print(f"  Expected:                 {ORIGINAL_PROVIDER_RANKING}")
    print(f"  Original:                 {original['provider_ranking']}")
    print(f"  Bootstrap mean ρ:         {results['bootstrap_distributions']['rank_correlation']['mean']:.4f}")
    print(f"  95% CI:                   [{rank_correlation_ci[0]:.4f}, {rank_correlation_ci[1]:.4f}]")
    print(f"  Samples ρ≥0.8:            {h3_samples_pass}/{N_BOOTSTRAP} ({h3_pass_rate*100:.1f}%)")
    print(f"  NaN samples (missing prov): {np.sum(np.isnan(rank_correlation_dist))}")
    print(f"  Result:                   {'✅ PASS' if h3_pass else '❌ FAIL'}")

    print(f"\n[Effect Size Stability]")
    print(f"  Shift decline within ±20%:    {shift_decline_within_20pct*100:.1f}%")
    print(f"  Variance growth within ±20%:  {variance_growth_within_20pct*100:.1f}%")

    print(f"\n{'='*80}")
    print(f"N≥3 VALIDATION: {'✅ PROMOTED' if results['n3_validation']['promoted_to_n3'] else '❌ NOT PROMOTED'}")
    print(f"{'='*80}")

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for Observation 5: Reflection Stabilization Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 scripts/bootstrap_obs5.py --batch-id full-scale-2025-11-20-v2
        """
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="Batch ID (e.g., full-scale-2025-11-20-v2)"
    )
    args = parser.parse_args()

    results = run_bootstrap(args.batch_id)
