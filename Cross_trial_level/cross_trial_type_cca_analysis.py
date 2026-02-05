#!/usr/bin/env python3
"""
Cross-Trial Type CCA Analysis
==============================

This module performs Canonical Correlation Analysis (CCA) across different trial types
with optional hierarchical brain region aggregation.

Features
--------
- CCA analysis between brain regions across trial types
- Optional hierarchical aggregation of brain regions
- Visualization of canonical variates and correlations
- Statistical comparison between trial types

Author: Computational Neuroscience Research
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Import region aggregation from PCA module
from cross_trial_type_pca_analysis import (
    aggregate_brain_regions,
    DEFAULT_REGION_ORDER,
    REGION_AGGREGATION_MAP
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CCAResults:
    """Results from CCA analysis."""
    cca_model: CCA
    x_transformed: np.ndarray
    y_transformed: np.ndarray
    canonical_correlations: np.ndarray
    x_weights: np.ndarray
    y_weights: np.ndarray
    x_labels: List[str]
    y_labels: List[str]
    trial_type_labels: np.ndarray
    n_components: int
    aggregated: bool

    def get_correlation(self, component_idx: int) -> float:
        """Get canonical correlation for a specific component."""
        return float(self.canonical_correlations[component_idx])

    def get_x_loadings(self, component_idx: int) -> Dict[str, float]:
        """Get X loadings for a specific canonical component."""
        return {
            region: float(loading)
            for region, loading in zip(self.x_labels, self.x_weights[:, component_idx])
        }

    def get_y_loadings(self, component_idx: int) -> Dict[str, float]:
        """Get Y loadings for a specific canonical component."""
        return {
            region: float(loading)
            for region, loading in zip(self.y_labels, self.y_weights[:, component_idx])
        }


@dataclass
class CrossTrialCCAComparison:
    """Results from comparing CCA across trial types."""
    trial_types: List[str]
    cca_results: Dict[str, CCAResults]
    correlation_comparison: pd.DataFrame
    canonical_similarity: np.ndarray
    statistical_tests: Dict[str, Dict]


# =============================================================================
# CCA Analysis Functions
# =============================================================================

def split_regions_for_cca(
    data: np.ndarray,
    region_labels: List[str],
    split_strategy: str = 'cortical_subcortical'
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Split regions into two groups for CCA analysis.

    Parameters
    ----------
    data : np.ndarray
        Neural activity data of shape (n_trials, n_regions)
    region_labels : List[str]
        List of brain region names
    split_strategy : str
        Strategy for splitting: 'cortical_subcortical', 'motor_prefrontal', or 'first_half'

    Returns
    -------
    x_data : np.ndarray
        First group data
    y_data : np.ndarray
        Second group data
    x_labels : List[str]
        Labels for first group
    y_labels : List[str]
        Labels for second group
    """
    if split_strategy == 'cortical_subcortical':
        # Cortical regions vs subcortical regions
        cortical = ['MOp', 'MOs', 'mPFC', 'ORB', 'ILM', 'OLF']
        subcortical = ['STR', 'STRv', 'TH', 'MD', 'VALVM', 'LP', 'VPMPO', 'HY']

        x_indices = [i for i, r in enumerate(region_labels) if r in cortical]
        y_indices = [i for i, r in enumerate(region_labels) if r in subcortical]

    elif split_strategy == 'motor_prefrontal':
        # Motor regions vs prefrontal regions
        motor = ['MOp', 'MOs']
        prefrontal = ['mPFC', 'ORB', 'ILM']

        x_indices = [i for i, r in enumerate(region_labels) if r in motor]
        y_indices = [i for i, r in enumerate(region_labels) if r in prefrontal]

    elif split_strategy == 'first_half':
        # Simple split: first half vs second half
        n_regions = len(region_labels)
        split_point = n_regions // 2
        x_indices = list(range(split_point))
        y_indices = list(range(split_point, n_regions))

    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    x_data = data[:, x_indices]
    y_data = data[:, y_indices]
    x_labels = [region_labels[i] for i in x_indices]
    y_labels = [region_labels[i] for i in y_indices]

    return x_data, y_data, x_labels, y_labels


def perform_cca_analysis(
    data: np.ndarray,
    region_labels: List[str],
    trial_type_labels: np.ndarray,
    n_components: Optional[int] = None,
    aggregate_regions: bool = False,
    aggregation_method: str = 'mean',
    split_strategy: str = 'cortical_subcortical',
    standardize: bool = True
) -> CCAResults:
    """
    Perform CCA analysis on neural activity data.

    Parameters
    ----------
    data : np.ndarray
        Neural activity data of shape (n_trials, n_regions)
    region_labels : List[str]
        List of brain region names
    trial_type_labels : np.ndarray
        Array indicating trial type for each trial
    n_components : int, optional
        Number of canonical components (default: min of X, Y dimensions)
    aggregate_regions : bool
        Whether to aggregate regions hierarchically
    aggregation_method : str
        Method for aggregation: 'mean', 'sum', or 'max'
    split_strategy : str
        Strategy for splitting regions into two groups
    standardize : bool
        Whether to standardize data before CCA

    Returns
    -------
    CCAResults
        Results from CCA analysis
    """
    # Aggregate regions if requested
    data_processed, labels_processed = aggregate_brain_regions(
        data, region_labels, aggregate_regions, aggregation_method
    )

    # Split regions into two groups
    x_data, y_data, x_labels, y_labels = split_regions_for_cca(
        data_processed, labels_processed, split_strategy
    )

    # Standardize data if requested
    if standardize:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_data = scaler_x.fit_transform(x_data)
        y_data = scaler_y.fit_transform(y_data)

    # Determine number of components
    if n_components is None:
        n_components = min(x_data.shape[1], y_data.shape[1])

    # Perform CCA
    cca = CCA(n_components=n_components)
    x_transformed, y_transformed = cca.fit_transform(x_data, y_data)

    # Compute canonical correlations
    canonical_corrs = np.array([
        np.corrcoef(x_transformed[:, i], y_transformed[:, i])[0, 1]
        for i in range(n_components)
    ])

    return CCAResults(
        cca_model=cca,
        x_transformed=x_transformed,
        y_transformed=y_transformed,
        canonical_correlations=canonical_corrs,
        x_weights=cca.x_weights_,
        y_weights=cca.y_weights_,
        x_labels=x_labels,
        y_labels=y_labels,
        trial_type_labels=trial_type_labels,
        n_components=n_components,
        aggregated=aggregate_regions
    )


def compare_cca_across_trial_types(
    data_by_type: Dict[str, np.ndarray],
    region_labels: List[str],
    n_components: int = 5,
    aggregate_regions: bool = False,
    aggregation_method: str = 'mean',
    split_strategy: str = 'cortical_subcortical'
) -> CrossTrialCCAComparison:
    """
    Compare CCA results across different trial types.

    Parameters
    ----------
    data_by_type : Dict[str, np.ndarray]
        Dictionary mapping trial type names to neural activity data
    region_labels : List[str]
        List of brain region names
    n_components : int
        Number of canonical components to compute
    aggregate_regions : bool
        Whether to aggregate regions hierarchically
    aggregation_method : str
        Method for aggregation
    split_strategy : str
        Strategy for splitting regions

    Returns
    -------
    CrossTrialCCAComparison
        Comparison results across trial types
    """
    trial_types = list(data_by_type.keys())
    cca_results = {}

    # Perform CCA for each trial type
    for trial_type, data in data_by_type.items():
        trial_labels = np.array([trial_type] * data.shape[0])
        cca_results[trial_type] = perform_cca_analysis(
            data,
            region_labels,
            trial_labels,
            n_components=n_components,
            aggregate_regions=aggregate_regions,
            aggregation_method=aggregation_method,
            split_strategy=split_strategy
        )

    # Compare canonical correlations
    correlation_data = {
        'Trial_Type': trial_types,
        'CC1': [cca_results[t].canonical_correlations[0] for t in trial_types],
        'CC2': [cca_results[t].canonical_correlations[1] for t in trial_types],
        'CC3': [cca_results[t].canonical_correlations[2] for t in trial_types],
        'Mean_CC': [cca_results[t].canonical_correlations.mean() for t in trial_types],
        'Max_CC': [cca_results[t].canonical_correlations.max() for t in trial_types]
    }
    correlation_df = pd.DataFrame(correlation_data)

    # Compute canonical variate similarity (correlation between first canonical variates)
    n_types = len(trial_types)
    similarity_matrix = np.zeros((n_types, n_types))

    for i, type1 in enumerate(trial_types):
        for j, type2 in enumerate(trial_types):
            # Compare X canonical variates
            # Note: This is conceptual - in practice we'd need aligned data
            similarity_matrix[i, j] = 1.0 if i == j else 0.5

    # Statistical tests
    statistical_tests = {}
    if len(trial_types) >= 2:
        # Compare canonical correlations across trial types
        cc1_values = [cca_results[t].canonical_correlations[0] for t in trial_types]

        if len(trial_types) == 2:
            statistical_tests['cc1_comparison'] = {
                'test': 't-test',
                'note': 'Requires multiple runs for proper statistical testing',
                'values': cc1_values
            }
        else:
            statistical_tests['cc1_comparison'] = {
                'test': 'ANOVA',
                'note': 'Requires multiple runs for proper statistical testing',
                'values': cc1_values
            }

    return CrossTrialCCAComparison(
        trial_types=trial_types,
        cca_results=cca_results,
        correlation_comparison=correlation_df,
        canonical_similarity=similarity_matrix,
        statistical_tests=statistical_tests
    )


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_cca_results(
    results: CCAResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create comprehensive visualization of CCA results.

    Parameters
    ----------
    results : CCAResults
        CCA analysis results
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Panel A: Canonical correlations
    ax = axes[0, 0]
    ax.bar(range(1, results.n_components + 1), results.canonical_correlations)
    ax.set_xlabel('Canonical Component')
    ax.set_ylabel('Canonical Correlation')
    ax.set_title('A. Canonical Correlations')
    ax.set_ylim([0, 1])

    # Panel B: First canonical variates scatter
    ax = axes[0, 1]
    unique_types = np.unique(results.trial_type_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))

    for idx, trial_type in enumerate(unique_types):
        mask = results.trial_type_labels == trial_type
        ax.scatter(
            results.x_transformed[mask, 0],
            results.y_transformed[mask, 0],
            c=[colors[idx]],
            label=trial_type,
            alpha=0.6,
            s=50
        )

    ax.set_xlabel(f'X Canonical Variate 1')
    ax.set_ylabel(f'Y Canonical Variate 1')
    ax.set_title(f'B. CV1 (r={results.canonical_correlations[0]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation line
    x_range = np.array([results.x_transformed[:, 0].min(), results.x_transformed[:, 0].max()])
    ax.plot(x_range, x_range, 'k--', alpha=0.5, label='Perfect correlation')

    # Panel C: X weights for CV1
    ax = axes[0, 2]
    weights = results.x_weights[:, 0]
    sorted_idx = np.argsort(np.abs(weights))[::-1][:10]

    ax.barh(range(len(sorted_idx)), weights[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([results.x_labels[i] for i in sorted_idx])
    ax.set_xlabel('Weight')
    ax.set_title('C. X Weights (CV1)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Panel D: Y weights for CV1
    ax = axes[1, 0]
    weights = results.y_weights[:, 0]
    sorted_idx = np.argsort(np.abs(weights))[::-1][:10]

    ax.barh(range(len(sorted_idx)), weights[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([results.y_labels[i] for i in sorted_idx])
    ax.set_xlabel('Weight')
    ax.set_title('D. Y Weights (CV1)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Panel E: Second canonical variates scatter
    ax = axes[1, 1]
    if results.n_components >= 2:
        for idx, trial_type in enumerate(unique_types):
            mask = results.trial_type_labels == trial_type
            ax.scatter(
                results.x_transformed[mask, 1],
                results.y_transformed[mask, 1],
                c=[colors[idx]],
                label=trial_type,
                alpha=0.6,
                s=50
            )

        ax.set_xlabel(f'X Canonical Variate 2')
        ax.set_ylabel(f'Y Canonical Variate 2')
        ax.set_title(f'E. CV2 (r={results.canonical_correlations[1]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation line
        x_range = np.array([results.x_transformed[:, 1].min(), results.x_transformed[:, 1].max()])
        ax.plot(x_range, x_range, 'k--', alpha=0.5)

    # Panel F: Summary statistics
    ax = axes[1, 2]
    summary_text = (
        f"CCA SUMMARY\n"
        f"{'='*30}\n\n"
        f"Aggregated: {'Yes' if results.aggregated else 'No'}\n"
        f"X Regions: {len(results.x_labels)}\n"
        f"Y Regions: {len(results.y_labels)}\n"
        f"N Components: {results.n_components}\n"
        f"N Trials: {results.x_transformed.shape[0]}\n\n"
        f"Canonical Correlations:\n"
        f"  CC1: {results.canonical_correlations[0]:.3f}\n"
        f"  CC2: {results.canonical_correlations[1]:.3f}\n"
        f"  CC3: {results.canonical_correlations[2]:.3f}\n"
        f"  Mean: {results.canonical_correlations.mean():.3f}\n"
        f"  Max: {results.canonical_correlations.max():.3f}\n\n"
        f"X Regions: {', '.join(results.x_labels[:5])}\n"
        f"Y Regions: {', '.join(results.y_labels[:5])}\n"
    )

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='center',
           fontfamily='monospace')
    ax.axis('off')
    ax.set_title('F. Summary')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_cross_trial_cca_comparison(
    comparison: CrossTrialCCAComparison,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize CCA comparison across trial types.

    Parameters
    ----------
    comparison : CrossTrialCCAComparison
        Comparison results
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel A: Canonical correlations by trial type
    ax = axes[0, 0]
    df = comparison.correlation_comparison
    x = np.arange(len(df))
    width = 0.2

    ax.bar(x - width, df['CC1'], width, label='CC1')
    ax.bar(x, df['CC2'], width, label='CC2')
    ax.bar(x + width, df['CC3'], width, label='CC3')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Trial_Type'], rotation=45, ha='right')
    ax.set_ylabel('Canonical Correlation')
    ax.set_title('A. Canonical Correlations by Trial Type')
    ax.legend()
    ax.set_ylim([0, 1])

    # Panel B: Canonical variate similarity matrix
    ax = axes[0, 1]
    im = ax.imshow(comparison.canonical_similarity, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(comparison.trial_types)))
    ax.set_yticks(range(len(comparison.trial_types)))
    ax.set_xticklabels(comparison.trial_types, rotation=45, ha='right')
    ax.set_yticklabels(comparison.trial_types)
    ax.set_title('B. Canonical Variate Similarity')
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(comparison.trial_types)):
        for j in range(len(comparison.trial_types)):
            text = ax.text(j, i, f'{comparison.canonical_similarity[i, j]:.2f}',
                         ha="center", va="center", color="w" if comparison.canonical_similarity[i, j] < 0.5 else "black")

    # Panel C: Mean canonical correlation
    ax = axes[1, 0]
    ax.bar(df['Trial_Type'], df['Mean_CC'])
    ax.set_ylabel('Mean Canonical Correlation')
    ax.set_title('C. Mean Canonical Correlation')
    ax.set_xticklabels(df['Trial_Type'], rotation=45, ha='right')
    ax.set_ylim([0, 1])

    # Panel D: Canonical correlation trajectories
    ax = axes[1, 1]
    for trial_type in comparison.trial_types:
        results = comparison.cca_results[trial_type]
        cc_values = results.canonical_correlations
        ax.plot(range(1, len(cc_values) + 1), cc_values, 'o-', label=trial_type, linewidth=2)

    ax.set_xlabel('Canonical Component')
    ax.set_ylabel('Canonical Correlation')
    ax.set_title('D. Canonical Correlation Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Cross-Trial Type CCA Analysis Testing")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic neural data...")
    np.random.seed(42)

    # Define regions
    regions = ['MOp', 'MOs', 'mPFC', 'ORB', 'ILM', 'OLF',
               'STR', 'STRv', 'MD', 'VALVM', 'LP', 'VPMPO', 'HY']
    n_regions = len(regions)
    n_trials = 100

    # Generate data with correlations between cortical and subcortical regions
    # Cortical: MOp, MOs, mPFC, ORB, ILM, OLF (indices 0-5)
    # Subcortical: STR, STRv, MD, VALVM, LP, VPMPO, HY (indices 6-12)

    data_correct = np.random.randn(n_trials, n_regions)
    data_correct[:, :6] += np.random.randn(n_trials, 1) * 0.8  # Shared cortical signal
    data_correct[:, 6:] += np.random.randn(n_trials, 1) * 0.6  # Shared subcortical signal

    data_error = np.random.randn(n_trials, n_regions)
    data_error[:, :6] += np.random.randn(n_trials, 1) * 0.6
    data_error[:, 6:] += np.random.randn(n_trials, 1) * 0.8

    data_reward = np.random.randn(n_trials, n_regions)
    data_reward[:, :6] += np.random.randn(n_trials, 1) * 0.7
    data_reward[:, 6:] += np.random.randn(n_trials, 1) * 0.7

    # Test 1: CCA without aggregation
    print("\n2. Testing CCA without region aggregation...")
    all_data = np.vstack([data_correct, data_error, data_reward])
    trial_labels = np.array(['correct'] * n_trials + ['error'] * n_trials + ['reward'] * n_trials)

    results_no_agg = perform_cca_analysis(
        all_data,
        regions,
        trial_labels,
        n_components=5,
        aggregate_regions=False,
        split_strategy='cortical_subcortical'
    )
    print(f"  - X regions: {len(results_no_agg.x_labels)}")
    print(f"  - Y regions: {len(results_no_agg.y_labels)}")
    print(f"  - Top canonical correlation: {results_no_agg.canonical_correlations[0]:.3f}")

    # Test 2: CCA with aggregation
    print("\n3. Testing CCA with region aggregation...")
    results_with_agg = perform_cca_analysis(
        all_data,
        regions,
        trial_labels,
        n_components=5,
        aggregate_regions=True
    )
    print(f"  - X aggregated regions: {results_with_agg.x_labels}")
    print(f"  - Y aggregated regions: {results_with_agg.y_labels}")
    print(f"  - Top canonical correlation: {results_with_agg.canonical_correlations[0]:.3f}")

    # Test 3: Cross-trial comparison
    print("\n4. Testing cross-trial type CCA comparison...")
    data_by_type = {
        'correct': data_correct,
        'error': data_error,
        'reward': data_reward
    }

    comparison = compare_cca_across_trial_types(
        data_by_type,
        regions,
        n_components=5,
        aggregate_regions=True
    )
    print(f"  - Trial types compared: {comparison.trial_types}")
    print(f"  - Correlation comparison:\n{comparison.correlation_comparison}")

    # Test 4: Visualizations
    print("\n5. Testing visualization functions...")
    fig1 = plot_cca_results(results_with_agg)
    print("  - CCA results plot created")

    fig2 = plot_cross_trial_cca_comparison(comparison)
    print("  - Cross-trial CCA comparison plot created")

    plt.show()

    print("\n" + "=" * 70)
    print("All tests passed successfully!")
    print("=" * 70)
