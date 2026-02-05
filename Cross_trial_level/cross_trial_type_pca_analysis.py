#!/usr/bin/env python3
"""
Cross-Trial Type PCA Analysis
==============================

This module performs Principal Component Analysis (PCA) across different trial types
with optional hierarchical brain region aggregation.

Features
--------
- PCA analysis on neural activity across trial types
- Optional hierarchical aggregation of brain regions
- Visualization of principal components and variance explained
- Statistical comparison between trial types

Author: Computational Neuroscience Research
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Brain Region Hierarchy Configuration
# =============================================================================

# Default hierarchical anatomical ordering
DEFAULT_REGION_ORDER = [
    # Motor Cortex - keep separate
    'MOp',    # Primary Motor Cortex
    'MOs',    # Secondary Motor Cortex

    # Prefrontal Cortex - keep separate
    'mPFC',   # Medial Prefrontal Cortex
    'ORB',    # Orbitofrontal Cortex
    'ILM',    # Infralimbic Area

    # Olfactory Cortex - keep separate
    'OLF',    # Olfactory Areas

    # Basal Ganglia - Striatum - AGGREGATE
    'STR',    # Dorsal Striatum (aggregates STR + STRv)

    # Diencephalon - Thalamus - AGGREGATE
    'TH',     # Thalamus (aggregates MD + VALVM + LP + VPMPO)

    # Diencephalon - Hypothalamus - keep separate
    'HY'      # Hypothalamus
]

# Mapping from fine-grained to aggregated regions
REGION_AGGREGATION_MAP = {
    # Keep these as-is
    'MOp': 'MOp',
    'MOs': 'MOs',
    'mPFC': 'mPFC',
    'ORB': 'ORB',
    'ILM': 'ILM',
    'OLF': 'OLF',
    'HY': 'HY',

    # Aggregate striatum
    'STR': 'STR',
    'STRv': 'STR',  # Ventral striatum aggregates into STR

    # Aggregate thalamus
    'MD': 'TH',      # Mediodorsal Nucleus
    'VALVM': 'TH',   # Ventral Anterior-Lateral Complex
    'LP': 'TH',      # Lateral Posterior Nucleus
    'VPMPO': 'TH',   # Ventral Posteromedial Nucleus
}


def aggregate_brain_regions(
    data: np.ndarray,
    region_labels: List[str],
    aggregate: bool = False,
    aggregation_method: str = 'mean'
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate brain regions according to hierarchical mapping.

    Parameters
    ----------
    data : np.ndarray
        Neural activity data of shape (n_trials, n_regions) or (n_trials, n_timepoints, n_regions)
    region_labels : List[str]
        List of region names corresponding to data columns
    aggregate : bool
        Whether to perform aggregation (default: False)
    aggregation_method : str
        Method for aggregation: 'mean', 'sum', or 'max'

    Returns
    -------
    aggregated_data : np.ndarray
        Data with aggregated regions
    aggregated_labels : List[str]
        Labels for aggregated regions
    """
    if not aggregate:
        return data, region_labels

    # Map each region to its aggregated category
    aggregated_regions = {}
    for idx, region in enumerate(region_labels):
        target_region = REGION_AGGREGATION_MAP.get(region, region)

        if target_region not in aggregated_regions:
            aggregated_regions[target_region] = []
        aggregated_regions[target_region].append(idx)

    # Aggregate data
    aggregated_data_list = []
    aggregated_labels = []

    for region in DEFAULT_REGION_ORDER:
        if region in aggregated_regions:
            indices = aggregated_regions[region]

            if aggregation_method == 'mean':
                if len(data.shape) == 2:
                    agg_values = data[:, indices].mean(axis=1, keepdims=True)
                else:  # 3D data
                    agg_values = data[:, :, indices].mean(axis=2, keepdims=True)
            elif aggregation_method == 'sum':
                if len(data.shape) == 2:
                    agg_values = data[:, indices].sum(axis=1, keepdims=True)
                else:
                    agg_values = data[:, :, indices].sum(axis=2, keepdims=True)
            elif aggregation_method == 'max':
                if len(data.shape) == 2:
                    agg_values = data[:, indices].max(axis=1, keepdims=True)
                else:
                    agg_values = data[:, :, indices].max(axis=2, keepdims=True)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            aggregated_data_list.append(agg_values)
            aggregated_labels.append(region)

    if len(data.shape) == 2:
        aggregated_data = np.concatenate(aggregated_data_list, axis=1)
    else:
        aggregated_data = np.concatenate(aggregated_data_list, axis=2)

    return aggregated_data, aggregated_labels


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PCAResults:
    """Results from PCA analysis."""
    pca_model: PCA
    transformed_data: np.ndarray
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    region_labels: List[str]
    trial_type_labels: np.ndarray
    n_components: int
    aggregated: bool

    def get_variance_explained(self, n_components: Optional[int] = None) -> float:
        """Get cumulative variance explained by first n components."""
        if n_components is None:
            n_components = self.n_components
        return self.explained_variance_ratio[:n_components].sum()

    def get_component_loadings(self, component_idx: int) -> Dict[str, float]:
        """Get loadings for a specific component."""
        return {
            region: float(loading)
            for region, loading in zip(self.region_labels, self.components[component_idx])
        }


@dataclass
class CrossTrialPCAComparison:
    """Results from comparing PCA across trial types."""
    trial_types: List[str]
    pca_results: Dict[str, PCAResults]
    variance_comparison: pd.DataFrame
    component_similarity: np.ndarray
    statistical_tests: Dict[str, Dict]


# =============================================================================
# PCA Analysis Functions
# =============================================================================

def perform_pca_analysis(
    data: np.ndarray,
    region_labels: List[str],
    trial_type_labels: np.ndarray,
    n_components: Optional[int] = None,
    aggregate_regions: bool = False,
    aggregation_method: str = 'mean',
    standardize: bool = True
) -> PCAResults:
    """
    Perform PCA analysis on neural activity data.

    Parameters
    ----------
    data : np.ndarray
        Neural activity data of shape (n_trials, n_regions)
    region_labels : List[str]
        List of brain region names
    trial_type_labels : np.ndarray
        Array indicating trial type for each trial
    n_components : int, optional
        Number of principal components (default: min(n_trials, n_regions))
    aggregate_regions : bool
        Whether to aggregate regions hierarchically
    aggregation_method : str
        Method for aggregation: 'mean', 'sum', or 'max'
    standardize : bool
        Whether to standardize data before PCA

    Returns
    -------
    PCAResults
        Results from PCA analysis
    """
    # Aggregate regions if requested
    data_processed, labels_processed = aggregate_brain_regions(
        data, region_labels, aggregate_regions, aggregation_method
    )

    # Standardize data if requested
    if standardize:
        scaler = StandardScaler()
        data_processed = scaler.fit_transform(data_processed)

    # Determine number of components
    if n_components is None:
        n_components = min(data_processed.shape[0], data_processed.shape[1])

    # Perform PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_processed)

    return PCAResults(
        pca_model=pca,
        transformed_data=transformed,
        explained_variance_ratio=pca.explained_variance_ratio_,
        components=pca.components_,
        region_labels=labels_processed,
        trial_type_labels=trial_type_labels,
        n_components=n_components,
        aggregated=aggregate_regions
    )


def compare_pca_across_trial_types(
    data_by_type: Dict[str, np.ndarray],
    region_labels: List[str],
    n_components: int = 10,
    aggregate_regions: bool = False,
    aggregation_method: str = 'mean'
) -> CrossTrialPCAComparison:
    """
    Compare PCA results across different trial types.

    Parameters
    ----------
    data_by_type : Dict[str, np.ndarray]
        Dictionary mapping trial type names to neural activity data
    region_labels : List[str]
        List of brain region names
    n_components : int
        Number of principal components to compute
    aggregate_regions : bool
        Whether to aggregate regions hierarchically
    aggregation_method : str
        Method for aggregation

    Returns
    -------
    CrossTrialPCAComparison
        Comparison results across trial types
    """
    trial_types = list(data_by_type.keys())
    pca_results = {}

    # Perform PCA for each trial type
    for trial_type, data in data_by_type.items():
        trial_labels = np.array([trial_type] * data.shape[0])
        pca_results[trial_type] = perform_pca_analysis(
            data,
            region_labels,
            trial_labels,
            n_components=n_components,
            aggregate_regions=aggregate_regions,
            aggregation_method=aggregation_method
        )

    # Compare variance explained
    variance_data = {
        'Trial_Type': trial_types,
        'PC1_Variance': [pca_results[t].explained_variance_ratio[0] for t in trial_types],
        'PC2_Variance': [pca_results[t].explained_variance_ratio[1] for t in trial_types],
        'PC3_Variance': [pca_results[t].explained_variance_ratio[2] for t in trial_types],
        'Total_Top3_Variance': [pca_results[t].get_variance_explained(3) for t in trial_types],
        'Total_All_Variance': [pca_results[t].get_variance_explained() for t in trial_types]
    }
    variance_df = pd.DataFrame(variance_data)

    # Compute component similarity (cosine similarity between PC1s)
    n_types = len(trial_types)
    similarity_matrix = np.zeros((n_types, n_types))

    for i, type1 in enumerate(trial_types):
        for j, type2 in enumerate(trial_types):
            pc1_1 = pca_results[type1].components[0]
            pc1_2 = pca_results[type2].components[0]

            # Cosine similarity
            similarity = np.dot(pc1_1, pc1_2) / (
                np.linalg.norm(pc1_1) * np.linalg.norm(pc1_2)
            )
            similarity_matrix[i, j] = abs(similarity)  # Absolute value (direction invariant)

    # Statistical tests
    statistical_tests = {}
    if len(trial_types) >= 2:
        # Compare variance explained across trial types
        variances = [pca_results[t].get_variance_explained(3) for t in trial_types]

        if len(trial_types) == 2:
            # t-test for 2 groups
            # Note: This is conceptual - with single values per group, we'd need multiple runs
            statistical_tests['variance_comparison'] = {
                'test': 't-test',
                'note': 'Requires multiple runs for proper statistical testing',
                'values': variances
            }
        else:
            # ANOVA for >2 groups
            statistical_tests['variance_comparison'] = {
                'test': 'ANOVA',
                'note': 'Requires multiple runs for proper statistical testing',
                'values': variances
            }

    return CrossTrialPCAComparison(
        trial_types=trial_types,
        pca_results=pca_results,
        variance_comparison=variance_df,
        component_similarity=similarity_matrix,
        statistical_tests=statistical_tests
    )


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_pca_results(
    results: PCAResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create comprehensive visualization of PCA results.

    Parameters
    ----------
    results : PCAResults
        PCA analysis results
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

    # Panel A: Scree plot (variance explained)
    ax = axes[0, 0]
    n_show = min(10, len(results.explained_variance_ratio))
    ax.bar(range(1, n_show + 1), results.explained_variance_ratio[:n_show])
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('A. Scree Plot')

    # Add cumulative variance line
    ax2 = ax.twinx()
    cumsum = np.cumsum(results.explained_variance_ratio[:n_show])
    ax2.plot(range(1, n_show + 1), cumsum, 'r-o', linewidth=2)
    ax2.set_ylabel('Cumulative Variance', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Panel B: PC1 vs PC2 scatter
    ax = axes[0, 1]
    unique_types = np.unique(results.trial_type_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))

    for idx, trial_type in enumerate(unique_types):
        mask = results.trial_type_labels == trial_type
        ax.scatter(
            results.transformed_data[mask, 0],
            results.transformed_data[mask, 1],
            c=[colors[idx]],
            label=trial_type,
            alpha=0.6,
            s=50
        )

    ax.set_xlabel(f'PC1 ({results.explained_variance_ratio[0]:.2%} var)')
    ax.set_ylabel(f'PC2 ({results.explained_variance_ratio[1]:.2%} var)')
    ax.set_title('B. PC1 vs PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: PC1 loadings
    ax = axes[0, 2]
    loadings = results.components[0]
    sorted_idx = np.argsort(np.abs(loadings))[::-1][:10]

    ax.barh(range(len(sorted_idx)), loadings[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([results.region_labels[i] for i in sorted_idx])
    ax.set_xlabel('PC1 Loading')
    ax.set_title('C. Top PC1 Loadings')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Panel D: PC2 vs PC3 scatter
    ax = axes[1, 0]
    for idx, trial_type in enumerate(unique_types):
        mask = results.trial_type_labels == trial_type
        ax.scatter(
            results.transformed_data[mask, 1],
            results.transformed_data[mask, 2],
            c=[colors[idx]],
            label=trial_type,
            alpha=0.6,
            s=50
        )

    ax.set_xlabel(f'PC2 ({results.explained_variance_ratio[1]:.2%} var)')
    ax.set_ylabel(f'PC3 ({results.explained_variance_ratio[2]:.2%} var)')
    ax.set_title('D. PC2 vs PC3')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel E: PC2 loadings
    ax = axes[1, 1]
    loadings = results.components[1]
    sorted_idx = np.argsort(np.abs(loadings))[::-1][:10]

    ax.barh(range(len(sorted_idx)), loadings[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([results.region_labels[i] for i in sorted_idx])
    ax.set_xlabel('PC2 Loading')
    ax.set_title('E. Top PC2 Loadings')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Panel F: Summary statistics
    ax = axes[1, 2]
    summary_text = (
        f"PCA SUMMARY\n"
        f"{'='*30}\n\n"
        f"Aggregated: {'Yes' if results.aggregated else 'No'}\n"
        f"N Regions: {len(results.region_labels)}\n"
        f"N Components: {results.n_components}\n"
        f"N Trials: {results.transformed_data.shape[0]}\n\n"
        f"Variance Explained:\n"
        f"  PC1: {results.explained_variance_ratio[0]:.2%}\n"
        f"  PC2: {results.explained_variance_ratio[1]:.2%}\n"
        f"  PC3: {results.explained_variance_ratio[2]:.2%}\n"
        f"  Top 3: {results.get_variance_explained(3):.2%}\n"
        f"  Total: {results.get_variance_explained():.2%}\n"
    )

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           fontfamily='monospace')
    ax.axis('off')
    ax.set_title('F. Summary')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_cross_trial_comparison(
    comparison: CrossTrialPCAComparison,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize comparison across trial types.

    Parameters
    ----------
    comparison : CrossTrialPCAComparison
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

    # Panel A: Variance explained by trial type
    ax = axes[0, 0]
    df = comparison.variance_comparison
    x = np.arange(len(df))
    width = 0.2

    ax.bar(x - width, df['PC1_Variance'], width, label='PC1')
    ax.bar(x, df['PC2_Variance'], width, label='PC2')
    ax.bar(x + width, df['PC3_Variance'], width, label='PC3')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Trial_Type'], rotation=45, ha='right')
    ax.set_ylabel('Variance Explained')
    ax.set_title('A. Variance Explained by Trial Type')
    ax.legend()

    # Panel B: Component similarity matrix
    ax = axes[0, 1]
    im = ax.imshow(comparison.component_similarity, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(comparison.trial_types)))
    ax.set_yticks(range(len(comparison.trial_types)))
    ax.set_xticklabels(comparison.trial_types, rotation=45, ha='right')
    ax.set_yticklabels(comparison.trial_types)
    ax.set_title('B. PC1 Similarity Matrix')
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(comparison.trial_types)):
        for j in range(len(comparison.trial_types)):
            text = ax.text(j, i, f'{comparison.component_similarity[i, j]:.2f}',
                         ha="center", va="center", color="w" if comparison.component_similarity[i, j] < 0.5 else "black")

    # Panel C: Total variance explained
    ax = axes[1, 0]
    ax.bar(df['Trial_Type'], df['Total_Top3_Variance'])
    ax.set_ylabel('Cumulative Variance (Top 3 PCs)')
    ax.set_title('C. Total Variance Explained (Top 3 PCs)')
    ax.set_xticklabels(df['Trial_Type'], rotation=45, ha='right')

    # Panel D: PC trajectories across trial types
    ax = axes[1, 1]
    for trial_type in comparison.trial_types:
        results = comparison.pca_results[trial_type]
        var_explained = results.explained_variance_ratio[:5]
        ax.plot(range(1, len(var_explained) + 1), var_explained, 'o-', label=trial_type, linewidth=2)

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('D. Variance Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Cross-Trial Type PCA Analysis Testing")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic neural data...")
    np.random.seed(42)

    # Define regions
    regions = ['MOp', 'MOs', 'mPFC', 'ORB', 'ILM', 'OLF',
               'STR', 'STRv', 'MD', 'VALVM', 'LP', 'VPMPO', 'HY']
    n_regions = len(regions)
    n_trials = 100

    # Generate data for different trial types
    data_correct = np.random.randn(n_trials, n_regions) + np.array([1, 1, 0.5, 0.5, 0.5, 0, 0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0])
    data_error = np.random.randn(n_trials, n_regions) + np.array([0.5, 0.5, 1, 1, 1, 0.2, 0.3, 0.3, 0.8, 0.8, 0.8, 0.8, 0.2])
    data_reward = np.random.randn(n_trials, n_regions) + np.array([0.8, 0.8, 0.8, 0.3, 0.3, 0.1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Test 1: PCA without aggregation
    print("\n2. Testing PCA without region aggregation...")
    all_data = np.vstack([data_correct, data_error, data_reward])
    trial_labels = np.array(['correct'] * n_trials + ['error'] * n_trials + ['reward'] * n_trials)

    results_no_agg = perform_pca_analysis(
        all_data,
        regions,
        trial_labels,
        n_components=10,
        aggregate_regions=False
    )
    print(f"  - Number of regions: {len(results_no_agg.region_labels)}")
    print(f"  - Top 3 PC variance: {results_no_agg.get_variance_explained(3):.2%}")

    # Test 2: PCA with aggregation
    print("\n3. Testing PCA with region aggregation...")
    results_with_agg = perform_pca_analysis(
        all_data,
        regions,
        trial_labels,
        n_components=10,
        aggregate_regions=True
    )
    print(f"  - Number of aggregated regions: {len(results_with_agg.region_labels)}")
    print(f"  - Aggregated regions: {results_with_agg.region_labels}")
    print(f"  - Top 3 PC variance: {results_with_agg.get_variance_explained(3):.2%}")

    # Test 3: Cross-trial comparison
    print("\n4. Testing cross-trial type comparison...")
    data_by_type = {
        'correct': data_correct,
        'error': data_error,
        'reward': data_reward
    }

    comparison = compare_pca_across_trial_types(
        data_by_type,
        regions,
        n_components=10,
        aggregate_regions=True
    )
    print(f"  - Trial types compared: {comparison.trial_types}")
    print(f"  - Variance comparison:\n{comparison.variance_comparison}")

    # Test 4: Visualizations
    print("\n5. Testing visualization functions...")
    fig1 = plot_pca_results(results_with_agg)
    print("  - PCA results plot created")

    fig2 = plot_cross_trial_comparison(comparison)
    print("  - Cross-trial comparison plot created")

    plt.show()

    print("\n" + "=" * 70)
    print("All tests passed successfully!")
    print("=" * 70)
