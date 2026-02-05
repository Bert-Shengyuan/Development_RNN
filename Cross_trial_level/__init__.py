"""
Cross-Trial Type Analysis Package
==================================

This package provides comprehensive tools for analyzing neural data across trial types
with support for hierarchical brain region aggregation.

Main Modules
------------
- cross_trial_type_pca_analysis: PCA analysis with region aggregation
- cross_trial_type_cca_analysis: CCA analysis with region aggregation
- run_cross_trial_type_pca_analysis: CLI for PCA analysis
- run_cross_trial_type_cca_analysis: CLI for CCA analysis

Quick Start
-----------
>>> from Cross_trial_level.cross_trial_type_pca_analysis import perform_pca_analysis
>>> from Cross_trial_level.cross_trial_type_cca_analysis import perform_cca_analysis
>>>
>>> # Perform PCA with region aggregation
>>> results = perform_pca_analysis(
...     data, region_labels, trial_types,
...     aggregate_regions=True
... )

Region Hierarchy
----------------
The package supports hierarchical aggregation:
- Kept separate: MOp, MOs, mPFC, ORB, ILM, OLF, HY
- STR: Combines STR + STRv (Striatum)
- TH: Combines MD + VALVM + LP + VPMPO (Thalamus)

Author: Computational Neuroscience Research
"""

__version__ = '1.0.0'
__all__ = [
    'perform_pca_analysis',
    'perform_cca_analysis',
    'compare_pca_across_trial_types',
    'compare_cca_across_trial_types',
    'aggregate_brain_regions',
    'DEFAULT_REGION_ORDER',
    'REGION_AGGREGATION_MAP'
]

# Version info
VERSION = __version__

# Try to import main functions (will fail if dependencies not installed)
try:
    from .cross_trial_type_pca_analysis import (
        perform_pca_analysis,
        compare_pca_across_trial_types,
        plot_pca_results,
        plot_cross_trial_comparison,
        aggregate_brain_regions,
        DEFAULT_REGION_ORDER,
        REGION_AGGREGATION_MAP
    )
except ImportError:
    pass

try:
    from .cross_trial_type_cca_analysis import (
        perform_cca_analysis,
        compare_cca_across_trial_types,
        plot_cca_results,
        plot_cross_trial_cca_comparison
    )
except ImportError:
    pass
