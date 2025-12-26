"""
================================================================================
Brain-Inspired Developmental RNN Framework
================================================================================

A comprehensive framework for comparing developmentally-constrained recurrent
neural networks on cognitive decision-making tasks.

Theoretical Foundation
----------------------
This framework relates fMRI dynamics to low-rank RNN structure, investigating
how developmental differences in brain organization affect computation:

    PREMATURE: Higher effective rank, slower singular value decay,
               lower response heterogeneity
    
    MATURE: Lower effective rank, faster singular value decay,
            higher response heterogeneity (more specialization)

The key insight is that mature brains exhibit a paradoxical relationship:
lower dimensionality (compression) coexists with higher functional
specialization—a computational trade-off captured by this framework.

Mathematical Framework
----------------------
The training objective combines task performance with developmental constraints:

    L_total = L_task + α||W||_* + β D_eff(W) + L_spectral - λ H(W)

where:
    - ||W||_* = Σσ_i is the nuclear norm (promotes low-rank)
    - D_eff(W) = (Σσ_i)² / Σ(σ_i²) is effective rank
    - L_spectral matches target singular value decay: σ_i ~ exp(-γi)
    - H(W) = Var(||∂h/∂I_j||₂) is response heterogeneity

Modules
-------
brain_inspired_rnn : Core RNN architectures
    - BrainInspiredRNN: Main model class
    - Multiple factorization approaches (SVD, Eigenvalue, SLIN)
    - Multiple cell types (Leaky, DevGRU, SwitchingGRU, SLIN)

cognitive_tasks : Cognitive task implementations
    - ReversalLearningTask: Contingency reversal adaptation
    - TwoStageTask: Model-based vs model-free dissociation
    - ProbabilisticRewardTask: Volatile reward learning

analysis : Dynamical systems analysis
    - PhasePortraitGenerator: L(t) vs ΔL(t) visualization
    - VectorFieldGenerator: 2D state space flow
    - FixedPointFinder: Attractor identification
    - DevelopmentalComparisonAnalyzer: Comprehensive comparison

training : Training pipeline
    - Trainer: Complete training loop with regularization
    - run_developmental_comparison: Main experiment function

visualization : Publication-quality figures
    - Singular value spectra
    - Effective rank comparison
    - Response heterogeneity analysis
    - Phase portraits and vector fields

Usage
-----
Basic usage for developmental comparison:

    >>> from brain_inspired_rnn import create_model, FactorizationType
    >>> from cognitive_tasks import TaskType, TaskDataset
    >>> from main import run_developmental_comparison
    >>> 
    >>> # Run comparison experiment
    >>> results = run_developmental_comparison(
    ...     task_type=TaskType.REVERSAL_LEARNING,
    ...     n_epochs=100
    ... )
    >>> 
    >>> # Access results
    >>> print(f"Premature accuracy: {results['premature']['test_metrics']['accuracy']:.3f}")
    >>> print(f"Mature accuracy: {results['mature']['test_metrics']['accuracy']:.3f}")

Creating individual models:

    >>> # Create mature brain-inspired model
    >>> model = create_model(
    ...     developmental_stage="mature",
    ...     n_hidden=32,
    ...     factorization_type=FactorizationType.SVD_DIRECT
    ... )
    >>> 
    >>> # Forward pass
    >>> outputs, h_final = model(inputs)
    >>> 
    >>> # Get metrics
    >>> print(f"Effective rank: {model.compute_effective_rank():.2f}")

References
----------
1. Ji-An et al. (2025) "Discovering cognitive strategies with tiny RNNs"
2. Sussillo & Barak (2013) "Opening the Black Box"
3. Mastrogiuseppe & Ostojic (2018) "Linking Connectivity, Dynamics, and 
   Computations in Low-Rank Recurrent Neural Networks"

Author: Computational Neuroscience Research
================================================================================
"""

__version__ = "1.0.0"
__author__ = "Computational Neuroscience Research"

# =============================================================================
# Core Architecture Imports
# =============================================================================

from brain_inspired_rnn import (
    # Configuration
    DevelopmentalConfig,
    FactorizationType,
    CellType,
    create_premature_config,
    create_mature_config,
    
    # Factory function
    create_model,
    
    # Main model
    BrainInspiredRNN,
    
    # Layer components
    LowRankLinear,
    SVDLowRankLayer,
    EigenConstrainedLayer,
    SLINLayer,
    
    # Cell types
    DevGRUCell,
    SwitchingDevGRUCell,
    SLINCell,
    
    # Analysis tools
    ResponseHeterogeneityAnalyzer,
)

# =============================================================================
# Cognitive Task Imports
# =============================================================================

from cognitive_tasks import (
    # Enums and data structures
    TaskType,
    TrialData,
    SessionData,
    
    # Task implementations
    ReversalLearningTask,
    TwoStageTask,
    ProbabilisticRewardTask,
    
    # Dataset class
    TaskDataset,
    
    # Classical models
    ModelFreeRL,
    BayesianAgent,
    
    # Utility functions
    compute_choice_accuracy,
    compute_negative_log_likelihood,
)

# =============================================================================
# Analysis Imports
# =============================================================================

from analysis import (
    # Data structures
    PhasePortraitData,
    VectorFieldData,
    DevelopmentalComparison,
    
    # Analysis classes
    FixedPointFinder,
    LogitAnalyzer,
    PhasePortraitGenerator,
    VectorFieldGenerator,
    DevelopmentalComparisonAnalyzer,
)

# =============================================================================
# Visualization Imports
# =============================================================================

from visualization import (
    # Individual plot functions
    plot_learning_curves,
    plot_singular_value_spectra,
    plot_effective_rank_comparison,
    plot_response_heterogeneity,
    plot_hidden_state_pca,
    plot_weight_matrices,
    plot_phase_portrait,
    plot_vector_field,
    plot_summary_table,
    
    # Comprehensive figures
    create_comprehensive_figure,
    generate_all_figures,
    
    # Color palettes
    COLORS,
    PHASE_COLORS,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Configuration
    "DevelopmentalConfig",
    "FactorizationType", 
    "CellType",
    "create_premature_config",
    "create_mature_config",
    
    # Models
    "BrainInspiredRNN",
    "create_model",
    "LowRankLinear",
    "SVDLowRankLayer",
    "EigenConstrainedLayer",
    "SLINLayer",
    "DevGRUCell",
    "SwitchingDevGRUCell",
    "SLINCell",
    "ResponseHeterogeneityAnalyzer",
    
    # Tasks
    "TaskType",
    "TrialData",
    "SessionData",
    "ReversalLearningTask",
    "TwoStageTask",
    "ProbabilisticRewardTask",
    "TaskDataset",
    "ModelFreeRL",
    "BayesianAgent",
    "compute_choice_accuracy",
    "compute_negative_log_likelihood",
    
    # Analysis
    "PhasePortraitData",
    "VectorFieldData",
    "DevelopmentalComparison",
    "FixedPointFinder",
    "LogitAnalyzer",
    "PhasePortraitGenerator",
    "VectorFieldGenerator",
    "DevelopmentalComparisonAnalyzer",
    
    # Visualization
    "plot_learning_curves",
    "plot_singular_value_spectra",
    "plot_effective_rank_comparison",
    "plot_response_heterogeneity",
    "plot_hidden_state_pca",
    "plot_weight_matrices",
    "plot_phase_portrait",
    "plot_vector_field",
    "plot_summary_table",
    "create_comprehensive_figure",
    "generate_all_figures",
    "COLORS",
    "PHASE_COLORS",
]


# =============================================================================
# Package Initialization
# =============================================================================

def get_version():
    """Return the package version."""
    return __version__


def get_available_factorizations():
    """Return list of available factorization types."""
    return [ft.value for ft in FactorizationType]


def get_available_cell_types():
    """Return list of available cell types."""
    return [ct.value for ct in CellType]


def get_available_tasks():
    """Return list of available cognitive tasks."""
    return [tt.value for tt in TaskType]


# Print welcome message when imported interactively
import sys
if hasattr(sys, 'ps1'):  # Interactive mode
    print(f"Brain-Inspired Developmental RNN Framework v{__version__}")
    print("Use 'help(module_name)' for documentation.")
