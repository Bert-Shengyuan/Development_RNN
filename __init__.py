"""
Brain-Inspired Low-Rank RNN Framework for Cognitive Tasks
==========================================================

A comprehensive Python implementation connecting fMRI dynamics to low-rank RNN structure,
enabling comparison between developmentally-constrained neural network architectures.

Theoretical Foundation:
-----------------------
Based on empirical findings that mature infant brains exhibit:
- Lower effective rank (D ≈ 5-6 vs D ≈ 7-8 for premature)
- Higher response heterogeneity
- Faster singular value decay (γ_mature > γ_premature)

Mathematical Framework:
-----------------------
Single-step dynamics: x_{t+1} = A * x_t + η_t
Low-rank factorization: A ≈ U Σ V^T

RNN Implementation:
    h_{t+1} = (1-α)h_t + α * tanh(m n^T h_t + M N^T h_t + W_in u_t)
    y_t = W_out h_t

Modules:
--------
- brain_inspired_rnn: Core RNN architectures with developmental constraints
- cognitive_tasks: Reversal learning, two-stage, and probabilistic reward tasks
- training: Training pipeline with specialized regularization
- visualization: Publication-quality figure generation
- analysis: Dynamical systems analysis and mechanism extraction
- main: Command-line interface and experiment orchestration

Reference:
----------
Ji-An, L., Benna, M.K., & Mattar, M.G. (2025). 
"Discovering cognitive strategies with tiny recurrent neural networks." Nature.

Usage:
------
    python main.py --task reversal --n_epochs 100
    python main.py --all_tasks --output_dir ./results
"""

from .brain_inspired_rnn import (
    DevelopmentalConfig,
    create_premature_config,
    create_mature_config,
    LowRankRecurrentLayer,
    BrainInspiredRNN,
    TinyGRU,
    get_model_metrics,
    compute_effective_rank,
    compute_nuclear_norm,
    compute_response_heterogeneity
)

from .cognitive_tasks import (
    TaskType,
    ReversalLearningTask,
    TwoStageTask,
    ProbabilisticRewardTask,
    TaskDataset,
    generate_reversal_session,
    generate_two_stage_session,
    generate_probabilistic_session
)

from .training import (
    TrainingConfig,
    Trainer,
    run_developmental_comparison,
    analyze_learning_dynamics
)

from .visualization import (
    COLORS,
    create_comprehensive_figure,
    generate_all_figures,
    plot_learning_curves,
    plot_singular_value_spectra,
    plot_effective_rank_comparison,
    plot_response_heterogeneity,
    plot_task_performance_comparison
)

from .analysis import (
    FixedPointFinder,
    DynamicsAnalyzer,
    compare_developmental_mechanisms,
    compute_jacobian,
    analyze_fixed_point_structure
)

__version__ = "1.0.0"
__author__ = "Computational Neuroscience Research"

__all__ = [
    # Configuration
    'DevelopmentalConfig',
    'create_premature_config', 
    'create_mature_config',
    'TrainingConfig',
    
    # RNN Models
    'LowRankRecurrentLayer',
    'BrainInspiredRNN',
    'TinyGRU',
    
    # Tasks
    'TaskType',
    'ReversalLearningTask',
    'TwoStageTask', 
    'ProbabilisticRewardTask',
    'TaskDataset',
    
    # Training
    'Trainer',
    'run_developmental_comparison',
    'analyze_learning_dynamics',
    
    # Visualization
    'COLORS',
    'create_comprehensive_figure',
    'generate_all_figures',
    
    # Analysis
    'FixedPointFinder',
    'DynamicsAnalyzer',
    'compare_developmental_mechanisms',
    
    # Metrics
    'get_model_metrics',
    'compute_effective_rank',
    'compute_nuclear_norm',
    'compute_response_heterogeneity'
]
