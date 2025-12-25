"""
================================================================================
COMPREHENSIVE FRAMEWORK DOCUMENTATION
Brain-Inspired Developmental RNN Analysis
================================================================================

This document explains the relationship between the first-round files and the
current merged implementation.

================================================================================
HISTORY AND RELATIONSHIP
================================================================================

FIRST ROUND (Exploratory Phase)
-------------------------------
Five files were created to explore different architectural approaches:

1. enhanced_low_rank_rnn.py
   - Multiple factorization approaches (SVD, Eigenvalue-constrained, SLIN)
   - ResponseHeterogeneityAnalyzer class
   - EnhancedBrainInspiredRNN with selectable factorization
   
2. developmental_gru.py
   - Low-rank GRU variants (DevGRU, SwitchingDevGRU)
   - LowRankLinear layer implementation
   - SLINCell following Ji-An et al. Eq. 4
   
3. refined_cognitive_tasks.py
   - Detailed SessionData structure
   - ModelFreeRL and BayesianAgent classical models
   - Comprehensive input encoding documentation
   
4. dynamical_analysis.py
   - LogitAnalyzer for L(t) trajectories
   - PhasePortraitGenerator (Figure 3 style)
   - VectorFieldGenerator (Figure 4 style)
   - DevelopmentalComparisonAnalyzer
   
5. framework_summary.py
   - Theoretical documentation
   - Usage guide


SECOND ROUND (Polished Framework)
---------------------------------
Four consolidated files with improved documentation:

1. brain_inspired_rnn.py - Core architectures (simplified)
2. cognitive_tasks.py - Task implementations
3. training.py - Training pipeline
4. analysis.py - Dynamics analysis

PROBLEM: The polished framework OMITTED several key components:
   - EigenConstrainedLayer (eigenvalue distribution control)
   - SLINLayer and SLINCell (switching linear dynamics)
   - DevGRUCell and SwitchingDevGRUCell (low-rank GRU)
   - VectorFieldGenerator (for d>1 models)
   - Comprehensive ResponseHeterogeneityAnalyzer


MERGED FRAMEWORK (Current)
--------------------------
This merged implementation combines ALL functionality:

/home/claude/merged_framework/
├── brain_inspired_rnn.py  (1000+ lines)
│   ├── Configuration classes with all parameters
│   ├── FactorizationType enum (SVD_DIRECT, EIGENVALUE_CONSTRAINED, SLIN, HYBRID)
│   ├── CellType enum (LEAKY_TANH, DEV_GRU, SWITCHING_DEV_GRU, SLIN_CELL)
│   ├── LowRankLinear layer
│   ├── SVDLowRankLayer (W = m n^T + M N^T)
│   ├── EigenConstrainedLayer (W = V Lambda V^T)
│   ├── SLINLayer (h = W^{(x)} h + b^{(x)})
│   ├── DevGRUCell (low-rank GRU gates)
│   ├── SwitchingDevGRUCell (input-dependent GRU)
│   ├── SLINCell (pure switching linear)
│   ├── ResponseHeterogeneityAnalyzer (multi-metric)
│   └── BrainInspiredRNN (unified model)
│
├── cognitive_tasks.py (from polished)
│   ├── ReversalLearningTask
│   ├── TwoStageTask
│   ├── ModelFreeRL
│   ├── BayesianAgent
│   └── TaskDataset
│
├── training.py (from polished)
│   ├── TrainingConfig
│   ├── Trainer class
│   └── run_developmental_comparison()
│
├── analysis.py (1000+ lines, merged)
│   ├── PhasePortraitData, VectorFieldData, DevelopmentalComparison
│   ├── FixedPointFinder
│   ├── LogitAnalyzer
│   ├── PhasePortraitGenerator (Figure 3)
│   ├── VectorFieldGenerator (Figure 4) <-- RESTORED
│   ├── DynamicsAnalyzer (Lyapunov, geometry)
│   └── DevelopmentalComparisonAnalyzer
│
└── DOCUMENTATION.py (this file)


================================================================================
MATHEMATICAL FRAMEWORK
================================================================================

1. SINGLE-STEP DYNAMICS
-----------------------
The fMRI dynamics follow a linear dynamical system approximation:

    x_{t+1} = A x_t + eta_t

where:
    - x_t in R^n is the neural state
    - A in R^{n x n} is the effective connectivity matrix
    - eta_t captures noise and external inputs


2. FACTORIZATION APPROACHES
---------------------------

SVD Direct:
    W_rec = m n^T + M N^T
    
    - m, n in R^N define a rank-1 "common mode"
    - M in R^{N x r}, N in R^{N x r} define rank-r component
    - Explicit rank constraint via matrix dimensions

Eigenvalue-Constrained:
    W_rec = V Lambda V^{-1}  (or V Lambda V^T for orthogonal V)
    
    - Constrains eigenvalue distribution rather than rank
    - Better captures spectral properties
    - Mature: concentrated eigenvalues
    - Premature: spread eigenvalues

Switching Linear (SLIN):
    h_t = W^{(x_{t-1})} h_{t-1} + b^{(x_{t-1})}
    
    - Input-dependent weight matrices
    - Following Ji-An et al. (2025) Eq. 4
    - Each W^{(x)} is low-rank factorized


3. GRU VARIANTS
---------------

Standard GRU:
    r_t = sigma(W_ir x_t + W_hr h_{t-1} + b_r)
    z_t = sigma(W_iz x_t + W_hz h_{t-1} + b_z)
    n_t = tanh(W_in x_t + r_t * (W_hn h_{t-1} + b_n))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}

DevGRU:
    Same equations, but W_h* are LOW-RANK factorized:
    W_hr = U_r V_r^T, W_hz = U_z V_z^T, W_hn = U_n V_n^T

SwitchingDevGRU:
    All weights are INPUT-DEPENDENT:
    W_hr^{(x)}, W_hz^{(x)}, W_hn^{(x)}, b^{(x)}
    
    Following Ji-An et al. Eq. 2


4. TRAINING OBJECTIVE
---------------------
    L_total = L_task + alpha * ||W||_* + beta * D_eff(W) 
              + L_spectral - lambda * L_hetero

where:
    - ||W||_* = sum(sigma_i) is the nuclear norm
    - D_eff(W) = (sum sigma_i)^2 / sum(sigma_i^2) is effective rank
    - L_spectral = sum_i (sigma_i - sigma_i^target)^2
    - L_hetero = Var(||dh/dI_j||_2) is response heterogeneity


5. DEVELOPMENTAL CONSTRAINTS
----------------------------

                    PREMATURE              MATURE
                    ---------              ------
Effective rank      Higher (7-8)           Lower (5-6)
SV decay (gamma)    0.15-0.2               0.35-0.4
Heterogeneity       Lower                  Higher
Eigenvalue spread   Broader                Concentrated
Nuclear norm reg    Weaker (5e-4)          Stronger (2e-3)


================================================================================
ANALYSIS METHODOLOGY (Following Ji-An et al.)
================================================================================

1. PHASE PORTRAIT ANALYSIS (Figure 3)
-------------------------------------
For d=1 models:
    L(t) = log(P(A1)/P(A2))

Phase portrait: Plot Delta-L(t) vs L(t), colored by input condition
    - Fixed points: where Delta-L = 0
    - Slopes: indicate effective learning rate
    - Curvature: indicates state-dependent learning

2. PREFERENCE SETPOINTS
-----------------------
    u_I = L*_I / max_I |L*_I|

where L*_I is the fixed point for input I.

3. VECTOR FIELD ANALYSIS (Figure 4)
-----------------------------------
For d=2 models:
    - Plot (Delta-h1, Delta-h2) arrows on (h1, h2) grid
    - Identify attractors (line attractors, point attractors)
    - Compare axis-aligned vs tilted arrows

4. MODEL-BASED vs MODEL-FREE SIGNATURES
---------------------------------------
Two-stage task analysis:
    MF_index = mean(P(stay|rew) - P(stay|no_rew))
    MB_index = P(stay|common,rew) - P(stay|rare,rew)


================================================================================
USAGE GUIDE
================================================================================

1. CREATE MODELS WITH DIFFERENT FACTORIZATIONS
----------------------------------------------

from brain_inspired_rnn import (
    create_model, FactorizationType, CellType
)

# SVD Direct (default)
model_svd = create_model(
    "mature", 
    factorization_type=FactorizationType.SVD_DIRECT
)

# Eigenvalue-constrained
model_eigen = create_model(
    "mature", 
    factorization_type=FactorizationType.EIGENVALUE_CONSTRAINED
)

# With DevGRU cell
model_gru = create_model(
    "mature", 
    cell_type=CellType.DEV_GRU
)


2. GENERATE PHASE PORTRAITS
---------------------------

from analysis import PhasePortraitGenerator

phase_gen = PhasePortraitGenerator(model)
phase_data = phase_gen.generate_from_data(inputs, actions, rewards)
ax = phase_gen.plot_phase_portrait(phase_data, title="Phase Portrait")


3. GENERATE VECTOR FIELDS (d>1)
-------------------------------

from analysis import VectorFieldGenerator

vector_gen = VectorFieldGenerator(model)
vf_data = vector_gen.generate_vector_field(n_grid=15)
fig = vector_gen.plot_all_conditions(vf_data)


4. COMPREHENSIVE COMPARISON
---------------------------

from analysis import DevelopmentalComparisonAnalyzer

comparator = DevelopmentalComparisonAnalyzer(premature_model, mature_model)
comparison = comparator.run_comparison(inputs, actions, rewards)
fig = comparator.plot_comparison(comparison, save_path="comparison.png")


================================================================================
FILE SUMMARY
================================================================================

MERGED FRAMEWORK FILES:

brain_inspired_rnn.py (~1000 lines)
    - ALL factorization approaches
    - ALL cell types
    - Response heterogeneity analyzer
    - Factory functions

analysis.py (~1000 lines)
    - Phase portraits (Figure 3)
    - Vector fields (Figure 4)
    - Fixed point analysis
    - Lyapunov exponents
    - MB/MF signatures
    - Developmental comparison

cognitive_tasks.py (~965 lines)
    - Reversal learning
    - Two-stage task
    - Classical models
    - Dataset utilities

training.py (~833 lines)
    - Training configuration
    - Trainer class
    - Comparison pipeline


================================================================================
REFERENCES
================================================================================

1. Ji-An, L., Benna, M.K., & Mattar, M.G. (2025).
   "Discovering cognitive strategies with tiny recurrent neural networks."
   Nature.

2. Sussillo, D., & Barak, O. (2013).
   "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional
   Recurrent Neural Networks." Neural Computation.

3. Mastrogiuseppe, F., & Ostojic, S. (2018).
   "Linking Connectivity, Dynamics, and Computations in Low-Rank
   Recurrent Neural Networks." Neuron.

4. Daw, N.D., et al. (2011).
   "Model-Based Influences on Humans' Choices and Striatal Prediction Errors."
   Neuron.

================================================================================
"""

if __name__ == "__main__":
    print("Framework Documentation")
    print("=" * 60)
    print("\nThis document explains the relationship between:")
    print("  - First round files (5 exploratory modules)")
    print("  - Second round files (4 polished modules)")
    print("  - Merged framework (comprehensive implementation)")
    print("\nKey restored functionality:")
    print("  1. EigenConstrainedLayer")
    print("  2. SLINLayer and SLINCell")
    print("  3. DevGRUCell and SwitchingDevGRUCell")
    print("  4. VectorFieldGenerator")
    print("  5. ResponseHeterogeneityAnalyzer")
    print("\nSee the docstring for complete documentation.")
    print("=" * 60)
