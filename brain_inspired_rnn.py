#!/usr/bin/env python3
"""
Comprehensive Brain-Inspired Low-Rank RNN Framework
=====================================================

This module implements developmentally-constrained recurrent neural networks
with MULTIPLE FACTORIZATION APPROACHES, merging functionality from:
- enhanced_low_rank_rnn.py (multiple factorization types)
- developmental_gru.py (low-rank GRU variants)

Factorization Approaches
------------------------
1. SVD_DIRECT: W_rec = m n^T + M N^T (explicit low-rank)
2. EIGENVALUE_CONSTRAINED: W_rec = V Lambda V^{-1} (spectral control)
3. SLIN: h_t = W^{(x)} h_{t-1} + b^{(x)} (switching linear dynamics)

GRU Variants
------------
1. DevGRU: Low-rank factorization of hidden-to-hidden weights
2. SwitchingDevGRU: Input-dependent GRU weights (Ji-An et al. Eq. 2)
3. SLINCell: Pure switching linear dynamics (Ji-An et al. Eq. 4)

Theoretical Foundation
----------------------
Mature brains exhibit:
    - Lower effective rank (r_mature ~ 5-6 vs r_premature ~ 7-8)
    - Faster singular value decay: sigma_i ~ exp(-gamma * i)
    - Higher response heterogeneity

References
----------
- Ji-An et al. (2025) "Discovering cognitive strategies with tiny RNNs"
- Sussillo & Barak (2013) "Opening the Black Box"

Author: Computational Neuroscience Research
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import math
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Enums and Configuration
# =============================================================================

class FactorizationType(Enum):
    """Available factorization approaches for recurrent weights."""
    SVD_DIRECT = "svd_direct"           # Direct low-rank factorization
    EIGENVALUE_CONSTRAINED = "eigen"    # Eigenvalue distribution constraint
    SLIN = "slin"                       # Switching linear dynamics
    HYBRID = "hybrid"                   # Combination approach


class CellType(Enum):
    """Available RNN cell types."""
    LEAKY_TANH = "leaky_tanh"           # Simple leaky tanh dynamics
    DEV_GRU = "dev_gru"                 # Low-rank GRU
    SWITCHING_DEV_GRU = "switching_gru" # Input-dependent GRU
    SLIN_CELL = "slin"                  # Switching linear


@dataclass
class DevelopmentalConfig:
    """
    Configuration for developmental stage-specific RNN constraints.
    
    Mathematical Framework
    ----------------------
    The training objective is:
        L_total = L_task + alpha * ||W||_* + beta * D_eff(W) 
                  + L_spectral - lambda * L_hetero
    
    Parameters
    ----------
    n_hidden : int
        Number of hidden units (d dynamical variables).
    n_input : int
        Input dimension.
    n_output : int
        Output dimension (typically 2 for action selection).
    rank : int
        Target effective rank for low-rank factorization.
    sv_decay_gamma : float
        Singular value decay rate: sigma_i ~ exp(-gamma * i).
        Higher gamma -> faster decay -> lower effective dimensionality.
    eigenvalue_spread : float
        Variance of eigenvalue distribution (for EIGENVALUE_CONSTRAINED).
    max_spectral_radius : float
        Maximum spectral radius for stability.
    response_hetero_weight : float
        Weight for response heterogeneity loss term (lambda).
    nuclear_norm_alpha : float
        Nuclear norm regularization strength ||W||_*.
    effective_rank_beta : float
        Effective rank penalty weight.
    leak_alpha : float
        Leak parameter for slow dynamics: (1-alpha)*h + alpha*f(h).
    noise_std : float
        Process noise standard deviation.
    factorization_type : FactorizationType
        Type of weight matrix factorization.
    cell_type : CellType
        Type of RNN cell.
    developmental_stage : str
        Label: "premature" or "mature".
    """
    # Network dimensions
    n_hidden: int = 32
    n_input: int = 3
    n_output: int = 2
    
    # Rank constraints
    rank: int = 6
    rank_tolerance: float = 0.5
    
    # Spectral constraints
    sv_decay_gamma: float = 0.3
    eigenvalue_spread: float = 0.5
    max_spectral_radius: float = 0.95
    
    # Response heterogeneity
    response_hetero_weight: float = 0.1
    selectivity_target: float = 1.5
    
    # Regularization
    nuclear_norm_alpha: float = 1e-3
    effective_rank_beta: float = 1e-2
    weight_decay: float = 1e-4
    
    # Dynamics
    leak_alpha: float = 0.1
    noise_std: float = 0.01
    
    # Architecture choices
    factorization_type: FactorizationType = FactorizationType.SVD_DIRECT
    cell_type: CellType = CellType.LEAKY_TANH
    
    # Identification
    developmental_stage: str = "mature"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.n_hidden > 0, "n_hidden must be positive"
        assert 0 < self.rank <= self.n_hidden, "rank must be in (0, n_hidden]"
        assert 0 < self.leak_alpha <= 1, "leak_alpha must be in (0, 1]"
        assert self.sv_decay_gamma >= 0, "sv_decay_gamma must be non-negative"


def create_premature_config(
    n_hidden: int = 32,
    n_input: int = 3,
    n_output: int = 2,
    factorization_type: FactorizationType = FactorizationType.SVD_DIRECT,
    cell_type: CellType = CellType.LEAKY_TANH
) -> DevelopmentalConfig:
    """
    Create configuration for premature brain-inspired RNN.
    
    Characteristics:
        - Higher effective rank (less compression)
        - Slower singular value decay
        - Lower response heterogeneity
        - Broader eigenvalue spread
    """
    return DevelopmentalConfig(
        n_hidden=n_hidden,
        n_input=n_input,
        n_output=n_output,
        rank=8,                           # Higher effective rank
        sv_decay_gamma=0.2,               # Slower decay
        eigenvalue_spread=0.7,            # Broader distribution
        response_hetero_weight=0.05,      # Lower heterogeneity emphasis
        selectivity_target=1.0,
        nuclear_norm_alpha=5e-4,          # Weaker rank regularization
        effective_rank_beta=5e-3,
        factorization_type=factorization_type,
        cell_type=cell_type,
        developmental_stage="premature"
    )


def create_mature_config(
    n_hidden: int = 32,
    n_input: int = 3,
    n_output: int = 2,
    factorization_type: FactorizationType = FactorizationType.SVD_DIRECT,
    cell_type: CellType = CellType.LEAKY_TANH
) -> DevelopmentalConfig:
    """
    Create configuration for mature brain-inspired RNN.
    
    Characteristics:
        - Lower effective rank (more compression)
        - Faster singular value decay
        - Higher response heterogeneity (specialization)
        - Concentrated eigenvalue spectrum
    """
    return DevelopmentalConfig(
        n_hidden=n_hidden,
        n_input=n_input,
        n_output=n_output,
        rank=5,                           # Lower effective rank
        sv_decay_gamma=0.4,               # Faster decay
        eigenvalue_spread=0.4,            # Concentrated distribution
        response_hetero_weight=0.15,      # Higher heterogeneity emphasis
        selectivity_target=2.0,
        nuclear_norm_alpha=2e-3,          # Stronger rank regularization
        effective_rank_beta=2e-2,
        factorization_type=factorization_type,
        cell_type=cell_type,
        developmental_stage="mature"
    )


# =============================================================================
# Low-Rank Linear Layer (for GRU variants)
# =============================================================================

class LowRankLinear(nn.Module):
    """
    Linear layer with explicit low-rank factorization.
    
    W = U V^T where U in R^{m x r}, V in R^{n x r}
    
    This ensures rank(W) <= r.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int, 
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))
        
        # Low-rank factors
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        self.V = nn.Parameter(torch.empty(in_features, self.rank))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize with scaled random values."""
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.normal_(self.U, std=std)
        nn.init.normal_(self.V, std=std)
        
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def get_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix W = U V^T."""
        return self.U @ self.V.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Efficient forward: y = x @ V @ U^T + b."""
        intermediate = x @ self.V  # [batch, rank]
        output = intermediate @ self.U.T  # [batch, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


# =============================================================================
# Factorization Approach 1: SVD Direct (Low-Rank)
# =============================================================================

class SVDLowRankLayer(nn.Module):
    """
    Low-rank recurrent layer via direct SVD factorization.
    
    W_rec = m n^T + M N^T
    
    where:
    - m, n in R^N define a rank-1 "common mode" component
    - M in R^{N x r}, N in R^{N x r} define the rank-r component
    """
    
    def __init__(self, config: DevelopmentalConfig):
        super().__init__()
        self.config = config
        N = config.n_hidden
        r = config.rank
        
        # Rank-1 component (common mode)
        self.m = nn.Parameter(torch.randn(N, 1) * 0.1)
        self.n = nn.Parameter(torch.randn(N, 1) * 0.1)
        
        # Rank-r component
        self.M = nn.Parameter(torch.randn(N, r) * 0.1)
        self.N = nn.Parameter(torch.randn(N, r) * 0.1)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(N))
        
        self._initialize_spectral_structure()
    
    def _initialize_spectral_structure(self):
        """Initialize M, N with target singular value decay."""
        r = self.config.rank
        gamma = self.config.sv_decay_gamma
        
        target_sv = torch.exp(-gamma * torch.arange(r, dtype=torch.float32))
        target_sv = target_sv / target_sv.sum() * r
        
        with torch.no_grad():
            sqrt_sv = torch.sqrt(target_sv)
            self.M.data = self.M.data * sqrt_sv.unsqueeze(0)
            self.N.data = self.N.data * sqrt_sv.unsqueeze(0)
    
    def get_weight_matrix(self) -> torch.Tensor:
        """Compute W = m n^T + M N^T."""
        return self.m @ self.n.T + self.M @ self.N.T
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Single step with leaky integration."""
        alpha = self.config.leak_alpha
        W = self.get_weight_matrix()
        
        pre_activation = h @ W.T + x + self.bias
        h_new = (1 - alpha) * h + alpha * torch.tanh(pre_activation)
        
        return h_new


# =============================================================================
# Factorization Approach 2: Eigenvalue-Constrained
# =============================================================================

class EigenConstrainedLayer(nn.Module):
    """
    Recurrent layer with eigenvalue distribution constraints.
    
    W_rec = V Lambda V^{-1}
    
    For orthogonal V: W = V Lambda V^T
    
    Mature networks: concentrated eigenvalues (low variance)
    Premature networks: spread eigenvalues (high variance)
    """
    
    def __init__(self, config: DevelopmentalConfig):
        super().__init__()
        self.config = config
        N = config.n_hidden
        
        # Parameterize eigenvalues directly
        eigenvalue_init = self._initialize_eigenvalues()
        self.eigenvalues = nn.Parameter(eigenvalue_init)
        
        # Parameterize eigenvectors via orthogonal matrix
        self.V = nn.Parameter(torch.linalg.qr(torch.randn(N, N))[0])
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(N))
    
    def _initialize_eigenvalues(self) -> torch.Tensor:
        """Initialize eigenvalues with developmental stage-specific distribution."""
        N = self.config.n_hidden
        gamma = self.config.sv_decay_gamma
        spread = self.config.eigenvalue_spread
        max_radius = self.config.max_spectral_radius
        
        # Base eigenvalues with exponential decay
        base_eigenvalues = max_radius * torch.exp(
            -gamma * torch.arange(N, dtype=torch.float32)
        )
        
        # Add noise based on developmental stage
        noise = torch.randn(N) * spread * 0.1
        eigenvalues = base_eigenvalues + noise
        
        # Ensure stability
        eigenvalues = eigenvalues.clamp(max=max_radius)
        
        return eigenvalues
    
    def get_weight_matrix(self) -> torch.Tensor:
        """Compute W = V Lambda V^T."""
        Lambda = torch.diag(self.eigenvalues)
        return self.V @ Lambda @ self.V.T
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Single step with leaky integration."""
        alpha = self.config.leak_alpha
        W = self.get_weight_matrix()
        
        pre_activation = h @ W.T + x + self.bias
        h_new = (1 - alpha) * h + alpha * torch.tanh(pre_activation)
        
        return h_new
    
    def compute_eigenvalue_concentration(self) -> float:
        """
        Compute eigenvalue concentration metric.
        
        Higher values -> more concentrated (characteristic of mature).
        Metric: (sum lambda_i)^2 / sum(lambda_i^2)
        """
        eigenvalues = self.eigenvalues.detach().abs()
        sum_sq = (eigenvalues.sum()) ** 2
        sq_sum = (eigenvalues ** 2).sum()
        return float(sum_sq / sq_sum)


# =============================================================================
# Factorization Approach 3: Switching Linear (SLIN)
# =============================================================================

class SLINLayer(nn.Module):
    """
    Switching Linear Neural Network layer following Ji-An et al. (2025) Eq. 4.
    
    h_t = W^{(x_{t-1})} h_{t-1} + b^{(x_{t-1})}
    
    Input-dependent weight matrices, each low-rank factorized.
    """
    
    def __init__(self, config: DevelopmentalConfig, n_input_conditions: int = 4):
        super().__init__()
        self.config = config
        self.n_conditions = n_input_conditions
        N = config.n_hidden
        r = config.rank
        
        # Input-dependent weight matrices (low-rank factorized)
        self.M = nn.Parameter(torch.randn(n_input_conditions, N, r) * 0.1)
        self.N = nn.Parameter(torch.randn(n_input_conditions, N, r) * 0.1)
        
        # Input-dependent biases
        self.bias = nn.Parameter(torch.zeros(n_input_conditions, N))
        
        # Optional symmetry constraint
        self.symmetric = False
        
        self._initialize_spectral_structure()
    
    def _initialize_spectral_structure(self):
        """Initialize all condition-specific weight matrices."""
        r = self.config.rank
        gamma = self.config.sv_decay_gamma
        
        target_sv = torch.exp(-gamma * torch.arange(r, dtype=torch.float32))
        target_sv = target_sv / target_sv.sum() * r
        
        with torch.no_grad():
            sqrt_sv = torch.sqrt(target_sv)
            for c in range(self.n_conditions):
                self.M.data[c] = self.M.data[c] * sqrt_sv.unsqueeze(0)
                self.N.data[c] = self.N.data[c] * sqrt_sv.unsqueeze(0)
    
    def get_weight_matrix(self, condition_idx: int) -> torch.Tensor:
        """Get weight matrix for specific input condition."""
        W = self.M[condition_idx] @ self.N[condition_idx].T
        
        if self.symmetric:
            W = (W + W.T) / 2
        
        return W
    
    def forward(self, h: torch.Tensor, x: torch.Tensor, 
                condition_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input-dependent dynamics.
        
        For one-hot input x, this selects the appropriate W^{(x)}.
        """
        batch_size = h.shape[0]
        h_new = torch.zeros_like(h)
        
        for c in range(self.n_conditions):
            mask = (condition_indices == c)
            if mask.any():
                W_c = self.get_weight_matrix(c)
                b_c = self.bias[c]
                
                h_c = h[mask]
                h_new[mask] = h_c @ W_c.T + b_c
        
        return h_new


# =============================================================================
# GRU Cell Variants
# =============================================================================

class DevGRUCell(nn.Module):
    """
    Developmental GRU Cell with low-rank recurrent constraints.
    
    The recurrent weight matrices (W_hr, W_hz, W_hn) are factorized
    as low-rank matrices to implement developmental constraints.
    
    Standard GRU dynamics:
        r_t = sigma(W_ir x_t + W_hr h_{t-1} + b_r)
        z_t = sigma(W_iz x_t + W_hz h_{t-1} + b_z)
        n_t = tanh(W_in x_t + r_t * (W_hn h_{t-1} + b_n))
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """
    
    def __init__(self, config: DevelopmentalConfig):
        super().__init__()
        self.config = config
        self.input_size = config.n_input
        self.hidden_size = config.n_hidden
        self.rank = config.rank
        
        # Input-to-hidden weights (full rank)
        self.W_ir = nn.Linear(config.n_input, config.n_hidden)
        self.W_iz = nn.Linear(config.n_input, config.n_hidden)
        self.W_in = nn.Linear(config.n_input, config.n_hidden)
        
        # Hidden-to-hidden weights (LOW RANK for developmental constraint)
        self.W_hr = LowRankLinear(config.n_hidden, config.n_hidden, self.rank)
        self.W_hz = LowRankLinear(config.n_hidden, config.n_hidden, self.rank)
        self.W_hn = LowRankLinear(config.n_hidden, config.n_hidden, self.rank)
        
        self._init_with_spectral_constraints()
    
    def _init_with_spectral_constraints(self):
        """Initialize recurrent weights with target spectral properties."""
        gamma = self.config.sv_decay_gamma
        
        target_sv = torch.exp(-gamma * torch.arange(self.rank, dtype=torch.float32))
        target_sv = target_sv / target_sv.max()
        
        with torch.no_grad():
            sqrt_sv = torch.sqrt(target_sv)
            for layer in [self.W_hr, self.W_hz, self.W_hn]:
                layer.U.data = layer.U.data * sqrt_sv.unsqueeze(0)
                layer.V.data = layer.V.data * sqrt_sv.unsqueeze(0)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single step of GRU dynamics."""
        # Reset gate
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        
        # Update gate
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h))
        
        # Candidate activation
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h))
        
        # New hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new
    
    def get_recurrent_weights(self) -> Dict[str, torch.Tensor]:
        """Return reconstructed recurrent weight matrices."""
        return {
            'W_hr': self.W_hr.get_weight(),
            'W_hz': self.W_hz.get_weight(),
            'W_hn': self.W_hn.get_weight()
        }


class SwitchingDevGRUCell(nn.Module):
    """
    Switching GRU with developmental constraints (Ji-An et al. Eq. 2).
    
    All weights and biases are input-dependent:
        r_t = sigma(b_ir^{(x)} + W_hr^{(x)} h_{t-1} + b_hr^{(x)})
        z_t = sigma(b_iz^{(x)} + W_hz^{(x)} h_{t-1} + b_hz^{(x)})
        n_t = tanh(b_in^{(x)} + r_t * (W_hn^{(x)} h_{t-1} + b_hn^{(x)}))
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """
    
    def __init__(self, config: DevelopmentalConfig, n_input_conditions: int = 4):
        super().__init__()
        self.config = config
        self.n_conditions = n_input_conditions
        self.hidden_size = config.n_hidden
        self.rank = config.rank
        
        # Low-rank factors for each input condition
        # W^{(i)} = U^{(i)} @ V^{(i)T}
        self.U = nn.Parameter(
            torch.randn(n_input_conditions, 3 * config.n_hidden, self.rank) * 0.1
        )
        self.V = nn.Parameter(
            torch.randn(n_input_conditions, config.n_hidden, self.rank) * 0.1
        )
        
        # Input-dependent biases
        self.bias_ih = nn.Parameter(
            torch.zeros(n_input_conditions, 3 * config.n_hidden)
        )
        self.bias_hh = nn.Parameter(
            torch.zeros(n_input_conditions, 3 * config.n_hidden)
        )
        
        self._init_with_spectral_constraints()
    
    def _init_with_spectral_constraints(self):
        """Initialize with target spectral properties."""
        gamma = self.config.sv_decay_gamma
        target_sv = torch.exp(-gamma * torch.arange(self.rank, dtype=torch.float32))
        target_sv = target_sv / target_sv.max()
        
        with torch.no_grad():
            sqrt_sv = torch.sqrt(target_sv)
            for i in range(self.n_conditions):
                self.U.data[i] = self.U.data[i] * sqrt_sv.unsqueeze(0)
                self.V.data[i] = self.V.data[i] * sqrt_sv.unsqueeze(0)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input-dependent dynamics.
        
        x should be one-hot encoded [batch, n_conditions].
        """
        batch_size = x.shape[0]
        
        # Compute W @ V^T for each condition
        W_hh = torch.bmm(self.U, self.V.transpose(1, 2))  # [n_cond, 3H, H]
        
        # Select based on input
        x_expand = x.unsqueeze(-1).unsqueeze(-1)  # [batch, n_cond, 1, 1]
        W_hh_expand = W_hh.unsqueeze(0)  # [1, n_cond, 3H, H]
        
        trial_W_hh = (x_expand * W_hh_expand).sum(1)  # [batch, 3H, H]
        
        # Select biases
        trial_bias_ih = (x.unsqueeze(-1) * self.bias_ih.unsqueeze(0)).sum(1)
        trial_bias_hh = (x.unsqueeze(-1) * self.bias_hh.unsqueeze(0)).sum(1)
        
        # Compute recurrent term
        rec_term = torch.bmm(trial_W_hh, h.unsqueeze(-1)).squeeze(-1)
        
        # Gate computations
        gates_i = trial_bias_ih
        gates_h = rec_term + trial_bias_hh
        
        r_i, z_i, n_i = gates_i.chunk(3, dim=1)
        r_h, z_h, n_h = gates_h.chunk(3, dim=1)
        
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        n = torch.tanh(n_i + r * n_h)
        
        h_new = (1 - z) * n + z * h
        
        return h_new


class SLINCell(nn.Module):
    """
    Switching Linear Neural Network (SLIN) cell.
    
    Following Ji-An et al. Eq. 4:
        h_t = W^{(x_{t-1})} h_{t-1} + b^{(x_{t-1})}
    
    Pure linear dynamics with input-dependent weights.
    """
    
    def __init__(self, config: DevelopmentalConfig, n_input_conditions: int = 4):
        super().__init__()
        self.config = config
        self.n_conditions = n_input_conditions
        self.hidden_size = config.n_hidden
        self.rank = config.rank
        self.symmetric = False
        
        # Low-rank factors: W^{(i)} = U^{(i)} @ V^{(i)T}
        self.U = nn.Parameter(
            torch.randn(n_input_conditions, config.n_hidden, self.rank) * 0.1
        )
        self.V = nn.Parameter(
            torch.randn(n_input_conditions, config.n_hidden, self.rank) * 0.1
        )
        
        # Input-dependent biases
        self.bias = nn.Parameter(torch.zeros(n_input_conditions, config.n_hidden))
        
        self._init_with_spectral_constraints()
    
    def _init_with_spectral_constraints(self):
        """Initialize with target spectral properties."""
        gamma = self.config.sv_decay_gamma
        target_sv = torch.exp(-gamma * torch.arange(self.rank, dtype=torch.float32))
        target_sv = target_sv / target_sv.max() * self.config.max_spectral_radius
        
        with torch.no_grad():
            sqrt_sv = torch.sqrt(target_sv)
            for i in range(self.n_conditions):
                self.U.data[i] = self.U.data[i] * sqrt_sv.unsqueeze(0)
                self.V.data[i] = self.V.data[i] * sqrt_sv.unsqueeze(0)
    
    def get_weight(self, condition_idx: int) -> torch.Tensor:
        """Get weight matrix for specific condition."""
        W = self.U[condition_idx] @ self.V[condition_idx].T
        
        if self.symmetric:
            W = (W + W.T) / 2
        
        return W
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: h_t = W^{(x)} h_{t-1} + b^{(x)}.
        
        x should be one-hot encoded [batch, n_conditions].
        """
        # Reconstruct all W matrices
        W_all = torch.bmm(self.U, self.V.transpose(1, 2))  # [n_cond, H, H]
        
        if self.symmetric:
            W_all = (W_all + W_all.transpose(1, 2)) / 2
        
        # Select based on input
        x_expand = x.unsqueeze(-1).unsqueeze(-1)  # [batch, n_cond, 1, 1]
        W_selected = (x_expand * W_all.unsqueeze(0)).sum(1)  # [batch, H, H]
        
        # Select bias
        b_selected = (x.unsqueeze(-1) * self.bias.unsqueeze(0)).sum(1)
        
        # Linear dynamics
        h_new = torch.bmm(W_selected, h.unsqueeze(-1)).squeeze(-1) + b_selected
        
        return h_new


# =============================================================================
# Response Heterogeneity Analyzer
# =============================================================================

class ResponseHeterogeneityAnalyzer:
    """
    Comprehensive analysis of response heterogeneity in RNN dynamics.
    
    Metrics:
    1. Jacobian-based: Var(||dh/dI_j||_2) - response to localized perturbations
    2. Selectivity Index: (max - mean) / std of weight magnitudes per unit
    3. Coefficient of Variation: std / mean of response magnitudes
    4. Sparsity Index: proportion of near-zero responses
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
    
    def compute_jacobian_heterogeneity(self, h: torch.Tensor) -> float:
        """
        Compute heterogeneity via Jacobian analysis.
        
        Heterogeneity = Var_j(||dh/dI_j||_2)
        
        Higher values -> more specialized responses (mature characteristic).
        """
        h = h.detach().requires_grad_(True)
        n_hidden = h.shape[-1]
        
        if hasattr(self.model, 'recurrent'):
            layer = self.model.recurrent
        else:
            return 0.0
        
        jacobian_norms = []
        
        for j in range(n_hidden):
            I_j = torch.zeros(1, n_hidden, device=self.device)
            I_j[0, j] = 1.0
            
            h_sample = h[0:1].detach().requires_grad_(True)
            x_zero = torch.zeros(1, n_hidden, device=self.device)
            
            if hasattr(layer, 'forward'):
                h_next = layer(h_sample, x_zero + I_j)
            else:
                continue
            
            jacobian_col = torch.autograd.grad(
                h_next.sum(), h_sample, create_graph=False
            )[0]
            
            norm = jacobian_col.norm().item()
            jacobian_norms.append(norm)
        
        return float(np.var(jacobian_norms)) if jacobian_norms else 0.0
    
    def compute_selectivity_index(self, W: torch.Tensor) -> Dict[str, float]:
        """
        Compute selectivity index for each unit.
        
        S_j = (max_k |W_{jk}| - mean_k |W_{jk}|) / std_k |W_{jk}|
        """
        W = W.detach().cpu().numpy()
        W_abs = np.abs(W)
        
        selectivities = []
        for j in range(W.shape[0]):
            row = W_abs[j, :]
            S_j = (row.max() - row.mean()) / (row.std() + 1e-8)
            selectivities.append(S_j)
        
        return {
            'mean_selectivity': float(np.mean(selectivities)),
            'std_selectivity': float(np.std(selectivities)),
            'max_selectivity': float(np.max(selectivities)),
            'min_selectivity': float(np.min(selectivities))
        }
    
    def compute_response_cv(self, responses: torch.Tensor) -> float:
        """Coefficient of variation: std/mean."""
        responses = responses.detach().cpu().numpy().flatten()
        responses_abs = np.abs(responses)
        return float(responses_abs.std() / (responses_abs.mean() + 1e-8))
    
    def compute_sparsity_index(self, responses: torch.Tensor, 
                               threshold: float = 0.1) -> float:
        """Sparsity: proportion of near-zero responses."""
        responses = responses.detach().cpu().numpy().flatten()
        n_sparse = np.sum(np.abs(responses) < threshold)
        return float(n_sparse / len(responses))
    
    def full_analysis(self, h: torch.Tensor) -> Dict[str, float]:
        """Comprehensive heterogeneity analysis combining all metrics."""
        results = {}
        
        results['jacobian_heterogeneity'] = self.compute_jacobian_heterogeneity(h)
        
        if hasattr(self.model, 'recurrent'):
            if hasattr(self.model.recurrent, 'get_weight_matrix'):
                W = self.model.recurrent.get_weight_matrix()
            elif hasattr(self.model.recurrent, 'get_recurrent_weights'):
                W = self.model.recurrent.get_recurrent_weights()
            else:
                W = None
            
            if W is not None:
                if isinstance(W, dict):
                    W = list(W.values())[0]
                selectivity = self.compute_selectivity_index(W)
                results.update({f'weight_{k}': v for k, v in selectivity.items()})
        
        results['response_cv'] = self.compute_response_cv(h)
        results['sparsity_index'] = self.compute_sparsity_index(h)
        
        return results


# =============================================================================
# Unified Brain-Inspired RNN Model
# =============================================================================

class BrainInspiredRNN(nn.Module):
    """
    Brain-Inspired RNN with selectable factorization and cell type.
    
    Supports:
    - Multiple factorization approaches (SVD, Eigenvalue, SLIN)
    - Multiple cell types (Leaky, GRU, Switching GRU, SLIN)
    
    Architecture:
        h_{t+1} = f(h_t, x_t; W_rec, W_in, b)
        y_t = W_out h_t
    """
    
    def __init__(self, config: DevelopmentalConfig):
        super().__init__()
        self.config = config
        
        N = config.n_hidden
        n_in = config.n_input
        n_out = config.n_output
        
        # Input layer
        self.W_in = nn.Linear(n_in, N)
        
        # Recurrent layer (depends on factorization type)
        if config.factorization_type == FactorizationType.SVD_DIRECT:
            self.recurrent = SVDLowRankLayer(config)
        elif config.factorization_type == FactorizationType.EIGENVALUE_CONSTRAINED:
            self.recurrent = EigenConstrainedLayer(config)
        elif config.factorization_type == FactorizationType.SLIN:
            self.recurrent = SLINLayer(config, n_input_conditions=4)
        else:
            self.recurrent = SVDLowRankLayer(config)
        
        # GRU cell (optional, for cell_type variants)
        if config.cell_type == CellType.DEV_GRU:
            self.gru_cell = DevGRUCell(config)
        elif config.cell_type == CellType.SWITCHING_DEV_GRU:
            self.gru_cell = SwitchingDevGRUCell(config)
        elif config.cell_type == CellType.SLIN_CELL:
            self.gru_cell = SLINCell(config)
        else:
            self.gru_cell = None
        
        # Output layer
        self.W_out = nn.Linear(N, n_out)
        
        # Hidden state initialization
        self.h0 = nn.Parameter(torch.zeros(1, N), requires_grad=False)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state for batch."""
        return self.h0.expand(batch_size, -1).clone().to(device)
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        h0: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        condition_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the RNN.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence [batch, seq_len, n_input]
        h0 : torch.Tensor, optional
            Initial hidden state
        return_hidden : bool
            Whether to return hidden state history
        condition_indices : torch.Tensor, optional
            For SLIN, discrete condition indices [batch, seq_len]
        
        Returns
        -------
        outputs : torch.Tensor
            Output sequence [batch, seq_len, n_output]
        h_final : torch.Tensor
            Final hidden state
        hidden_history : torch.Tensor, optional
            Hidden states over time
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        
        h = h0 if h0 is not None else self.init_hidden(batch_size, device)
        
        outputs = []
        hidden_states = []
        
        for t in range(seq_len):
            x_t = self.W_in(inputs[:, t, :])
            
            # Update hidden state
            if self.gru_cell is not None:
                if isinstance(self.gru_cell, (SwitchingDevGRUCell, SLINCell)):
                    h = self.gru_cell(inputs[:, t, :], h)
                else:
                    h = self.gru_cell(inputs[:, t, :], h)
            elif isinstance(self.recurrent, SLINLayer) and condition_indices is not None:
                h = self.recurrent(h, x_t, condition_indices[:, t])
            else:
                h = self.recurrent(h, x_t)
            
            y_t = self.W_out(h)
            outputs.append(y_t)
            
            if return_hidden:
                hidden_states.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        
        if return_hidden:
            hidden_history = torch.stack(hidden_states, dim=1)
            return outputs, h, hidden_history
        
        return outputs, h
    
    def compute_effective_rank(self) -> float:
        """Compute effective rank: D(W) = (sum s_i)^2 / sum(s_i^2)."""
        if hasattr(self.recurrent, 'get_weight_matrix'):
            W = self.recurrent.get_weight_matrix()
        elif hasattr(self.recurrent, 'get_recurrent_weights'):
            W = self.recurrent.get_recurrent_weights()
        else:
            return 0.0
        
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        S_sq = S ** 2
        return float((S_sq.sum() ** 2) / (S_sq ** 2).sum())
    
    def compute_sv_decay_rate(self) -> float:
        """Fit singular value decay: log(sigma_i) = -gamma * i + c."""
        if hasattr(self.recurrent, 'get_weight_matrix'):
            W = self.recurrent.get_weight_matrix()
        else:
            return 0.0
        
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        S = S.detach().cpu().numpy()
        
        S_pos = S[S > 1e-10]
        if len(S_pos) < 2:
            return 0.0
        
        log_S = np.log(S_pos)
        indices = np.arange(len(S_pos))
        
        slope, _ = np.polyfit(indices, log_S, 1)
        return float(-slope)
    
    def compute_spectral_radius(self) -> float:
        """Compute spectral radius of recurrent weight matrix."""
        if hasattr(self.recurrent, 'get_weight_matrix'):
            W = self.recurrent.get_weight_matrix()
        else:
            return 0.0

        eigenvalues = torch.linalg.eigvals(W)
        return float(eigenvalues.abs().max().item())

    def compute_jacobian_heterogeneity(self, h: torch.Tensor) -> float:
        """
        Compute heterogeneity via Jacobian analysis.

        Heterogeneity = Var_j(||dh/dI_j||_2)

        Higher values -> more specialized responses (mature characteristic).

        Parameters
        ----------
        h : torch.Tensor
            Hidden state tensor [batch, n_hidden]

        Returns
        -------
        float
            Variance of Jacobian norms across units
        """
        if not hasattr(self.recurrent, 'forward'):
            return 0.0

        # Enable gradients for Jacobian computation
        with torch.enable_grad():
            h = h.detach().requires_grad_(True)
            n_hidden = h.shape[-1]
            device = h.device

            jacobian_norms = []

            for j in range(n_hidden):
                I_j = torch.zeros(1, n_hidden, device=device)
                I_j[0, j] = 1.0

                h_sample = h[0:1].detach().requires_grad_(True)
                x_zero = torch.zeros(1, n_hidden, device=device)

                h_next = self.recurrent(h_sample, x_zero + I_j)

                jacobian_col = torch.autograd.grad(
                    h_next.sum(), h_sample, create_graph=False
                )[0]

                norm = jacobian_col.norm().item()
                jacobian_norms.append(norm)

        return float(np.var(jacobian_norms)) if jacobian_norms else 0.0

    def compute_regularization_losses(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all regularization losses for training."""
        losses = {}
        
        if hasattr(self.recurrent, 'get_weight_matrix'):
            W = self.recurrent.get_weight_matrix()
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            
            # Nuclear norm
            losses['nuclear_norm'] = self.config.nuclear_norm_alpha * S.sum()
            
            # Spectral loss
            r = min(self.config.rank, len(S))
            gamma = self.config.sv_decay_gamma
            target_sv = torch.exp(-gamma * torch.arange(r, dtype=torch.float32, device=S.device))
            target_sv = target_sv * (S[:r].sum() / target_sv.sum())
            losses['spectral'] = F.mse_loss(S[:r], target_sv)
        
        # Response heterogeneity (maximize, so negative)
        unit_magnitudes = h.abs().mean(dim=0)
        losses['heterogeneity'] = -self.config.response_hetero_weight * unit_magnitudes.var()
        
        return losses
    
    def get_model_metrics(self) -> Dict[str, Union[float, str]]:
        """Extract comprehensive metrics for model comparison."""
        return {
            'effective_rank': self.compute_effective_rank(),
            'sv_decay_rate': self.compute_sv_decay_rate(),
            'spectral_radius': self.compute_spectral_radius(),
            'developmental_stage': self.config.developmental_stage,
            'factorization_type': self.config.factorization_type.value,
            'cell_type': self.config.cell_type.value
        }


class TinyGRU(nn.Module):
    """
    Minimal GRU model following Ji-An et al. architecture.

    This serves as a baseline comparison without developmental constraints.

    Architecture:
        - GRU cell with small hidden dimension (d=4 as in the paper)
        - Linear readout layer

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_dim : int
        Number of hidden units (typically 2-4 for interpretability)
    output_dim : int
        Dimension of output (typically 2 for binary choice)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # GRU cell
        self.gru = nn.GRUCell(input_dim, hidden_dim)

        # Output layer
        self.readout = nn.Linear(hidden_dim, output_dim)

        # Initial hidden state (learnable)
        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(
            self,
            inputs: torch.Tensor,
            h0: Optional[torch.Tensor] = None,
            return_hidden: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [batch, seq_len, input_dim]
        h0 : torch.Tensor, optional
            Initial hidden state [batch, hidden_dim]
        return_hidden : bool
            Whether to return hidden state history

        Returns
        -------
        outputs : torch.Tensor
            Output logits [batch, seq_len, output_dim]
        h_final : torch.Tensor
            Final hidden state [batch, hidden_dim]
        hidden_history : torch.Tensor, optional
            All hidden states [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # Initialize hidden state
        if h0 is None:
            h = self.h0.expand(batch_size, -1).contiguous()
        else:
            h = h0

        outputs = []
        hidden_states = []

        for t in range(seq_len):
            h = self.gru(inputs[:, t, :], h)
            y = self.readout(h)
            outputs.append(y)

            if return_hidden:
                hidden_states.append(h)

        outputs = torch.stack(outputs, dim=1)

        if return_hidden:
            hidden_history = torch.stack(hidden_states, dim=1)
            return outputs, h, hidden_history

        return outputs, h


def get_model_metrics(model: nn.Module) -> Dict:
    """
    Extract comprehensive metrics from a trained model.

    For BrainInspiredRNN, computes:
        - Effective rank D(W)
        - Singular value decay rate γ
        - Spectral radius ρ(W)
        - Full singular value spectrum

    For TinyGRU, computes basic GRU weight statistics.

    Parameters
    ----------
    model : nn.Module
        The trained RNN model

    Returns
    -------
    Dict
        Dictionary containing all computed metrics
    """
    metrics = {}

    if isinstance(model, BrainInspiredRNN):
        # Get metrics from the model's built-in methods
        metrics['effective_rank'] = model.compute_effective_rank()
        metrics['sv_decay_rate'] = model.compute_sv_decay_rate()
        metrics['spectral_radius'] = model.compute_spectral_radius()

        # Extract singular values for visualization
        if hasattr(model.recurrent, 'get_weight_matrix'):
            W = model.recurrent.get_weight_matrix()
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            S = S.detach().cpu().numpy()
            S_normalized = S / S[0] if S[0] > 0 else S
            metrics['singular_values'] = S_normalized
        else:
            metrics['singular_values'] = np.array([1.0])

        # Get model configuration info
        model_info = model.get_model_metrics()
        metrics.update(model_info)

    elif isinstance(model, TinyGRU):
        # For baseline GRU, compute basic statistics
        W_hh = model.gru.weight_hh.detach()
        _, S, _ = torch.linalg.svd(W_hh, full_matrices=False)
        S = S.cpu().numpy()

        metrics['effective_rank'] = float(
            (S.sum() ** 2) / (S ** 2).sum()
        )

        # Decay rate from log-linear fit
        S_pos = S[S > 1e-10]
        if len(S_pos) > 1:
            log_S = np.log(S_pos)
            indices = np.arange(len(S_pos))
            slope, _ = np.polyfit(indices, log_S, 1)
            metrics['sv_decay_rate'] = float(-slope)
        else:
            metrics['sv_decay_rate'] = 0.0

        S_normalized = S / S[0] if S[0] > 0 else S
        metrics['singular_values'] = S_normalized
        metrics['spectral_radius'] = float(np.abs(np.linalg.eigvals(W_hh.cpu().numpy())).max())
        metrics['developmental_stage'] = 'baseline'
        metrics['cell_type'] = 'gru'

    return metrics
# =============================================================================
# Model Factory Functions
# =============================================================================

def create_model(
    developmental_stage: str = "mature",
    n_hidden: int = 32,
    n_input: int = 3,
    n_output: int = 2,
    factorization_type: FactorizationType = FactorizationType.SVD_DIRECT,
    cell_type: CellType = CellType.LEAKY_TANH
) -> BrainInspiredRNN:
    """
    Factory function to create developmental RNNs.
    
    Parameters
    ----------
    developmental_stage : str
        "premature" or "mature"
    n_hidden : int
        Number of hidden units
    n_input : int
        Input dimension
    n_output : int
        Output dimension
    factorization_type : FactorizationType
        Type of weight matrix factorization
    cell_type : CellType
        Type of RNN cell
    
    Returns
    -------
    BrainInspiredRNN
        Configured model
    """
    if developmental_stage == "premature":
        config = create_premature_config(
            n_hidden, n_input, n_output, factorization_type, cell_type
        )
    else:
        config = create_mature_config(
            n_hidden, n_input, n_output, factorization_type, cell_type
        )
    
    return BrainInspiredRNN(config)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Comprehensive Brain-Inspired RNN Framework Testing")
    print("=" * 70)
    
    # Test 1: SVD Direct Factorization
    print("\n1. Testing SVD Direct Factorization")
    print("-" * 50)
    
    premature_svd = create_model("premature", factorization_type=FactorizationType.SVD_DIRECT)
    mature_svd = create_model("mature", factorization_type=FactorizationType.SVD_DIRECT)
    
    batch_size, seq_len = 16, 50
    x = torch.randn(batch_size, seq_len, 3)
    
    out_prem, h_prem = premature_svd(x)
    out_mat, h_mat = mature_svd(x)
    
    print(f"  Premature - Effective Rank: {premature_svd.compute_effective_rank():.3f}")
    print(f"  Mature    - Effective Rank: {mature_svd.compute_effective_rank():.3f}")
    print(f"  Premature - SV Decay Rate: {premature_svd.compute_sv_decay_rate():.3f}")
    print(f"  Mature    - SV Decay Rate: {mature_svd.compute_sv_decay_rate():.3f}")
    
    # Test 2: Eigenvalue-Constrained Factorization
    print("\n2. Testing Eigenvalue-Constrained Factorization")
    print("-" * 50)
    
    premature_eigen = create_model("premature", factorization_type=FactorizationType.EIGENVALUE_CONSTRAINED)
    mature_eigen = create_model("mature", factorization_type=FactorizationType.EIGENVALUE_CONSTRAINED)
    
    out_prem_e, h_prem_e = premature_eigen(x)
    out_mat_e, h_mat_e = mature_eigen(x)
    
    print(f"  Premature - Eigenvalue Concentration: "
          f"{premature_eigen.recurrent.compute_eigenvalue_concentration():.3f}")
    print(f"  Mature    - Eigenvalue Concentration: "
          f"{mature_eigen.recurrent.compute_eigenvalue_concentration():.3f}")
    
    # Test 3: Response Heterogeneity
    print("\n3. Testing Response Heterogeneity Analysis")
    print("-" * 50)
    
    analyzer_prem = ResponseHeterogeneityAnalyzer(premature_svd)
    analyzer_mat = ResponseHeterogeneityAnalyzer(mature_svd)
    
    hetero_prem = analyzer_prem.full_analysis(h_prem)
    hetero_mat = analyzer_mat.full_analysis(h_mat)
    
    print(f"  Premature - Response CV: {hetero_prem['response_cv']:.3f}")
    print(f"  Mature    - Response CV: {hetero_mat['response_cv']:.3f}")
    print(f"  Premature - Sparsity: {hetero_prem['sparsity_index']:.3f}")
    print(f"  Mature    - Sparsity: {hetero_mat['sparsity_index']:.3f}")
    
    # Test 4: DevGRU Cell
    print("\n4. Testing Developmental GRU Cell")
    print("-" * 50)
    
    config_gru = create_mature_config(n_hidden=32, cell_type=CellType.DEV_GRU)
    gru_model = BrainInspiredRNN(config_gru)
    
    out_gru, h_gru = gru_model(x)
    print(f"  DevGRU output shape: {out_gru.shape}")
    print(f"  DevGRU metrics: {gru_model.get_model_metrics()}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
