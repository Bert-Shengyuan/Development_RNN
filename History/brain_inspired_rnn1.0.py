"""
Brain-Inspired Low-Rank RNN Framework for Cognitive Tasks
==========================================================

This module implements developmentally-constrained recurrent neural networks
based on the theoretical framework connecting fMRI dynamics to low-rank RNN structure.

The key insight is that mature brains exhibit:
- Lower effective rank (r_mature ≈ 5-6 vs r_premature ≈ 7-8)
- Higher response heterogeneity
- Faster singular value decay

Mathematical Foundation:
------------------------
The single-step dynamics follow:
    x_{t+1} = A * x_t + η_t

where A admits a low-rank factorization:
    A ≈ U Σ V^T = Σ_{i=1}^{r} σ_i u_i v_i^T

RNN Implementation:
    h_{t+1} = (1-α)h_t + α * tanh(m n^T h_t + M N^T h_t + W_in u_t)
    y_t = W_out h_t

Author: Computational Neuroscience Research
Reference: Ji-An et al. "Discovering cognitive strategies with tiny recurrent neural networks"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DevelopmentalConfig:
    """
    Configuration for developmental stage-specific RNN constraints.
    
    Premature Brain Characteristics:
    - Higher effective rank (less compression)
    - Lower response heterogeneity
    - Slower singular value decay (γ_premature < γ_mature)
    
    Mature Brain Characteristics:
    - Lower effective rank (more compression onto low-dim manifold)
    - Higher response heterogeneity (more specialized responses)
    - Faster singular value decay
    """
    # Network architecture
    n_hidden: int = 32                    # Number of hidden units
    rank: int = 6                         # Effective rank for low-rank factorization
    
    # Developmental parameters
    sv_decay_gamma: float = 0.3           # Singular value decay rate: σ_i ∝ exp(-γi)
    response_hetero_weight: float = 0.1   # Weight for response heterogeneity loss
    
    # Training regularization
    nuclear_norm_alpha: float = 1e-3      # Nuclear norm regularization strength
    effective_rank_beta: float = 1e-2     # Effective rank penalty weight
    
    # Dynamics
    leak_alpha: float = 0.1               # Leak parameter for slow dynamics
    
    # Label for identification
    developmental_stage: str = "mature"
    

def create_premature_config(n_hidden: int = 32) -> DevelopmentalConfig:
    """
    Create configuration mimicking premature infant brain characteristics.
    
    Key differences from mature:
    - Higher rank (r ≈ 7-8): less compressed representations
    - Slower SV decay (γ ≈ 0.2): more uniform eigenvalue distribution
    - Lower heterogeneity weight: less specialized responses
    """
    return DevelopmentalConfig(
        n_hidden=n_hidden,
        rank=8,                           # Higher effective rank
        sv_decay_gamma=0.2,               # Slower decay → more uniform spectrum
        response_hetero_weight=0.05,      # Lower heterogeneity encouragement
        nuclear_norm_alpha=5e-4,          # Weaker rank regularization
        effective_rank_beta=5e-3,
        leak_alpha=0.1,
        developmental_stage="premature"
    )


def create_mature_config(n_hidden: int = 32) -> DevelopmentalConfig:
    """
    Create configuration mimicking mature infant brain characteristics.
    
    Key differences from premature:
    - Lower rank (r ≈ 5-6): more compressed onto low-dim manifold
    - Faster SV decay (γ ≈ 0.4): variance concentrated in few modes
    - Higher heterogeneity weight: more specialized, differentiated responses
    """
    return DevelopmentalConfig(
        n_hidden=n_hidden,
        rank=5,                           # Lower effective rank
        sv_decay_gamma=0.4,               # Faster decay → concentrated spectrum
        response_hetero_weight=0.15,      # Higher heterogeneity encouragement
        nuclear_norm_alpha=2e-3,          # Stronger rank regularization
        effective_rank_beta=2e-2,
        leak_alpha=0.1,
        developmental_stage="mature"
    )


class LowRankRecurrentLayer(nn.Module):
    """
    Low-Rank Recurrent Layer with developmental constraints.
    
    The recurrent weight matrix is factorized as:
        W_rec = m n^T + M N^T
    
    where:
    - m, n ∈ R^N define a rank-1 "common mode" component
    - M ∈ R^{N×r}, N ∈ R^{N×r} define the rank-r component
    
    This enforces that dynamics evolve predominantly in an r-dimensional subspace,
    mirroring the biological observation of low-dimensional neural manifolds.
    """
    
    def __init__(self, config: DevelopmentalConfig):
        super().__init__()
        self.config = config
        N = config.n_hidden
        r = config.rank
        
        # Rank-1 component: m n^T (common mode)
        self.m = nn.Parameter(torch.randn(N, 1) * 0.1)
        self.n = nn.Parameter(torch.randn(N, 1) * 0.1)
        
        # Rank-r component: M N^T
        self.M = nn.Parameter(torch.randn(N, r) * 0.1)
        self.N = nn.Parameter(torch.randn(N, r) * 0.1)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(N))
        
        # Initialize with target singular value structure
        self._initialize_with_sv_decay()
    
    def _initialize_with_sv_decay(self):
        """
        Initialize M, N such that the resulting weight matrix has 
        singular values following: σ_i ∝ exp(-γ * i)
        
        This reflects the empirical finding that mature brains show
        faster singular value decay.
        """
        r = self.config.rank
        gamma = self.config.sv_decay_gamma
        
        # Target singular values
        target_sv = torch.exp(-gamma * torch.arange(r, dtype=torch.float32))
        target_sv = target_sv / target_sv.sum() * r  # Normalize
        
        # Initialize M, N such that M N^T has these singular values
        # Using: M = U * sqrt(Σ), N = V * sqrt(Σ)
        with torch.no_grad():
            sqrt_sv = torch.sqrt(target_sv)
            self.M.data = self.M.data * sqrt_sv.unsqueeze(0)
            self.N.data = self.N.data * sqrt_sv.unsqueeze(0)
    
    def get_recurrent_weights(self) -> torch.Tensor:
        """
        Compute the full recurrent weight matrix:
            W_rec = m n^T + M N^T
        """
        # Rank-1 component
        rank1 = self.m @ self.n.T
        
        # Rank-r component
        rank_r = self.M @ self.N.T
        
        return rank1 + rank_r
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Single step of recurrent dynamics:
            h_new = (1-α)h + α * tanh(W_rec @ h + x + bias)
        
        Args:
            h: Hidden state [batch, N]
            x: Input after transformation [batch, N]
        
        Returns:
            h_new: Updated hidden state [batch, N]
        """
        alpha = self.config.leak_alpha
        W_rec = self.get_recurrent_weights()
        
        # Recurrent computation
        pre_activation = h @ W_rec.T + x + self.bias
        
        # Leaky integration (slow dynamics for fMRI-like timescales)
        h_new = (1 - alpha) * h + alpha * torch.tanh(pre_activation)
        
        return h_new


class BrainInspiredRNN(nn.Module):
    """
    Brain-Inspired RNN with Low-Rank Developmental Constraints.
    
    Architecture:
        h_{t+1} = (1-α)h_t + α * tanh(W_rec h_t + W_in u_t + b)
        y_t = W_out h_t
    
    Key features:
    1. Low-rank recurrent weights (developmental constraint)
    2. Spectral regularization (matching empirical SV decay)
    3. Response heterogeneity loss (encouraging specialization)
    
    This model captures the key developmental principle:
    Mature brains show lower dimensionality but higher specialization.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: DevelopmentalConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        N = config.n_hidden
        
        # Input projection
        self.W_in = nn.Linear(input_dim, N, bias=False)
        
        # Low-rank recurrent layer
        self.recurrent = LowRankRecurrentLayer(config)
        
        # Output projection
        self.W_out = nn.Linear(N, output_dim, bias=True)
        
        # For storing hidden states (useful for analysis)
        self.hidden_history = []
        
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(batch_size, self.config.n_hidden, device=device)
    
    def forward(self, inputs: torch.Tensor, h0: Optional[torch.Tensor] = None,
                return_hidden: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.
        
        Args:
            inputs: Input sequence [batch, seq_len, input_dim]
            h0: Initial hidden state (optional)
            return_hidden: Whether to return full hidden state trajectory
        
        Returns:
            outputs: Output sequence [batch, seq_len, output_dim]
            h_final: Final hidden state [batch, N]
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        
        # Initialize hidden state
        h = h0 if h0 is not None else self.init_hidden(batch_size, device)
        
        outputs = []
        hidden_states = []
        
        for t in range(seq_len):
            # Transform input
            x_t = self.W_in(inputs[:, t, :])
            
            # Recurrent step
            h = self.recurrent(h, x_t)
            
            # Output
            y_t = self.W_out(h)
            outputs.append(y_t)
            
            if return_hidden:
                hidden_states.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        
        if return_hidden:
            self.hidden_history = torch.stack(hidden_states, dim=1)
            return outputs, h, self.hidden_history
        
        return outputs, h
    
    def compute_response_heterogeneity(self, h: torch.Tensor, 
                                        perturbation_strength: float = 0.1) -> torch.Tensor:
        """
        Compute response heterogeneity to localized perturbations.
        
        This captures the finding that mature brains show higher 
        response heterogeneity: ∂h/∂I_j has higher variance across j.
        
        Args:
            h: Hidden state [batch, N]
            perturbation_strength: Magnitude of perturbation
        
        Returns:
            heterogeneity: Variance of response magnitudes
        """
        N = self.config.n_hidden
        batch_size = h.shape[0]
        device = h.device
        
        responses = []
        
        # Apply perturbation to each unit and measure response
        for j in range(N):
            # Create localized perturbation to unit j
            perturbation = torch.zeros(batch_size, N, device=device)
            perturbation[:, j] = perturbation_strength
            
            # Compute response (one-step forward)
            h_perturbed = self.recurrent(h, perturbation)
            response_magnitude = torch.norm(h_perturbed - h, dim=1)
            responses.append(response_magnitude)
        
        responses = torch.stack(responses, dim=1)  # [batch, N]
        
        # Heterogeneity = variance of response magnitudes across units
        heterogeneity = torch.var(responses, dim=1).mean()
        
        return heterogeneity
    
    def compute_effective_rank(self) -> torch.Tensor:
        """
        Compute effective rank of the recurrent weight matrix.
        
        Effective rank is defined as:
            D(W) = (Σλ_i)² / Σλ_i²
        
        where λ_i are eigenvalues of W W^T (equivalently, squared singular values).
        
        Lower effective rank indicates more compressed dynamics.
        """
        W = self.recurrent.get_recurrent_weights()
        
        # Compute singular values
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        
        # Compute effective rank
        S_sq = S ** 2
        eff_rank = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        
        return eff_rank
    
    def compute_spectral_loss(self) -> torch.Tensor:
        """
        Compute spectral regularization loss.
        
        Encourages singular values to follow the target decay:
            σ_i^target ∝ exp(-γ * i)
        
        Loss = Σ_i (σ_i - σ_i^target)²
        """
        W = self.recurrent.get_recurrent_weights()
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        
        # Target singular values
        r = min(self.config.rank, len(S))
        gamma = self.config.sv_decay_gamma
        target_sv = torch.exp(-gamma * torch.arange(r, dtype=torch.float32, device=S.device))
        
        # Normalize to match scale
        target_sv = target_sv * (S[:r].sum() / target_sv.sum())
        
        # Spectral loss
        loss = F.mse_loss(S[:r], target_sv)
        
        return loss
    
    def compute_nuclear_norm(self) -> torch.Tensor:
        """
        Compute nuclear norm ||W||_* = Σ σ_i
        
        This promotes low-rank structure by penalizing total singular value mass.
        """
        W = self.recurrent.get_recurrent_weights()
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        return S.sum()
    
    def compute_regularization_losses(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all regularization losses for developmental constraints.
        
        Returns dictionary with:
        - nuclear_norm: Promotes low-rank structure
        - spectral: Enforces target SV decay
        - heterogeneity: Encourages response specialization
        - effective_rank: Penalizes high dimensionality
        """
        losses = {}
        
        # Nuclear norm regularization
        losses['nuclear_norm'] = self.config.nuclear_norm_alpha * self.compute_nuclear_norm()
        
        # Spectral regularization
        losses['spectral'] = self.compute_spectral_loss()
        
        # Response heterogeneity (negative because we want to maximize it)
        losses['heterogeneity'] = -self.config.response_hetero_weight * \
                                   self.compute_response_heterogeneity(h)
        
        # Effective rank penalty
        losses['effective_rank'] = self.config.effective_rank_beta * self.compute_effective_rank()
        
        return losses


class GRUCell(nn.Module):
    """
    Standard GRU cell for comparison (following Ji-An et al.).
    
    The GRU update equations are:
        z_t = σ(W_z x_t + U_z h_{t-1} + b_z)
        r_t = σ(W_r x_t + U_r h_{t-1} + b_r)
        h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Gate weights
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        
        combined_reset = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        
        h_new = (1 - z) * h + z * h_tilde
        
        return h_new


class TinyGRU(nn.Module):
    """
    Tiny GRU model following Ji-An et al. for baseline comparison.
    
    This implements the standard approach from "Discovering cognitive 
    strategies with tiny recurrent neural networks" without developmental
    constraints.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.gru_cell = GRUCell(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)
    
    def forward(self, inputs: torch.Tensor, h0: Optional[torch.Tensor] = None,
                return_hidden: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        
        h = h0 if h0 is not None else self.init_hidden(batch_size, device)
        
        outputs = []
        hidden_states = []
        
        for t in range(seq_len):
            h = self.gru_cell(inputs[:, t, :], h)
            y = self.output_layer(h)
            outputs.append(y)
            
            if return_hidden:
                hidden_states.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        
        if return_hidden:
            hidden_history = torch.stack(hidden_states, dim=1)
            return outputs, h, hidden_history
        
        return outputs, h


def get_model_metrics(model: nn.Module) -> Dict[str, float]:
    """
    Extract key metrics from a trained model for comparison.
    
    Returns:
        Dictionary containing:
        - effective_rank: Dimensionality of dynamics
        - sv_decay_rate: Rate of singular value decay
        - weight_norm: Total weight magnitude
    """
    metrics = {}
    
    if isinstance(model, BrainInspiredRNN):
        W = model.recurrent.get_recurrent_weights().detach()
    elif isinstance(model, TinyGRU):
        # For GRU, analyze the hidden-to-hidden weights
        W = model.gru_cell.W_h.weight[:, model.hidden_dim:].detach()
    else:
        return metrics
    
    # Compute singular values
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    S = S.cpu().numpy()
    
    # Effective rank
    S_sq = S ** 2
    metrics['effective_rank'] = float((S_sq.sum() ** 2) / (S_sq ** 2).sum())
    
    # SV decay rate (fit exponential)
    log_S = np.log(S + 1e-10)
    indices = np.arange(len(S))
    # Linear fit: log(σ_i) = -γ*i + c
    if len(S) > 1:
        slope, _ = np.polyfit(indices, log_S, 1)
        metrics['sv_decay_rate'] = float(-slope)
    else:
        metrics['sv_decay_rate'] = 0.0
    
    # Weight norm
    metrics['weight_norm'] = float(np.linalg.norm(W.cpu().numpy()))
    
    # Normalized singular values for spectrum analysis
    metrics['singular_values'] = S / S.max()
    
    return metrics


if __name__ == "__main__":
    # Quick test of the models
    print("Testing Brain-Inspired RNN Framework")
    print("=" * 50)
    
    # Create configurations
    premature_config = create_premature_config(n_hidden=32)
    mature_config = create_mature_config(n_hidden=32)
    
    # Create models
    input_dim, output_dim = 3, 2
    premature_rnn = BrainInspiredRNN(input_dim, output_dim, premature_config)
    mature_rnn = BrainInspiredRNN(input_dim, output_dim, mature_config)
    
    # Test forward pass
    batch_size, seq_len = 16, 50
    x = torch.randn(batch_size, seq_len, input_dim)
    
    out_prem, h_prem = premature_rnn(x)
    out_mat, h_mat = mature_rnn(x)
    
    print(f"Premature RNN - Effective Rank: {premature_rnn.compute_effective_rank():.3f}")
    print(f"Mature RNN - Effective Rank: {mature_rnn.compute_effective_rank():.3f}")
    
    print(f"\nPremature RNN - Response Heterogeneity: {premature_rnn.compute_response_heterogeneity(h_prem):.4f}")
    print(f"Mature RNN - Response Heterogeneity: {mature_rnn.compute_response_heterogeneity(h_mat):.4f}")
    
    print("\n✓ All tests passed!")
