#!/usr/bin/env python3
"""
Comprehensive Dynamical Systems Analysis for Developmental RNNs
================================================================

This module merges functionality from:
- dynamical_analysis.py (Phase portraits, vector fields, comparisons)
- analysis.py (Fixed points, Lyapunov, MB/MF signatures)

Analyses Include
----------------
1. Fixed Point Analysis: Find and characterize attractors
2. Phase Portrait Generation: L(t) vs Delta-L(t) plots (Figure 3)
3. Vector Field Generation: 2D state space flow (Figure 4)
4. Lyapunov Exponent: Global dynamical stability
5. Preference Setpoints: Normalized asymptotic preferences
6. Model-Based vs Model-Free Signatures: Two-stage task analysis
7. Developmental Comparison: Comprehensive metrics

Mathematical Framework
----------------------
For d=1 models:
    L(t) = log(P(A1)/P(A2)) = beta * h_t^{(1)}
    Delta-L(t) = L(t+1) - L(t) = f(L(t), x_t)

Fixed points: L* such that Delta-L(L*, x) = 0

For d>1 models:
    Vector field: (Delta-h1, Delta-h2) on (h1, h2) grid

References
----------
- Ji-An et al. (2025) "Discovering cognitive strategies with tiny RNNs"
- Sussillo & Barak (2013) "Opening the Black Box"
- Mastrogiuseppe & Ostojic (2018) "Linking Connectivity, Dynamics, and Computations"

Author: Computational Neuroscience Research
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA


# =============================================================================
# Data Structures for Analysis Results
# =============================================================================

@dataclass
class PhasePortraitData:
    """Data structure for phase portrait analysis."""
    logits: np.ndarray              # L(t) values
    logit_changes: np.ndarray       # Delta-L(t) = L(t+1) - L(t)
    actions: np.ndarray             # Actions taken
    rewards: np.ndarray             # Rewards received
    input_conditions: np.ndarray    # Discrete input condition indices
    
    # Fixed point analysis
    fixed_points: Dict[int, float]  # Per-condition fixed points
    preference_setpoints: Dict[int, float]  # Normalized fixed points


@dataclass
class VectorFieldData:
    """Data structure for 2D vector field analysis."""
    h1_grid: np.ndarray            # Grid of h1 values
    h2_grid: np.ndarray            # Grid of h2 values
    delta_h1: Dict[int, np.ndarray]  # Per-condition Delta-h1
    delta_h2: Dict[int, np.ndarray]  # Per-condition Delta-h2
    fixed_points: Dict[int, List[Tuple[float, float]]]  # Per-condition FPs


@dataclass
class DevelopmentalComparison:
    """Results of comparing premature vs mature RNN dynamics."""
    # Phase portrait metrics
    premature_fixed_points: Dict[int, float]
    mature_fixed_points: Dict[int, float]
    
    # Preference setpoints
    premature_setpoints: Dict[int, float]
    mature_setpoints: Dict[int, float]
    
    # Learning rate analysis
    premature_learning_rates: Dict[str, float]
    mature_learning_rates: Dict[str, float]
    
    # Stability analysis
    premature_stability: Dict[str, float]
    mature_stability: Dict[str, float]
    
    # Summary metrics
    dimensionality_premature: float
    dimensionality_mature: float
    specialization_premature: float
    specialization_mature: float


# =============================================================================
# Fixed Point Finder
# =============================================================================

class FixedPointFinder:
    """
    Find and analyze fixed points of RNN dynamics.
    
    Fixed points h* satisfy: h* = f(h*, x=0)
    
    We use optimization:
        min_h ||h - f(h, 0)||^2
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        n_points: int = 20, 
        n_iters: int = 1000, 
        lr: float = 0.1, 
        tol: float = 1e-6
    ):
        self.model = model
        self.n_points = n_points
        self.n_iters = n_iters
        self.lr = lr
        self.tol = tol
        self.device = next(model.parameters()).device
    
    def _dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Single step of autonomous dynamics (no input)."""
        if hasattr(self.model, 'config'):
            n_input = self.model.config.n_input
        else:
            n_input = 3
        
        x = torch.zeros(h.shape[0], n_input, device=self.device)
        
        if hasattr(self.model, 'W_in') and hasattr(self.model, 'recurrent'):
            x_transformed = self.model.W_in(x)
            h_new = self.model.recurrent(h, x_transformed)
        else:
            h_new = h
        
        return h_new
    
    def find_fixed_points(self) -> Dict[str, Union[np.ndarray, int]]:
        """Find fixed points by optimization."""
        if hasattr(self.model, 'config'):
            n_hidden = self.model.config.n_hidden
        else:
            n_hidden = 32
        
        h_init = torch.randn(self.n_points, n_hidden, device=self.device) * 0.5
        h_init.requires_grad_(True)
        
        optimizer = torch.optim.Adam([h_init], lr=self.lr)
        
        for _ in range(self.n_iters):
            optimizer.zero_grad()
            h_new = self._dynamics(h_init)
            loss = ((h_init - h_new) ** 2).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
        
        # Find converged points
        with torch.no_grad():
            h_final = h_init.detach()
            h_new = self._dynamics(h_final)
            velocities = ((h_final - h_new) ** 2).sum(dim=1).sqrt()
            
            converged = velocities < self.tol
            fixed_points = h_final[converged].cpu().numpy()
            
            # Remove duplicates
            if len(fixed_points) > 0:
                fixed_points = self._remove_duplicates(fixed_points)
        
        return {
            'fixed_points': fixed_points,
            'velocities': velocities.cpu().numpy(),
            'n_found': len(fixed_points)
        }
    
    def _remove_duplicates(self, points: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Remove duplicate fixed points."""
        if len(points) == 0:
            return points
        
        unique = [points[0]]
        for p in points[1:]:
            distances = [np.linalg.norm(p - u) for u in unique]
            if min(distances) > threshold:
                unique.append(p)
        
        return np.array(unique)
    
    def compute_jacobian(self, h_star: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian at a fixed point."""
        h = h_star.clone().requires_grad_(True)
        h_new = self._dynamics(h.unsqueeze(0)).squeeze(0)
        
        n_hidden = h.shape[0]
        jacobian = torch.zeros(n_hidden, n_hidden, device=self.device)
        
        for i in range(n_hidden):
            grad = torch.autograd.grad(h_new[i], h, retain_graph=True)[0]
            jacobian[i] = grad
        
        return jacobian
    
    def analyze_stability(self, fixed_points: np.ndarray) -> List[Dict]:
        """Analyze stability of each fixed point."""
        stability_results = []
        
        for fp in fixed_points:
            h_star = torch.tensor(fp, dtype=torch.float32, device=self.device)
            J = self.compute_jacobian(h_star)
            
            eigenvalues = torch.linalg.eigvals(J)
            max_abs_eig = eigenvalues.abs().max().item()
            
            if max_abs_eig < 1:
                stability = "stable"
            elif max_abs_eig > 1:
                stability = "unstable"
            else:
                stability = "neutral"
            
            stability_results.append({
                'fixed_point': fp,
                'max_eigenvalue': max_abs_eig,
                'stability': stability,
                'eigenvalues': eigenvalues.cpu().numpy()
            })
        
        return stability_results


# =============================================================================
# Logit Analyzer
# =============================================================================

class LogitAnalyzer:
    """
    Analyze RNN dynamics through logit space.
    
    L(t) = log(P(A1)/P(A2)) = beta * (output_1 - output_2)
    """
    
    def __init__(self, model: nn.Module, inverse_temp: float = 1.0):
        self.model = model
        self.beta = inverse_temp
        self.device = next(model.parameters()).device
    
    def compute_logit(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute policy logit from model outputs."""
        if outputs.shape[-1] >= 2:
            logit = self.beta * (outputs[..., 0] - outputs[..., 1])
        else:
            logit = self.beta * outputs[..., 0]
        
        return logit
    
    def extract_trajectories(
        self, 
        inputs: torch.Tensor, 
        return_hidden: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Extract logit trajectories from input sequences."""
        self.model.eval()
        
        with torch.no_grad():
            if return_hidden:
                outputs, h_final, hidden_history = self.model(inputs, return_hidden=True)
            else:
                outputs, h_final = self.model(inputs)
                hidden_history = None
        
        logits = self.compute_logit(outputs)
        
        logit_changes = torch.zeros_like(logits)
        logit_changes[:, :-1] = logits[:, 1:] - logits[:, :-1]
        
        result = {
            'logits': logits,
            'logit_changes': logit_changes,
            'outputs': outputs
        }
        
        if hidden_history is not None:
            result['hidden_states'] = hidden_history
        
        return result


# =============================================================================
# Phase Portrait Generator (Figure 3)
# =============================================================================

class PhasePortraitGenerator:
    """
    Generate phase portraits for RNN dynamics visualization.
    
    Following Figure 3 of Ji-An et al.:
    - X-axis: Logit L(t)
    - Y-axis: Logit change Delta-L(t)
    - Colors: Input condition (action, reward)
    """
    
    def __init__(self, model: nn.Module, task_name: str = "reversal"):
        self.model = model
        self.task_name = task_name
        self.device = next(model.parameters()).device
        self.analyzer = LogitAnalyzer(model)
        
        # Color scheme following paper
        self.colors = {
            0: '#ADD8E6',  # A1, R=0 (light blue)
            1: '#00008B',  # A1, R=1 (dark blue)
            2: '#FFC0CB',  # A2, R=0 (light pink)
            3: '#8B0000',  # A2, R=1 (dark red)
            # For 8-condition two-stage task
            4: '#90EE90',  # A1, S2, R=0 (light green)
            5: '#006400',  # A1, S2, R=1 (dark green)
            6: '#FFD700',  # A2, S1, R=0 (light yellow)
            7: '#FF8C00',  # A2, S1, R=1 (dark orange)
        }
    
    def generate_from_data(
        self, 
        inputs: torch.Tensor, 
        actions: np.ndarray,
        rewards: np.ndarray,
        second_stages: Optional[np.ndarray] = None
    ) -> PhasePortraitData:
        """Generate phase portrait data from experimental data."""
        trajectories = self.analyzer.extract_trajectories(inputs)
        
        logits = trajectories['logits'].cpu().numpy().flatten()
        logit_changes = trajectories['logit_changes'].cpu().numpy().flatten()
        
        actions_flat = actions.flatten()
        rewards_flat = rewards.flatten()
        
        # Compute input conditions
        if second_stages is not None:
            # 8 conditions for two-stage task
            second_stages_flat = second_stages.flatten()
            input_conditions = 4 * actions_flat + 2 * second_stages_flat + rewards_flat
        else:
            # 4 conditions for reversal
            input_conditions = 2 * actions_flat + rewards_flat
        
        # Find fixed points
        fixed_points = self._estimate_fixed_points(
            logits, logit_changes, input_conditions
        )
        
        # Compute preference setpoints
        max_fp = max(abs(fp) for fp in fixed_points.values()) if fixed_points else 1.0
        setpoints = {k: v / max_fp for k, v in fixed_points.items()}
        
        return PhasePortraitData(
            logits=logits,
            logit_changes=logit_changes,
            actions=actions_flat,
            rewards=rewards_flat,
            input_conditions=input_conditions.astype(int),
            fixed_points=fixed_points,
            preference_setpoints=setpoints
        )
    
    def _estimate_fixed_points(
        self, 
        logits: np.ndarray, 
        logit_changes: np.ndarray,
        conditions: np.ndarray,
        n_bins: int = 20
    ) -> Dict[int, float]:
        """Estimate fixed points from empirical data."""
        fixed_points = {}
        unique_conditions = np.unique(conditions)
        
        for cond in unique_conditions:
            mask = conditions == cond
            L = logits[mask]
            dL = logit_changes[mask]
            
            if len(L) < 10:
                continue
            
            # Bin the data
            bins = np.linspace(L.min(), L.max(), n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_means = []
            
            for i in range(n_bins):
                bin_mask = (L >= bins[i]) & (L < bins[i + 1])
                if bin_mask.sum() > 0:
                    bin_means.append(dL[bin_mask].mean())
                else:
                    bin_means.append(np.nan)
            
            bin_means = np.array(bin_means)
            
            # Find zero crossings
            valid = ~np.isnan(bin_means)
            if valid.sum() < 2:
                continue
            
            for i in range(len(bin_means) - 1):
                if not valid[i] or not valid[i + 1]:
                    continue
                if bin_means[i] * bin_means[i + 1] < 0:
                    t = -bin_means[i] / (bin_means[i + 1] - bin_means[i])
                    fp = bin_centers[i] + t * (bin_centers[i + 1] - bin_centers[i])
                    fixed_points[int(cond)] = fp
                    break
        
        return fixed_points
    
    def plot_phase_portrait(
        self, 
        data: PhasePortraitData,
        ax: Optional[plt.Axes] = None,
        title: str = "",
        show_curves: bool = True
    ) -> plt.Axes:
        """Create phase portrait visualization."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        for cond in sorted(np.unique(data.input_conditions)):
            mask = data.input_conditions == cond
            
            action = cond // 2 if cond < 4 else (cond - 4) // 2
            reward = cond % 2
            label = f"$A_{action+1}$, $R={reward}$"
            
            color = self.colors.get(cond, '#808080')
            
            ax.scatter(
                data.logits[mask], 
                data.logit_changes[mask],
                c=color,
                alpha=0.5,
                s=10,
                label=label
            )
            
            if show_curves:
                L = data.logits[mask]
                dL = data.logit_changes[mask]
                
                sort_idx = np.argsort(L)
                L_sorted = L[sort_idx]
                dL_sorted = dL[sort_idx]
                
                if len(L_sorted) > 20:
                    window = len(L_sorted) // 10
                    L_smooth = np.convolve(L_sorted, np.ones(window)/window, mode='valid')
                    dL_smooth = np.convolve(dL_sorted, np.ones(window)/window, mode='valid')
                    
                    ax.plot(L_smooth, dL_smooth, color=color, linewidth=2, alpha=0.8)
        
        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Mark fixed points
        for cond, fp in data.fixed_points.items():
            color = self.colors.get(cond, '#808080')
            ax.axvline(x=fp, color=color, linestyle=':', alpha=0.7)
            ax.scatter([fp], [0], color=color, s=100, marker='x', zorder=5)
        
        ax.set_xlabel('Logit $L(t)$', fontsize=12)
        ax.set_ylabel('Logit Change $\\Delta L(t)$', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        
        return ax


# =============================================================================
# Vector Field Generator (Figure 4) - FOR d>1 MODELS
# =============================================================================

class VectorFieldGenerator:
    """
    Generate vector field visualizations for d>1 models.
    
    Following Figure 4 of Ji-An et al.:
    - 2D state space with h1, h2 axes
    - Arrows show (Delta-h1, Delta-h2) for each input condition
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
    
    def _single_step(
        self, 
        h: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Perform single step of dynamics."""
        if hasattr(self.model, 'gru_cell') and self.model.gru_cell is not None:
            return self.model.gru_cell(x, h)
        elif hasattr(self.model, 'recurrent'):
            if hasattr(self.model, 'W_in'):
                x_transformed = self.model.W_in(x)
            else:
                x_transformed = x
            return self.model.recurrent(h, x_transformed)
        else:
            return h
    
    def generate_vector_field(
        self,
        h_range: Tuple[float, float] = (-1, 1),
        n_grid: int = 15,
        input_conditions: List[int] = [0, 1, 2, 3],
        n_input_dim: int = 4
    ) -> VectorFieldData:
        """
        Generate vector field data on a grid of hidden states.
        
        Parameters
        ----------
        h_range : tuple
            Range for h1, h2 values
        n_grid : int
            Number of grid points per dimension
        input_conditions : list
            List of input condition indices to compute
        n_input_dim : int
            Dimension of one-hot input
        
        Returns
        -------
        VectorFieldData
            Grid and vectors for visualization
        """
        h_vals = np.linspace(h_range[0], h_range[1], n_grid)
        h1_grid, h2_grid = np.meshgrid(h_vals, h_vals)
        
        delta_h1 = {}
        delta_h2 = {}
        
        for cond in input_conditions:
            dh1 = np.zeros((n_grid, n_grid))
            dh2 = np.zeros((n_grid, n_grid))
            
            for i in range(n_grid):
                for j in range(n_grid):
                    # Create hidden state (assume d=2)
                    h = torch.tensor(
                        [[h1_grid[i, j], h2_grid[i, j]]], 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    
                    # Create one-hot input for condition
                    x = torch.zeros(1, n_input_dim, device=self.device)
                    x[0, cond] = 1.0
                    
                    with torch.no_grad():
                        h_new = self._single_step(h, x)
                    
                    dh1[i, j] = (h_new[0, 0] - h[0, 0]).item()
                    if h.shape[1] > 1:
                        dh2[i, j] = (h_new[0, 1] - h[0, 1]).item()
            
            delta_h1[cond] = dh1
            delta_h2[cond] = dh2
        
        return VectorFieldData(
            h1_grid=h1_grid,
            h2_grid=h2_grid,
            delta_h1=delta_h1,
            delta_h2=delta_h2,
            fixed_points={}
        )
    
    def plot_vector_field(
        self, 
        data: VectorFieldData,
        condition: int,
        ax: Optional[plt.Axes] = None,
        title: str = ""
    ) -> plt.Axes:
        """Plot vector field for a specific input condition."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Color by magnitude
        magnitude = np.sqrt(
            data.delta_h1[condition]**2 + data.delta_h2[condition]**2
        )
        
        ax.quiver(
            data.h1_grid, data.h2_grid,
            data.delta_h1[condition], data.delta_h2[condition],
            magnitude,
            cmap='viridis',
            alpha=0.7
        )
        
        ax.set_xlabel('$h_1$ (dynamical variable 1)', fontsize=12)
        ax.set_ylabel('$h_2$ (dynamical variable 2)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        return ax
    
    def plot_all_conditions(
        self, 
        data: VectorFieldData,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """Plot vector fields for all conditions."""
        n_conditions = len(data.delta_h1)
        n_cols = 2
        n_rows = (n_conditions + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_conditions > 1 else [axes]
        
        colors = ['blue', 'darkblue', 'red', 'darkred']
        labels = ['A1, R=0', 'A1, R=1', 'A2, R=0', 'A2, R=1']
        
        for idx, cond in enumerate(sorted(data.delta_h1.keys())):
            if idx < len(axes):
                self.plot_vector_field(data, cond, ax=axes[idx], title=labels[idx])
        
        plt.tight_layout()
        return fig


# =============================================================================
# Dynamics Analyzer
# =============================================================================

class DynamicsAnalyzer:
    """
    Comprehensive analysis of RNN dynamics.
    
    Includes:
    - Lyapunov exponent estimation
    - Representational geometry
    - Adaptation speed measurement
    - Two-stage task signatures
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
    
    def compute_lyapunov_exponent(
        self, 
        inputs: torch.Tensor,
        n_steps: int = 1000,
        n_trials: int = 10,
        epsilon: float = 1e-6
    ) -> float:
        """
        Estimate maximal Lyapunov exponent.
        
        lambda > 0: chaotic
        lambda < 0: convergent
        lambda ~ 0: edge of chaos
        """
        lyapunov_estimates = []
        
        for trial in range(n_trials):
            # Initial state
            h = torch.randn(1, self.model.config.n_hidden, device=self.device) * 0.1
            
            # Nearby state
            perturbation = torch.randn_like(h) * epsilon
            perturbation = perturbation / perturbation.norm() * epsilon
            h_perturbed = h + perturbation
            
            lyapunov_sum = 0.0
            
            seq_len = min(n_steps, inputs.shape[1])
            
            for t in range(seq_len):
                x_t = self.model.W_in(inputs[0:1, t, :])
                
                h = self.model.recurrent(h, x_t)
                h_perturbed = self.model.recurrent(h_perturbed, x_t)
                
                # Measure divergence
                delta = (h_perturbed - h).norm().item()
                
                if delta > 0:
                    lyapunov_sum += np.log(delta / epsilon)
                    
                    # Renormalize
                    h_perturbed = h + (h_perturbed - h) / delta * epsilon
            
            lyapunov_estimates.append(lyapunov_sum / seq_len)
        
        return float(np.mean(lyapunov_estimates))
    
    def analyze_representational_geometry(
        self,
        inputs: torch.Tensor,
        n_components: int = 3
    ) -> Dict[str, float]:
        """Analyze geometry of hidden state representations."""
        with torch.no_grad():
            _, _, hidden_states = self.model(inputs, return_hidden=True)
        
        h_flat = hidden_states.view(-1, hidden_states.shape[-1]).cpu().numpy()
        
        pca = PCA(n_components=min(n_components, h_flat.shape[1]))
        pca.fit(h_flat)
        
        # Participation ratio
        explained_var = pca.explained_variance_ratio_
        participation_ratio = 1.0 / (explained_var ** 2).sum()
        
        return {
            'variance_explained_top3': explained_var[:3].sum() if len(explained_var) >= 3 else explained_var.sum(),
            'participation_ratio': participation_ratio,
            'effective_dimensionality': participation_ratio,
            'total_variance': h_flat.var()
        }
    
    def analyze_two_stage_signatures(
        self,
        outputs: torch.Tensor,
        actions: np.ndarray,
        transition_types: np.ndarray,
        rewards: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze model-based vs model-free signatures.
        
        Model-free: P(stay|rew) > P(stay|no_rew), independent of transition
        Model-based: P(stay|common,rew) > P(stay|rare,rew)
        """
        actions = actions.flatten()
        transition_types = transition_types.flatten()
        rewards = rewards.flatten()
        
        stay_probs = {
            'common_rewarded': [],
            'common_unrewarded': [],
            'rare_rewarded': [],
            'rare_unrewarded': []
        }
        
        for t in range(len(actions) - 1):
            stayed = int(actions[t + 1] == actions[t])
            
            is_common = (transition_types[t] == 0)
            is_rewarded = (rewards[t] == 1)
            
            if is_common:
                if is_rewarded:
                    stay_probs['common_rewarded'].append(stayed)
                else:
                    stay_probs['common_unrewarded'].append(stayed)
            else:
                if is_rewarded:
                    stay_probs['rare_rewarded'].append(stayed)
                else:
                    stay_probs['rare_unrewarded'].append(stayed)
        
        results = {k: np.mean(v) if v else 0.5 for k, v in stay_probs.items()}
        
        # Compute indices
        results['mf_index'] = (
            (results['common_rewarded'] - results['common_unrewarded'] +
             results['rare_rewarded'] - results['rare_unrewarded']) / 2
        )
        results['mb_index'] = (
            results['common_rewarded'] - results['rare_rewarded']
        )
        
        return results


# =============================================================================
# Developmental Comparison Analyzer
# =============================================================================

class DevelopmentalComparisonAnalyzer:
    """
    Compare dynamical properties between premature and mature models.
    """
    
    def __init__(self, premature_model: nn.Module, mature_model: nn.Module):
        self.premature = premature_model
        self.mature = mature_model
        
        self.prem_phase = PhasePortraitGenerator(premature_model)
        self.mat_phase = PhasePortraitGenerator(mature_model)
    
    def run_comparison(
        self, 
        inputs: torch.Tensor,
        actions: np.ndarray,
        rewards: np.ndarray
    ) -> DevelopmentalComparison:
        """Run comprehensive comparison."""
        # Generate phase portrait data
        prem_data = self.prem_phase.generate_from_data(inputs, actions, rewards)
        mat_data = self.mat_phase.generate_from_data(inputs, actions, rewards)
        
        # Extract metrics
        prem_lr = self._compute_learning_rates(prem_data)
        mat_lr = self._compute_learning_rates(mat_data)
        
        prem_stability = self._analyze_stability(prem_data)
        mat_stability = self._analyze_stability(mat_data)
        
        prem_dim = self._compute_dimensionality(self.premature)
        mat_dim = self._compute_dimensionality(self.mature)
        
        prem_spec = self._compute_specialization(prem_data)
        mat_spec = self._compute_specialization(mat_data)
        
        return DevelopmentalComparison(
            premature_fixed_points=prem_data.fixed_points,
            mature_fixed_points=mat_data.fixed_points,
            premature_setpoints=prem_data.preference_setpoints,
            mature_setpoints=mat_data.preference_setpoints,
            premature_learning_rates=prem_lr,
            mature_learning_rates=mat_lr,
            premature_stability=prem_stability,
            mature_stability=mat_stability,
            dimensionality_premature=prem_dim,
            dimensionality_mature=mat_dim,
            specialization_premature=prem_spec,
            specialization_mature=mat_spec
        )
    
    def _compute_learning_rates(self, data: PhasePortraitData) -> Dict[str, float]:
        """Compute effective learning rate from logit change curves."""
        learning_rates = {}
        
        for cond in np.unique(data.input_conditions):
            mask = data.input_conditions == cond
            L = data.logits[mask]
            dL = data.logit_changes[mask]
            
            if len(L) < 10:
                continue
            
            neutral_mask = (np.abs(L) < 2)
            if neutral_mask.sum() > 5:
                slope, _, _, _, _ = stats.linregress(L[neutral_mask], dL[neutral_mask])
                learning_rates[f'cond_{cond}'] = abs(slope)
        
        return learning_rates
    
    def _analyze_stability(self, data: PhasePortraitData) -> Dict[str, float]:
        """Analyze stability of dynamics."""
        return {
            'n_fixed_points': float(len(data.fixed_points)),
            'logit_change_variance': float(np.var(data.logit_changes)),
            'mean_logit_magnitude': float(np.abs(data.logits).mean())
        }
    
    def _compute_dimensionality(self, model: nn.Module) -> float:
        """Compute effective dimensionality."""
        if hasattr(model, 'compute_effective_rank'):
            return model.compute_effective_rank()
        return 0.0
    
    def _compute_specialization(self, data: PhasePortraitData) -> float:
        """Compute specialization from setpoint variance."""
        setpoint_values = list(data.preference_setpoints.values())
        if len(setpoint_values) > 1:
            return float(np.var(setpoint_values))
        return 0.0
    
    def plot_comparison(
        self, 
        comparison: DevelopmentalComparison,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create comprehensive comparison visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Panel A: Preference setpoints
        ax = axes[0, 0]
        conditions = sorted(set(comparison.premature_setpoints.keys()) | 
                          set(comparison.mature_setpoints.keys()))
        x = np.arange(len(conditions))
        width = 0.35
        
        prem_vals = [comparison.premature_setpoints.get(c, 0) for c in conditions]
        mat_vals = [comparison.mature_setpoints.get(c, 0) for c in conditions]
        
        ax.bar(x - width/2, prem_vals, width, label='Premature', color='#E74C3C')
        ax.bar(x + width/2, mat_vals, width, label='Mature', color='#2ECC71')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in conditions])
        ax.set_ylabel('Preference Setpoint $u_I$')
        ax.set_title('A. Preference Setpoints')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Panel B: Learning rates
        ax = axes[0, 1]
        prem_lr = list(comparison.premature_learning_rates.values())
        mat_lr = list(comparison.mature_learning_rates.values())
        
        if prem_lr and mat_lr:
            ax.bar(['Premature', 'Mature'], 
                  [np.mean(prem_lr), np.mean(mat_lr)],
                  color=['#E74C3C', '#2ECC71'],
                  yerr=[np.std(prem_lr) if len(prem_lr) > 1 else 0, 
                        np.std(mat_lr) if len(mat_lr) > 1 else 0],
                  capsize=5)
        ax.set_ylabel('Effective Learning Rate')
        ax.set_title('B. Effective Learning Rates')
        
        # Panel C: Dimensionality
        ax = axes[0, 2]
        ax.bar(['Premature', 'Mature'],
              [comparison.dimensionality_premature, comparison.dimensionality_mature],
              color=['#E74C3C', '#2ECC71'])
        ax.set_ylabel('Effective Rank')
        ax.set_title('C. Dimensionality')
        
        # Panel D: Stability
        ax = axes[1, 0]
        metrics = ['n_fixed_points', 'logit_change_variance']
        prem_stab = [comparison.premature_stability.get(m, 0) for m in metrics]
        mat_stab = [comparison.mature_stability.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        ax.bar(x - width/2, prem_stab, width, label='Premature', color='#E74C3C')
        ax.bar(x + width/2, mat_stab, width, label='Mature', color='#2ECC71')
        ax.set_xticks(x)
        ax.set_xticklabels(['Fixed Points', '$\\Delta L$ Var'])
        ax.set_title('D. Stability Metrics')
        ax.legend()
        
        # Panel E: Specialization
        ax = axes[1, 1]
        ax.bar(['Premature', 'Mature'],
              [comparison.specialization_premature, comparison.specialization_mature],
              color=['#E74C3C', '#2ECC71'])
        ax.set_ylabel('Response Heterogeneity')
        ax.set_title('E. Specialization')
        
        # Panel F: Summary
        ax = axes[1, 2]
        summary_text = (
            f"DEVELOPMENTAL COMPARISON\n"
            f"{'='*30}\n\n"
            f"Dimensionality:\n"
            f"  Premature: {comparison.dimensionality_premature:.2f}\n"
            f"  Mature: {comparison.dimensionality_mature:.2f}\n\n"
            f"Specialization:\n"
            f"  Premature: {comparison.specialization_premature:.3f}\n"
            f"  Mature: {comparison.specialization_mature:.3f}\n\n"
            f"Key Finding:\n"
            f"{'Mature: lower dim + higher spec' if comparison.dimensionality_mature < comparison.dimensionality_premature and comparison.specialization_mature > comparison.specialization_premature else 'Analysis ongoing'}"
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


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Comprehensive Dynamical Analysis Module Testing")
    print("=" * 70)
    
    # Create simple test model
    class SimpleRNN(nn.Module):
        def __init__(self, hidden_size=2):
            super().__init__()
            self.hidden_size = hidden_size
            self.n_input = 4
            self.cell = nn.GRUCell(4, hidden_size)
            self.W_in = nn.Linear(3, 4)  # Dummy
            self.recurrent = self  # Self-reference for compatibility
            self.output_layer = nn.Linear(hidden_size, 2)
            
            # Dummy config
            class Config:
                n_hidden = hidden_size
                n_input = 3
            self.config = Config()
        
        def forward(self, inputs, return_hidden=False):
            batch, seq_len, _ = inputs.shape
            h = torch.zeros(batch, self.hidden_size)
            
            outputs = []
            hiddens = []
            
            for t in range(seq_len):
                h = self.cell(inputs[:, t, :].repeat(1, 2)[:, :4], h)
                y = self.output_layer(h)
                outputs.append(y)
                hiddens.append(h)
            
            outputs = torch.stack(outputs, dim=1)
            
            if return_hidden:
                return outputs, h, torch.stack(hiddens, dim=1)
            return outputs, h
        
        def get_weight_matrix(self):
            return self.cell.weight_hh
        
        def compute_effective_rank(self):
            W = self.cell.weight_hh
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            S_sq = S ** 2
            return float((S_sq.sum() ** 2) / (S_sq ** 2).sum())
    
    # Create test models
    print("\n1. Creating test models...")
    premature_model = SimpleRNN(hidden_size=2)
    mature_model = SimpleRNN(hidden_size=2)
    
    # Test data
    print("\n2. Generating test data...")
    batch_size, seq_len = 100, 50
    inputs = torch.randn(batch_size, seq_len, 4)
    inputs = F.softmax(inputs, dim=-1)
    
    actions = np.random.randint(0, 2, (batch_size, seq_len))
    rewards = np.random.randint(0, 2, (batch_size, seq_len))
    
    # Test PhasePortraitGenerator
    print("\n3. Testing PhasePortraitGenerator...")
    phase_gen = PhasePortraitGenerator(premature_model)
    phase_data = phase_gen.generate_from_data(inputs, actions, rewards)
    print(f"  Fixed points found: {phase_data.fixed_points}")
    print(f"  Preference setpoints: {phase_data.preference_setpoints}")
    
    # Test VectorFieldGenerator
    print("\n4. Testing VectorFieldGenerator...")
    vector_gen = VectorFieldGenerator(premature_model)
    vf_data = vector_gen.generate_vector_field(n_grid=5)
    print(f"  Grid shape: {vf_data.h1_grid.shape}")
    print(f"  Conditions computed: {list(vf_data.delta_h1.keys())}")
    
    # Test FixedPointFinder
    print("\n5. Testing FixedPointFinder...")
    fp_finder = FixedPointFinder(premature_model, n_points=5, n_iters=100)
    fp_results = fp_finder.find_fixed_points()
    print(f"  Fixed points found: {fp_results['n_found']}")
    
    # Test comparison
    print("\n6. Testing Developmental Comparison...")
    comparator = DevelopmentalComparisonAnalyzer(premature_model, mature_model)
    comparison = comparator.run_comparison(inputs, actions, rewards)
    
    print(f"  Premature dimensionality: {comparison.dimensionality_premature:.2f}")
    print(f"  Mature dimensionality: {comparison.dimensionality_mature:.2f}")
    print(f"  Premature specialization: {comparison.specialization_premature:.4f}")
    print(f"  Mature specialization: {comparison.specialization_mature:.4f}")
    
    print("\n" + "=" * 70)
    print("All dynamical analysis tests passed!")
    print("=" * 70)
