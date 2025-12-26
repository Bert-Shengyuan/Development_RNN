#!/usr/bin/env python3
"""
================================================================================
VISUALIZATION MODULE FOR BRAIN-INSPIRED RNN ANALYSIS
================================================================================

This module generates publication-quality figures comparing premature and 
mature brain-inspired RNNs on cognitive tasks.

Figure Components
-----------------
A. Model Architecture Comparison
B. Training Dynamics (Learning Curves)
C. Singular Value Spectra
D. Effective Rank Evolution
E. Response Heterogeneity Analysis
F. Task Performance Comparison
G. Hidden State Dynamics (PCA visualization)
H. Reversal Learning Adaptation
I. Phase Portraits (following Ji-An et al. Figure 3)
J. Vector Fields (following Ji-An et al. Figure 4)
K. Developmental Comparison Summary

Mathematical Framework
----------------------
The visualizations capture key developmental differences:

    PREMATURE: D_eff ≈ 7-8, γ ≈ 0.2, lower specialization
    MATURE: D_eff ≈ 5-6, γ ≈ 0.4, higher specialization

where:
    - D_eff = (Σσ_i)² / Σ(σ_i²) is effective rank
    - γ is the singular value decay rate in σ_i ∼ exp(-γi)

Reference
---------
Ji-An et al. (2025) "Discovering cognitive strategies with tiny RNNs"

Author: Computational Neuroscience Research
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.decomposition import PCA
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# =============================================================================
# Style Configuration
# =============================================================================

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.0,
})

# Color palette for developmental stages
COLORS = {
    'premature': '#E74C3C',         # Red
    'mature': '#2ECC71',            # Green
    'baseline': '#3498DB',          # Blue
    'premature_light': '#F5B7B1',
    'mature_light': '#ABEBC6',
    'baseline_light': '#AED6F1',
    'premature_dark': '#922B21',
    'mature_dark': '#1D8348',
}

# Phase portrait colors (following Ji-An et al.)
PHASE_COLORS = {
    0: '#ADD8E6',   # A1, R=0 (light blue)
    1: '#00008B',   # A1, R=1 (dark blue)
    2: '#FFC0CB',   # A2, R=0 (light pink)
    3: '#8B0000',   # A2, R=1 (dark red)
    4: '#90EE90',   # A1, S2, R=0 (light green)
    5: '#006400',   # A1, S2, R=1 (dark green)
    6: '#FFD700',   # A2, S1, R=0 (light yellow)
    7: '#FF8C00',   # A2, S1, R=1 (dark orange)
}


# =============================================================================
# Learning Curves
# =============================================================================

def plot_learning_curves(
    results: Dict, 
    ax: plt.Axes = None,
    show_legend: bool = True,
    show_train: bool = False
) -> plt.Axes:
    """
    Plot training and validation learning curves.
    
    Displays validation accuracy over epochs for each model,
    optionally with training loss overlay.
    
    Parameters
    ----------
    results : Dict
        Results dictionary from run_developmental_comparison
    ax : plt.Axes, optional
        Matplotlib axes (creates new if None)
    show_legend : bool
        Whether to show legend
    show_train : bool
        Whether to show training loss curve
    
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['premature', 'mature', 'baseline_gru']
    labels = ['Premature RNN', 'Mature RNN', 'Baseline GRU']
    colors = [COLORS['premature'], COLORS['mature'], COLORS['baseline']]
    
    for model, label, color in zip(models, labels, colors):
        if model not in results:
            continue
        
        history = results[model]['history']
        val_acc = np.array(history['val_accuracy'])
        epochs = np.arange(len(val_acc))
        
        # Plot validation accuracy
        ax.plot(epochs, val_acc, color=color, linewidth=2, label=label)
        
        # Add final point marker
        ax.scatter(epochs[-1], val_acc[-1], color=color, s=50, zorder=5)
        
        # Optional: show training loss on secondary axis
        if show_train and 'train_total' in history:
            ax2 = ax.twinx()
            train_loss = np.array(history['train_total'])
            ax2.plot(epochs, train_loss, color=color, linewidth=1, 
                    linestyle='--', alpha=0.5)
            ax2.set_ylabel('Training Loss', color='gray')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Learning Curves')
    ax.set_ylim([0.4, 1.0])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    if show_legend:
        ax.legend(loc='lower right', frameon=True, fancybox=True)
    
    return ax


# =============================================================================
# Singular Value Spectra
# =============================================================================

def plot_singular_value_spectra(
    results: Dict, 
    ax: plt.Axes = None,
    log_scale: bool = True,
    show_theoretical: bool = True
) -> plt.Axes:
    """
    Plot singular value spectra of recurrent weight matrices.
    
    This visualizes the key developmental difference:
    - Mature brains show faster SV decay (more compressed)
    - Premature brains show slower decay (higher effective rank)
    
    Mathematical interpretation:
    The decay rate γ in σ_i ∼ exp(-γi) indicates how quickly
    variance is concentrated in dominant modes.
    
    Parameters
    ----------
    results : Dict
        Results dictionary
    ax : plt.Axes, optional
        Matplotlib axes
    log_scale : bool
        Whether to use log scale for y-axis
    show_theoretical : bool
        Whether to show theoretical decay lines
    
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    for model_name, color in [('premature', COLORS['premature']), 
                               ('mature', COLORS['mature'])]:
        if model_name not in results:
            continue
        
        metrics = results[model_name].get('model_metrics', {})
        sv = metrics.get('singular_values', np.array([1.0]))
        decay_rate = metrics.get('sv_decay_rate', 0.3)
        
        if isinstance(sv, list):
            sv = np.array(sv)
        
        # Plot normalized singular values
        if log_scale:
            ax.semilogy(np.arange(len(sv)) + 1, sv, 'o-', 
                        color=color, linewidth=2, markersize=6,
                        label=f'{model_name.capitalize()} (γ={decay_rate:.2f})')
        else:
            ax.plot(np.arange(len(sv)) + 1, sv, 'o-', 
                   color=color, linewidth=2, markersize=6,
                   label=f'{model_name.capitalize()} (γ={decay_rate:.2f})')
    
    # Add theoretical decay lines for reference
    if show_theoretical:
        x = np.arange(1, 15)
        if log_scale:
            ax.semilogy(x, np.exp(-0.2 * x), '--', color=COLORS['premature_light'], 
                        alpha=0.7, label='γ=0.2 (slow)')
            ax.semilogy(x, np.exp(-0.4 * x), '--', color=COLORS['mature_light'], 
                        alpha=0.7, label='γ=0.4 (fast)')
        else:
            ax.plot(x, np.exp(-0.2 * x), '--', color=COLORS['premature_light'], alpha=0.7)
            ax.plot(x, np.exp(-0.4 * x), '--', color=COLORS['mature_light'], alpha=0.7)
    
    ax.set_xlabel('Singular Value Index $i$')
    ax.set_ylabel('Normalized $\\sigma_i / \\sigma_1$')
    ax.set_title('Singular Value Spectrum of $\\mathbf{W}_{rec}$')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim([0, 15])
    
    if log_scale:
        ax.set_ylim([1e-3, 1.5])
    
    return ax


# =============================================================================
# Effective Rank Comparison
# =============================================================================

def plot_effective_rank_comparison(
    results: Dict, 
    ax: plt.Axes = None,
    show_annotation: bool = True
) -> plt.Axes:
    """
    Plot effective rank comparison between developmental stages.
    
    Effective rank D(W) = (Σλ_i)² / Σλ_i² measures the intrinsic
    dimensionality of the dynamics.
    
    Key prediction: D(W_mature) < D(W_premature)
    
    Parameters
    ----------
    results : Dict
        Results dictionary
    ax : plt.Axes, optional
        Matplotlib axes
    show_annotation : bool
        Whether to show difference annotation
    
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Extract effective ranks
    models = []
    ranks = []
    colors = []
    
    for model_name in ['premature', 'mature']:
        if model_name in results:
            test_metrics = results[model_name].get('test_metrics', {})
            model_metrics = results[model_name].get('model_metrics', {})
            
            rank = test_metrics.get('effective_rank', 
                   model_metrics.get('effective_rank', 0.0))
            
            models.append(model_name.capitalize())
            ranks.append(rank)
            colors.append(COLORS[model_name])
    
    if not ranks:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes)
        return ax
    
    # Create bar plot
    bars = ax.bar(models, ranks, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{rank:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Effective Rank $D(\\mathbf{W})$')
    ax.set_title('Dimensionality Comparison')
    ax.set_ylim([0, max(ranks) * 1.3])
    
    # Add interpretation annotation
    if show_annotation and len(ranks) >= 2:
        diff = ranks[0] - ranks[1]
        if diff > 0:
            ax.annotate(f'Δ = {diff:.2f}\n(Lower = Compressed)',
                        xy=(0.5, max(ranks) * 0.5), ha='center',
                        fontsize=9, color='gray',
                        transform=ax.get_xaxis_transform())
    
    return ax


# =============================================================================
# Response Heterogeneity
# =============================================================================

def plot_response_heterogeneity(
    results: Dict, 
    ax: plt.Axes = None,
    metric_type: str = 'response_cv'
) -> plt.Axes:
    """
    Plot response heterogeneity comparison.
    
    Response heterogeneity measures how differently the network responds
    to perturbations at different units:
        Var(||∂h/∂I_j||₂) across j
    
    Higher heterogeneity indicates more specialized responses.
    Key prediction: Heterogeneity_mature > Heterogeneity_premature
    
    Parameters
    ----------
    results : Dict
        Results dictionary
    ax : plt.Axes, optional
        Matplotlib axes
    metric_type : str
        Type of heterogeneity metric ('response_cv', 'jacobian', 'sparsity')
    
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    models = []
    heterogeneities = []
    colors = []
    
    for model_name in ['premature', 'mature']:
        if model_name in results:
            test_metrics = results[model_name].get('test_metrics', {})
            
            hetero = test_metrics.get('response_heterogeneity', 0.0)
            
            models.append(model_name.capitalize())
            heterogeneities.append(hetero)
            colors.append(COLORS[model_name])
    
    if not heterogeneities:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes)
        return ax
    
    # Create bar plot
    bars = ax.bar(models, heterogeneities, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, hetero in zip(bars, heterogeneities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{hetero:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Response Heterogeneity')
    ax.set_title('Functional Specialization')
    
    if heterogeneities:
        ax.set_ylim([0, max(heterogeneities) * 1.4])
    
    return ax


# =============================================================================
# Hidden State PCA Visualization
# =============================================================================

def plot_hidden_state_pca(
    model: nn.Module,
    inputs: torch.Tensor,
    ax: plt.Axes = None,
    n_components: int = 2,
    color_by_time: bool = True,
    title: str = ''
) -> plt.Axes:
    """
    Visualize hidden state trajectories using PCA.
    
    Projects the high-dimensional hidden state dynamics onto
    the first two principal components.
    
    Parameters
    ----------
    model : nn.Module
        Trained RNN model
    inputs : torch.Tensor
        Input tensor [batch, seq_len, input_dim]
    ax : plt.Axes, optional
        Matplotlib axes
    n_components : int
        Number of PCA components (typically 2)
    color_by_time : bool
        Whether to color by time step
    title : str
        Plot title
    
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    model.eval()
    
    with torch.no_grad():
        outputs, h_final, hidden_history = model(inputs, return_hidden=True)
    
    # Get hidden states [batch, seq_len, hidden_dim]
    hidden = hidden_history.cpu().numpy()
    
    # Reshape for PCA
    batch_size, seq_len, hidden_dim = hidden.shape
    hidden_flat = hidden.reshape(-1, hidden_dim)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    hidden_pca = pca.fit_transform(hidden_flat)
    hidden_pca = hidden_pca.reshape(batch_size, seq_len, n_components)
    
    # Plot trajectories
    if color_by_time:
        cmap = plt.cm.viridis
        for b in range(min(batch_size, 10)):  # Plot up to 10 trajectories
            colors = cmap(np.linspace(0, 1, seq_len))
            for t in range(seq_len - 1):
                ax.plot(hidden_pca[b, t:t+2, 0], hidden_pca[b, t:t+2, 1],
                       color=colors[t], alpha=0.5, linewidth=0.8)
            
            # Mark start and end
            ax.scatter(hidden_pca[b, 0, 0], hidden_pca[b, 0, 1], 
                      color='green', s=30, marker='o', zorder=5)
            ax.scatter(hidden_pca[b, -1, 0], hidden_pca[b, -1, 1], 
                      color='red', s=30, marker='x', zorder=5)
    else:
        for b in range(min(batch_size, 10)):
            ax.plot(hidden_pca[b, :, 0], hidden_pca[b, :, 1], 
                   alpha=0.5, linewidth=0.8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Hidden State Trajectories\n{title}')
    
    # Add colorbar for time
    if color_by_time:
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                    norm=plt.Normalize(vmin=0, vmax=seq_len))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Time step')
    
    return ax


# =============================================================================
# Weight Matrix Visualization
# =============================================================================

def plot_weight_matrices(
    results: Dict,
    figsize: Tuple[float, float] = (12, 5)
) -> plt.Figure:
    """
    Visualize recurrent weight matrices for both developmental stages.
    
    Shows the structure of W_rec for premature and mature models,
    highlighting differences in sparsity and organization.
    
    Parameters
    ----------
    results : Dict
        Results dictionary containing trained models
    figsize : Tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for idx, (model_name, color) in enumerate([('premature', COLORS['premature']), 
                                                 ('mature', COLORS['mature'])]):
        ax = axes[idx]
        
        if model_name not in results:
            ax.text(0.5, 0.5, f'No {model_name} model', ha='center', va='center',
                   transform=ax.transAxes)
            continue
        
        model = results[model_name].get('model', None)
        
        if model is None:
            ax.text(0.5, 0.5, 'Model not available', ha='center', va='center',
                   transform=ax.transAxes)
            continue
        
        # Extract weight matrix
        W = None
        if hasattr(model, 'recurrent') and hasattr(model.recurrent, 'get_weight_matrix'):
            W = model.recurrent.get_weight_matrix().detach().cpu().numpy()
        elif hasattr(model, 'gru'):
            W = model.gru.weight_hh.detach().cpu().numpy()
        
        if W is None:
            ax.text(0.5, 0.5, 'Weights not accessible', ha='center', va='center',
                   transform=ax.transAxes)
            continue
        
        # Plot heatmap
        vmax = np.abs(W).max()
        im = ax.imshow(W, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        
        ax.set_title(f'{model_name.capitalize()} $\\mathbf{{W}}_{{rec}}$')
        ax.set_xlabel('Pre-synaptic unit')
        ax.set_ylabel('Post-synaptic unit')
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Phase Portrait Visualization (Figure 3 style)
# =============================================================================

def plot_phase_portrait(
    model: nn.Module,
    inputs: torch.Tensor,
    actions: np.ndarray,
    rewards: np.ndarray,
    ax: plt.Axes = None,
    title: str = '',
    show_fixed_points: bool = True,
    show_curves: bool = True
) -> plt.Axes:
    """
    Create phase portrait visualization following Ji-An et al. Figure 3.
    
    Phase portrait shows:
    - X-axis: Logit L(t) = log(P(A1)/P(A2))
    - Y-axis: Logit change ΔL(t) = L(t+1) - L(t)
    - Colors: Input condition (action × reward)
    
    Fixed points occur where ΔL = 0 (nullclines).
    
    Parameters
    ----------
    model : nn.Module
        Trained RNN model
    inputs : torch.Tensor
        Input tensor [batch, seq_len, input_dim]
    actions : np.ndarray
        Action sequence [batch, seq_len]
    rewards : np.ndarray
        Reward sequence [batch, seq_len]
    ax : plt.Axes, optional
        Matplotlib axes
    title : str
        Plot title
    show_fixed_points : bool
        Whether to mark fixed points
    show_curves : bool
        Whether to show smoothed curves
    
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    model.eval()
    
    with torch.no_grad():
        outputs, _ = model(inputs)
    
    # Compute logits: L(t) = output_1 - output_2
    logits = (outputs[:, :, 0] - outputs[:, :, 1]).cpu().numpy()
    
    # Compute logit changes
    logit_changes = np.zeros_like(logits)
    logit_changes[:, :-1] = logits[:, 1:] - logits[:, :-1]
    
    # Flatten
    logits_flat = logits.flatten()
    logit_changes_flat = logit_changes.flatten()
    actions_flat = actions.flatten()
    rewards_flat = rewards.flatten()
    
    # Compute input conditions: 2*action + reward
    conditions = 2 * actions_flat + rewards_flat
    
    # Plot each condition
    for cond in np.unique(conditions):
        mask = conditions == cond
        
        action = int(cond // 2)
        reward = int(cond % 2)
        label = f'$A_{action+1}$, $R={reward}$'
        
        color = PHASE_COLORS.get(int(cond), '#808080')
        
        ax.scatter(
            logits_flat[mask], 
            logit_changes_flat[mask],
            c=color,
            alpha=0.4,
            s=8,
            label=label
        )
        
        if show_curves:
            L = logits_flat[mask]
            dL = logit_changes_flat[mask]
            
            sort_idx = np.argsort(L)
            L_sorted = L[sort_idx]
            dL_sorted = dL[sort_idx]
            
            if len(L_sorted) > 20:
                window = max(len(L_sorted) // 15, 3)
                L_smooth = np.convolve(L_sorted, np.ones(window)/window, mode='valid')
                dL_smooth = np.convolve(dL_sorted, np.ones(window)/window, mode='valid')
                
                ax.plot(L_smooth, dL_smooth, color=color, linewidth=2, alpha=0.8)
    
    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Logit $L(t)$', fontsize=12)
    ax.set_ylabel('Logit Change $\\Delta L(t)$', fontsize=12)
    ax.set_title(f'Phase Portrait\n{title}', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    
    return ax


# =============================================================================
# Vector Field Visualization (Figure 4 style)
# =============================================================================

def plot_vector_field(
    model: nn.Module,
    ax: plt.Axes = None,
    n_grid: int = 15,
    condition: int = 0,
    title: str = '',
    h_range: Tuple[float, float] = (-2, 2)
) -> plt.Axes:
    """
    Create 2D vector field visualization for d=2 models.
    
    Following Ji-An et al. Figure 4:
    - Shows (Δh1, Δh2) arrows on (h1, h2) grid
    - Identifies attractors and flow patterns
    
    Parameters
    ----------
    model : nn.Module
        Trained RNN model with hidden_dim = 2
    ax : plt.Axes, optional
        Matplotlib axes
    n_grid : int
        Number of grid points per dimension
    condition : int
        Input condition index
    title : str
        Plot title
    h_range : Tuple
        Range for hidden state values
    
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create grid
    h1_vals = np.linspace(h_range[0], h_range[1], n_grid)
    h2_vals = np.linspace(h_range[0], h_range[1], n_grid)
    H1, H2 = np.meshgrid(h1_vals, h2_vals)
    
    # Compute dynamics at each grid point
    delta_h1 = np.zeros_like(H1)
    delta_h2 = np.zeros_like(H2)
    
    model.eval()
    
    # Determine input dimension
    if hasattr(model, 'config'):
        n_input = model.config.n_input
    else:
        n_input = 4  # Default for switching models
    
    # Create input for this condition
    x = torch.zeros(1, n_input)
    if condition < n_input:
        x[0, condition] = 1.0
    
    with torch.no_grad():
        for i in range(n_grid):
            for j in range(n_grid):
                h = torch.tensor([[H1[i, j], H2[i, j]]], dtype=torch.float32)
                
                # Get next hidden state
                if hasattr(model, 'recurrent'):
                    if hasattr(model, 'W_in'):
                        x_transformed = model.W_in(x)
                        h_next = model.recurrent(h, x_transformed)
                    else:
                        h_next = model.recurrent(h, x)
                elif hasattr(model, 'gru'):
                    h_next = model.gru(x, h)
                else:
                    h_next = h
                
                h_next = h_next.numpy()
                delta_h1[i, j] = h_next[0, 0] - H1[i, j]
                delta_h2[i, j] = h_next[0, 1] - H2[i, j]
    
    # Normalize arrows for visualization
    magnitude = np.sqrt(delta_h1**2 + delta_h2**2)
    max_mag = magnitude.max()
    if max_mag > 0:
        delta_h1_norm = delta_h1 / max_mag
        delta_h2_norm = delta_h2 / max_mag
    else:
        delta_h1_norm = delta_h1
        delta_h2_norm = delta_h2
    
    # Plot vector field
    color = PHASE_COLORS.get(condition, '#808080')
    ax.quiver(H1, H2, delta_h1_norm, delta_h2_norm, 
              color=color, alpha=0.7, scale=20, width=0.005)
    
    # Mark origin
    ax.scatter([0], [0], color='black', s=50, marker='o', zorder=5)
    
    ax.set_xlabel('$h_1$', fontsize=12)
    ax.set_ylabel('$h_2$', fontsize=12)
    ax.set_title(f'Vector Field (Condition {condition})\n{title}', fontsize=12)
    ax.set_xlim(h_range)
    ax.set_ylim(h_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


# =============================================================================
# Summary Statistics Table
# =============================================================================

def plot_summary_table(
    results: Dict, 
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Create a summary statistics table.
    
    Parameters
    ----------
    results : Dict
        Results dictionary
    ax : plt.Axes, optional
        Matplotlib axes
    
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.axis('off')
    
    columns = ['Metric', 'Premature', 'Mature', 'Δ']
    rows = []
    
    if 'premature' in results and 'mature' in results:
        prem = results['premature'].get('test_metrics', {})
        mat = results['mature'].get('test_metrics', {})
        prem_model = results['premature'].get('model_metrics', {})
        mat_model = results['mature'].get('model_metrics', {})
        
        # Effective rank
        prem_rank = prem.get('effective_rank', prem_model.get('effective_rank', 0.0))
        mat_rank = mat.get('effective_rank', mat_model.get('effective_rank', 0.0))
        rows.append(['Effective Rank $D(\\mathbf{W})$', 
                     f'{prem_rank:.2f}', f'{mat_rank:.2f}',
                     f'{mat_rank-prem_rank:+.2f}'])
        
        # Accuracy
        prem_acc = prem.get('accuracy', 0.0)
        mat_acc = mat.get('accuracy', 0.0)
        rows.append(['Test Accuracy', f'{prem_acc:.3f}', f'{mat_acc:.3f}',
                     f'{mat_acc-prem_acc:+.3f}'])
        
        # Loss
        prem_loss = prem.get('loss', 0.0)
        mat_loss = mat.get('loss', 0.0)
        rows.append(['Test Loss', f'{prem_loss:.3f}', f'{mat_loss:.3f}',
                     f'{mat_loss-prem_loss:+.3f}'])
        
        # SV decay rate
        prem_sv = prem_model.get('sv_decay_rate', 0.0)
        mat_sv = mat_model.get('sv_decay_rate', 0.0)
        rows.append(['SV Decay Rate $\\gamma$', f'{prem_sv:.3f}', f'{mat_sv:.3f}',
                     f'{mat_sv-prem_sv:+.3f}'])
        
        # Response heterogeneity
        prem_hetero = prem.get('response_heterogeneity', 0.0)
        mat_hetero = mat.get('response_heterogeneity', 0.0)
        rows.append(['Response Heterogeneity', f'{prem_hetero:.4f}', f'{mat_hetero:.4f}',
                     f'{mat_hetero-prem_hetero:+.4f}'])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=columns,
                     loc='center', cellLoc='center',
                     colColours=['#E8E8E8']*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')
    
    ax.set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=20)
    
    return ax


# =============================================================================
# Comprehensive Figure
# =============================================================================

def create_comprehensive_figure(
    results: Dict,
    figsize: Tuple[float, float] = (16, 12)
) -> plt.Figure:
    """
    Create comprehensive comparison figure with all panels.
    
    Layout:
    ┌────────────────┬────────────────┬────────────────┐
    │ A. Learning    │ B. SV Spectrum │ C. Eff. Rank   │
    ├────────────────┼────────────────┼────────────────┤
    │ D. Heterog.    │ E. Weights     │ F. Weights     │
    ├────────────────┴────────────────┴────────────────┤
    │                G. Summary Table                  │
    └──────────────────────────────────────────────────┘
    
    Parameters
    ----------
    results : Dict
        Results dictionary from run_developmental_comparison
    figsize : Tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.6], 
                           hspace=0.35, wspace=0.3)
    
    # A. Learning curves
    ax_learn = fig.add_subplot(gs[0, 0])
    plot_learning_curves(results, ax=ax_learn)
    ax_learn.set_title('A. Learning Curves', fontweight='bold')
    
    # B. Singular value spectra
    ax_sv = fig.add_subplot(gs[0, 1])
    plot_singular_value_spectra(results, ax=ax_sv)
    ax_sv.set_title('B. Singular Value Spectrum', fontweight='bold')
    
    # C. Effective rank
    ax_rank = fig.add_subplot(gs[0, 2])
    plot_effective_rank_comparison(results, ax=ax_rank)
    ax_rank.set_title('C. Effective Rank', fontweight='bold')
    
    # D. Response heterogeneity
    ax_hetero = fig.add_subplot(gs[1, 0])
    plot_response_heterogeneity(results, ax=ax_hetero)
    ax_hetero.set_title('D. Response Heterogeneity', fontweight='bold')
    
    # E-F. Weight matrices
    ax_w_prem = fig.add_subplot(gs[1, 1])
    ax_w_mat = fig.add_subplot(gs[1, 2])
    
    for ax, model_name, title_letter in [(ax_w_prem, 'premature', 'E'),
                                          (ax_w_mat, 'mature', 'F')]:
        if model_name in results and 'model' in results[model_name]:
            model = results[model_name]['model']
            W = None
            
            if hasattr(model, 'recurrent') and hasattr(model.recurrent, 'get_weight_matrix'):
                W = model.recurrent.get_weight_matrix().detach().cpu().numpy()
            
            if W is not None:
                vmax = np.abs(W).max()
                im = ax.imshow(W, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                ax.set_title(f'{title_letter}. {model_name.capitalize()} $\\mathbf{{W}}_{{rec}}$',
                           fontweight='bold')
                ax.set_xlabel('Pre-synaptic')
                ax.set_ylabel('Post-synaptic')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'Weights not available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{title_letter}. {model_name.capitalize()} Weights', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Model not available', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{title_letter}. {model_name.capitalize()} Weights', fontweight='bold')
    
    # G. Summary table
    ax_table = fig.add_subplot(gs[2, :])
    plot_summary_table(results, ax=ax_table)
    ax_table.set_title('G. Summary Statistics', fontweight='bold')
    
    plt.suptitle('Brain-Inspired RNN: Premature vs Mature Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    return fig


# =============================================================================
# Generate All Figures
# =============================================================================

def generate_all_figures(
    results: Dict, 
    output_dir: str = './figures'
) -> List[str]:
    """
    Generate all figures for the paper.
    
    Creates:
    1. Main comprehensive figure
    2. Weight matrix comparison
    3. Detailed learning curves
    
    Parameters
    ----------
    results : Dict
        Results from run_developmental_comparison
    output_dir : str
        Directory to save figures
    
    Returns
    -------
    List[str]
        List of saved figure paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    # Figure 1: Comprehensive comparison
    try:
        fig1 = create_comprehensive_figure(results)
        path1 = os.path.join(output_dir, 'figure_comprehensive.png')
        fig1.savefig(path1, dpi=300, bbox_inches='tight', facecolor='white')
        saved_paths.append(path1)
        plt.close(fig1)
    except Exception as e:
        print(f"Warning: Could not generate comprehensive figure: {e}")
    
    # Figure 2: Weight matrices
    try:
        fig2 = plot_weight_matrices(results)
        path2 = os.path.join(output_dir, 'figure_weight_matrices.png')
        fig2.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white')
        saved_paths.append(path2)
        plt.close(fig2)
    except Exception as e:
        print(f"Warning: Could not generate weight matrix figure: {e}")
    
    # Figure 3: Detailed learning curves
    try:
        fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        for model_name, color in [('premature', COLORS['premature']), 
                                   ('mature', COLORS['mature']),
                                   ('baseline_gru', COLORS['baseline'])]:
            if model_name in results:
                history = results[model_name]['history']
                train_loss = history.get('train_total', [])
                val_loss = history.get('val_loss', [])
                label = model_name.replace('_', ' ').capitalize()
                
                if train_loss:
                    axes[0].plot(train_loss, color=color, linestyle='-', 
                                label=f'{label} (train)', alpha=0.7)
                if val_loss:
                    axes[0].plot(val_loss, color=color, linestyle='--',
                                label=f'{label} (val)')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (NLL)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend(loc='upper right', fontsize=8)
        
        # Accuracy curves
        for model_name, color in [('premature', COLORS['premature']), 
                                   ('mature', COLORS['mature']),
                                   ('baseline_gru', COLORS['baseline'])]:
            if model_name in results:
                val_acc = results[model_name]['history'].get('val_accuracy', [])
                label = model_name.replace('_', ' ').capitalize()
                if val_acc:
                    axes[1].plot(val_acc, color=color, linewidth=2, label=label)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Accuracy')
        axes[1].set_title('Learning Progress')
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1].legend(loc='lower right')
        
        plt.tight_layout()
        path3 = os.path.join(output_dir, 'figure_learning_details.png')
        fig3.savefig(path3, dpi=300, bbox_inches='tight', facecolor='white')
        saved_paths.append(path3)
        plt.close(fig3)
    except Exception as e:
        print(f"Warning: Could not generate learning details figure: {e}")
    
    print(f"\nGenerated {len(saved_paths)} figures:")
    for path in saved_paths:
        print(f"  - {path}")
    
    return saved_paths


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Visualization Module")
    print("=" * 50)
    
    # Create mock results for testing
    mock_results = {
        'premature': {
            'history': {
                'train_total': list(np.exp(-np.linspace(0, 2, 80)) * 0.5 + 0.2),
                'val_loss': list(np.exp(-np.linspace(0, 2, 80)) * 0.5 + 0.22),
                'val_accuracy': list(1 - np.exp(-np.linspace(0, 2, 80)) * 0.4 - 0.1)
            },
            'test_metrics': {
                'accuracy': 0.72,
                'loss': 0.35,
                'effective_rank': 6.8,
                'response_heterogeneity': 0.012
            },
            'model_metrics': {
                'effective_rank': 6.8,
                'sv_decay_rate': 0.22,
                'singular_values': list(np.exp(-0.22 * np.arange(15)))
            }
        },
        'mature': {
            'history': {
                'train_total': list(np.exp(-np.linspace(0, 2.5, 80)) * 0.5 + 0.18),
                'val_loss': list(np.exp(-np.linspace(0, 2.5, 80)) * 0.5 + 0.20),
                'val_accuracy': list(1 - np.exp(-np.linspace(0, 2.5, 80)) * 0.35 - 0.08)
            },
            'test_metrics': {
                'accuracy': 0.76,
                'loss': 0.30,
                'effective_rank': 4.5,
                'response_heterogeneity': 0.018
            },
            'model_metrics': {
                'effective_rank': 4.5,
                'sv_decay_rate': 0.38,
                'singular_values': list(np.exp(-0.38 * np.arange(15)))
            }
        },
        'baseline_gru': {
            'history': {
                'train_total': list(np.exp(-np.linspace(0, 1.8, 80)) * 0.5 + 0.25),
                'val_loss': list(np.exp(-np.linspace(0, 1.8, 80)) * 0.5 + 0.28),
                'val_accuracy': list(1 - np.exp(-np.linspace(0, 1.8, 80)) * 0.45 - 0.12)
            },
            'test_metrics': {
                'accuracy': 0.68,
                'loss': 0.40
            },
            'model_metrics': {
                'effective_rank': 3.2,
                'sv_decay_rate': 0.30,
                'singular_values': list(np.exp(-0.30 * np.arange(4)))
            }
        }
    }
    
    # Generate test figures
    fig = create_comprehensive_figure(mock_results)
    
    print("\n✓ Visualization tests complete!")
    print("Close the figure window to exit.")
    
    plt.show()
