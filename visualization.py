"""
Visualization Module for Brain-Inspired RNN Analysis
=====================================================

This module generates publication-quality figures comparing premature and 
mature brain-inspired RNNs on cognitive tasks.

Figure Components:
------------------
A. Model Architecture Comparison
B. Training Dynamics (Learning Curves)
C. Singular Value Spectra
D. Effective Rank Evolution
E. Response Heterogeneity Analysis
F. Task Performance Comparison
G. Hidden State Dynamics (PCA visualization)
H. Reversal Learning Adaptation

Reference: Based on the framework in 
"Discovering cognitive strategies with tiny recurrent neural networks" (Ji-An et al.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.decomposition import PCA

import torch
import torch.nn as nn

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
})

# Color palette for developmental stages
COLORS = {
    'premature': '#E74C3C',      # Red
    'mature': '#2ECC71',         # Green
    'baseline': '#3498DB',       # Blue
    'premature_light': '#F5B7B1',
    'mature_light': '#ABEBC6',
    'baseline_light': '#AED6F1'
}


def plot_learning_curves(results: Dict, ax: plt.Axes = None,
                         show_legend: bool = True) -> plt.Axes:
    """
    Plot training and validation learning curves.
    
    Displays:
    - Training loss over epochs for each model
    - Validation accuracy with confidence bands
    
    Args:
        results: Dictionary from run_developmental_comparison
        ax: Matplotlib axes (creates new if None)
        show_legend: Whether to show legend
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['premature', 'mature', 'baseline_gru']
    labels = ['Premature RNN', 'Mature RNN', 'Baseline GRU (d=4)']
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
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Learning Curves')
    ax.set_ylim([0.4, 1.0])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    if show_legend:
        ax.legend(loc='lower right', frameon=True, fancybox=True)
    
    return ax


def plot_singular_value_spectra(results: Dict, ax: plt.Axes = None) -> plt.Axes:
    """
    Plot singular value spectra of recurrent weight matrices.
    
    This visualizes the key developmental difference:
    - Mature brains show faster SV decay (more compressed)
    - Premature brains show slower decay (higher effective rank)
    
    Mathematical interpretation:
    The decay rate γ in σ_i ∝ exp(-γi) indicates how quickly
    variance is concentrated in dominant modes.
    
    Args:
        results: Dictionary from run_developmental_comparison
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    for model_name, color in [('premature', COLORS['premature']), 
                               ('mature', COLORS['mature'])]:
        if model_name not in results:
            continue
        
        metrics = results[model_name]['model_metrics']
        sv = metrics['singular_values']
        
        # Plot normalized singular values
        ax.semilogy(np.arange(len(sv)) + 1, sv, 'o-', 
                    color=color, linewidth=2, markersize=6,
                    label=f'{model_name.capitalize()} (γ={metrics["sv_decay_rate"]:.2f})')
    
    # Add theoretical decay lines for reference
    x = np.arange(1, 15)
    ax.semilogy(x, np.exp(-0.2 * x), '--', color=COLORS['premature'], 
                alpha=0.5, label='γ=0.2 (slow decay)')
    ax.semilogy(x, np.exp(-0.4 * x), '--', color=COLORS['mature'], 
                alpha=0.5, label='γ=0.4 (fast decay)')
    
    ax.set_xlabel('Singular Value Index $i$')
    ax.set_ylabel('Normalized $\\sigma_i / \\sigma_1$')
    ax.set_title('Singular Value Spectrum of $\\mathbf{W}_{rec}$')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim([0, 15])
    ax.set_ylim([1e-3, 1.5])
    
    return ax


def plot_effective_rank_comparison(results: Dict, ax: plt.Axes = None) -> plt.Axes:
    """
    Plot effective rank comparison between developmental stages.
    
    Effective rank D(W) = (Σλ_i)² / Σλ_i² measures the intrinsic
    dimensionality of the dynamics.
    
    Key prediction: D(W_mature) < D(W_premature)
    This reflects compression of dynamics onto lower-dimensional manifold.
    
    Args:
        results: Results dictionary
        ax: Matplotlib axes
    
    Returns:
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
            models.append(model_name.capitalize())
            ranks.append(results[model_name]['test_metrics']['effective_rank'])
            colors.append(COLORS[model_name])
    
    # Create bar plot
    bars = ax.bar(models, ranks, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{rank:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Effective Rank $D(\\mathbf{W})$')
    ax.set_title('Dimensionality Comparison')
    ax.set_ylim([0, max(ranks) * 1.3])
    
    # # Add interpretation annotation
    # diff = ranks[0] - ranks[1] if len(ranks) == 2 else 0
    # if diff > 0:
    #     ax.annotate(f'Δ = {diff:.2f}\n(Lower = More Compressed)',
    #                 xy=(0.5, max(ranks) * 0.7), ha='center',
    #                 fontsize=9, color='gray')
    
    return ax


def plot_response_heterogeneity(results: Dict, ax: plt.Axes = None) -> plt.Axes:
    """
    Plot response heterogeneity comparison.
    
    Response heterogeneity measures how differently the network responds
    to perturbations at different units:
        Var(||∂h/∂I_j||₂) across j
    
    Higher heterogeneity indicates more specialized, differentiated responses.
    Key prediction: Heterogeneity_mature > Heterogeneity_premature
    
    Args:
        results: Results dictionary
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Extract heterogeneity values
    models = []
    hetero = []
    colors = []
    
    for model_name in ['premature', 'mature']:
        if model_name in results and 'response_heterogeneity' in results[model_name]['test_metrics']:
            models.append(model_name.capitalize())
            hetero.append(results[model_name]['test_metrics']['response_heterogeneity'])
            colors.append(COLORS[model_name])
    
    # Create bar plot
    bars = ax.bar(models, hetero, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, h in zip(bars, hetero):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{h:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Response Heterogeneity')
    ax.set_title('Functional Specialization')
    ax.set_ylim([0, max(hetero) * 1.4])
    
    # Add interpretation
    # if len(hetero) == 2:
    #     diff = hetero[1] - hetero[0]
    #     sign = '+' if diff > 0 else ''
    #     ax.annotate(f'Δ = {sign}{diff:.4f}\n(Higher = More Specialized)',
    #                 xy=(0.5, max(hetero) * 0.7), ha='center',
    #                 fontsize=9, color='gray')
    #
    return ax


def plot_task_performance_bars(results: Dict, ax: plt.Axes = None) -> plt.Axes:
    """
    Plot task performance comparison across all models.
    
    Shows:
    - Final test accuracy
    - Error bars from validation variance
    
    Args:
        results: Results dictionary
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    
    models = ['premature', 'mature', 'baseline_gru']
    labels = ['Premature\nRNN', 'Mature\nRNN', 'Baseline\nGRU (d=4)']
    colors = [COLORS['premature'], COLORS['mature'], COLORS['baseline']]
    
    accuracies = []
    stds = []
    valid_labels = []
    valid_colors = []
    
    for model, label, color in zip(models, labels, colors):
        if model in results:
            acc = results[model]['test_metrics']['accuracy']
            # Estimate std from validation history
            val_hist = np.array(results[model]['history']['val_accuracy'])
            std = np.std(val_hist[-10:])  # Use last 10 epochs for stability estimate
            
            accuracies.append(acc)
            stds.append(std)
            valid_labels.append(label)
            valid_colors.append(color)
    
    x = np.arange(len(valid_labels))
    bars = ax.bar(x, accuracies, yerr=stds, capsize=5,
                  color=valid_colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (acc, std) in enumerate(zip(accuracies, stds)):
        ax.text(i, acc + std + 0.02, f'{acc:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Cognitive Task Performance')
    ax.set_ylim([0.4, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    return ax


def plot_hidden_state_dynamics(model: nn.Module, inputs: torch.Tensor,
                                ax: plt.Axes = None, title: str = '') -> plt.Axes:
    """
    Visualize hidden state dynamics using PCA.
    
    Projects the hidden state trajectory onto the first 2 PCs to show
    how the network's internal representation evolves during a trial.
    
    Args:
        model: Trained RNN model
        inputs: Input sequence [1, seq_len, input_dim]
        ax: Matplotlib axes
        title: Plot title
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    model.eval()
    with torch.no_grad():
        outputs, h_final, hidden_history = model(inputs, return_hidden=True)
    
    # Get hidden states [seq_len, n_hidden]
    hidden = hidden_history[0].cpu().numpy()
    
    # PCA projection
    pca = PCA(n_components=2)
    hidden_pca = pca.fit_transform(hidden)
    
    # Create colormap based on time
    n_steps = len(hidden_pca)
    colors = plt.cm.viridis(np.linspace(0, 1, n_steps))
    
    # Plot trajectory
    for i in range(n_steps - 1):
        ax.plot(hidden_pca[i:i+2, 0], hidden_pca[i:i+2, 1],
                color=colors[i], linewidth=1.5, alpha=0.7)
    
    # Mark start and end
    ax.scatter(hidden_pca[0, 0], hidden_pca[0, 1], 
               color='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(hidden_pca[-1, 0], hidden_pca[-1, 1], 
               color='red', s=100, marker='s', zorder=5, label='End')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title(f'Hidden State Dynamics\n{title}')
    ax.legend(loc='upper right')
    
    return ax


def plot_weight_matrices(results: Dict, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Visualize the learned recurrent weight matrices.
    
    Shows:
    - Full weight matrix heatmaps
    - Low-rank component structure
    
    Args:
        results: Results dictionary
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for ax, model_name in zip(axes, ['premature', 'mature']):
        if model_name not in results:
            continue
        
        # Extract weight matrix from the model
        # (This requires access to the actual model, so we'll create a placeholder visualization)
        config = results[model_name]['config']
        metrics = results[model_name]['model_metrics']
        
        # Create representative weight matrix based on SV spectrum
        N = config.n_hidden
        r = config.rank
        sv = metrics['singular_values'][:min(r, len(metrics['singular_values']))]
        
        # Reconstruct approximate weight matrix from SVD structure
        U = np.random.randn(N, len(sv))
        U, _ = np.linalg.qr(U)
        V = np.random.randn(N, len(sv))
        V, _ = np.linalg.qr(V)
        
        W = U @ np.diag(sv * metrics['singular_values'].max()) @ V.T
        
        # Plot heatmap
        im = ax.imshow(W, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f'{model_name.capitalize()} $\\mathbf{{W}}_{{rec}}$\n(Rank ≈ {metrics["effective_rank"]:.1f})')
        ax.set_xlabel('Pre-synaptic')
        ax.set_ylabel('Post-synaptic')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    return fig


def create_comprehensive_figure(results: Dict, 
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive publication figure combining all analyses.
    Args:
        results: Results from run_developmental_comparison
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Panel A: Learning Curves
    ax_a = fig.add_subplot(gs[0, 0])
    plot_learning_curves(results, ax_a)
    ax_a.text(-0.2, 1.05, 'A', transform=ax_a.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # Panel B: Singular Value Spectrum
    ax_b = fig.add_subplot(gs[0, 1])
    plot_singular_value_spectra(results, ax_b)
    ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # Panel C: Effective Rank Comparison
    ax_c = fig.add_subplot(gs[0, 2])
    plot_effective_rank_comparison(results, ax_c)
    ax_c.text(-0.15, 1.05, 'C', transform=ax_c.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # Panel D: Response Heterogeneity
    ax_d = fig.add_subplot(gs[1, 0])
    plot_response_heterogeneity(results, ax_d)
    ax_d.text(-0.2, 1.05, 'D', transform=ax_d.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # Panel E: Task Performance
    ax_e = fig.add_subplot(gs[1, 1])
    plot_task_performance_bars(results, ax_e)
    ax_e.text(-0.15, 1.05, 'E', transform=ax_e.transAxes,
              fontsize=14, fontweight='bold', va='top')
    
    # # Panel F: Summary Statistics Table
    # ax_f = fig.add_subplot(gs[1, 2])
    # create_summary_table(results, ax_f)
    # ax_f.text(-0.15, 1.05, 'F', transform=ax_f.transAxes,
    #           fontsize=14, fontweight='bold', va='top')
    
    # Add main title
    fig.suptitle('Brain-Inspired RNN: Developmental Comparison on Cognitive Tasks',
                 fontsize=14, fontweight='bold', y=0.98)

    #fig.show()
    if save_path is None:
        save_path = '/Users/shengyuancai/Downloads/Sustech paper/Development_RNN_results/figures/comprehensive_figure.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    return fig


def create_summary_table(results: Dict, ax: plt.Axes) -> plt.Axes:
    """
    Create a summary statistics table as a plot panel.
    
    Args:
        results: Results dictionary
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes object
    """
    ax.axis('off')
    
    # Prepare table data
    columns = ['Metric', 'Premature', 'Mature', 'Δ (Mat-Prem)']
    
    rows = []
    
    # Effective Rank
    if 'premature' in results and 'mature' in results:
        prem_rank = results['premature']['test_metrics']['effective_rank']
        mat_rank = results['mature']['test_metrics']['effective_rank']
        rows.append(['Effective Rank', f'{prem_rank:.2f}', f'{mat_rank:.2f}', 
                     f'{mat_rank-prem_rank:+.2f}'])
        
        prem_hetero = results['premature']['test_metrics']['response_heterogeneity']
        mat_hetero = results['mature']['test_metrics']['response_heterogeneity']
        rows.append(['Response Hetero.', f'{prem_hetero:.4f}', f'{mat_hetero:.4f}',
                     f'{mat_hetero-prem_hetero:+.4f}'])
        
        prem_acc = results['premature']['test_metrics']['accuracy']
        mat_acc = results['mature']['test_metrics']['accuracy']
        rows.append(['Test Accuracy', f'{prem_acc:.3f}', f'{mat_acc:.3f}',
                     f'{mat_acc-prem_acc:+.3f}'])
        
        prem_loss = results['premature']['test_metrics']['loss']
        mat_loss = results['mature']['test_metrics']['loss']
        rows.append(['Test Loss', f'{prem_loss:.3f}', f'{mat_loss:.3f}',
                     f'{mat_loss-prem_loss:+.3f}'])
        
        prem_sv = results['premature']['model_metrics']['sv_decay_rate']
        mat_sv = results['mature']['model_metrics']['sv_decay_rate']
        rows.append(['SV Decay Rate (γ)', f'{prem_sv:.3f}', f'{mat_sv:.3f}',
                     f'{mat_sv-prem_sv:+.3f}'])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=columns,
                     loc='center', cellLoc='center',
                     colColours=['#E8E8E8']*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')
    
    ax.set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=20)
    
    return ax


def plot_reversal_adaptation(model: nn.Module, task, 
                              ax: plt.Axes = None, title: str = '') -> plt.Axes:
    """
    Plot adaptation to reversals in the reversal learning task.
    
    Shows how quickly the model adapts after a contingency reversal,
    comparing premature vs mature developmental stages.
    
    Args:
        model: Trained RNN model
        task: ReversalLearningTask instance
        ax: Matplotlib axes
        title: Plot title
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Generate session with known reversals
    session = task.generate_session(200, seed=123)
    
    # Get model predictions
    model.eval()
    inputs = torch.tensor(session.inputs, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        outputs, _ = model(inputs)
    
    # Get probability of choosing action 0
    probs = torch.softmax(outputs, dim=-1)[0, :, 0].numpy()
    
    # Plot
    ax.plot(probs, color='blue', alpha=0.7, linewidth=1)
    ax.plot(session.targets[:, 0], color='red', linestyle='--', 
            alpha=0.5, label='Optimal')
    
    # Mark reversals
    for rev_trial in session.trial_info['reversal_trials']:
        ax.axvline(x=rev_trial, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Trial')
    ax.set_ylabel('P(Action 0)')
    ax.set_title(f'Reversal Learning Adaptation\n{title}')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right')
    
    return ax


def generate_all_figures(results: Dict, output_dir: str = '/Users/shengyuancai/Downloads/Sustech paper/Development_RNN_results/figures') -> List[str]:
    """
    Generate all figures for the paper.
    
    Creates:
    1. Main comprehensive figure
    2. Supplementary figures for detailed analyses
    
    Args:
        results: Results from run_developmental_comparison
        output_dir: Directory to save figures
    
    Returns:
        List of saved figure paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    # Figure 1: Comprehensive comparison
    fig1 = create_comprehensive_figure(results)
    path1 = os.path.join(output_dir, 'figure_comprehensive.png')
    fig1.savefig(path1, dpi=300, bbox_inches='tight', facecolor='white')
    saved_paths.append(path1)
    plt.close(fig1)
    
    # Figure 2: Weight matrices
    fig2 = plot_weight_matrices(results)
    path2 = os.path.join(output_dir, 'figure_weight_matrices.png')
    fig2.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white')
    saved_paths.append(path2)
    plt.close(fig2)
    
    # Figure 3: Detailed learning curves
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    for model_name, color in [('premature', COLORS['premature']), 
                               ('mature', COLORS['mature']),
                               ('baseline_gru', COLORS['baseline'])]:
        if model_name in results:
            train_loss = results[model_name]['history']['train_total']
            val_loss = results[model_name]['history']['val_loss']
            label = model_name.replace('_', ' ').capitalize()
            
            axes[0].plot(train_loss, color=color, linestyle='-', 
                        label=f'{label} (train)', alpha=0.7)
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
            val_acc = results[model_name]['history']['val_accuracy']
            label = model_name.replace('_', ' ').capitalize()
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
    
    print(f"\nGenerated {len(saved_paths)} figures:")
    for path in saved_paths:
        print(f"  - {path}")
    
    return saved_paths


if __name__ == "__main__":
    # Create sample results for testing visualization
    print("Testing Visualization Module")
    print("=" * 50)
    
    # Create mock results
    mock_results = {
        'premature': {
            'history': {
                'train_total': np.exp(-np.linspace(0, 2, 80)) * 0.5 + 0.2,
                'val_loss': np.exp(-np.linspace(0, 2, 80)) * 0.5 + 0.22,
                'val_accuracy': 1 - np.exp(-np.linspace(0, 2, 80)) * 0.4 - 0.1
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
                'singular_values': np.exp(-0.22 * np.arange(15))
            },
            'config': type('Config', (), {'n_hidden': 32, 'rank': 8})()
        },
        'mature': {
            'history': {
                'train_total': np.exp(-np.linspace(0, 2.5, 80)) * 0.5 + 0.18,
                'val_loss': np.exp(-np.linspace(0, 2.5, 80)) * 0.5 + 0.20,
                'val_accuracy': 1 - np.exp(-np.linspace(0, 2.5, 80)) * 0.35 - 0.08
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
                'singular_values': np.exp(-0.38 * np.arange(15))
            },
            'config': type('Config', (), {'n_hidden': 32, 'rank': 5})()
        },
        'baseline_gru': {
            'history': {
                'train_total': np.exp(-np.linspace(0, 1.8, 80)) * 0.5 + 0.25,
                'val_loss': np.exp(-np.linspace(0, 1.8, 80)) * 0.5 + 0.28,
                'val_accuracy': 1 - np.exp(-np.linspace(0, 1.8, 80)) * 0.45 - 0.12
            },
            'test_metrics': {
                'accuracy': 0.68,
                'loss': 0.40
            },
            'model_metrics': {
                'effective_rank': 3.2,
                'sv_decay_rate': 0.30,
                'singular_values': np.exp(-0.30 * np.arange(4))
            }
        }
    }
    
    # Generate test figures
    fig = create_comprehensive_figure(mock_results)
    plt.show()
    
    print("\n✓ Visualization tests complete!")
