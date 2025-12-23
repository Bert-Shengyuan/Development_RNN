"""
Detailed Analysis Module: Cognitive Mechanisms in Developmental RNNs
=====================================================================

This module provides in-depth analysis of the computational mechanisms
underlying developmental differences between premature and mature
brain-inspired RNNs.

Analyses Include:
-----------------
1. Fixed Point Analysis: Characterize attractor structure
2. Jacobian Spectrum: Local linearization stability
3. Information Flow: How information propagates through the network
4. Reversal Dynamics: Adaptation speed after contingency changes
5. Model-Based vs Model-Free Signatures: Two-stage task analysis

Mathematical Framework:
-----------------------
For a trained RNN with dynamics h_{t+1} = f(h_t, x_t), we analyze:

1. Fixed Points: h* such that h* = f(h*, 0)
2. Jacobian: J = ∂f/∂h |_{h=h*}
3. Eigenvalue spectrum of J determines local stability
4. Effective connectivity: How perturbations propagate

Reference: 
- Sussillo & Barak (2013) "Opening the Black Box"
- Mastrogiuseppe & Ostojic (2018) "Linking Connectivity, Dynamics, and Computations"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.linalg import schur
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from brain_inspired_rnn import BrainInspiredRNN, DevelopmentalConfig
from cognitive_tasks import (
    ReversalLearningTask, TwoStageTask, 
    TaskDataset, TaskType, TrialData
)


class FixedPointFinder:
    """
    Find and analyze fixed points of RNN dynamics.
    
    Fixed points h* satisfy: h* = f(h*, x=0)
    
    We use optimization to find these:
        min_h ||h - f(h, 0)||²
    
    The Jacobian at fixed points determines stability:
    - Eigenvalues |λ| < 1: stable (attractor)
    - Eigenvalues |λ| > 1: unstable (saddle/repeller)
    """
    
    def __init__(self, model: BrainInspiredRNN, n_points: int = 20, 
                 n_iters: int = 1000, lr: float = 0.1, tol: float = 1e-6):
        """
        Initialize the fixed point finder.
        
        Args:
            model: Trained RNN model
            n_points: Number of initial conditions to try
            n_iters: Optimization iterations per initial condition
            lr: Learning rate for optimization
            tol: Tolerance for considering a point fixed
        """
        self.model = model
        self.n_points = n_points
        self.n_iters = n_iters
        self.lr = lr
        self.tol = tol
        self.device = next(model.parameters()).device
        
    def _dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Single step of autonomous dynamics (no input)."""
        x = torch.zeros(h.shape[0], self.model.input_dim, device=self.device)
        h_new = self.model.recurrent(h, self.model.W_in(x))
        return h_new
    
    def find_fixed_points(self) -> Dict[str, np.ndarray]:
        """
        Find fixed points by optimization from random initial conditions.
        
        Returns:
            Dictionary containing:
            - 'fixed_points': Array of found fixed points
            - 'velocities': Residual velocities (should be ~0)
            - 'stability': Stability classification
        """
        n_hidden = self.model.config.n_hidden
        
        # Generate random initial conditions
        h_init = torch.randn(self.n_points, n_hidden, device=self.device) * 0.5
        h_init.requires_grad_(True)
        
        optimizer = torch.optim.Adam([h_init], lr=self.lr)
        
        for _ in range(self.n_iters):
            optimizer.zero_grad()
            
            # Compute dynamics
            h_next = self._dynamics(h_init)
            
            # Loss: ||h - f(h)||²
            loss = ((h_init - h_next) ** 2).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
        
        # Filter for actual fixed points
        with torch.no_grad():
            h_final = h_init.detach()
            h_next = self._dynamics(h_final)
            velocities = torch.norm(h_final - h_next, dim=1).cpu().numpy()
            
            # Keep points with small velocity
            mask = velocities < self.tol
            fixed_points = h_final[mask].cpu().numpy()
            velocities = velocities[mask]
        
        # Remove duplicates
        if len(fixed_points) > 0:
            unique_fps = self._remove_duplicates(fixed_points)
        else:
            unique_fps = np.array([])
        
        return {
            'fixed_points': unique_fps,
            'velocities': velocities,
            'n_found': len(unique_fps)
        }
    
    def _remove_duplicates(self, points: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Remove duplicate fixed points that are close together."""
        if len(points) == 0:
            return points
        
        unique = [points[0]]
        for p in points[1:]:
            is_duplicate = False
            for u in unique:
                if np.linalg.norm(p - u) < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(p)
        
        return np.array(unique)
    
    def compute_jacobian(self, h: torch.Tensor) -> np.ndarray:
        """
        Compute the Jacobian of the dynamics at a point.
        
        J_ij = ∂f_i/∂h_j
        
        Args:
            h: Hidden state [1, n_hidden]
        
        Returns:
            Jacobian matrix [n_hidden, n_hidden]
        """
        h = h.clone().requires_grad_(True)
        n_hidden = h.shape[1]
        
        jacobian = torch.zeros(n_hidden, n_hidden, device=self.device)
        
        for i in range(n_hidden):
            self.model.zero_grad()
            h_next = self._dynamics(h)
            h_next[0, i].backward(retain_graph=True)
            jacobian[i, :] = h.grad[0, :].clone()
            h.grad.zero_()
        
        return jacobian.cpu().numpy()
    
    def analyze_stability(self, fixed_point: np.ndarray) -> Dict:
        """
        Analyze the stability of a fixed point via Jacobian eigenvalues.
        
        Args:
            fixed_point: The fixed point to analyze
        
        Returns:
            Dictionary with eigenvalue analysis
        """
        h = torch.tensor(fixed_point, dtype=torch.float32, 
                        device=self.device).unsqueeze(0)
        
        J = self.compute_jacobian(h)
        eigenvalues = np.linalg.eigvals(J)
        
        # Spectral radius determines stability
        spectral_radius = np.max(np.abs(eigenvalues))
        
        # Classification
        if spectral_radius < 1:
            stability = 'stable'
        elif spectral_radius > 1:
            # Check if saddle or repeller
            n_unstable = np.sum(np.abs(eigenvalues) > 1)
            if n_unstable < len(eigenvalues):
                stability = 'saddle'
            else:
                stability = 'repeller'
        else:
            stability = 'neutral'
        
        return {
            'eigenvalues': eigenvalues,
            'spectral_radius': spectral_radius,
            'stability': stability,
            'jacobian': J
        }


class DynamicsAnalyzer:
    """
    Comprehensive analyzer for RNN dynamics and computational mechanisms.
    
    Provides methods to:
    - Characterize the dynamical landscape
    - Analyze information flow
    - Compare computational strategies
    """
    
    def __init__(self, model: BrainInspiredRNN):
        """
        Initialize the dynamics analyzer.
        
        Args:
            model: Trained BrainInspiredRNN model
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    def compute_lyapunov_exponent(self, n_steps: int = 1000, 
                                   n_trials: int = 10) -> float:
        """
        Estimate the maximal Lyapunov exponent.
        
        The Lyapunov exponent λ characterizes sensitivity to initial conditions:
        - λ > 0: chaotic dynamics
        - λ < 0: convergent dynamics (to fixed point or limit cycle)
        - λ ≈ 0: edge of chaos
        
        Args:
            n_steps: Number of time steps
            n_trials: Number of random trials to average
        
        Returns:
            Estimated maximal Lyapunov exponent
        """
        n_hidden = self.model.config.n_hidden
        lyap_estimates = []
        
        for _ in range(n_trials):
            # Random initial condition and small perturbation
            h = torch.randn(1, n_hidden, device=self.device) * 0.5
            delta = torch.randn(1, n_hidden, device=self.device) * 1e-8
            h_perturbed = h + delta
            
            log_divergence = 0
            
            for t in range(n_steps):
                # Random input
                x = torch.randn(1, self.model.input_dim, device=self.device) * 0.1
                x_transformed = self.model.W_in(x)
                
                # Evolve both trajectories
                h = self.model.recurrent(h, x_transformed)
                h_perturbed = self.model.recurrent(h_perturbed, x_transformed)
                
                # Measure divergence
                delta_new = h_perturbed - h
                delta_norm = torch.norm(delta_new).item()
                
                if delta_norm > 0:
                    log_divergence += np.log(delta_norm / 1e-8)
                    
                    # Rescale perturbation to avoid saturation
                    h_perturbed = h + delta_new / delta_norm * 1e-8
            
            lyap_estimates.append(log_divergence / n_steps)
        
        return np.mean(lyap_estimates)
    
    def analyze_representational_geometry(self, inputs: torch.Tensor, 
                                          n_components: int = 3) -> Dict:
        """
        Analyze the geometry of neural representations using PCA.
        
        Args:
            inputs: Input sequences [batch, seq_len, input_dim]
            n_components: Number of PCA components
        
        Returns:
            Dictionary with PCA results and dimensionality metrics
        """
        self.model.eval()
        with torch.no_grad():
            _, _, hidden_history = self.model(inputs, return_hidden=True)
        
        # Reshape: [batch * seq_len, n_hidden]
        hidden_flat = hidden_history.reshape(-1, self.model.config.n_hidden).cpu().numpy()
        
        # PCA
        pca = PCA(n_components=min(n_components, hidden_flat.shape[1]))
        hidden_pca = pca.fit_transform(hidden_flat)
        
        # Participation ratio (effective dimensionality)
        eigenvalues = pca.explained_variance_ratio_
        participation_ratio = 1 / np.sum(eigenvalues ** 2)
        
        return {
            'pca_coords': hidden_pca,
            'explained_variance': pca.explained_variance_ratio_,
            'participation_ratio': participation_ratio,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
    
    def measure_adaptation_speed(self, task: ReversalLearningTask, 
                                 window: int = 10) -> Dict:
        """
        Measure how quickly the model adapts after a reversal.
        
        Args:
            task: Reversal learning task instance
            window: Window size for measuring adaptation
        
        Returns:
            Dictionary with adaptation metrics
        """
        # Generate session with known reversals
        session = task.generate_session(300, seed=123)
        
        # Get model predictions
        self.model.eval()
        inputs = torch.tensor(session.inputs, dtype=torch.float32, 
                             device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            outputs, _ = self.model(inputs)
        
        # Get choice probabilities
        probs = torch.softmax(outputs, dim=-1)[0].cpu().numpy()
        
        # Analyze adaptation around each reversal
        reversal_trials = session.trial_info['reversal_trials']
        adaptations = []
        
        for rev_trial in reversal_trials:
            if rev_trial + window < len(probs) and rev_trial - window >= 0:
                # Get accuracy before and after reversal
                targets = session.targets
                
                # Pre-reversal accuracy
                pre_correct = np.argmax(probs[rev_trial-window:rev_trial], axis=1) == \
                              np.argmax(targets[rev_trial-window:rev_trial], axis=1)
                
                # Post-reversal accuracy (after adapting)
                post_correct = np.argmax(probs[rev_trial:rev_trial+window], axis=1) == \
                               np.argmax(targets[rev_trial:rev_trial+window], axis=1)
                
                adaptations.append({
                    'pre_accuracy': pre_correct.mean(),
                    'post_accuracy': post_correct.mean(),
                    'trials_to_adapt': np.argmax(post_correct) if np.any(post_correct) else window
                })
        
        if adaptations:
            mean_adaptation = {
                'mean_pre_accuracy': np.mean([a['pre_accuracy'] for a in adaptations]),
                'mean_post_accuracy': np.mean([a['post_accuracy'] for a in adaptations]),
                'mean_trials_to_adapt': np.mean([a['trials_to_adapt'] for a in adaptations])
            }
        else:
            mean_adaptation = {'mean_pre_accuracy': 0, 'mean_post_accuracy': 0, 
                              'mean_trials_to_adapt': window}
        
        return mean_adaptation
    
    def analyze_two_stage_signatures(self, task: TwoStageTask) -> Dict:
        """
        Analyze model-based vs model-free signatures in two-stage task.
        
        Key signatures:
        - Model-free: P(stay|rewarded) > P(stay|unrewarded) regardless of transition
        - Model-based: P(stay|rewarded, rare) < P(stay|rewarded, common)
        
        Args:
            task: Two-stage task instance
        
        Returns:
            Dictionary with stay probability analysis
        """
        # Generate sessions
        sessions = [task.generate_session(200, seed=i) for i in range(10)]
        
        # Aggregate stay probabilities
        all_stays = {
            'common_rewarded': [],
            'common_unrewarded': [],
            'rare_rewarded': [],
            'rare_unrewarded': []
        }
        
        self.model.eval()
        
        for session in sessions:
            inputs = torch.tensor(session.inputs, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                outputs, _ = self.model(inputs)
            
            # Get predicted actions
            pred_actions = torch.argmax(outputs, dim=-1)[0].cpu().numpy()
            transitions = session.trial_info['transitions']
            
            for t in range(len(session.rewards) - 1):
                is_common = transitions[t] == 0
                is_rewarded = session.rewards[t] > 0
                is_stay = pred_actions[t] == pred_actions[t + 1]
                
                key = f"{'common' if is_common else 'rare'}_{'rewarded' if is_rewarded else 'unrewarded'}"
                all_stays[key].append(int(is_stay))
        
        # Compute mean stay probabilities
        stay_probs = {k: np.mean(v) if v else 0.5 for k, v in all_stays.items()}
        
        # Compute indices
        # Model-free index: (rewarded - unrewarded) averaging over transition
        mf_common = stay_probs['common_rewarded'] - stay_probs['common_unrewarded']
        mf_rare = stay_probs['rare_rewarded'] - stay_probs['rare_unrewarded']
        model_free_index = (mf_common + mf_rare) / 2
        
        # Model-based index: interaction effect
        # MB agents show: P(stay|rew,rare) < P(stay|rew,common)
        model_based_index = (stay_probs['common_rewarded'] - stay_probs['rare_rewarded'])
        
        return {
            'stay_probabilities': stay_probs,
            'model_free_index': model_free_index,
            'model_based_index': model_based_index
        }


def compare_developmental_mechanisms(premature_model: BrainInspiredRNN,
                                      mature_model: BrainInspiredRNN,
                                      task_type: TaskType = TaskType.REVERSAL_LEARNING) -> Dict:
    """
    Comprehensive comparison of computational mechanisms between developmental stages.
    
    Args:
        premature_model: Trained premature RNN
        mature_model: Trained mature RNN
        task_type: Type of cognitive task for analysis
    
    Returns:
        Dictionary with comprehensive comparison results
    """
    print("\n" + "=" * 60)
    print("COMPUTATIONAL MECHANISM ANALYSIS")
    print("=" * 60)
    
    results = {'premature': {}, 'mature': {}}
    
    for model, name in [(premature_model, 'premature'), (mature_model, 'mature')]:
        print(f"\nAnalyzing {name.upper()} model...")
        analyzer = DynamicsAnalyzer(model)
        
        # 1. Fixed point analysis
        print("  - Finding fixed points...")
        fp_finder = FixedPointFinder(model, n_points=30)
        fp_results = fp_finder.find_fixed_points()
        results[name]['n_fixed_points'] = fp_results['n_found']
        
        # Analyze stability of found fixed points
        if fp_results['n_found'] > 0:
            stabilities = []
            for fp in fp_results['fixed_points'][:5]:  # Analyze first 5
                stability = fp_finder.analyze_stability(fp)
                stabilities.append(stability['stability'])
            results[name]['fixed_point_stability'] = stabilities
        
        # 2. Lyapunov exponent
        print("  - Computing Lyapunov exponent...")
        lyap = analyzer.compute_lyapunov_exponent(n_steps=500, n_trials=5)
        results[name]['lyapunov_exponent'] = lyap
        
        # 3. Representational geometry
        print("  - Analyzing representational geometry...")
        test_inputs = torch.randn(10, 100, model.input_dim, 
                                  device=next(model.parameters()).device)
        geom = analyzer.analyze_representational_geometry(test_inputs)
        results[name]['participation_ratio'] = geom['participation_ratio']
        results[name]['explained_variance'] = geom['explained_variance'][:3].tolist()
        
        # 4. Task-specific analysis
        if task_type == TaskType.REVERSAL_LEARNING:
            print("  - Analyzing reversal adaptation...")
            task = ReversalLearningTask()
            adaptation = analyzer.measure_adaptation_speed(task)
            results[name]['adaptation'] = adaptation
            
        elif task_type == TaskType.TWO_STAGE:
            print("  - Analyzing two-stage signatures...")
            task = TwoStageTask()
            signatures = analyzer.analyze_two_stage_signatures(task)
            results[name]['two_stage'] = signatures
    
    # Summary comparison
    print("\n" + "-" * 60)
    print("MECHANISM COMPARISON SUMMARY")
    print("-" * 60)
    
    print(f"\n{'Metric':<30} {'Premature':<15} {'Mature':<15} {'Interpretation'}")
    print("-" * 80)
    
    # Fixed points
    n_fp_prem = results['premature']['n_fixed_points']
    n_fp_mat = results['mature']['n_fixed_points']
    interp = "More attractors" if n_fp_prem > n_fp_mat else "Fewer attractors"
    print(f"{'Fixed Points':<30} {n_fp_prem:<15} {n_fp_mat:<15} {interp}")
    
    # Lyapunov exponent
    lyap_prem = results['premature']['lyapunov_exponent']
    lyap_mat = results['mature']['lyapunov_exponent']
    interp = "More chaotic" if lyap_prem > lyap_mat else "More stable"
    print(f"{'Lyapunov Exponent':<30} {lyap_prem:<15.4f} {lyap_mat:<15.4f} {interp}")
    
    # Participation ratio
    pr_prem = results['premature']['participation_ratio']
    pr_mat = results['mature']['participation_ratio']
    interp = "Higher dim" if pr_prem > pr_mat else "Lower dim"
    print(f"{'Participation Ratio':<30} {pr_prem:<15.2f} {pr_mat:<15.2f} {interp}")
    
    return results


def plot_mechanism_comparison(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of mechanism comparison.
    
    Args:
        results: Results from compare_developmental_mechanisms
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    colors = {'premature': '#E74C3C', 'mature': '#2ECC71'}
    
    # Panel A: Lyapunov exponents
    ax = axes[0, 0]
    lyap_vals = [results['premature']['lyapunov_exponent'], 
                 results['mature']['lyapunov_exponent']]
    bars = ax.bar(['Premature', 'Mature'], lyap_vals, 
                  color=[colors['premature'], colors['mature']],
                  edgecolor='black')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Lyapunov Exponent $\\lambda$')
    ax.set_title('A. Dynamical Stability')
    for bar, val in zip(bars, lyap_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Panel B: Participation ratio
    ax = axes[0, 1]
    pr_vals = [results['premature']['participation_ratio'],
               results['mature']['participation_ratio']]
    bars = ax.bar(['Premature', 'Mature'], pr_vals,
                  color=[colors['premature'], colors['mature']],
                  edgecolor='black')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('B. Representational Dimensionality')
    for bar, val in zip(bars, pr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Panel C: Explained variance
    ax = axes[1, 0]
    x = np.arange(1, 4)
    width = 0.35
    ev_prem = results['premature']['explained_variance']
    ev_mat = results['mature']['explained_variance']
    ax.bar(x - width/2, ev_prem, width, label='Premature', color=colors['premature'])
    ax.bar(x + width/2, ev_mat, width, label='Mature', color=colors['mature'])
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_title('C. Variance Distribution')
    ax.set_xticks(x)
    ax.legend()
    
    # Panel D: Fixed points
    ax = axes[1, 1]
    fp_vals = [results['premature']['n_fixed_points'],
               results['mature']['n_fixed_points']]
    bars = ax.bar(['Premature', 'Mature'], fp_vals,
                  color=[colors['premature'], colors['mature']],
                  edgecolor='black')
    ax.set_ylabel('Number of Fixed Points')
    ax.set_title('D. Attractor Landscape')
    for bar, val in zip(bars, fp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


if __name__ == "__main__":
    print("Testing Analysis Module")
    print("=" * 50)
    
    # Create test models
    from brain_inspired_rnn import create_premature_config, create_mature_config
    
    premature_config = create_premature_config(n_hidden=16)
    mature_config = create_mature_config(n_hidden=16)
    
    premature_model = BrainInspiredRNN(3, 2, premature_config)
    mature_model = BrainInspiredRNN(3, 2, mature_config)
    
    # Test fixed point finder
    print("\nTesting Fixed Point Finder...")
    fp_finder = FixedPointFinder(premature_model, n_points=10, n_iters=500)
    fp_results = fp_finder.find_fixed_points()
    print(f"  Found {fp_results['n_found']} fixed points")
    
    # Test dynamics analyzer
    print("\nTesting Dynamics Analyzer...")
    analyzer = DynamicsAnalyzer(premature_model)
    lyap = analyzer.compute_lyapunov_exponent(n_steps=200, n_trials=3)
    print(f"  Lyapunov exponent: {lyap:.4f}")
    
    print("\n✓ Analysis module tests complete!")
