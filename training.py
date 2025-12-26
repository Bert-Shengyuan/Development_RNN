#!/usr/bin/env python3
"""
Training and Evaluation Module for Brain-Inspired RNNs
======================================================

This module handles the training pipeline for comparing premature vs mature
brain-inspired RNNs on cognitive tasks.

Key Components
--------------
1. Training loop with developmental regularization
2. Performance comparison between developmental stages
3. Metrics computation and logging
4. Learning dynamics analysis

Training Objective
------------------
The total loss combines task performance and developmental constraints:

    L_total = L_task + alpha*||W||_* + beta*D_eff(W) + L_spectral - lambda*H(W)

where:
    - L_task: Negative log-likelihood of correct choices
    - ||W||_*: Nuclear norm (promotes low-rank)
    - D_eff(W): Effective rank penalty
    - L_spectral: Singular value decay matching
    - H(W): Response heterogeneity (maximized for mature)

Reference
---------
Ji-An et al. "Discovering cognitive strategies with tiny recurrent neural networks"
Nature (2025)

Author: Computational Neuroscience Research
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
from collections import defaultdict

from brain_inspired_rnn import (
    BrainInspiredRNN,
    TinyGRU,  # ✓ Now correctly available
    create_premature_config,
    create_mature_config,
    DevelopmentalConfig,
    get_model_metrics  # ✓ Now correctly available
)


from cognitive_tasks import (
    TaskDataset, TaskType, 
    compute_choice_accuracy, compute_negative_log_likelihood
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for the training process.
    
    Contains hyperparameters for optimization, regularization,
    early stopping, and logging.
    
    Attributes
    ----------
    learning_rate : float
        Initial learning rate for Adam optimizer
    weight_decay : float
        L2 regularization strength
    n_epochs : int
        Maximum number of training epochs
    batch_size : int
        Number of sessions per batch
    reg_warmup_epochs : int
        Epochs to gradually increase regularization strength
    patience : int
        Early stopping patience (epochs without improvement)
    min_delta : float
        Minimum improvement to reset patience counter
    log_interval : int
        Epochs between logging
    device : str
        Computation device ('cuda' or 'cpu')
    """
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 100
    batch_size: int = 16
    
    # Regularization schedule
    reg_warmup_epochs: int = 10
    
    # Early stopping
    patience: int = 500
    min_delta: float = 1e-4
    
    # Logging
    log_interval: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Trainer class for brain-inspired RNNs.
    
    Handles the complete training pipeline including:
        - Training with developmental constraints
        - Validation monitoring
        - Metrics computation
        - Model checkpointing
        - Early stopping
    
    Parameters
    ----------
    model : Union[BrainInspiredRNN, TinyGRU]
        The model to train
    dataset : TaskDataset
        Dataset with cognitive task data
    config : TrainingConfig
        Training hyperparameters
    
    Attributes
    ----------
    optimizer : Adam
        Optimizer instance
    scheduler : CosineAnnealingLR
        Learning rate scheduler
    history : Dict[str, List[float]]
        Training history
    best_val_loss : float
        Best validation loss seen
    best_model_state : Dict
        State dict of best model
    
    Examples
    --------
    >>> model = BrainInspiredRNN(3, 2, create_mature_config())
    >>> dataset = TaskDataset(TaskType.REVERSAL_LEARNING)
    >>> trainer = Trainer(model, dataset, TrainingConfig())
    >>> history = trainer.train()
    >>> test_metrics = trainer.test()
    """
    
    def __init__(
        self, 
        model: Union[BrainInspiredRNN, TinyGRU], 
        dataset: TaskDataset, 
        config: TrainingConfig
    ):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.n_epochs, 
            eta_min=1e-5
        )
        
        # Data splits
        self.train_idx, self.val_idx, self.test_idx = dataset.split()
        
        # Training history
        self.history: Dict[str, List[float]] = defaultdict(list)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state: Optional[Dict] = None
        self.patience_counter = 0
    
    def compute_loss(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss including task loss and regularization.
        
        The loss combines:
            1. Task loss: NLL of correct choice prediction
            2. Developmental regularization (for BrainInspiredRNN only)
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input sequences, shape [batch, seq_len, input_dim]
        targets : torch.Tensor
            Target probabilities, shape [batch, seq_len, output_dim]
        epoch : int
            Current epoch (for regularization warmup)
        
        Returns
        -------
        total_loss : torch.Tensor
            Combined loss tensor
        loss_components : Dict[str, float]
            Dictionary of individual loss components
        """
        # Forward pass
        outputs, h_final = self.model(inputs)
        
        # Task loss (NLL)
        task_loss = compute_negative_log_likelihood(outputs, targets)
        
        loss_components = {'task': task_loss.item()}
        total_loss = task_loss
        
        # Add developmental regularization for BrainInspiredRNN
        if isinstance(self.model, BrainInspiredRNN):
            # Warmup factor for regularization
            warmup_factor = min(1.0, epoch / max(1, self.config.reg_warmup_epochs))
            
            # Compute regularization losses
            reg_losses = self.model.compute_regularization_losses(h_final)
            
            for name, loss in reg_losses.items():
                loss_components[name] = loss.item()
                total_loss = total_loss + warmup_factor * loss
        
        loss_components['total'] = total_loss.item()
        
        return total_loss, loss_components
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        
        Returns
        -------
        Dict[str, float]
            Dictionary of mean loss components for this epoch
        """
        self.model.train()
        
        epoch_losses: Dict[str, List[float]] = defaultdict(list)
        n_batches = len(self.train_idx) // self.config.batch_size
        
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * self.config.batch_size
            end_idx = start_idx + self.config.batch_size
            batch_indices = self.train_idx[start_idx:end_idx]
            
            # Get batch data
            inputs, targets, _ = self.dataset.get_batch(
                len(batch_indices), 
                batch_indices,
                device=torch.device(self.config.device)
            )
            
            # Forward and backward
            self.optimizer.zero_grad()
            loss, loss_components = self.compute_loss(inputs, targets, epoch)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record losses
            for name, value in loss_components.items():
                epoch_losses[name].append(value)
        
        # Compute means
        mean_losses = {name: np.mean(values) for name, values in epoch_losses.items()}
        
        return mean_losses
    
    def validate(self) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Returns
        -------
        val_loss : float
            Mean validation loss
        val_accuracy : float
            Mean validation accuracy
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs, targets, _ = self.dataset.get_batch(
                len(self.val_idx), 
                self.val_idx,
                device=torch.device(self.config.device)
            )
            
            outputs, _ = self.model(inputs)
            
            val_loss = compute_negative_log_likelihood(outputs, targets).item()
            val_accuracy = compute_choice_accuracy(outputs, targets)
        
        return val_loss, val_accuracy
    
    def test(self) -> Dict[str, float]:
        """
        Evaluate on test set.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of test metrics including:
            - 'loss': Test loss (NLL)
            - 'accuracy': Test accuracy
            - 'effective_rank': Effective rank (BrainInspiredRNN only)
            - 'response_heterogeneity': Response heterogeneity (BrainInspiredRNN only)
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs, targets, _ = self.dataset.get_batch(
                len(self.test_idx),
                self.test_idx,
                device=torch.device(self.config.device)
            )

            outputs, h_final = self.model(inputs)

            test_loss = compute_negative_log_likelihood(outputs, targets).item()
            test_accuracy = compute_choice_accuracy(outputs, targets)

        # Core metrics
        metrics = {
            'loss': test_loss,
            'accuracy': test_accuracy
        }

        # Additional metrics for BrainInspiredRNN (require gradients)
        if isinstance(self.model, BrainInspiredRNN):
            metrics['effective_rank'] = self.model.compute_effective_rank()
            metrics['response_heterogeneity'] = (
                self.model.compute_jacobian_heterogeneity(h_final)
            )
        
        return metrics
    
    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Parameters
        ----------
        verbose : bool
            Whether to print progress
        
        Returns
        -------
        Dict[str, List[float]]
            Training history with keys:
            - 'train_task', 'train_total', etc.
            - 'val_loss', 'val_accuracy'
        """
        start_time = time.time()
        
        for epoch in range(self.config.n_epochs):
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_accuracy = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            for name, value in train_losses.items():
                self.history[f'train_{name}'].append(value)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    k: v.cpu().clone() 
                    for k, v in self.model.state_dict().items()
                }
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Logging
            if verbose and (epoch % self.config.log_interval == 0 or 
                           epoch == self.config.n_epochs - 1):
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_losses['total']:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.3f} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return dict(self.history)


# =============================================================================
# Main Comparison Function
# =============================================================================

def run_developmental_comparison(
    task_type: TaskType = TaskType.REVERSAL_LEARNING,
    n_sessions: int = 80,
    trials_per_session: int = 150,
    n_hidden: int = 32,
    n_epochs: int = 80,
    seed: int = 42
) -> Dict:
    """
    Run a full comparison between premature and mature brain-inspired RNNs.
    
    This is the main experiment function that:
        1. Creates datasets for the cognitive task
        2. Trains premature and mature RNNs
        3. Trains a baseline GRU for comparison
        4. Compares performance and structural metrics
    
    Parameters
    ----------
    task_type : TaskType
        Type of cognitive task
    n_sessions : int
        Number of training sessions
    trials_per_session : int
        Trials per session
    n_hidden : int
        Number of hidden units
    n_epochs : int
        Training epochs
    seed : int
        Random seed
    
    Returns
    -------
    Dict
        Dictionary with all comparison results:
        - 'premature': Results for premature RNN
        - 'mature': Results for mature RNN
        - 'baseline_gru': Results for baseline GRU
        
        Each contains 'history', 'test_metrics', 'model_metrics', 'config'
    
    Examples
    --------
    >>> results = run_developmental_comparison(
    ...     task_type=TaskType.REVERSAL_LEARNING,
    ...     n_epochs=50
    ... )
    >>> print(results['mature']['test_metrics']['accuracy'])
    """
    print("=" * 70)
    print("BRAIN-INSPIRED RNN DEVELOPMENTAL COMPARISON")
    print("=" * 70)
    print(f"\nTask: {task_type.value}")
    print(f"Sessions: {n_sessions}, Trials/session: {trials_per_session}")
    print(f"Hidden units: {n_hidden}, Epochs: {n_epochs}")
    print()
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    dataset = TaskDataset(task_type, n_sessions, trials_per_session, seed)
    input_dim = dataset.input_dim
    output_dim = dataset.output_dim
    
    # Training configuration
    train_config = TrainingConfig(
        n_epochs=n_epochs,
        batch_size=16,
        learning_rate=1e-3
    )
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. Train Premature RNN
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("Training PREMATURE Brain-Inspired RNN")
    print("-" * 70)
    
    premature_config = create_premature_config(n_hidden)
    premature_model = BrainInspiredRNN(premature_config)
    premature_trainer = Trainer(premature_model, dataset, train_config)
    premature_history = premature_trainer.train(verbose=True)
    premature_test = premature_trainer.test()
    premature_metrics = get_model_metrics(premature_model)
    
    results['premature'] = {
        'history': premature_history,
        'test_metrics': premature_test,
        'model_metrics': premature_metrics,
        'config': premature_config
    }
    
    print(f"\nPremature Test Results:")
    print(f"  Accuracy: {premature_test['accuracy']:.4f}")
    print(f"  Loss: {premature_test['loss']:.4f}")
    print(f"  Effective Rank: {premature_test['effective_rank']:.3f}")
    print(f"  Response Heterogeneity: {premature_test['response_heterogeneity']:.4f}")
    
    # -------------------------------------------------------------------------
    # 2. Train Mature RNN
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Training MATURE Brain-Inspired RNN")
    print("-" * 70)
    
    # Reset seeds for fair comparison
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = TaskDataset(task_type, n_sessions, trials_per_session, seed)
    
    mature_config = create_mature_config(n_hidden)
    mature_model = BrainInspiredRNN(input_dim, output_dim, mature_config)
    mature_trainer = Trainer(mature_model, dataset, train_config)
    mature_history = mature_trainer.train(verbose=True)
    mature_test = mature_trainer.test()
    mature_metrics = get_model_metrics(mature_model)
    
    results['mature'] = {
        'history': mature_history,
        'test_metrics': mature_test,
        'model_metrics': mature_metrics,
        'config': mature_config
    }
    
    print(f"\nMature Test Results:")
    print(f"  Accuracy: {mature_test['accuracy']:.4f}")
    print(f"  Loss: {mature_test['loss']:.4f}")
    print(f"  Effective Rank: {mature_test['effective_rank']:.3f}")
    print(f"  Response Heterogeneity: {mature_test['response_heterogeneity']:.4f}")
    
    # -------------------------------------------------------------------------
    # 3. Train Baseline GRU (Ji-An et al. style)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Training BASELINE GRU (Ji-An et al. style)")
    print("-" * 70)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = TaskDataset(task_type, n_sessions, trials_per_session, seed)
    
    # Use smaller hidden size for tiny RNN as in Ji-An et al.
    gru_hidden = 4  # Tiny RNN
    gru_model = TinyGRU(input_dim, gru_hidden, output_dim)
    gru_trainer = Trainer(gru_model, dataset, train_config)
    gru_history = gru_trainer.train(verbose=True)
    gru_test = gru_trainer.test()
    gru_metrics = get_model_metrics(gru_model)
    
    results['baseline_gru'] = {
        'history': gru_history,
        'test_metrics': gru_test,
        'model_metrics': gru_metrics
    }
    
    print(f"\nBaseline GRU Test Results:")
    print(f"  Accuracy: {gru_test['accuracy']:.4f}")
    print(f"  Loss: {gru_test['loss']:.4f}")
    
    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    header = f"{'Model':<25} {'Accuracy':<12} {'Loss':<12} {'Eff. Rank':<12} {'Heterogeneity':<15}"
    print(f"\n{header}")
    print("-" * 76)
    
    print(
        f"{'Premature RNN':<25} "
        f"{premature_test['accuracy']:<12.4f} "
        f"{premature_test['loss']:<12.4f} "
        f"{premature_test['effective_rank']:<12.3f} "
        f"{premature_test['response_heterogeneity']:<15.4f}"
    )
    
    print(
        f"{'Mature RNN':<25} "
        f"{mature_test['accuracy']:<12.4f} "
        f"{mature_test['loss']:<12.4f} "
        f"{mature_test['effective_rank']:<12.3f} "
        f"{mature_test['response_heterogeneity']:<15.4f}"
    )
    
    print(
        f"{'Baseline GRU (d=4)':<25} "
        f"{gru_test['accuracy']:<12.4f} "
        f"{gru_test['loss']:<12.4f} "
        f"{'N/A':<12} "
        f"{'N/A':<15}"
    )
    
    # -------------------------------------------------------------------------
    # Key findings
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY DEVELOPMENTAL DIFFERENCES")
    print("=" * 70)
    
    eff_rank_diff = premature_test['effective_rank'] - mature_test['effective_rank']
    hetero_diff = mature_test['response_heterogeneity'] - premature_test['response_heterogeneity']
    acc_diff = mature_test['accuracy'] - premature_test['accuracy']
    
    print(f"\n1. Effective Rank: Mature is {eff_rank_diff:.3f} LOWER than Premature")
    print("   -> Mature brain dynamics are more compressed (lower dimensionality)")
    
    print(f"\n2. Response Heterogeneity: Mature is {hetero_diff:.4f} HIGHER than Premature")
    print("   -> Mature brain shows more specialized, differentiated responses")
    
    print(f"\n3. Task Accuracy: Mature is {acc_diff:+.4f} compared to Premature")
    if acc_diff > 0.01:
        print("   -> Lower dimensionality with higher specialization improves performance")
    elif acc_diff < -0.01:
        print("   -> Over-compression may limit flexibility in this task")
    else:
        print("   -> Similar task performance with different computational strategies")
    
    return results


def analyze_learning_dynamics(results: Dict, save_data: bool = False) -> Dict:
    """
    Analyze the learning dynamics of trained models.
    
    Computes metrics including:
        - Learning curve statistics
        - Convergence rates
        - Stability metrics
    
    Parameters
    ----------
    results : Dict
        Results from run_developmental_comparison
    save_data : bool
        Whether to save analysis data (not implemented)
    
    Returns
    -------
    Dict
        Dictionary of learning dynamics analysis for each model
    """
    analysis = {}
    
    for model_name in ['premature', 'mature', 'baseline_gru']:
        if model_name not in results:
            continue
            
        history = results[model_name]['history']
        
        # Learning curve analysis
        train_loss = np.array(history['train_total'])
        val_loss = np.array(history['val_loss'])
        val_acc = np.array(history['val_accuracy'])
        
        model_analysis = {
            'final_train_loss': train_loss[-1],
            'final_val_loss': val_loss[-1],
            'final_accuracy': val_acc[-1],
            'best_accuracy': np.max(val_acc),
            'epochs_to_best': int(np.argmax(val_acc)),
            'train_loss_std': np.std(train_loss[-10:]),  # Stability
            'val_loss_std': np.std(val_loss[-10:]),
            'convergence_rate': (train_loss[0] - train_loss[-1]) / len(train_loss)
        }
        
        analysis[model_name] = model_analysis
    
    return analysis


def save_model_checkpoint(
    model: nn.Module, 
    filepath: str,
    config: Optional[DevelopmentalConfig] = None,
    metrics: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint with configuration and metrics.
    
    Parameters
    ----------
    model : nn.Module
        Model to save
    filepath : str
        Path for checkpoint file
    config : DevelopmentalConfig, optional
        Model configuration
    metrics : Dict, optional
        Training metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    if config is not None:
        checkpoint['config'] = {
            'n_hidden': config.n_hidden,
            'rank': config.rank,
            'sv_decay_gamma': config.sv_decay_gamma,
            'developmental_stage': config.developmental_stage
        }
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to: {filepath}")


def load_model_checkpoint(
    filepath: str,
    model_class: type = BrainInspiredRNN,
    input_dim: int = 3,
    output_dim: int = 2
) -> Tuple[nn.Module, Optional[Dict]]:
    """
    Load model from checkpoint.
    
    Parameters
    ----------
    filepath : str
        Path to checkpoint file
    model_class : type
        Model class to instantiate
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    
    Returns
    -------
    model : nn.Module
        Loaded model
    config_dict : Dict, optional
        Configuration dictionary if saved
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    config_dict = checkpoint.get('config', None)
    if config_dict is not None:
        config = DevelopmentalConfig(**config_dict)
        model = model_class(input_dim, output_dim, config)
    else:
        model = model_class(input_dim, output_dim, create_mature_config())
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config_dict


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Training Module")
    print("=" * 60)
    
    # Quick test with reduced parameters
    print("\nRunning quick developmental comparison test...")
    
    results = run_developmental_comparison(
        task_type=TaskType.REVERSAL_LEARNING,
        n_sessions=30,
        trials_per_session=100,
        n_hidden=16,
        n_epochs=20,
        seed=42
    )
    
    # Analyze learning dynamics
    dynamics = analyze_learning_dynamics(results)
    
    print("\n" + "=" * 60)
    print("LEARNING DYNAMICS ANALYSIS")
    print("=" * 60)
    
    for model_name, metrics in dynamics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Epochs to best: {metrics['epochs_to_best']}")
        print(f"  Best accuracy: {metrics['best_accuracy']:.4f}")
        print(f"  Convergence rate: {metrics['convergence_rate']:.5f}")
        print(f"  Final stability (val loss std): {metrics['val_loss_std']:.5f}")
    
    print("\n" + "=" * 60)
    print("Training module tests complete!")
    print("=" * 60)
