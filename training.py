"""
Training and Evaluation Module for Brain-Inspired RNNs
======================================================

This module handles the training pipeline for comparing premature vs mature
brain-inspired RNNs on cognitive tasks.

Key Components:
1. Training loop with developmental regularization
2. Performance comparison between developmental stages
3. Metrics computation and logging
4. Learning dynamics analysis

Training Objective:
-------------------
The total loss combines task performance and developmental constraints:

    L_total = L_task + α||W||_* + β*EffRank(W) + L_spectral - λ*Heterogeneity

where:
- L_task: Negative log-likelihood of correct choices
- ||W||_*: Nuclear norm (promotes low-rank)
- EffRank(W): Effective rank penalty
- L_spectral: Singular value decay matching
- Heterogeneity: Response differentiation (maximized for mature)

Author: Computational Neuroscience Research
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from collections import defaultdict

from brain_inspired_rnn import (
    BrainInspiredRNN, TinyGRU,
    create_premature_config, create_mature_config,
    DevelopmentalConfig, get_model_metrics
)
from cognitive_tasks import (
    TaskDataset, TaskType, 
    compute_choice_accuracy, compute_negative_log_likelihood
)


@dataclass
class TrainingConfig:
    """
    Configuration for the training process.
    
    Contains hyperparameters for:
    - Optimization (learning rate, epochs, batch size)
    - Regularization strengths
    - Early stopping criteria
    - Logging frequency
    """
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 100
    batch_size: int = 16
    
    # Regularization schedule
    reg_warmup_epochs: int = 10  # Gradually increase regularization
    
    # Early stopping
    patience: int = 500
    min_delta: float = 1e-4
    
    # Logging
    log_interval: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """
    Trainer class for brain-inspired RNNs.
    
    Handles:
    - Training with developmental constraints
    - Validation monitoring
    - Metrics computation
    - Model checkpointing
    """
    
    def __init__(self, model: BrainInspiredRNN, dataset: TaskDataset, 
                 config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            model: BrainInspiredRNN or TinyGRU model
            dataset: TaskDataset with cognitive task data
            config: TrainingConfig with hyperparameters
        """
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
            self.optimizer, T_max=config.n_epochs, eta_min=1e-5
        )
        
        # Data splits
        self.train_idx, self.val_idx, self.test_idx = dataset.split()
        
        # Training history
        self.history = defaultdict(list)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                     epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss including task loss and regularization.
        
        The loss combines:
        1. Task loss: NLL of correct choice prediction
        2. Developmental regularization (for BrainInspiredRNN)
        
        Args:
            inputs: Input sequences [batch, seq_len, input_dim]
            targets: Target probabilities [batch, seq_len, output_dim]
            epoch: Current epoch (for regularization warmup)
        
        Returns:
            total_loss: Combined loss tensor
            loss_components: Dictionary of individual loss components
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
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of mean loss components for this epoch
        """
        self.model.train()
        
        epoch_losses = defaultdict(list)
        n_batches = len(self.train_idx) // self.config.batch_size
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * self.config.batch_size
            end_idx = start_idx + self.config.batch_size
            batch_indices = self.train_idx[start_idx:end_idx]
            
            inputs, targets, _ = self.dataset.get_batch(
                len(batch_indices), batch_indices,
                device=torch.device(self.config.device)
            )
            
            # Forward and backward
            self.optimizer.zero_grad()
            loss, loss_components = self.compute_loss(inputs, targets, epoch)
            loss.backward()
            
            # Gradient clipping
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
        
        Returns:
            val_loss: Mean validation loss
            val_accuracy: Mean validation accuracy
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs, targets, _ = self.dataset.get_batch(
                len(self.val_idx), self.val_idx,
                device=torch.device(self.config.device)
            )
            
            outputs, _ = self.model(inputs)
            
            val_loss = compute_negative_log_likelihood(outputs, targets).item()
            val_accuracy = compute_choice_accuracy(outputs, targets)
        
        return val_loss, val_accuracy
    
    def test(self) -> Dict[str, float]:
        """
        Evaluate on test set.
        
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs, targets, _ = self.dataset.get_batch(
                len(self.test_idx), self.test_idx,
                device=torch.device(self.config.device)
            )
            
            outputs, h_final = self.model(inputs)
            
            test_loss = compute_negative_log_likelihood(outputs, targets).item()
            test_accuracy = compute_choice_accuracy(outputs, targets)
            
            # Additional metrics for BrainInspiredRNN
            metrics = {
                'loss': test_loss,
                'accuracy': test_accuracy
            }
            
            if isinstance(self.model, BrainInspiredRNN):
                metrics['effective_rank'] = self.model.compute_effective_rank().item()
                metrics['response_heterogeneity'] = \
                    self.model.compute_response_heterogeneity(h_final).item()
        
        return metrics
    
    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            verbose: Whether to print progress
        
        Returns:
            Training history dictionary
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
                self.best_model_state = {k: v.cpu().clone() 
                                         for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Logging
            if verbose and (epoch % self.config.log_interval == 0 or 
                           epoch == self.config.n_epochs - 1):
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_losses['total']:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.3f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return dict(self.history)


def run_developmental_comparison(task_type: TaskType = TaskType.REVERSAL_LEARNING,
                                 n_sessions: int = 80,
                                 trials_per_session: int = 150,
                                 n_hidden: int = 32,
                                 n_epochs: int = 80,
                                 seed: int = 42) -> Dict:
    """
    Run a full comparison between premature and mature brain-inspired RNNs.
    
    This is the main experiment function that:
    1. Creates datasets for the cognitive task
    2. Trains premature and mature RNNs
    3. Trains a baseline GRU for comparison
    4. Compares performance and structural metrics
    
    Args:
        task_type: Type of cognitive task
        n_sessions: Number of training sessions
        trials_per_session: Trials per session
        n_hidden: Number of hidden units
        n_epochs: Training epochs
        seed: Random seed
    
    Returns:
        Dictionary with all comparison results
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
    
    # 1. Train Premature RNN
    print("-" * 70)
    print("Training PREMATURE Brain-Inspired RNN")
    print("-" * 70)
    
    premature_config = create_premature_config(n_hidden)
    premature_model = BrainInspiredRNN(input_dim, output_dim, premature_config)
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
    
    # 2. Train Mature RNN
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
    
    # 3. Train Baseline GRU (for comparison with Ji-An et al.)
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
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Loss':<12} {'Eff. Rank':<12} {'Heterogeneity':<15}")
    print("-" * 76)
    
    print(f"{'Premature RNN':<25} "
          f"{premature_test['accuracy']:<12.4f} "
          f"{premature_test['loss']:<12.4f} "
          f"{premature_test['effective_rank']:<12.3f} "
          f"{premature_test['response_heterogeneity']:<15.4f}")
    
    print(f"{'Mature RNN':<25} "
          f"{mature_test['accuracy']:<12.4f} "
          f"{mature_test['loss']:<12.4f} "
          f"{mature_test['effective_rank']:<12.3f} "
          f"{mature_test['response_heterogeneity']:<15.4f}")
    
    print(f"{'Baseline GRU (d=4)':<25} "
          f"{gru_test['accuracy']:<12.4f} "
          f"{gru_test['loss']:<12.4f} "
          f"{'N/A':<12} "
          f"{'N/A':<15}")
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY DEVELOPMENTAL DIFFERENCES")
    print("=" * 70)
    
    eff_rank_diff = premature_test['effective_rank'] - mature_test['effective_rank']
    hetero_diff = mature_test['response_heterogeneity'] - premature_test['response_heterogeneity']
    acc_diff = mature_test['accuracy'] - premature_test['accuracy']
    
    print(f"\n1. Effective Rank: Mature is {eff_rank_diff:.3f} LOWER than Premature")
    print(f"   → Mature brain dynamics are more compressed (lower dimensionality)")
    
    print(f"\n2. Response Heterogeneity: Mature is {hetero_diff:.4f} HIGHER than Premature")
    print(f"   → Mature brain shows more specialized, differentiated responses")
    
    print(f"\n3. Task Accuracy: Mature is {acc_diff:+.4f} compared to Premature")
    if acc_diff > 0.01:
        print(f"   → Lower dimensionality with higher specialization improves performance")
    elif acc_diff < -0.01:
        print(f"   → Over-compression may limit flexibility in this task")
    else:
        print(f"   → Similar task performance with different computational strategies")
    
    return results


def analyze_learning_dynamics(results: Dict, save_data: bool = True) -> Dict:
    """
    Analyze the learning dynamics of trained models.
    
    Computes:
    - Learning curve statistics
    - Convergence rates
    - Stability metrics
    
    Args:
        results: Results from run_developmental_comparison
        save_data: Whether to save analysis data
    
    Returns:
        Dictionary of learning dynamics analysis
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


if __name__ == "__main__":
    # Run the main comparison experiment
    results = run_developmental_comparison(
        task_type=TaskType.REVERSAL_LEARNING,
        n_sessions=80,
        trials_per_session=150,
        n_hidden=32,
        n_epochs=80,
        seed=42
    )
    
    # Analyze learning dynamics
    dynamics = analyze_learning_dynamics(results)
    
    print("\n" + "=" * 70)
    print("LEARNING DYNAMICS ANALYSIS")
    print("=" * 70)
    
    for model_name, metrics in dynamics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Epochs to best: {metrics['epochs_to_best']}")
        print(f"  Best accuracy: {metrics['best_accuracy']:.4f}")
        print(f"  Convergence rate: {metrics['convergence_rate']:.5f}")
        print(f"  Final stability (val loss std): {metrics['val_loss_std']:.5f}")
