#!/usr/bin/env python3
"""
Cognitive Tasks Module for Brain-Inspired RNN Framework
========================================================

This module implements cognitive tasks used to train and evaluate
brain-inspired RNNs with developmental constraints.

Tasks Implemented
-----------------
1. Reversal Learning Task
   - Binary choice with probabilistic rewards
   - Contingency reversals test adaptation
   
2. Two-Stage Task (Daw et al., 2011)
   - Dissociates model-based vs model-free learning
   - Stage 1: choice -> transition -> Stage 2 state
   - Stage 2: probabilistic reward
   
3. Probabilistic Reward Task
   - Volatile reward probabilities
   - Tests learning rate and uncertainty tracking

Input Encoding
--------------
Following Ji-An et al. (2025), inputs encode previous trial information:
    x_t = [a_{t-1}, s_{t-1}, r_{t-1}]
    
where:
    - a_{t-1}: Previous action (0 or 1)
    - s_{t-1}: Previous state (task-specific)
    - r_{t-1}: Previous reward (0 or 1)

One-Hot Encoding
----------------
For switching models (SLIN), inputs are one-hot encoded:
    - Reversal: 4 conditions (2 actions x 2 rewards)
    - Two-stage: 8 conditions (2 actions x 2 states x 2 rewards)

Reference
---------
Ji-An et al. "Discovering cognitive strategies with tiny recurrent neural networks"
Nature (2025)

Author: Computational Neuroscience Research
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrialData:
    """
    Container for single trial data.
    
    Attributes
    ----------
    action : int
        Chosen action (0 or 1)
    state : int
        Current state
    reward : float
        Received reward
    optimal_action : int
        Optimal action for this trial
    """
    action: int
    state: int
    reward: float
    optimal_action: int


@dataclass
class SessionData:
    """
    Container for a complete session of trials.
    
    Attributes
    ----------
    inputs : np.ndarray
        Input sequences, shape [n_trials, input_dim]
    targets : np.ndarray
        Target probabilities, shape [n_trials, output_dim]
    actions : np.ndarray
        Action sequence
    rewards : np.ndarray
        Reward sequence
    states : np.ndarray
        State sequence
    trial_info : Dict
        Additional trial information (reversals, transitions, etc.)
    """
    inputs: np.ndarray
    targets: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    states: np.ndarray
    trial_info: Dict = field(default_factory=dict)


class TaskType(Enum):
    """Enumeration of available task types."""
    REVERSAL_LEARNING = "reversal_learning"
    TWO_STAGE = "two_stage"
    PROBABILISTIC_REWARD = "probabilistic_reward"


# =============================================================================
# Reversal Learning Task
# =============================================================================

class ReversalLearningTask:
    """
    Reversal learning task with probabilistic rewards.
    
    The agent chooses between two actions. One action has higher
    reward probability (e.g., 0.8 vs 0.2). Periodically, the
    contingencies reverse.
    
    Parameters
    ----------
    reward_probs : Tuple[float, float]
        Reward probabilities for (better, worse) option
    min_trials_before_reversal : int
        Minimum trials before reversal can occur
    reversal_prob : float
        Probability of reversal after minimum trials
    
    Attributes
    ----------
    input_dim : int
        Input dimension (3: action, state, reward)
    output_dim : int
        Output dimension (2: action probabilities)
    n_onehot : int
        Number of one-hot conditions (4)
    
    Examples
    --------
    >>> task = ReversalLearningTask()
    >>> session = task.generate_session(100, seed=42)
    >>> print(session.inputs.shape)  # (100, 3)
    """
    
    def __init__(
        self,
        reward_probs: Tuple[float, float] = (0.8, 0.2),
        min_trials_before_reversal: int = 20,
        reversal_prob: float = 0.03
    ):
        self.reward_probs = reward_probs
        self.min_trials_before_reversal = min_trials_before_reversal
        self.reversal_prob = reversal_prob
        
        # Dimensions
        self.input_dim = 3  # [a_{t-1}, s_{t-1}, r_{t-1}]
        self.output_dim = 2  # P(A1), P(A2)
        self.n_onehot = 4  # 2 actions x 2 rewards
    
    def generate_session(
        self,
        n_trials: int,
        seed: Optional[int] = None
    ) -> SessionData:
        """
        Generate a session of reversal learning trials.
        
        Parameters
        ----------
        n_trials : int
            Number of trials
        seed : int, optional
            Random seed
        
        Returns
        -------
        SessionData
            Complete session data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize
        inputs = np.zeros((n_trials, self.input_dim))
        targets = np.zeros((n_trials, self.output_dim))
        actions = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials)
        states = np.zeros(n_trials, dtype=int)
        
        # Track contingency
        better_action = 0
        trials_since_reversal = 0
        reversal_trials = []
        
        for t in range(n_trials):
            # Set input from previous trial
            if t > 0:
                inputs[t] = [actions[t-1], states[t-1], rewards[t-1]]
            
            # Set target (optimal policy)
            targets[t, better_action] = 1.0
            
            # Simulate action (epsilon-greedy for data generation)
            if np.random.random() < 0.1:
                actions[t] = np.random.randint(2)
            else:
                actions[t] = better_action
            
            # Generate reward
            if actions[t] == better_action:
                rewards[t] = float(np.random.random() < self.reward_probs[0])
            else:
                rewards[t] = float(np.random.random() < self.reward_probs[1])
            
            # Check for reversal
            trials_since_reversal += 1
            if trials_since_reversal >= self.min_trials_before_reversal:
                if np.random.random() < self.reversal_prob:
                    better_action = 1 - better_action
                    reversal_trials.append(t)
                    trials_since_reversal = 0
        
        return SessionData(
            inputs=inputs,
            targets=targets,
            actions=actions,
            rewards=rewards,
            states=states,
            trial_info={'reversal_trials': reversal_trials}
        )
    
    def get_onehot_inputs(self, session: SessionData) -> np.ndarray:
        """
        Convert session inputs to one-hot encoding.
        
        One-hot index = 2 * action + reward
        
        Parameters
        ----------
        session : SessionData
            Session data
        
        Returns
        -------
        np.ndarray
            One-hot encoded inputs, shape [n_trials, 4]
        """
        n_trials = len(session.actions)
        onehot = np.zeros((n_trials, self.n_onehot))
        
        for t in range(1, n_trials):
            idx = int(2 * session.actions[t-1] + session.rewards[t-1])
            onehot[t, idx] = 1.0
        
        return onehot
    
    def compute_optimal_policy(
        self,
        session: SessionData,
        learning_rate: float = 0.3
    ) -> np.ndarray:
        """
        Compute optimal policy using simple Q-learning.
        
        Parameters
        ----------
        session : SessionData
            Session data
        learning_rate : float
            Learning rate for Q-learning
        
        Returns
        -------
        np.ndarray
            Optimal action probabilities, shape [n_trials, 2]
        """
        n_trials = len(session.actions)
        Q = np.array([0.5, 0.5])
        policy = np.zeros((n_trials, 2))
        
        for t in range(n_trials):
            # Softmax policy
            exp_Q = np.exp(5 * Q)
            policy[t] = exp_Q / exp_Q.sum()
            
            # Update Q-values
            if t > 0:
                a = session.actions[t-1]
                r = session.rewards[t-1]
                Q[a] += learning_rate * (r - Q[a])
        
        return policy


# =============================================================================
# Two-Stage Task
# =============================================================================

class TwoStageTask:
    """
    Two-stage Markov decision task (Daw et al., 2011).
    
    Stage 1: Choose A1 or A2
    Transition: Common (0.7) or rare (0.3) to Stage 2 state
    Stage 2: Receive probabilistic reward
    
    This task dissociates model-based and model-free learning:
    - Model-free: Stay after reward regardless of transition
    - Model-based: Account for transition structure
    
    Parameters
    ----------
    common_prob : float
        Common transition probability
    reward_drift_sd : float
        Standard deviation of reward probability drift
    
    Attributes
    ----------
    input_dim : int
        Input dimension (3)
    output_dim : int
        Output dimension (2)
    n_onehot : int
        Number of one-hot conditions (8)
    """
    
    def __init__(
        self,
        common_prob: float = 0.7,
        reward_drift_sd: float = 0.025
    ):
        self.common_prob = common_prob
        self.reward_drift_sd = reward_drift_sd
        
        # Dimensions
        self.input_dim = 3
        self.output_dim = 2
        self.n_onehot = 8  # 2 actions x 2 states x 2 rewards
    
    def generate_session(
        self,
        n_trials: int,
        seed: Optional[int] = None
    ) -> SessionData:
        """
        Generate a session of two-stage task trials.
        
        Parameters
        ----------
        n_trials : int
            Number of trials
        seed : int, optional
            Random seed
        
        Returns
        -------
        SessionData
            Complete session data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize reward probabilities with random walk
        reward_probs = np.array([0.5, 0.5])  # For state 0 and state 1
        
        # Storage
        inputs = np.zeros((n_trials, self.input_dim))
        targets = np.zeros((n_trials, self.output_dim))
        actions = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials)
        states = np.zeros(n_trials, dtype=int)
        transitions = np.zeros(n_trials, dtype=int)  # 0=common, 1=rare
        
        for t in range(n_trials):
            # Set input from previous trial
            if t > 0:
                inputs[t] = [actions[t-1], states[t-1], rewards[t-1]]
            
            # Random action for data generation
            actions[t] = np.random.randint(2)
            
            # Transition
            is_common = np.random.random() < self.common_prob
            transitions[t] = 0 if is_common else 1
            
            if is_common:
                states[t] = actions[t]  # A1->S1, A2->S2
            else:
                states[t] = 1 - actions[t]  # A1->S2, A2->S1
            
            # Reward
            rewards[t] = float(np.random.random() < reward_probs[states[t]])
            
            # Drift reward probabilities
            reward_probs += np.random.randn(2) * self.reward_drift_sd
            reward_probs = np.clip(reward_probs, 0.25, 0.75)
            
            # Target: uniform for now (actual optimal policy is complex)
            targets[t] = [0.5, 0.5]
        
        return SessionData(
            inputs=inputs,
            targets=targets,
            actions=actions,
            rewards=rewards,
            states=states,
            trial_info={'transitions': transitions}
        )
    
    def get_onehot_inputs(self, session: SessionData) -> np.ndarray:
        """
        Convert to one-hot encoding.
        
        Index = 4*action + 2*state + reward
        
        Parameters
        ----------
        session : SessionData
            Session data
        
        Returns
        -------
        np.ndarray
            One-hot encoded inputs, shape [n_trials, 8]
        """
        n_trials = len(session.actions)
        onehot = np.zeros((n_trials, self.n_onehot))
        
        for t in range(1, n_trials):
            idx = int(
                4 * session.actions[t-1] + 
                2 * session.states[t-1] + 
                session.rewards[t-1]
            )
            onehot[t, idx] = 1.0
        
        return onehot
    
    def compute_stay_probabilities(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        transitions: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute stay probabilities for each condition.
        
        Parameters
        ----------
        actions : np.ndarray
            Action sequence
        rewards : np.ndarray
            Reward sequence
        transitions : np.ndarray
            Transition type (0=common, 1=rare)
        
        Returns
        -------
        Dict[str, float]
            Stay probabilities for each condition
        """
        conditions = {
            'common_rewarded': [],
            'common_unrewarded': [],
            'rare_rewarded': [],
            'rare_unrewarded': []
        }
        
        for t in range(len(actions) - 1):
            is_common = transitions[t] == 0
            is_rewarded = rewards[t] > 0
            is_stay = actions[t] == actions[t + 1]
            
            key = f"{'common' if is_common else 'rare'}_{'rewarded' if is_rewarded else 'unrewarded'}"
            conditions[key].append(int(is_stay))
        
        return {k: np.mean(v) if v else 0.5 for k, v in conditions.items()}
    
    def compute_mb_mf_indices(
        self,
        stay_probs: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Compute model-based and model-free indices.
        
        Parameters
        ----------
        stay_probs : Dict[str, float]
            Stay probabilities from compute_stay_probabilities
        
        Returns
        -------
        Tuple[float, float]
            (model_free_index, model_based_index)
        
        Notes
        -----
        MF index = mean reward effect = P(stay|rew) - P(stay|no_rew)
        MB index = interaction = (common_rew - common_no) - (rare_rew - rare_no)
        """
        # Model-free: main effect of reward
        mf_common = stay_probs['common_rewarded'] - stay_probs['common_unrewarded']
        mf_rare = stay_probs['rare_rewarded'] - stay_probs['rare_unrewarded']
        model_free_index = (mf_common + mf_rare) / 2
        
        # Model-based: interaction effect
        model_based_index = mf_common - mf_rare
        
        return model_free_index, model_based_index


# =============================================================================
# Probabilistic Reward Task
# =============================================================================

class ProbabilisticRewardTask:
    """
    Simple probabilistic reward task with volatile probabilities.
    
    Parameters
    ----------
    volatility : float
        Rate of reward probability change
    
    Attributes
    ----------
    input_dim : int
        Input dimension (3)
    output_dim : int
        Output dimension (2)
    """
    
    def __init__(self, volatility: float = 0.02):
        self.volatility = volatility
        self.input_dim = 3
        self.output_dim = 2
        self.n_onehot = 4
    
    def generate_session(
        self,
        n_trials: int,
        seed: Optional[int] = None
    ) -> SessionData:
        """
        Generate a session of probabilistic reward trials.
        
        Parameters
        ----------
        n_trials : int
            Number of trials
        seed : int, optional
            Random seed
        
        Returns
        -------
        SessionData
            Complete session data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize
        inputs = np.zeros((n_trials, self.input_dim))
        targets = np.zeros((n_trials, self.output_dim))
        actions = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials)
        states = np.zeros(n_trials, dtype=int)
        
        # Reward probabilities for each action
        reward_probs = np.array([0.6, 0.4])
        
        for t in range(n_trials):
            if t > 0:
                inputs[t] = [actions[t-1], states[t-1], rewards[t-1]]
            
            # Target: optimal action
            targets[t, np.argmax(reward_probs)] = 1.0
            
            # Random action
            actions[t] = np.random.randint(2)
            
            # Reward
            rewards[t] = float(np.random.random() < reward_probs[actions[t]])
            
            # Drift probabilities
            reward_probs += np.random.randn(2) * self.volatility
            reward_probs = np.clip(reward_probs, 0.2, 0.8)
        
        return SessionData(
            inputs=inputs,
            targets=targets,
            actions=actions,
            rewards=rewards,
            states=states,
            trial_info={'volatility': self.volatility}
        )


# =============================================================================
# Classical Cognitive Models
# =============================================================================

class ModelFreeRL:
    """
    Model-free reinforcement learning agent using Q-learning.
    
    Parameters
    ----------
    learning_rate : float
        Learning rate alpha
    temperature : float
        Softmax temperature beta
    
    Examples
    --------
    >>> agent = ModelFreeRL(learning_rate=0.3, temperature=5.0)
    >>> probs = agent.get_action_probabilities()
    >>> agent.update(action=0, reward=1.0)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.3,
        temperature: float = 5.0
    ):
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.Q = np.array([0.5, 0.5])
    
    def get_action_probabilities(self) -> np.ndarray:
        """Get softmax action probabilities."""
        exp_Q = np.exp(self.temperature * self.Q)
        return exp_Q / exp_Q.sum()
    
    def update(self, action: int, reward: float) -> None:
        """Update Q-values based on experience."""
        self.Q[action] += self.learning_rate * (reward - self.Q[action])
    
    def reset(self) -> None:
        """Reset Q-values."""
        self.Q = np.array([0.5, 0.5])


class BayesianAgent:
    """
    Bayesian agent that infers reward probabilities.
    
    Uses Beta distribution as conjugate prior for Bernoulli rewards.
    
    Parameters
    ----------
    prior_alpha : float
        Prior alpha parameter
    prior_beta : float
        Prior beta parameter
    """
    
    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = np.array([prior_alpha, prior_alpha])
        self.beta = np.array([prior_beta, prior_beta])
    
    def get_action_probabilities(self, temperature: float = 5.0) -> np.ndarray:
        """
        Get action probabilities based on posterior mean.
        
        Parameters
        ----------
        temperature : float
            Softmax temperature
        
        Returns
        -------
        np.ndarray
            Action probabilities
        """
        # Posterior mean of reward probability
        posterior_mean = self.alpha / (self.alpha + self.beta)
        exp_vals = np.exp(temperature * posterior_mean)
        return exp_vals / exp_vals.sum()
    
    def update(self, action: int, reward: float) -> None:
        """Update posterior based on observation."""
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
    
    def reset(self) -> None:
        """Reset to prior."""
        self.alpha = np.array([self.prior_alpha, self.prior_alpha])
        self.beta = np.array([self.prior_beta, self.prior_beta])


# =============================================================================
# Dataset Class
# =============================================================================

class TaskDataset:
    """
    Dataset class for cognitive tasks.
    
    Generates and manages multiple sessions for training.
    
    Parameters
    ----------
    task_type : TaskType
        Type of cognitive task
    n_sessions : int
        Number of sessions to generate
    trials_per_session : int
        Trials per session
    seed : int, optional
        Random seed
    
    Attributes
    ----------
    sessions : List[SessionData]
        Generated sessions
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    
    Examples
    --------
    >>> dataset = TaskDataset(TaskType.REVERSAL_LEARNING, n_sessions=50)
    >>> inputs, targets, info = dataset.get_batch(16)
    >>> train_idx, val_idx, test_idx = dataset.split()
    """
    
    def __init__(
        self,
        task_type: TaskType = TaskType.REVERSAL_LEARNING,
        n_sessions: int = 100,
        trials_per_session: int = 150,
        seed: Optional[int] = None
    ):
        self.task_type = task_type
        self.n_sessions = n_sessions
        self.trials_per_session = trials_per_session
        
        # Create task
        if task_type == TaskType.REVERSAL_LEARNING:
            self.task = ReversalLearningTask()
        elif task_type == TaskType.TWO_STAGE:
            self.task = TwoStageTask()
        elif task_type == TaskType.PROBABILISTIC_REWARD:
            self.task = ProbabilisticRewardTask()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        self.input_dim = self.task.input_dim
        self.output_dim = self.task.output_dim
        
        # Generate sessions
        if seed is not None:
            np.random.seed(seed)
        
        self.sessions = [
            self.task.generate_session(trials_per_session, seed=seed + i if seed else None)
            for i in range(n_sessions)
        ]
    
    def get_batch(
        self,
        batch_size: int,
        indices: Optional[List[int]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[SessionData]]:
        """
        Get a batch of sessions.
        
        Parameters
        ----------
        batch_size : int
            Number of sessions
        indices : List[int], optional
            Specific indices to use
        device : torch.device, optional
            Device for tensors
        
        Returns
        -------
        inputs : torch.Tensor
            Input sequences, shape [batch, seq_len, input_dim]
        targets : torch.Tensor
            Target probabilities, shape [batch, seq_len, output_dim]
        sessions : List[SessionData]
            Session data objects
        """
        if indices is None:
            indices = np.random.choice(len(self.sessions), batch_size, replace=False)
        
        sessions = [self.sessions[i] for i in indices]
        
        inputs = np.stack([s.inputs for s in sessions])
        targets = np.stack([s.targets for s in sessions])
        
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        if device is not None:
            inputs_tensor = inputs_tensor.to(device)
            targets_tensor = targets_tensor.to(device)
        
        return inputs_tensor, targets_tensor, sessions
    
    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset into train/val/test indices.
        
        Parameters
        ----------
        train_ratio : float
            Fraction for training
        val_ratio : float
            Fraction for validation
        
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            (train_indices, val_indices, test_indices)
        """
        n = len(self.sessions)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return (
            list(indices[:train_end]),
            list(indices[train_end:val_end]),
            list(indices[val_end:])
        )


# =============================================================================
# Utility Functions
# =============================================================================

def compute_choice_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute choice accuracy.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Model outputs (logits), shape [batch, seq_len, 2]
    targets : torch.Tensor
        Target probabilities, shape [batch, seq_len, 2]
    
    Returns
    -------
    float
        Accuracy (0 to 1)
    """
    pred = torch.argmax(outputs, dim=-1)
    target = torch.argmax(targets, dim=-1)
    return (pred == target).float().mean().item()


def compute_negative_log_likelihood(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute negative log-likelihood loss.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Model outputs (logits)
    targets : torch.Tensor
        Target probabilities
    
    Returns
    -------
    torch.Tensor
        NLL loss
    """
    probs = torch.softmax(outputs, dim=-1)
    # Avoid log(0)
    probs = torch.clamp(probs, min=1e-10)
    nll = -torch.sum(targets * torch.log(probs), dim=-1)
    return nll.mean()


# =============================================================================
# Main Execution (Testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Cognitive Tasks Module")
    print("=" * 60)
    
    # Test Reversal Learning
    print("\n1. Testing Reversal Learning Task...")
    task = ReversalLearningTask()
    session = task.generate_session(200, seed=42)
    print(f"   Input shape: {session.inputs.shape}")
    print(f"   Number of reversals: {len(session.trial_info['reversal_trials'])}")
    
    onehot = task.get_onehot_inputs(session)
    print(f"   One-hot shape: {onehot.shape}")
    
    # Test Two-Stage Task
    print("\n2. Testing Two-Stage Task...")
    task2 = TwoStageTask()
    session2 = task2.generate_session(200, seed=42)
    print(f"   Input shape: {session2.inputs.shape}")
    
    stay_probs = task2.compute_stay_probabilities(
        session2.actions,
        session2.rewards,
        session2.trial_info['transitions']
    )
    mf_idx, mb_idx = task2.compute_mb_mf_indices(stay_probs)
    print(f"   Model-free index: {mf_idx:.3f}")
    print(f"   Model-based index: {mb_idx:.3f}")
    
    # Test Dataset
    print("\n3. Testing TaskDataset...")
    dataset = TaskDataset(TaskType.REVERSAL_LEARNING, n_sessions=50, seed=42)
    inputs, targets, _ = dataset.get_batch(8)
    print(f"   Batch inputs shape: {inputs.shape}")
    print(f"   Batch targets shape: {targets.shape}")
    
    train_idx, val_idx, test_idx = dataset.split()
    print(f"   Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    # Test classical models
    print("\n4. Testing Classical Models...")
    mf_agent = ModelFreeRL()
    bayesian = BayesianAgent()
    
    print(f"   MF initial probs: {mf_agent.get_action_probabilities()}")
    mf_agent.update(0, 1.0)
    print(f"   MF after A0 reward: {mf_agent.get_action_probabilities()}")
    
    print("\n" + "=" * 60)
    print("All cognitive tasks tests passed!")
    print("=" * 60)
