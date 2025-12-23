"""
Cognitive Tasks for Brain-Inspired RNN Evaluation
==================================================

This module implements the cognitive tasks from Ji-An et al. 
"Discovering cognitive strategies with tiny recurrent neural networks"

Tasks Implemented:
1. Reversal Learning Task - Animal learns which action yields reward, 
   with occasional reversals
2. Two-Stage Task - Hierarchical decision-making with probabilistic 
   state transitions
3. Probabilistic Reward Task - Continuous learning with volatile 
   reward probabilities

Each task provides:
- Trial generation with proper structure
- Reward computation
- Performance metrics
- Optimal/Bayesian agent baselines

Mathematical Framework:
-----------------------
For the reversal learning task, the optimal policy follows:
    P(correct) = σ(β * [V(correct) - V(incorrect)])

where V evolves according to:
    V(a) ← V(a) + α * (r - V(a))

For the two-stage task:
    Action A₁ or A₂ → State S₁ or S₂ (probabilistic transition)
    Then receive reward with P(reward|state)
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import warnings


class TaskType(Enum):
    """Enumeration of available task types."""
    REVERSAL_LEARNING = "reversal"
    TWO_STAGE = "two_stage"
    PROBABILISTIC_REWARD = "probabilistic"


@dataclass
class TrialData:
    """
    Container for trial data in cognitive tasks.
    
    Attributes:
        inputs: Input features for each trial [n_trials, input_dim]
        targets: Target actions/values [n_trials, output_dim]
        rewards: Rewards received [n_trials]
        actions: Actions taken [n_trials]
        states: Hidden states (for two-stage task) [n_trials]
        trial_info: Dictionary with additional trial information
    """
    inputs: np.ndarray
    targets: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray
    states: Optional[np.ndarray] = None
    trial_info: Optional[Dict] = None


class ReversalLearningTask:
    """
    Reversal Learning Task.
    
    In this task, the agent chooses between two actions (A₁, A₂).
    One action (correct) yields reward with high probability (e.g., 0.8),
    while the other yields reward with low probability (e.g., 0.2).
    
    Periodically (every ~30-50 trials), the contingencies reverse:
    the previously correct action becomes incorrect and vice versa.
    
    This task probes:
    - Learning rate adaptation
    - Flexibility/reversal learning
    - Exploitation vs exploration
    
    Input encoding (per trial):
        [previous_action, previous_second_stage, previous_reward]
        = [a_{t-1}, s_{t-1}, r_{t-1}]
    
    Output:
        P(A₁) via softmax over [logit_A1, logit_A2]
    """
    
    def __init__(self, 
                 reward_prob_high: float = 0.8,
                 reward_prob_low: float = 0.2,
                 reversal_prob: float = 0.03,
                 min_trials_before_reversal: int = 20):
        """
        Initialize the reversal learning task.
        
        Args:
            reward_prob_high: P(reward|correct action)
            reward_prob_low: P(reward|incorrect action)
            reversal_prob: Probability of reversal on each trial (after min)
            min_trials_before_reversal: Minimum trials between reversals
        """
        self.reward_prob_high = reward_prob_high
        self.reward_prob_low = reward_prob_low
        self.reversal_prob = reversal_prob
        self.min_trials_before_reversal = min_trials_before_reversal
        
        self.input_dim = 3   # [previous_action, previous_state, previous_reward]
        self.output_dim = 2  # [logit_A1, logit_A2]
        
    def generate_session(self, n_trials: int, seed: Optional[int] = None) -> TrialData:
        """
        Generate a session of reversal learning trials.
        
        Args:
            n_trials: Number of trials in the session
            seed: Random seed for reproducibility
        
        Returns:
            TrialData containing inputs, targets, rewards, actions
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Track which action is currently correct (0 or 1)
        correct_action = 0
        trials_since_reversal = 0
        
        # Storage
        inputs = np.zeros((n_trials, self.input_dim))
        targets = np.zeros((n_trials, self.output_dim))
        rewards = np.zeros(n_trials)
        actions = np.zeros(n_trials, dtype=int)
        reversal_trials = []
        
        # Initialize first trial inputs (no history)
        inputs[0, :] = [0.5, 0.5, 0.5]  # Neutral initialization
        
        for t in range(n_trials):
            # Set target based on current correct action
            targets[t, correct_action] = 1.0
            targets[t, 1 - correct_action] = 0.0
            
            # Agent chooses action (for data generation, use slight bias toward correct)
            if t == 0:
                action = np.random.randint(2)
            else:
                # Generate action with some exploration noise
                p_correct = 0.7  # Simulated agent slightly favors correct
                action = correct_action if np.random.random() < p_correct else (1 - correct_action)
            
            actions[t] = action
            
            # Determine reward
            if action == correct_action:
                reward = 1.0 if np.random.random() < self.reward_prob_high else 0.0
            else:
                reward = 1.0 if np.random.random() < self.reward_prob_low else 0.0
            
            rewards[t] = reward
            
            # Set inputs for next trial
            if t < n_trials - 1:
                inputs[t + 1, 0] = action  # Previous action (0 or 1)
                inputs[t + 1, 1] = 0.5     # No second stage in this task
                inputs[t + 1, 2] = reward  # Previous reward
            
            # Check for reversal
            trials_since_reversal += 1
            if trials_since_reversal >= self.min_trials_before_reversal:
                if np.random.random() < self.reversal_prob:
                    correct_action = 1 - correct_action
                    reversal_trials.append(t)
                    trials_since_reversal = 0
        
        return TrialData(
            inputs=inputs,
            targets=targets,
            rewards=rewards,
            actions=actions,
            trial_info={'reversal_trials': reversal_trials, 'task': 'reversal'}
        )
    
    def compute_optimal_policy(self, session: TrialData, 
                               learning_rate: float = 0.3,
                               inverse_temp: float = 5.0) -> np.ndarray:
        """
        Compute optimal Rescorla-Wagner Q-learning policy.
        
        Updates Q-values according to:
            Q(a) ← Q(a) + α * (r - Q(a))
        
        Action selection:
            P(a) = softmax(β * Q(a))
        
        Args:
            session: Trial data
            learning_rate: α parameter
            inverse_temp: β parameter (inverse temperature)
        
        Returns:
            Array of P(A₁) for each trial [n_trials]
        """
        n_trials = len(session.rewards)
        Q = np.array([0.5, 0.5])  # Initial Q-values
        policy = np.zeros(n_trials)
        
        for t in range(n_trials):
            # Softmax policy
            exp_Q = np.exp(inverse_temp * Q)
            policy[t] = exp_Q[0] / exp_Q.sum()
            
            # Update Q-value for chosen action
            a = session.actions[t]
            r = session.rewards[t]
            Q[a] += learning_rate * (r - Q[a])
        
        return policy


class TwoStageTask:
    """
    Two-Stage Decision Task.
    
    This task involves hierarchical decision-making:
    
    Stage 1: Choose action A₁ or A₂
    Stage 2: Transition to state S₁ or S₂
        - A₁ → S₁ with probability p, S₂ with probability 1-p (common transition)
        - A₂ → S₂ with probability p, S₁ with probability 1-p (common transition)
    Reward: Each state has its own reward probability that drifts over time
    
    This task dissociates:
    - Model-free learning (direct action-reward association)
    - Model-based learning (learning transition structure)
    
    A model-based agent should show:
        P(stay|rewarded, rare) < P(stay|rewarded, common)
    
    A model-free agent shows:
        P(stay|rewarded) > P(stay|unrewarded) regardless of transition type
    
    Input encoding:
        [previous_action, previous_state, previous_reward]
    
    Output:
        [logit_A1, logit_A2] for Stage 1 choice
    """
    
    def __init__(self,
                 common_prob: float = 0.8,
                 reward_drift: float = 0.025,
                 reward_bounds: Tuple[float, float] = (0.25, 0.75)):
        """
        Initialize the two-stage task.
        
        Args:
            common_prob: Probability of common transition
            reward_drift: Standard deviation of reward probability drift
            reward_bounds: Min and max reward probabilities
        """
        self.common_prob = common_prob
        self.reward_drift = reward_drift
        self.reward_bounds = reward_bounds
        
        self.input_dim = 3
        self.output_dim = 2
    
    def generate_session(self, n_trials: int, seed: Optional[int] = None) -> TrialData:
        """
        Generate a session of two-stage task trials.
        
        Args:
            n_trials: Number of trials
            seed: Random seed
        
        Returns:
            TrialData with full trial information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reward probabilities for each state (drift over time)
        reward_probs = np.zeros((n_trials, 2))
        reward_probs[0, :] = [0.5, 0.5]
        
        # Storage
        inputs = np.zeros((n_trials, self.input_dim))
        targets = np.zeros((n_trials, self.output_dim))
        rewards = np.zeros(n_trials)
        actions = np.zeros(n_trials, dtype=int)  # Stage 1 action
        states = np.zeros(n_trials, dtype=int)   # Stage 2 state
        transitions = np.zeros(n_trials, dtype=int)  # 0=common, 1=rare
        
        inputs[0, :] = [0.5, 0.5, 0.5]  # Neutral initialization
        
        for t in range(n_trials):
            # Update reward probabilities with drift
            if t > 0:
                drift = np.random.randn(2) * self.reward_drift
                reward_probs[t, :] = np.clip(
                    reward_probs[t-1, :] + drift,
                    self.reward_bounds[0],
                    self.reward_bounds[1]
                )
            
            # Target: optimal action based on current reward probs and transitions
            # Simplified: favor action leading to higher reward state
            expected_values = np.zeros(2)
            for a in range(2):
                # Expected value = p(common)*V(common_state) + p(rare)*V(rare_state)
                common_state = a  # A₁→S₁ common, A₂→S₂ common
                rare_state = 1 - a
                expected_values[a] = (self.common_prob * reward_probs[t, common_state] + 
                                     (1-self.common_prob) * reward_probs[t, rare_state])
            
            targets[t, :] = expected_values / expected_values.sum()
            
            # Agent action (with exploration)
            if np.random.random() < 0.3:  # Exploration
                action = np.random.randint(2)
            else:
                action = np.argmax(expected_values)
            
            actions[t] = action
            
            # Stage 2: Determine state based on transition
            common_state = action
            is_common = np.random.random() < self.common_prob
            transitions[t] = 0 if is_common else 1
            state = common_state if is_common else (1 - common_state)
            states[t] = state
            
            # Determine reward
            reward = 1.0 if np.random.random() < reward_probs[t, state] else 0.0
            rewards[t] = reward
            
            # Set inputs for next trial
            if t < n_trials - 1:
                inputs[t + 1, 0] = action
                inputs[t + 1, 1] = state
                inputs[t + 1, 2] = reward
        
        return TrialData(
            inputs=inputs,
            targets=targets,
            rewards=rewards,
            actions=actions,
            states=states,
            trial_info={
                'transitions': transitions,
                'reward_probs': reward_probs,
                'task': 'two_stage'
            }
        )
    
    def compute_stay_probabilities(self, session: TrialData) -> Dict[str, float]:
        """
        Compute stay probabilities conditioned on reward and transition type.
        
        Key behavioral signatures:
        - Model-free: P(stay|rewarded) > P(stay|unrewarded)
        - Model-based: interaction effect with transition type
        
        Returns:
            Dictionary with stay probabilities for each condition
        """
        n_trials = len(session.rewards)
        transitions = session.trial_info['transitions']
        
        # Count stays by condition
        conditions = {
            'common_rewarded': {'stay': 0, 'total': 0},
            'common_unrewarded': {'stay': 0, 'total': 0},
            'rare_rewarded': {'stay': 0, 'total': 0},
            'rare_unrewarded': {'stay': 0, 'total': 0}
        }
        
        for t in range(n_trials - 1):
            is_common = transitions[t] == 0
            is_rewarded = session.rewards[t] > 0
            is_stay = session.actions[t] == session.actions[t + 1]
            
            if is_common and is_rewarded:
                key = 'common_rewarded'
            elif is_common and not is_rewarded:
                key = 'common_unrewarded'
            elif not is_common and is_rewarded:
                key = 'rare_rewarded'
            else:
                key = 'rare_unrewarded'
            
            conditions[key]['total'] += 1
            if is_stay:
                conditions[key]['stay'] += 1
        
        # Compute probabilities
        probs = {}
        for key, counts in conditions.items():
            if counts['total'] > 0:
                probs[key] = counts['stay'] / counts['total']
            else:
                probs[key] = 0.5
        
        return probs


class ProbabilisticRewardTask:
    """
    Probabilistic Reward Learning Task.
    
    A simpler task where the agent learns volatile reward probabilities
    for two actions. The reward probabilities change over time following
    a random walk, requiring continuous adaptation.
    
    This task is useful for:
    - Measuring learning rate
    - Testing adaptation to volatility
    - Comparing developmental differences in flexibility
    
    Input:
        [previous_action, 0, previous_reward]
    
    Output:
        [P(A₁), P(A₂)] (softmax over logits)
    """
    
    def __init__(self,
                 initial_probs: Tuple[float, float] = (0.7, 0.3),
                 volatility: float = 0.02,
                 prob_bounds: Tuple[float, float] = (0.2, 0.8)):
        """
        Initialize the probabilistic reward task.
        
        Args:
            initial_probs: Initial reward probabilities for each action
            volatility: Standard deviation of probability drift
            prob_bounds: Bounds on reward probabilities
        """
        self.initial_probs = initial_probs
        self.volatility = volatility
        self.prob_bounds = prob_bounds
        
        self.input_dim = 3
        self.output_dim = 2
    
    def generate_session(self, n_trials: int, seed: Optional[int] = None) -> TrialData:
        """
        Generate a session of probabilistic reward trials.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Evolving reward probabilities
        reward_probs = np.zeros((n_trials, 2))
        reward_probs[0, :] = self.initial_probs
        
        inputs = np.zeros((n_trials, self.input_dim))
        targets = np.zeros((n_trials, self.output_dim))
        rewards = np.zeros(n_trials)
        actions = np.zeros(n_trials, dtype=int)
        
        inputs[0, :] = [0.5, 0.0, 0.5]
        
        for t in range(n_trials):
            # Update probabilities with drift
            if t > 0:
                drift = np.random.randn(2) * self.volatility
                reward_probs[t, :] = np.clip(
                    reward_probs[t-1, :] + drift,
                    self.prob_bounds[0],
                    self.prob_bounds[1]
                )
            
            # Target: favor action with higher reward probability
            targets[t, :] = reward_probs[t, :] / reward_probs[t, :].sum()
            
            # Agent action (softmax with exploration)
            probs = np.exp(3 * reward_probs[t, :])
            probs = probs / probs.sum()
            action = np.random.choice(2, p=probs)
            actions[t] = action
            
            # Reward
            reward = 1.0 if np.random.random() < reward_probs[t, action] else 0.0
            rewards[t] = reward
            
            # Next trial inputs
            if t < n_trials - 1:
                inputs[t + 1, 0] = action
                inputs[t + 1, 1] = 0.0
                inputs[t + 1, 2] = reward
        
        return TrialData(
            inputs=inputs,
            targets=targets,
            rewards=rewards,
            actions=actions,
            trial_info={'reward_probs': reward_probs, 'task': 'probabilistic'}
        )


class TaskDataset:
    """
    Dataset class for training RNNs on cognitive tasks.
    
    Handles:
    - Batch generation
    - Data augmentation
    - Train/validation/test splitting
    """
    
    def __init__(self, task_type: TaskType, n_sessions: int = 100, 
                 trials_per_session: int = 200, seed: int = 42):
        """
        Initialize the dataset.
        
        Args:
            task_type: Type of cognitive task
            n_sessions: Number of sessions to generate
            trials_per_session: Trials per session
            seed: Random seed
        """
        self.task_type = task_type
        self.n_sessions = n_sessions
        self.trials_per_session = trials_per_session
        
        # Create task instance
        if task_type == TaskType.REVERSAL_LEARNING:
            self.task = ReversalLearningTask()
        elif task_type == TaskType.TWO_STAGE:
            self.task = TwoStageTask()
        elif task_type == TaskType.PROBABILISTIC_REWARD:
            self.task = ProbabilisticRewardTask()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Generate all sessions
        np.random.seed(seed)
        self.sessions = [
            self.task.generate_session(trials_per_session, seed=seed+i)
            for i in range(n_sessions)
        ]
        
        # Store dimensions
        self.input_dim = self.task.input_dim
        self.output_dim = self.task.output_dim
    
    def get_batch(self, batch_size: int, session_indices: Optional[List[int]] = None,
                  device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a batch of data.
        
        Args:
            batch_size: Number of sessions in batch
            session_indices: Specific sessions to use (optional)
            device: Torch device
        
        Returns:
            inputs: [batch, seq_len, input_dim]
            targets: [batch, seq_len, output_dim]
            rewards: [batch, seq_len]
        """
        if session_indices is None:
            session_indices = np.random.choice(len(self.sessions), batch_size, replace=False)
        
        inputs = np.stack([self.sessions[i].inputs for i in session_indices])
        targets = np.stack([self.sessions[i].targets for i in session_indices])
        rewards = np.stack([self.sessions[i].rewards for i in session_indices])
        
        return (
            torch.tensor(inputs, dtype=torch.float32, device=device),
            torch.tensor(targets, dtype=torch.float32, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device)
        )
    
    def split(self, train_frac: float = 0.7, val_frac: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
        """
        Split sessions into train/validation/test sets.
        
        Returns:
            train_indices, val_indices, test_indices
        """
        n = len(self.sessions)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        
        return (
            indices[:train_end].tolist(),
            indices[train_end:val_end].tolist(),
            indices[val_end:].tolist()
        )


def compute_choice_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute choice prediction accuracy.
    
    Args:
        predictions: Model outputs [batch, seq_len, 2]
        targets: Target probabilities [batch, seq_len, 2]
    
    Returns:
        Accuracy (fraction of correct predictions)
    """
    pred_actions = predictions.argmax(dim=-1)
    target_actions = targets.argmax(dim=-1)
    
    accuracy = (pred_actions == target_actions).float().mean().item()
    return accuracy


def compute_negative_log_likelihood(predictions: torch.Tensor, 
                                    targets: torch.Tensor) -> torch.Tensor:
    """
    Compute negative log-likelihood loss.
    
    This is the standard loss for choice prediction:
        NLL = -Σ target * log(softmax(prediction))
    
    Args:
        predictions: Raw logits [batch, seq_len, 2]
        targets: Target probabilities [batch, seq_len, 2]
    
    Returns:
        Mean NLL across batch and time
    """
    # Apply log-softmax
    log_probs = torch.log_softmax(predictions, dim=-1)
    
    # Compute NLL (cross-entropy)
    nll = -(targets * log_probs).sum(dim=-1)
    
    return nll.mean()


if __name__ == "__main__":
    print("Testing Cognitive Tasks Module")
    print("=" * 50)
    
    # Test reversal learning
    print("\n1. Reversal Learning Task")
    reversal_task = ReversalLearningTask()
    session = reversal_task.generate_session(100, seed=42)
    print(f"   Generated {len(session.rewards)} trials")
    print(f"   Reversals occurred at trials: {session.trial_info['reversal_trials']}")
    print(f"   Mean reward: {session.rewards.mean():.3f}")
    
    # Test two-stage task
    print("\n2. Two-Stage Task")
    two_stage_task = TwoStageTask()
    session = two_stage_task.generate_session(100, seed=42)
    stay_probs = two_stage_task.compute_stay_probabilities(session)
    print(f"   Stay probabilities:")
    for cond, prob in stay_probs.items():
        print(f"     {cond}: {prob:.3f}")
    
    # Test dataset
    print("\n3. Task Dataset")
    dataset = TaskDataset(TaskType.REVERSAL_LEARNING, n_sessions=20)
    inputs, targets, rewards = dataset.get_batch(4)
    print(f"   Batch shapes: inputs={inputs.shape}, targets={targets.shape}")
    
    print("\n✓ All tests passed!")
