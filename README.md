# Brain-Inspired Developmental RNN Framework

A comprehensive framework for comparing developmentally-constrained recurrent neural networks (RNNs) on cognitive decision-making tasks. This project investigates how brain developmental differences—specifically between premature and mature infant brains—affect computational strategies in neural networks.

## Theoretical Foundation

This framework is grounded in the empirical observation that mature and premature infant brains exhibit fundamentally different dynamical properties:

| Property | Premature Brain | Mature Brain |
|----------|----------------|--------------|
| Effective Rank | Higher (~2.3-2.5) | Lower (~2.2-2.4) |
| SV Decay Rate (γ) | Slower (~0.2) | Faster (~0.4) |
| Response Heterogeneity | Lower | Higher |
| EBC-FC Correlation | r ≈ 0.59 | r ≈ 0.74 |

The **key paradox** we model: mature brains show *lower* dimensionality (more compression) but *higher* functional specialization. This reflects developmental processes like synaptic pruning that refine neural circuitry.

### Mathematical Framework

The RNN dynamics follow a linear dynamical system approximation:

$$\mathbf{x}_{t+1} = \mathbf{A} \mathbf{x}_t + \boldsymbol{\eta}_t$$

The training objective combines task performance with developmental constraints:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \alpha ||\mathbf{W}||_* + \beta D_{eff}(\mathbf{W}) + \mathcal{L}_{spectral} - \lambda \mathcal{H}(\mathbf{W})$$

where:
- $||\mathbf{W}||_* = \sum_i \sigma_i$ is the nuclear norm (promotes low-rank)
- $D_{eff}(\mathbf{W}) = (\sum_i \sigma_i)^2 / \sum_i \sigma_i^2$ is effective rank
- $\mathcal{L}_{spectral}$ matches target singular value decay: $\sigma_i \sim \exp(-\gamma i)$
- $\mathcal{H}(\mathbf{W}) = \text{Var}(||\partial \mathbf{h}/\partial \mathbf{I}_j||_2)$ is response heterogeneity

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/brain-inspired-rnn.git
cd brain-inspired-rnn

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.9
- NumPy
- SciPy
- Matplotlib
- Seaborn
- scikit-learn

## Quick Start

### Basic Usage

```python
from brain_inspired_rnn import create_model, FactorizationType, CellType
from cognitive_tasks import TaskType, TaskDataset
from main import run_developmental_comparison

# Run a full developmental comparison experiment
results = run_developmental_comparison(
    task_type=TaskType.REVERSAL_LEARNING,
    n_sessions=80,
    n_epochs=100,
    n_hidden=32,
    seed=42
)

# Access results
print(f"Premature accuracy: {results['premature']['test_metrics']['accuracy']:.3f}")
print(f"Mature accuracy: {results['mature']['test_metrics']['accuracy']:.3f}")
print(f"Premature effective rank: {results['premature']['test_metrics']['effective_rank']:.2f}")
print(f"Mature effective rank: {results['mature']['test_metrics']['effective_rank']:.2f}")
```

### Command Line Interface

```bash
# Run reversal learning experiment
python main.py --task reversal --n_epochs 100

# Run all tasks with eigenvalue-constrained factorization
python main.py --all_tasks --factorization eigen --output_dir ./results

# Use DevGRU cell type
python main.py --task two_stage --cell_type dev_gru --n_hidden 32
```

### Creating Individual Models

```python
from brain_inspired_rnn import (
    create_model,
    create_premature_config,
    create_mature_config,
    BrainInspiredRNN,
    FactorizationType,
    CellType
)

# Create a mature brain-inspired model with SVD factorization
mature_model = create_model(
    developmental_stage="mature",
    n_hidden=32,
    n_input=3,
    n_output=2,
    factorization_type=FactorizationType.SVD_DIRECT,
    cell_type=CellType.LEAKY_TANH
)

# Forward pass
import torch
inputs = torch.randn(16, 100, 3)  # [batch, seq_len, input_dim]
outputs, h_final = mature_model(inputs)

# Compute developmental metrics
print(f"Effective rank: {mature_model.compute_effective_rank():.2f}")
print(f"SV decay rate: {mature_model.compute_sv_decay_rate():.3f}")
print(f"Spectral radius: {mature_model.compute_spectral_radius():.3f}")
```

## Architecture Options

### Factorization Types

| Type | Description | Weight Structure |
|------|-------------|------------------|
| `SVD_DIRECT` | Direct low-rank factorization | $\mathbf{W} = \mathbf{m}\mathbf{n}^T + \mathbf{M}\mathbf{N}^T$ |
| `EIGENVALUE_CONSTRAINED` | Eigenvalue distribution control | $\mathbf{W} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$ |
| `SLIN` | Switching linear dynamics | $\mathbf{h}_t = \mathbf{W}^{(x_{t-1})}\mathbf{h}_{t-1} + \mathbf{b}^{(x_{t-1})}$ |
| `HYBRID` | Combination approach | Mixed factorization |

### Cell Types

| Type | Description | Based On |
|------|-------------|----------|
| `LEAKY_TANH` | Leaky integration with tanh | Standard RNN |
| `DEV_GRU` | Low-rank GRU gates | GRU with factorized weights |
| `SWITCHING_DEV_GRU` | Input-dependent GRU | Ji-An et al. Eq. 2 |
| `SLIN_CELL` | Pure switching linear | Ji-An et al. Eq. 4 |

## Cognitive Tasks

### Reversal Learning
Binary choice task with probabilistic rewards. Contingencies periodically reverse, testing adaptation.

```python
from cognitive_tasks import ReversalLearningTask

task = ReversalLearningTask(
    reward_probs=(0.8, 0.2),
    min_trials_before_reversal=20,
    reversal_prob=0.03
)
session = task.generate_session(200, seed=42)
```

### Two-Stage Task
Hierarchical decision-making task that dissociates model-based from model-free learning (Daw et al., 2011).

```python
from cognitive_tasks import TwoStageTask

task = TwoStageTask(
    common_prob=0.7,
    reward_probs=(0.25, 0.75)
)
session = task.generate_session(200, seed=42)
```

### Probabilistic Reward Task
Continuous learning under volatile reward probabilities.

```python
from cognitive_tasks import ProbabilisticRewardTask

task = ProbabilisticRewardTask(
    volatility=0.05,
    initial_probs=(0.7, 0.3)
)
session = task.generate_session(200, seed=42)
```

## Analysis Tools

### Phase Portrait Analysis (Figure 3 style)

Following Ji-An et al., visualize the dynamics in logit space:

```python
from analysis import PhasePortraitGenerator
from visualization import plot_phase_portrait

# Generate phase portrait data
phase_gen = PhasePortraitGenerator(model)
phase_data = phase_gen.generate_from_data(inputs, actions, rewards)

# Visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
phase_gen.plot_phase_portrait(phase_data, ax=ax, title="Mature RNN")
plt.show()
```

### Vector Field Analysis (Figure 4 style)

For d=2 models, visualize state space flow:

```python
from analysis import VectorFieldGenerator

vector_gen = VectorFieldGenerator(model)
vf_data = vector_gen.generate_vector_field(n_grid=15)
fig = vector_gen.plot_all_conditions(vf_data)
```

### Fixed Point Analysis

Identify and characterize attractors:

```python
from analysis import FixedPointFinder

fp_finder = FixedPointFinder(model, n_points=20, n_iters=1000)
results = fp_finder.find_fixed_points()

print(f"Found {results['n_found']} fixed points")
stability = fp_finder.analyze_stability(results['fixed_points'])
```

### Developmental Comparison

Comprehensive comparison of premature vs mature dynamics:

```python
from analysis import DevelopmentalComparisonAnalyzer

comparator = DevelopmentalComparisonAnalyzer(premature_model, mature_model)
comparison = comparator.run_comparison(inputs, actions, rewards)
fig = comparator.plot_comparison(comparison, save_path="comparison.png")
```

## Visualization

### Generate All Figures

```python
from visualization import generate_all_figures, create_comprehensive_figure

# Generate comprehensive figure
fig = create_comprehensive_figure(results)
fig.savefig('comprehensive.png', dpi=300)

# Generate all standard figures
paths = generate_all_figures(results, output_dir='./figures')
```

### Individual Plots

```python
from visualization import (
    plot_learning_curves,
    plot_singular_value_spectra,
    plot_effective_rank_comparison,
    plot_response_heterogeneity,
    plot_weight_matrices
)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_learning_curves(results, ax=axes[0, 0])
plot_singular_value_spectra(results, ax=axes[0, 1])
plot_effective_rank_comparison(results, ax=axes[1, 0])
plot_response_heterogeneity(results, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('analysis.png')
```

## Project Structure

```
brain-inspired-rnn/
├── brain_inspired_rnn.py    # Core RNN architectures (~1000 lines)
│   ├── DevelopmentalConfig  # Configuration dataclass
│   ├── FactorizationType    # SVD_DIRECT, EIGENVALUE_CONSTRAINED, SLIN
│   ├── CellType            # LEAKY_TANH, DEV_GRU, SWITCHING_DEV_GRU, SLIN
│   ├── SVDLowRankLayer     # W = m n^T + M N^T
│   ├── EigenConstrainedLayer # W = V Λ V^T
│   ├── SLINLayer           # Input-dependent weights
│   ├── DevGRUCell          # Low-rank GRU
│   ├── ResponseHeterogeneityAnalyzer
│   └── BrainInspiredRNN    # Main model class
│
├── cognitive_tasks.py       # Task implementations (~700 lines)
│   ├── ReversalLearningTask
│   ├── TwoStageTask
│   ├── ProbabilisticRewardTask
│   ├── ModelFreeRL, BayesianAgent
│   └── TaskDataset
│
├── analysis.py              # Dynamical analysis (~900 lines)
│   ├── FixedPointFinder
│   ├── LogitAnalyzer
│   ├── PhasePortraitGenerator
│   ├── VectorFieldGenerator
│   └── DevelopmentalComparisonAnalyzer
│
├── training.py              # Training pipeline (~800 lines)
│   ├── TrainingConfig
│   ├── Trainer
│   └── run_developmental_comparison
│
├── visualization.py         # Figure generation (~700 lines)
│   ├── plot_learning_curves
│   ├── plot_singular_value_spectra
│   ├── plot_phase_portrait
│   ├── plot_vector_field
│   └── create_comprehensive_figure
│
├── main.py                  # Orchestration script
├── __init__.py             # Package exports
├── DOCUMENTATION.py        # Mathematical documentation
└── README.md               # This file
```

## References

1. **Ji-An, L., Benna, M.K., & Mattar, M.G. (2025).** "Discovering cognitive strategies with tiny recurrent neural networks." *Nature*.

2. **Sussillo, D., & Barak, O. (2013).** "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks." *Neural Computation*, 25(3), 626-649.

3. **Mastrogiuseppe, F., & Ostojic, S. (2018).** "Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks." *Neuron*, 99(3), 609-623.

4. **Daw, N.D., Gershman, S.J., Seymour, B., Dayan, P., & Dolan, R.J. (2011).** "Model-Based Influences on Humans' Choices and Striatal Prediction Errors." *Neuron*, 69(6), 1204-1215.

## License

This project is licensed under the MIT License.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{brain_inspired_rnn,
  title = {Brain-Inspired Developmental RNN Framework},
  author = {Computational Neuroscience Research},
  year = {2025},
  url = {https://github.com/your-repo/brain-inspired-rnn}
}
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the main repository.
