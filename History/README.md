# Brain-Inspired Low-Rank RNN Framework for Cognitive Tasks

## Theoretical Foundation

This framework implements developmentally-constrained recurrent neural networks based on the theoretical connection between fMRI dynamics and low-rank RNN structure. The key empirical observation is that **mature infant brains** exhibit:

1. **Lower effective rank** ($D_{\text{mature}} \approx 5\text{-}6$ vs $D_{\text{premature}} \approx 7\text{-}8$)
2. **Higher response heterogeneity** (more specialized, differentiated responses)
3. **Faster singular value decay** ($\gamma_{\text{mature}} > \gamma_{\text{premature}}$)

### Mathematical Framework

#### Single-Step Dynamics

The fMRI dynamics follow a linear dynamical system approximation:

$$\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \boldsymbol{\eta}_t$$

where $\mathbf{x}_t \in \mathbb{R}^n$ is the neural state and $\mathbf{A}$ is the effective connectivity matrix.

#### Low-Rank Structure

The matrix $\mathbf{A}$ admits a low-rank factorization:

$$\mathbf{A} \approx \mathbf{U}{\Sigma}\mathbf{V}^T = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

where $r \ll n$ is the effective rank.

#### RNN Implementation

The recurrent weight matrix is parameterized as:

$$\mathbf{W}_{\text{rec}} = \mathbf{m} \mathbf{n}^T + \mathbf{M}\mathbf{N}^T$$

where:
- $\mathbf{m}, \mathbf{n} \in \mathbb{R}^N$ define a rank-1 component
- $\mathbf{M} \in \mathbb{R}^{N \times r}$, $\mathbf{N} \in \mathbb{R}^{N \times r}$ define the low-rank component

The full dynamics:

$$\mathbf{h}_{t+1} = (1-\alpha)\mathbf{h}_t + \alpha \tanh\left(\mathbf{m}\mathbf{n}^T\mathbf{h}_t + \mathbf{M}\mathbf{N}^T\mathbf{h}_t + \mathbf{W}_{\text{in}}\mathbf{u}_t\right)$$

#### Dimensionality Metrics

**Effective Rank** (based on eigenvalue dispersion):

$$D(\mathbf{W}) = \frac{\left(\sum_{i=1}^{n} \lambda_i\right)^2}{\sum_{i=1}^{n} \lambda_i^2}$$

**Nuclear Norm** (promoting low-rank):

$$\|\mathbf{W}\|_* = \sum_{i=1}^{n} \sigma_i$$

#### Training Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \|\mathbf{W}_{\text{rec}}\|_* + \beta D_{\text{eff}}(\mathbf{W}_{\text{rec}}) + \mathcal{L}_{\text{spectral}} - \lambda \mathcal{L}_{\text{hetero}}$$

where:
- $\mathcal{L}_{\text{spectral}} = \sum_{i} (\sigma_i - \sigma_i^{\text{target}})^2$ with $\sigma_i^{\text{target}} \propto e^{-\gamma i}$
- $\mathcal{L}_{\text{hetero}} = \text{Var}\left(\left\|\frac{\partial \mathbf{h}}{\partial \mathbf{I}_j}\right\|_2\right)$

---

## Cognitive Tasks

Following Ji-An et al. "Discovering cognitive strategies with tiny recurrent neural networks":

### 1. Reversal Learning Task

Subjects learn action-reward contingencies that periodically reverse:
- Two actions with reward probabilities $(p_{\text{high}}, p_{\text{low}}) = (0.8, 0.2)$
- Contingencies reverse every 20-40 trials
- Tests: Adaptation speed, state-dependent learning rates

### 2. Two-Stage Task

Hierarchical decision-making with probabilistic state transitions:
- First stage: Choose action → probabilistic transition to second stage
- Common transition probability: 0.8
- Tests: Model-free vs model-based learning strategies

### 3. Probabilistic Reward Task

Continuous learning under volatility:
- Reward probabilities drift according to random walk
- Volatility blocks with sudden changes
- Tests: Adaptive learning rate, uncertainty tracking

---

## Project Structure

```
brain_inspired_rnn_cognitive/
├── __init__.py              # Package initialization
├── brain_inspired_rnn.py    # Core RNN architectures
├── cognitive_tasks.py       # Task implementations
├── training.py              # Training pipeline
├── visualization.py         # Figure generation
├── analysis.py              # Dynamical systems analysis
├── main.py                  # CLI entry point
└── README.md                # This documentation
```

---

## Installation

```bash
pip install numpy torch matplotlib scipy
```

---

## Usage

### Basic Experiment

```bash
# Run reversal learning task
python main.py --task reversal --n_epochs 100

# Run all tasks
python main.py --all_tasks --output_dir ./results

# Custom configuration
python main.py --task two_stage --n_sessions 100 --seed 42
```

### Python API

```python
from brain_inspired_rnn_cognitive import (
    create_premature_config, create_mature_config,
    BrainInspiredRNN, TaskType, TaskDataset,
    run_developmental_comparison, create_comprehensive_figure
)

# Create developmental configurations
premature_cfg = create_premature_config(n_hidden=32)
mature_cfg = create_mature_config(n_hidden=32)

# Initialize models
premature_rnn = BrainInspiredRNN(premature_cfg, n_inputs=4, n_outputs=2)
mature_rnn = BrainInspiredRNN(mature_cfg, n_inputs=4, n_outputs=2)

# Generate task data
dataset = TaskDataset(TaskType.REVERSAL, n_sessions=80, trials_per_session=150)

# Run comparison experiment
results = run_developmental_comparison(
    dataset, 
    premature_cfg, 
    mature_cfg, 
    n_epochs=100
)

# Generate figures
create_comprehensive_figure(results, "developmental_comparison.png")
```

---

## Expected Results

Based on the theoretical framework:

| Metric | Premature RNN | Mature RNN | Interpretation |
|--------|---------------|------------|----------------|
| Effective Rank | ~7-8 | ~5-6 | More compressed representations |
| SV Decay (γ) | ~0.2 | ~0.4 | Faster spectral decay |
| Response Heterogeneity | Lower | Higher | More specialized responses |
| Adaptation Speed | Slower | Faster | Better reversal detection |

### Key Predictions

1. **Mature RNNs** should exhibit faster adaptation to contingency reversals due to more specialized, low-dimensional representations

2. **In two-stage tasks**, mature architecture may favor model-based strategies (exploiting low-rank transition structure)

3. **Response heterogeneity** paradoxically increases with dimensionality reduction, indicating functional specialization

---

## Module Documentation

### brain_inspired_rnn.py

Core classes:
- `DevelopmentalConfig`: Configuration dataclass for stage-specific parameters
- `LowRankRecurrentLayer`: Factorized recurrent layer $\mathbf{W} = \mathbf{mn}^T + \mathbf{MN}^T$
- `BrainInspiredRNN`: Full model with developmental constraints
- `TinyGRU`: Baseline comparison following Ji-An et al.

Key functions:
- `create_premature_config()`: Higher rank, slower decay
- `create_mature_config()`: Lower rank, faster decay
- `compute_effective_rank()`: Eigenvalue-based dimensionality
- `compute_response_heterogeneity()`: Variance of input-output Jacobians

### cognitive_tasks.py

Task classes:
- `ReversalLearningTask`: Periodic contingency reversals
- `TwoStageTask`: Hierarchical probabilistic transitions
- `ProbabilisticRewardTask`: Volatile reward environment
- `TaskDataset`: Unified data generation and batching

### training.py

Training components:
- `TrainingConfig`: Hyperparameters and regularization settings
- `Trainer`: Training loop with developmental regularization warmup
- `run_developmental_comparison()`: Main comparison experiment
- `analyze_learning_dynamics()`: Convergence and stability analysis

### visualization.py

Figure generation:
- `create_comprehensive_figure()`: 6-panel publication figure
- `plot_singular_value_spectra()`: SV decay visualization
- `plot_effective_rank_comparison()`: Dimensionality bar chart
- `plot_response_heterogeneity()`: Functional specialization

### analysis.py

Mechanistic analysis:
- `FixedPointFinder`: Optimization-based fixed point discovery
- `DynamicsAnalyzer`: Comprehensive dynamics characterization
- `compute_jacobian()`: Local linearization for stability
- `analyze_two_stage_signatures()`: Model-free vs model-based indices

---

## References

1. Ji-An, L., Benna, M.K., & Mattar, M.G. (2025). Discovering cognitive strategies with tiny recurrent neural networks. *Nature*.

2. Mastrogiuseppe, F. & Ostojic, S. (2018). Linking connectivity, dynamics, and computations in low-rank recurrent neural networks. *Neuron*, 99, 609-623.

3. Valente, A., Pillow, J.W. & Ostojic, S. (2022). Extracting computational mechanisms from neural data using low-rank RNNs. *NeurIPS*.

---

## License

Research code for academic purposes.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{your_paper,
  title={Developmental constraints on neural computation: 
         Low-rank RNNs for cognitive tasks},
  author={Your Name},
  journal={},
  year={2025}
}
```
