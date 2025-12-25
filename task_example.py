from History.brain_inspired_rnn import (
    create_premature_config, create_mature_config,
    BrainInspiredRNN

)
from History.cognitive_tasks import TaskType, TaskDataset
from visualization import create_comprehensive_figure
from training import run_developmental_comparison

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