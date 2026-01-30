#!/usr/bin/env python3
"""
================================================================================
MAIN EXECUTION SCRIPT: BRAIN-INSPIRED RNN FOR COGNITIVE TASKS
================================================================================

This script orchestrates the complete experimental pipeline for comparing
developmentally-constrained RNNs on cognitive decision-making tasks.

Theoretical Foundation
----------------------
Based on the framework connecting fMRI dynamics to low-rank RNN structure:


where the effective connectivity matrix A exhibits developmental differences:

    PREMATURE: Higher effective rank (~7-8), slower singular value decay
    MATURE: Lower effective rank (~5-6), faster decay, higher specialization

Empirical Constraints (from fMRI studies)
-----------------------------------------
- Mature infant brains: D_eff ≈ 2.2-2.4 (lower dimensionality)
- Premature infant brains: D_eff ≈ 2.3-2.5 (higher dimensionality)
- Mature EBC-FC correlation: r = 0.74 vs Premature r = 0.59

Tasks (following Ji-An et al., 2025)
------------------------------------
1. Reversal Learning: Adapt to changing reward contingencies
2. Two-Stage Task: Hierarchical decision-making (MB vs MF dissociation)
3. Probabilistic Reward: Continuous learning under volatility

Usage
-----
    python main.py --task reversal --n_epochs 100
    python main.py --all_tasks --output_dir ./results
    python main.py --task two_stage --factorization eigen

References
----------
- Ji-An et al. (2025) "Discovering cognitive strategies with tiny RNNs"
- Sussillo & Barak (2013) "Opening the Black Box"

Author: Computational Neuroscience Research
================================================================================
"""
import json
import os
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import time
from brain_inspired_rnn import (
    BrainInspiredRNN,
    TinyGRU,
    get_model_metrics,
    create_premature_config,
    create_mature_config,
    FactorizationType,
    CellType, DevelopmentalConfig
)

from cognitive_tasks import TaskType, TaskDataset

from training import (
    TrainingConfig,
    Trainer,
    run_developmental_comparison,
    analyze_learning_dynamics  # Use training.py's version
)

from visualization import generate_all_figures

from analysis import (
    PhasePortraitGenerator,
    VectorFieldGenerator,
    DynamicsAnalyzer,
    DevelopmentalComparisonAnalyzer
)

# =============================================================================
# Dynamical Systems Analysis Integration
# =============================================================================

def run_dynamical_analysis(
    results: dict,
    dataset: TaskDataset,
    output_dir: str,
    verbose: bool = True
) -> dict:
    """
    Run comprehensive dynamical systems analysis on trained models.

    Integrates analysis.py functions:
    - Phase portrait generation
    - Developmental comparison
    - Fixed point analysis

    Parameters
    ----------
    results : dict
        Results from run_developmental_comparison containing trained models
    dataset : TaskDataset
        Dataset used for generating test data
    output_dir : str
        Directory to save analysis figures
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    import os
    import torch

    analysis_results = {}

    # Check if we have both models for comparison
    if 'premature' not in results or 'mature' not in results:
        if verbose:
            print("Skipping dynamical analysis: need both premature and mature models")
        return analysis_results

    premature_model = results['premature'].get('model')
    mature_model = results['mature'].get('model')

    if premature_model is None or mature_model is None:
        if verbose:
            print("Skipping dynamical analysis: models not available in results")
        return analysis_results

    if verbose:
        print("\n" + "-" * 70)
        print("Running Dynamical Systems Analysis")
        print("-" * 70)

    # Get test data from dataset
    _, _, test_idx = dataset.split()
    inputs, targets, trial_info = dataset.get_batch(len(test_idx), test_idx)

    # Extract actions and rewards from trial_info
    # actions = trial_info['actions'].numpy() if isinstance(trial_info['actions'], torch.Tensor) else trial_info['actions']
    # rewards = trial_info['rewards'].numpy() if isinstance(trial_info['rewards'], torch.Tensor) else trial_info['rewards']
    actions_list = [t.actions for t in trial_info]
    rewards_list = [t.rewards for t in trial_info]

    # Stack them into a single array (Shape: [Batch_Size, Time_Steps])
    actions = np.vstack(actions_list)
    rewards = np.vstack(rewards_list)
    try:
        # 1. Run developmental comparison analysis
        if verbose:
            print("  Running developmental comparison...")

        comparator = DevelopmentalComparisonAnalyzer(premature_model, mature_model)
        comparison = comparator.run_comparison(inputs, actions, rewards)

        analysis_results['developmental_comparison'] = {
            'premature_fixed_points': comparison.premature_fixed_points,
            'mature_fixed_points': comparison.mature_fixed_points,
            'premature_setpoints': comparison.premature_setpoints,
            'mature_setpoints': comparison.mature_setpoints,
            'dimensionality_premature': comparison.dimensionality_premature,
            'dimensionality_mature': comparison.dimensionality_mature,
            'specialization_premature': comparison.specialization_premature,
            'specialization_mature': comparison.specialization_mature,
        }

        # Generate comparison figure
        fig_path = os.path.join(output_dir, 'figures', 'figure_developmental_comparison.png')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        comparator.plot_comparison(comparison, save_path=fig_path)

        if verbose:
            print(f"    Saved: {fig_path}")

        # 2. Generate phase portraits for each model
        if verbose:
            print("  Generating phase portraits...")

        for model_name in ['premature', 'mature']:
            model = results[model_name]['model']
            phase_gen = PhasePortraitGenerator(model)
            phase_data = phase_gen.generate_from_data(inputs, actions, rewards)

            analysis_results[f'{model_name}_phase_portrait'] = {
                'fixed_points': phase_data.fixed_points,
                'preference_setpoints': phase_data.preference_setpoints,
            }

            # Save phase portrait figure
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            phase_gen.plot_phase_portrait(phase_data, ax=ax,
                                         title=f'{model_name.capitalize()} Phase Portrait')
            fig_path = os.path.join(output_dir, 'figures', f'figure_phase_portrait_{model_name}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            if verbose:
                print(f"    Saved: {fig_path}")

        # 3. Summary
        if verbose:
            print("\n  Analysis Summary:")
            print(f"    Premature dimensionality: {comparison.dimensionality_premature:.2f}")
            print(f"    Mature dimensionality: {comparison.dimensionality_mature:.2f}")
            print(f"    Premature specialization: {comparison.specialization_premature:.4f}")
            print(f"    Mature specialization: {comparison.specialization_mature:.4f}")

    except Exception as e:
        if verbose:
            print(f"  Warning: Dynamical analysis failed: {e}")

    return analysis_results


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Brain-Inspired RNN Cognitive Task Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task reversal --n_epochs 100
  python main.py --all_tasks --output_dir ./results
  python main.py --task two_stage --n_sessions 100 --seed 123
  python main.py --factorization eigen --cell_type dev_gru
        """
    )
    
    parser.add_argument('--task', type=str, default='reversal',
                        choices=['reversal', 'two_stage', 'probabilistic'],
                        help='Cognitive task to run (default: reversal)')
    
    parser.add_argument('--all_tasks', action='store_true',
                        help='Run experiments on all tasks')
    
    parser.add_argument('--n_sessions', type=int, default=80,
                        help='Number of training sessions (default: 80)')
    
    parser.add_argument('--trials_per_session', type=int, default=60,
                        help='Trials per session (default: 150)')
    
    parser.add_argument('--n_hidden', type=int, default=32,
                        help='Number of hidden units (default: 32)')

    parser.add_argument('--n_epochs', type=int, default=80,
                        help='Training epochs (default: 80)')
    
    parser.add_argument('--factorization', type=str, default='svd_direct',
                        choices=['svd_direct', 'eigen', 'slin', 'hybrid'],
                        help='Factorization type (default: svd_direct)')
    
    parser.add_argument('--cell_type', type=str, default='leaky_tanh',
                        choices=['leaky_tanh', 'dev_gru', 'switching_gru', 'slin'],
                        help='RNN cell type (default: leaky_tanh)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory (default: ./output)')
    
    parser.add_argument('--no_figures', action='store_true',
                        help='Skip figure generation')
    
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    
    return parser.parse_args()


def get_task_type(task_name: str) -> TaskType:
    """Convert task name string to TaskType enum."""
    mapping = {
        'reversal': TaskType.REVERSAL_LEARNING,
        'two_stage': TaskType.TWO_STAGE,
        'probabilistic': TaskType.PROBABILISTIC_REWARD
    }
    return mapping[task_name]


def get_factorization_type(name: str) -> FactorizationType:
    """Convert factorization name to enum."""
    mapping = {
        'svd_direct': FactorizationType.SVD_DIRECT,
        'eigen': FactorizationType.EIGENVALUE_CONSTRAINED,
        'slin': FactorizationType.SLIN,
        'hybrid': FactorizationType.HYBRID
    }
    return mapping[name]


def get_cell_type(name: str) -> CellType:
    """Convert cell type name to enum."""
    mapping = {
        'leaky_tanh': CellType.LEAKY_TANH,
        'dev_gru': CellType.DEV_GRU,
        'switching_gru': CellType.SWITCHING_DEV_GRU,
        'slin': CellType.SLIN_CELL
    }
    return mapping[name]


def save_results(results: dict, output_dir: str, task_name: str):
    """Save experimental results to JSON format."""
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, DevelopmentalConfig):
            return {
                'n_hidden': obj.n_hidden,
                'rank': obj.rank,
                'sv_decay_gamma': obj.sv_decay_gamma,
                'developmental_stage': obj.developmental_stage
            }
        elif isinstance(obj, nn.Module):
            return f"<{obj.__class__.__name__}>"
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    filepath = os.path.join(output_dir, f'results_{task_name}.json')
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {filepath}")


def generate_summary_report(all_results: dict, output_dir: str):
    """Generate a comprehensive markdown summary report."""
    report_path = os.path.join(output_dir, 'SUMMARY_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Brain-Inspired RNN Cognitive Task Experiments\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes experiments comparing developmentally-constrained RNNs:\n\n")
        f.write("- **Premature RNN**: Higher effective rank, slower SV decay, lower heterogeneity\n")
        f.write("- **Mature RNN**: Lower effective rank, faster SV decay, higher heterogeneity\n\n")
        
        f.write("## Mathematical Framework\n\n")
        f.write("The training objective combines task performance with developmental constraints:\n\n")
        f.write("$$\\mathcal{L}_{total} = \\mathcal{L}_{task} + \\alpha ||W||_* + \\beta D_{eff}(W) + \\mathcal{L}_{spectral} - \\lambda \\mathcal{H}(W)$$\n\n")
        
        f.write("## Key Findings\n\n")
        
        for task_name, results in all_results.items():
            f.write(f"### {task_name.replace('_', ' ').title()} Task\n\n")
            
            if 'premature' in results and 'mature' in results:
                prem = results['premature']['test_metrics']
                mat = results['mature']['test_metrics']
                
                f.write("| Metric | Premature | Mature | Δ (Mat - Prem) |\n")
                f.write("|--------|-----------|--------|----------------|\n")
                f.write(f"| Accuracy | {prem['accuracy']:.4f} | {mat['accuracy']:.4f} | {mat['accuracy']-prem['accuracy']:+.4f} |\n")
                f.write(f"| Loss | {prem['loss']:.4f} | {mat['loss']:.4f} | {mat['loss']-prem['loss']:+.4f} |\n")
                
                if 'effective_rank' in prem:
                    f.write(f"| Effective Rank | {prem['effective_rank']:.3f} | {mat['effective_rank']:.3f} | {mat['effective_rank']-prem['effective_rank']:+.3f} |\n")
                
                if 'response_heterogeneity' in prem:
                    f.write(f"| Response Heterogeneity | {prem['response_heterogeneity']:.5f} | {mat['response_heterogeneity']:.5f} | {mat['response_heterogeneity']-prem['response_heterogeneity']:+.5f} |\n")
                
                f.write("\n")
        
        f.write("## Interpretation\n\n")
        f.write("1. **Lower Dimensionality in Mature Networks**: Mature-inspired RNNs show ")
        f.write("lower effective rank, reflecting dynamics evolving on a compressed ")
        f.write("low-dimensional manifold (analogous to synaptic pruning).\n\n")
        f.write("2. **Higher Specialization**: Despite lower dimensionality, mature networks ")
        f.write("exhibit higher response heterogeneity, indicating more specialized, ")
        f.write("differentiated functional responses.\n\n")
        f.write("3. **Task Performance**: The trade-off between compression and specialization ")
        f.write("affects cognitive task performance.\n\n")
        
        f.write("## References\n\n")
        f.write("- Ji-An et al. (2025) \"Discovering cognitive strategies with tiny RNNs\"\n")
        f.write("- Sussillo & Barak (2013) \"Opening the Black Box\"\n")
    
    print(f"Summary report saved to: {report_path}")


def print_banner():
    """Print experiment banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              BRAIN-INSPIRED RNN COGNITIVE EXPERIMENTS                        ║
║                                                                              ║
║  Framework: Relating fMRI Dynamics to Low-Rank RNN Structure                 ║
║  Reference: Ji-An et al. "Discovering cognitive strategies with tiny RNNs"  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main execution function."""
    args = parse_args()
    
    print_banner()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Set global seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print configuration
    print("\nExperiment Configuration:")
    print(f"  - Sessions: {args.n_sessions}")
    print(f"  - Trials/session: {args.trials_per_session}")
    print(f"  - Hidden units: {args.n_hidden}")
    print(f"  - Epochs: {args.n_epochs}")
    print(f"  - Factorization: {args.factorization}")
    print(f"  - Cell type: {args.cell_type}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Output: {args.output_dir}")
    
    # Determine which tasks to run
    if args.all_tasks:
        tasks = [TaskType.REVERSAL_LEARNING, TaskType.TWO_STAGE, TaskType.PROBABILISTIC_REWARD]
    else:
        tasks = [get_task_type(args.task)]
    
    # Run experiments
    all_results = {}
    start_time = time.time()
    
    for task_type in tasks:
        print(f"\n{'='*70}")
        print(f"TASK: {task_type.value.upper()}")
        print(f"{'='*70}")
        
        task_results = run_developmental_comparison(
            task_type=task_type,
            n_sessions=args.n_sessions,
            trials_per_session=args.trials_per_session,
            n_hidden=args.n_hidden,
            n_epochs=args.n_epochs,
            seed=args.seed
        )
        # Analyze learning dynamics
        dynamics = analyze_learning_dynamics(task_results)
        task_results['learning_dynamics'] = dynamics
        
        all_results[task_type.value] = task_results
        
        # Save results
        save_results(task_results, args.output_dir, task_type.value)
        
        # Generate figures
        if not args.no_figures:
            try:
                fig_paths = generate_all_figures(task_results, figures_dir)
                print(f"Figures generated: {len(fig_paths)}")
            except Exception as e:
                print(f"Warning: Figure generation failed: {e}")

            # Run dynamical systems analysis (using analysis.py functions)
            try:
                # Create a dataset for analysis
                analysis_dataset = TaskDataset(
                    task_type,
                    n_sessions=args.n_sessions,
                    trials_per_session=args.trials_per_session,
                    seed=args.seed
                )
                dynamical_analysis = run_dynamical_analysis(
                    task_results,
                    analysis_dataset,
                    args.output_dir,
                    verbose=args.verbose
                )
                task_results['dynamical_analysis'] = dynamical_analysis
            except Exception as e:
                print(f"Warning: Dynamical analysis failed: {e}")
    
    # Generate summary report
    generate_summary_report(all_results, args.output_dir)
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    print(f"Figures saved to: {figures_dir}")
    
    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)
    
    for task_name, results in all_results.items():
        print(f"\n{task_name.upper()}:")
        
        if 'premature' in results and 'mature' in results:
            prem_acc = results['premature']['test_metrics']['accuracy']
            mat_acc = results['mature']['test_metrics']['accuracy']
            
            print(f"  Premature Accuracy: {prem_acc:.4f}")
            print(f"  Mature Accuracy:    {mat_acc:.4f}")
            print(f"  Difference:         {mat_acc - prem_acc:+.4f}")
            
            if 'effective_rank' in results['premature']['test_metrics']:
                prem_rank = results['premature']['test_metrics']['effective_rank']
                mat_rank = results['mature']['test_metrics']['effective_rank']
                print(f"  Effective Rank (Prem→Mat): {prem_rank:.2f} → {mat_rank:.2f} (Δ={mat_rank-prem_rank:+.2f})")
    
    return all_results


if __name__ == "__main__":
    results = main()
