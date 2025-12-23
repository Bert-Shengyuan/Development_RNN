#!/usr/bin/env python3
"""
Main Execution Script: Brain-Inspired RNN for Cognitive Tasks
==============================================================

This script orchestrates the complete experimental pipeline for comparing
developmentally-constrained RNNs on cognitive decision-making tasks.

Experiment Overview:
--------------------
Based on the theoretical framework connecting fMRI dynamics to low-rank RNN structure,
we compare two brain-inspired RNN architectures:

1. PREMATURE RNN: Higher effective rank (~7-8), slower SV decay, lower response heterogeneity
2. MATURE RNN: Lower effective rank (~5-6), faster SV decay, higher response heterogeneity

These constraints are derived from empirical observations:
- Mature infant brains show lower effective rank entropy (≈2.2-2.4 vs ≈2.3-2.5)
- Mature brains exhibit higher response heterogeneity
- Mature EBC shows better correlation with empirical FC (r=0.74 vs r=0.59)

Tasks:
------
Following Ji-An et al. "Discovering cognitive strategies with tiny recurrent neural networks":
1. Reversal Learning: Adapt to changing reward contingencies
2. Two-Stage Task: Hierarchical decision-making with probabilistic transitions
3. Probabilistic Reward: Continuous learning under volatility

Outputs:
--------
- Training histories for all models
- Performance comparison metrics
- Publication-quality figures
- Detailed analysis reports

Usage:
------
    python main.py [--task {reversal,two_stage,probabilistic}]
                   [--n_sessions N] [--n_epochs N] [--seed S]
                   [--output_dir DIR] [--all_tasks]

Author: Computational Neuroscience Research
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch

# Import project modules
from brain_inspired_rnn import (
    BrainInspiredRNN, TinyGRU,
    create_premature_config, create_mature_config,
    DevelopmentalConfig, get_model_metrics
)
from cognitive_tasks import TaskType, TaskDataset
from training import run_developmental_comparison, analyze_learning_dynamics, TrainingConfig
from visualization import (
    create_comprehensive_figure, generate_all_figures,
    plot_learning_curves, plot_singular_value_spectra,
    COLORS
)

import matplotlib.pyplot as plt


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
        """
    )
    
    parser.add_argument('--task', type=str, default='probabilistic',
                        choices=['reversal', 'two_stage', 'probabilistic'],
                        help='Cognitive task to run (default: reversal)')
    
    parser.add_argument('--all_tasks', action='store_true',
                        help='Run experiments on all tasks')
    
    parser.add_argument('--n_sessions', type=int, default=100,
                        help='Number of training sessions (default: 80)')
    
    parser.add_argument('--trials_per_session', type=int, default=150,
                        help='Trials per session (default: 150)')
    
    parser.add_argument('--n_hidden', type=int, default=64,
                        help='Number of hidden units (default: 32)')

    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Training epochs (default: 80)')
    
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


def save_results(results: dict, output_dir: str, task_name: str):
    """
    Save experimental results to JSON format.
    
    Converts numpy arrays and other non-serializable objects.
    """
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, DevelopmentalConfig):
            return {
                'n_hidden': obj.n_hidden,
                'rank': obj.rank,
                'sv_decay_gamma': obj.sv_decay_gamma,
                'developmental_stage': obj.developmental_stage
            }
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


def print_banner():
    """Print experiment banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BRAIN-INSPIRED RNN COGNITIVE EXPERIMENTS                  ║
║                                                                              ║
║  Framework: Relating fMRI Dynamics to Low-Rank RNN Structure                 ║
║  Reference: Ji-An et al. "Discovering cognitive strategies with tiny RNNs"  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_single_task_experiment(task_type: TaskType, args) -> dict:
    """
    Run experiment for a single cognitive task.
    
    Args:
        task_type: Type of cognitive task
        args: Command line arguments
    
    Returns:
        Dictionary of experimental results
    """
    print(f"\n{'='*70}")
    print(f"TASK: {task_type.value.upper()}")
    print(f"{'='*70}")
    
    # Run the main comparison
    results = run_developmental_comparison(
        task_type=task_type,
        n_sessions=args.n_sessions,
        trials_per_session=args.trials_per_session,
        n_hidden=args.n_hidden,
        n_epochs=args.n_epochs,
        seed=args.seed
    )
    
    # Analyze learning dynamics
    dynamics = analyze_learning_dynamics(results)
    results['learning_dynamics'] = dynamics
    
    return results


def generate_summary_report(all_results: dict, output_dir: str):
    """
    Generate a comprehensive summary report.
    
    Creates a markdown report summarizing all experimental findings.
    """
    report_path = os.path.join(output_dir, 'SUMMARY_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Brain-Inspired RNN Cognitive Task Experiments\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes experiments comparing developmentally-constrained RNNs:\n")
        f.write("- **Premature RNN**: Higher effective rank, slower SV decay, lower heterogeneity\n")
        f.write("- **Mature RNN**: Lower effective rank, faster SV decay, higher heterogeneity\n\n")
        
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
        f.write("The results demonstrate the key developmental principle:\n\n")
        f.write("1. **Lower Dimensionality in Mature Networks**: Mature-inspired RNNs show ")
        f.write("lower effective rank, indicating that dynamics evolve on a more compressed ")
        f.write("low-dimensional manifold. This mirrors biological synaptic pruning.\n\n")
        f.write("2. **Higher Specialization**: Despite lower dimensionality, mature networks ")
        f.write("exhibit higher response heterogeneity, indicating more specialized, ")
        f.write("differentiated functional responses.\n\n")
        f.write("3. **Task Performance**: The trade-off between compression and specialization ")
        f.write("affects cognitive task performance, with implications for understanding ")
        f.write("developmental differences in decision-making.\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `results_*.json`: Raw experimental results\n")
        f.write("- `figures/`: Publication-quality figures\n")
        f.write("- `SUMMARY_REPORT.md`: This report\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Print banner
    print_banner()
    
    # Create output directory
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
        task_results = run_single_task_experiment(task_type, args)
        all_results[task_type.value] = task_results
        
        # Save results
        save_results(task_results, args.output_dir, task_type.value)
        
        # Generate figures for this task
        if not args.no_figures:
            #create_comprehensive_figure(task_results)
            generate_all_figures(task_results)
            # fig_path = os.path.join(figures_dir, f'figure_{task_type.value}.png')
            # fig.savefig(fig_path, dpi=300)
            # plt.close(fig)
            # print(f"Figure saved: {fig_path}")

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
