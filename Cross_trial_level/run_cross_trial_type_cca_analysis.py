#!/usr/bin/env python3
"""
Runner Script for Cross-Trial Type CCA Analysis
================================================

This script provides a command-line interface for running CCA analysis
on cross-trial neural data with optional brain region aggregation.

Usage Examples
--------------
# Basic usage with default settings
python run_cross_trial_type_cca_analysis.py --data_file data.npz

# With region aggregation enabled
python run_cross_trial_type_cca_analysis.py --data_file data.npz --aggregate

# Custom split strategy and output directory
python run_cross_trial_type_cca_analysis.py --data_file data.npz --split_strategy motor_prefrontal --output_dir ./results/

# Compare across trial types
python run_cross_trial_type_cca_analysis.py --data_file data.npz --compare_trials --aggregate

Author: Computational Neuroscience Research
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# Import analysis functions
from cross_trial_type_cca_analysis import (
    perform_cca_analysis,
    compare_cca_across_trial_types,
    plot_cca_results,
    plot_cross_trial_cca_comparison,
    DEFAULT_REGION_ORDER
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run cross-trial type CCA analysis with optional region aggregation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data_file data.npz
  %(prog)s --data_file data.npz --aggregate --n_components 5
  %(prog)s --data_file data.npz --compare_trials --split_strategy motor_prefrontal
        """
    )

    # Required arguments
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to input data file (.npz or .npy format)'
    )

    # Optional arguments
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Enable hierarchical brain region aggregation'
    )

    parser.add_argument(
        '--aggregation_method',
        type=str,
        default='mean',
        choices=['mean', 'sum', 'max'],
        help='Method for aggregating regions (default: mean)'
    )

    parser.add_argument(
        '--n_components',
        type=int,
        default=None,
        help='Number of canonical components (default: min(X_dims, Y_dims))'
    )

    parser.add_argument(
        '--split_strategy',
        type=str,
        default='cortical_subcortical',
        choices=['cortical_subcortical', 'motor_prefrontal', 'first_half'],
        help='Strategy for splitting regions into two groups (default: cortical_subcortical)'
    )

    parser.add_argument(
        '--compare_trials',
        action='store_true',
        help='Compare CCA across different trial types'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./cca_results',
        help='Directory for output files (default: ./cca_results)'
    )

    parser.add_argument(
        '--no_standardize',
        action='store_true',
        help='Disable data standardization before CCA'
    )

    parser.add_argument(
        '--save_transformed',
        action='store_true',
        help='Save transformed data (canonical variates) to file'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )

    return parser.parse_args()


def load_data(data_file):
    """
    Load neural data from file.

    Expected data format:
    - .npz file with keys: 'data', 'region_labels', 'trial_types' (optional)
    - 'data': shape (n_trials, n_regions) or (n_trials, n_timepoints, n_regions)
    - 'region_labels': list of region names
    - 'trial_types': array of trial type labels (optional)

    Returns
    -------
    data : np.ndarray
        Neural activity data
    region_labels : list
        Region names
    trial_types : np.ndarray
        Trial type labels (or None)
    """
    if data_file.endswith('.npz'):
        data_dict = np.load(data_file, allow_pickle=True)
        data = data_dict['data']
        region_labels = data_dict['region_labels'].tolist()
        trial_types = data_dict.get('trial_types', None)

        # If we have 3D data (trials x timepoints x regions), average over time
        if len(data.shape) == 3:
            print(f"  Averaging over {data.shape[1]} timepoints...")
            data = data.mean(axis=1)

    elif data_file.endswith('.npy'):
        data = np.load(data_file)
        # Assume default region labels
        n_regions = data.shape[-1]
        region_labels = [f'Region_{i+1}' for i in range(n_regions)]
        trial_types = None

        if len(data.shape) == 3:
            data = data.mean(axis=1)

    else:
        raise ValueError(f"Unsupported file format: {data_file}")

    return data, region_labels, trial_types


def save_results(results, comparison, output_dir, args):
    """Save analysis results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary report
    summary_path = os.path.join(output_dir, f'cca_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Cross-Trial Type CCA Analysis Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: {args.data_file}\n")
        f.write(f"Region aggregation: {'Yes' if args.aggregate else 'No'}\n")
        if args.aggregate:
            f.write(f"Aggregation method: {args.aggregation_method}\n")
        f.write(f"Split strategy: {args.split_strategy}\n")
        f.write(f"Standardization: {'Yes' if not args.no_standardize else 'No'}\n\n")

        if not args.compare_trials:
            f.write("SINGLE CCA ANALYSIS\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"X regions: {len(results.x_labels)}\n")
            f.write(f"X region labels: {', '.join(results.x_labels)}\n\n")
            f.write(f"Y regions: {len(results.y_labels)}\n")
            f.write(f"Y region labels: {', '.join(results.y_labels)}\n\n")
            f.write(f"Number of components: {results.n_components}\n")
            f.write(f"Number of trials: {results.x_transformed.shape[0]}\n\n")

            f.write("Canonical Correlations:\n")
            for i in range(results.n_components):
                f.write(f"  CC{i+1}: {results.canonical_correlations[i]:.4f}\n")
            f.write(f"\nMean canonical correlation: {results.canonical_correlations.mean():.4f}\n")
            f.write(f"Max canonical correlation: {results.canonical_correlations.max():.4f}\n\n")

            f.write("Top X Weights (CV1):\n")
            x_loadings = results.get_x_loadings(0)
            sorted_x = sorted(x_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
            for region, weight in sorted_x[:10]:
                f.write(f"  {region}: {weight:.4f}\n")

            f.write("\nTop Y Weights (CV1):\n")
            y_loadings = results.get_y_loadings(0)
            sorted_y = sorted(y_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
            for region, weight in sorted_y[:10]:
                f.write(f"  {region}: {weight:.4f}\n")

        else:
            f.write("CROSS-TRIAL TYPE COMPARISON\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"Trial types: {', '.join(comparison.trial_types)}\n\n")

            f.write("Canonical Correlation Comparison:\n")
            f.write(comparison.correlation_comparison.to_string())
            f.write("\n\n")

            f.write("Canonical Variate Similarity Matrix:\n")
            for i, type1 in enumerate(comparison.trial_types):
                f.write(f"\n{type1}:\n  ")
                for j, type2 in enumerate(comparison.trial_types):
                    f.write(f"{type2}: {comparison.canonical_similarity[i, j]:.3f}  ")

    print(f"\n✓ Summary saved to: {summary_path}")

    # Save correlation comparison as CSV (if comparison was run)
    if args.compare_trials and comparison is not None:
        csv_path = os.path.join(output_dir, f'correlation_comparison_{timestamp}.csv')
        comparison.correlation_comparison.to_csv(csv_path, index=False)
        print(f"✓ Correlation comparison saved to: {csv_path}")

    # Save transformed data if requested
    if args.save_transformed:
        if not args.compare_trials:
            transformed_path = os.path.join(output_dir, f'cca_transformed_{timestamp}.npz')
            np.savez(
                transformed_path,
                x_transformed=results.x_transformed,
                y_transformed=results.y_transformed,
                canonical_correlations=results.canonical_correlations,
                x_weights=results.x_weights,
                y_weights=results.y_weights,
                x_labels=results.x_labels,
                y_labels=results.y_labels
            )
            print(f"✓ Transformed data saved to: {transformed_path}")
        else:
            for trial_type, cca_results in comparison.cca_results.items():
                transformed_path = os.path.join(
                    output_dir,
                    f'cca_transformed_{trial_type}_{timestamp}.npz'
                )
                np.savez(
                    transformed_path,
                    x_transformed=cca_results.x_transformed,
                    y_transformed=cca_results.y_transformed,
                    canonical_correlations=cca_results.canonical_correlations,
                    x_weights=cca_results.x_weights,
                    y_weights=cca_results.y_weights,
                    x_labels=cca_results.x_labels,
                    y_labels=cca_results.y_labels
                )
            print(f"✓ Transformed data saved for all trial types")


def main():
    """Main execution function."""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Cross-Trial Type CCA Analysis")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    data, region_labels, trial_types = load_data(args.data_file)
    print(f"  ✓ Data shape: {data.shape}")
    print(f"  ✓ Number of regions: {len(region_labels)}")
    print(f"  ✓ Regions: {', '.join(region_labels[:5])}{'...' if len(region_labels) > 5 else ''}")

    if args.aggregate:
        print(f"\n  Region aggregation enabled (method: {args.aggregation_method})")

    print(f"  Split strategy: {args.split_strategy}")

    # Run analysis
    results = None
    comparison = None

    if not args.compare_trials or trial_types is None:
        # Single CCA analysis
        print("\n2. Performing CCA analysis...")

        if trial_types is None:
            trial_types = np.array(['all'] * data.shape[0])

        results = perform_cca_analysis(
            data,
            region_labels,
            trial_types,
            n_components=args.n_components,
            aggregate_regions=args.aggregate,
            aggregation_method=args.aggregation_method,
            split_strategy=args.split_strategy,
            standardize=not args.no_standardize
        )

        print(f"  ✓ CCA complete")
        print(f"  ✓ X regions: {len(results.x_labels)} ({', '.join(results.x_labels)})")
        print(f"  ✓ Y regions: {len(results.y_labels)} ({', '.join(results.y_labels)})")
        print(f"  ✓ Components computed: {results.n_components}")
        print(f"  ✓ Top canonical correlation: {results.canonical_correlations[0]:.3f}")
        print(f"  ✓ Mean canonical correlation: {results.canonical_correlations.mean():.3f}")

        # Generate visualization
        print("\n3. Generating visualizations...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(args.output_dir, f'cca_results_{timestamp}.png')
        fig = plot_cca_results(results, save_path=plot_path)
        print(f"  ✓ Plot saved to: {plot_path}")

    else:
        # Cross-trial comparison
        print("\n2. Performing cross-trial type CCA comparison...")

        # Organize data by trial type
        unique_types = np.unique(trial_types)
        data_by_type = {
            trial_type: data[trial_types == trial_type]
            for trial_type in unique_types
        }

        print(f"  ✓ Trial types found: {', '.join(unique_types)}")
        for trial_type, trial_data in data_by_type.items():
            print(f"    - {trial_type}: {trial_data.shape[0]} trials")

        comparison = compare_cca_across_trial_types(
            data_by_type,
            region_labels,
            n_components=args.n_components or 5,
            aggregate_regions=args.aggregate,
            aggregation_method=args.aggregation_method,
            split_strategy=args.split_strategy
        )

        print(f"\n  ✓ Comparison complete")

        # Generate visualization
        print("\n3. Generating visualizations...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(args.output_dir, f'cca_comparison_{timestamp}.png')
        fig = plot_cross_trial_cca_comparison(comparison, save_path=plot_path)
        print(f"  ✓ Comparison plot saved to: {plot_path}")

        # Also save individual plots
        for trial_type, cca_results in comparison.cca_results.items():
            plot_path = os.path.join(
                args.output_dir,
                f'cca_results_{trial_type}_{timestamp}.png'
            )
            fig = plot_cca_results(cca_results, save_path=plot_path)
        print(f"  ✓ Individual plots saved for all trial types")

    # Save results
    print("\n4. Saving results...")
    save_results(results, comparison, args.output_dir, args)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
