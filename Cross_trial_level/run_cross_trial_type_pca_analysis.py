#!/usr/bin/env python3
"""
Runner Script for Cross-Trial Type PCA Analysis
================================================

This script provides a command-line interface for running PCA analysis
on cross-trial neural data with optional brain region aggregation.

Usage Examples
--------------
# Basic usage with default settings
python run_cross_trial_type_pca_analysis.py --data_file data.npz

# With region aggregation enabled
python run_cross_trial_type_pca_analysis.py --data_file data.npz --aggregate

# Custom number of components and output directory
python run_cross_trial_type_pca_analysis.py --data_file data.npz --n_components 15 --output_dir ./results/

# Compare across trial types
python run_cross_trial_type_pca_analysis.py --data_file data.npz --compare_trials --aggregate

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
from cross_trial_type_pca_analysis import (
    perform_pca_analysis,
    compare_pca_across_trial_types,
    plot_pca_results,
    plot_cross_trial_comparison,
    DEFAULT_REGION_ORDER
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run cross-trial type PCA analysis with optional region aggregation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data_file data.npz
  %(prog)s --data_file data.npz --aggregate --n_components 10
  %(prog)s --data_file data.npz --compare_trials --output_dir results/
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
        help='Number of principal components (default: min(n_trials, n_regions))'
    )

    parser.add_argument(
        '--compare_trials',
        action='store_true',
        help='Compare PCA across different trial types'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./pca_results',
        help='Directory for output files (default: ./pca_results)'
    )

    parser.add_argument(
        '--no_standardize',
        action='store_true',
        help='Disable data standardization before PCA'
    )

    parser.add_argument(
        '--save_transformed',
        action='store_true',
        help='Save transformed data (PC scores) to file'
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
    summary_path = os.path.join(output_dir, f'pca_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Cross-Trial Type PCA Analysis Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: {args.data_file}\n")
        f.write(f"Region aggregation: {'Yes' if args.aggregate else 'No'}\n")
        if args.aggregate:
            f.write(f"Aggregation method: {args.aggregation_method}\n")
        f.write(f"Standardization: {'Yes' if not args.no_standardize else 'No'}\n\n")

        if not args.compare_trials:
            f.write("SINGLE PCA ANALYSIS\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"Number of regions: {len(results.region_labels)}\n")
            f.write(f"Region labels: {', '.join(results.region_labels)}\n\n")
            f.write(f"Number of components: {results.n_components}\n")
            f.write(f"Number of trials: {results.transformed_data.shape[0]}\n\n")

            f.write("Variance Explained:\n")
            for i in range(min(10, results.n_components)):
                f.write(f"  PC{i+1}: {results.explained_variance_ratio[i]:.4f} ({results.explained_variance_ratio[i]*100:.2f}%)\n")
            f.write(f"\nCumulative variance (top 3): {results.get_variance_explained(3):.4f}\n")
            f.write(f"Total variance explained: {results.get_variance_explained():.4f}\n\n")

            f.write("Top PC1 Loadings:\n")
            loadings = results.get_component_loadings(0)
            sorted_loadings = sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True)
            for region, loading in sorted_loadings[:10]:
                f.write(f"  {region}: {loading:.4f}\n")

        else:
            f.write("CROSS-TRIAL TYPE COMPARISON\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"Trial types: {', '.join(comparison.trial_types)}\n\n")

            f.write("Variance Comparison:\n")
            f.write(comparison.variance_comparison.to_string())
            f.write("\n\n")

            f.write("Component Similarity Matrix:\n")
            f.write("(Cosine similarity between PC1 vectors)\n")
            for i, type1 in enumerate(comparison.trial_types):
                f.write(f"\n{type1}:\n  ")
                for j, type2 in enumerate(comparison.trial_types):
                    f.write(f"{type2}: {comparison.component_similarity[i, j]:.3f}  ")

    print(f"\n✓ Summary saved to: {summary_path}")

    # Save variance comparison as CSV (if comparison was run)
    if args.compare_trials and comparison is not None:
        csv_path = os.path.join(output_dir, f'variance_comparison_{timestamp}.csv')
        comparison.variance_comparison.to_csv(csv_path, index=False)
        print(f"✓ Variance comparison saved to: {csv_path}")

    # Save transformed data if requested
    if args.save_transformed:
        if not args.compare_trials:
            transformed_path = os.path.join(output_dir, f'pca_transformed_{timestamp}.npz')
            np.savez(
                transformed_path,
                transformed_data=results.transformed_data,
                explained_variance_ratio=results.explained_variance_ratio,
                components=results.components,
                region_labels=results.region_labels
            )
            print(f"✓ Transformed data saved to: {transformed_path}")
        else:
            for trial_type, pca_results in comparison.pca_results.items():
                transformed_path = os.path.join(
                    output_dir,
                    f'pca_transformed_{trial_type}_{timestamp}.npz'
                )
                np.savez(
                    transformed_path,
                    transformed_data=pca_results.transformed_data,
                    explained_variance_ratio=pca_results.explained_variance_ratio,
                    components=pca_results.components,
                    region_labels=pca_results.region_labels
                )
            print(f"✓ Transformed data saved for all trial types")


def main():
    """Main execution function."""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Cross-Trial Type PCA Analysis")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    data, region_labels, trial_types = load_data(args.data_file)
    print(f"  ✓ Data shape: {data.shape}")
    print(f"  ✓ Number of regions: {len(region_labels)}")
    print(f"  ✓ Regions: {', '.join(region_labels[:5])}{'...' if len(region_labels) > 5 else ''}")

    if args.aggregate:
        print(f"\n  Region aggregation enabled (method: {args.aggregation_method})")

    # Run analysis
    results = None
    comparison = None

    if not args.compare_trials or trial_types is None:
        # Single PCA analysis
        print("\n2. Performing PCA analysis...")

        if trial_types is None:
            trial_types = np.array(['all'] * data.shape[0])

        results = perform_pca_analysis(
            data,
            region_labels,
            trial_types,
            n_components=args.n_components,
            aggregate_regions=args.aggregate,
            aggregation_method=args.aggregation_method,
            standardize=not args.no_standardize
        )

        print(f"  ✓ PCA complete")
        print(f"  ✓ Components computed: {results.n_components}")
        print(f"  ✓ Top 3 PC variance: {results.get_variance_explained(3):.2%}")
        print(f"  ✓ Total variance: {results.get_variance_explained():.2%}")

        # Generate visualization
        print("\n3. Generating visualizations...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(args.output_dir, f'pca_results_{timestamp}.png')
        fig = plot_pca_results(results, save_path=plot_path)
        print(f"  ✓ Plot saved to: {plot_path}")

    else:
        # Cross-trial comparison
        print("\n2. Performing cross-trial type PCA comparison...")

        # Organize data by trial type
        unique_types = np.unique(trial_types)
        data_by_type = {
            trial_type: data[trial_types == trial_type]
            for trial_type in unique_types
        }

        print(f"  ✓ Trial types found: {', '.join(unique_types)}")
        for trial_type, trial_data in data_by_type.items():
            print(f"    - {trial_type}: {trial_data.shape[0]} trials")

        comparison = compare_pca_across_trial_types(
            data_by_type,
            region_labels,
            n_components=args.n_components or 10,
            aggregate_regions=args.aggregate,
            aggregation_method=args.aggregation_method
        )

        print(f"\n  ✓ Comparison complete")

        # Generate visualization
        print("\n3. Generating visualizations...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(args.output_dir, f'pca_comparison_{timestamp}.png')
        fig = plot_cross_trial_comparison(comparison, save_path=plot_path)
        print(f"  ✓ Comparison plot saved to: {plot_path}")

        # Also save individual plots
        for trial_type, pca_results in comparison.pca_results.items():
            plot_path = os.path.join(
                args.output_dir,
                f'pca_results_{trial_type}_{timestamp}.png'
            )
            fig = plot_pca_results(pca_results, save_path=plot_path)
        print(f"  ✓ Individual plots saved for all trial types")

    # Save results
    print("\n4. Saving results...")
    save_results(results, comparison, args.output_dir, args)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
