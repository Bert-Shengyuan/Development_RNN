# Cross-Trial Type Analysis Scripts

This module provides comprehensive PCA and CCA analysis tools for cross-trial neural data with optional hierarchical brain region aggregation.

## Features

- **PCA Analysis**: Principal Component Analysis across trial types
- **CCA Analysis**: Canonical Correlation Analysis between brain region groups
- **Region Aggregation**: Optional hierarchical aggregation of brain regions
- **Visualization**: Comprehensive plots and summary statistics
- **CLI Interface**: Easy-to-use command-line interface

## Requirements

```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

## Brain Region Hierarchy

The scripts support hierarchical aggregation of brain regions:

### Regions Kept Separate
- **MOp**: Primary Motor Cortex
- **MOs**: Secondary Motor Cortex
- **mPFC**: Medial Prefrontal Cortex
- **ORB**: Orbitofrontal Cortex
- **ILM**: Infralimbic Area
- **OLF**: Olfactory Areas
- **HY**: Hypothalamus

### Regions Aggregated
- **STR** (Striatum): Combines `STR` + `STRv` (Dorsal + Ventral Striatum)
- **TH** (Thalamus): Combines `MD` + `VALVM` + `LP` + `VPMPO`
  - MD: Mediodorsal Nucleus
  - VALVM: Ventral Anterior-Lateral Complex
  - LP: Lateral Posterior Nucleus
  - VPMPO: Ventral Posteromedial Nucleus

## File Descriptions

### Core Analysis Modules

1. **cross_trial_type_pca_analysis.py**
   - PCA analysis functions
   - Region aggregation utilities
   - Visualization functions for PCA results
   - Can be imported or run standalone for testing

2. **cross_trial_type_cca_analysis.py**
   - CCA analysis functions
   - Region splitting strategies
   - Visualization functions for CCA results
   - Can be imported or run standalone for testing

### Runner Scripts

3. **run_cross_trial_type_pca_analysis.py**
   - Command-line interface for PCA analysis
   - Handles data loading and result saving
   - Supports both single and comparative analyses

4. **run_cross_trial_type_cca_analysis.py**
   - Command-line interface for CCA analysis
   - Handles data loading and result saving
   - Supports multiple splitting strategies

## Usage Examples

### PCA Analysis

#### Basic Usage
```bash
python run_cross_trial_type_pca_analysis.py --data_file data.npz
```

#### With Region Aggregation
```bash
python run_cross_trial_type_pca_analysis.py \
    --data_file data.npz \
    --aggregate \
    --aggregation_method mean
```

#### Compare Across Trial Types
```bash
python run_cross_trial_type_pca_analysis.py \
    --data_file data.npz \
    --compare_trials \
    --aggregate \
    --n_components 10 \
    --output_dir ./pca_results/
```

#### Save Transformed Data
```bash
python run_cross_trial_type_pca_analysis.py \
    --data_file data.npz \
    --aggregate \
    --save_transformed
```

### CCA Analysis

#### Basic Usage
```bash
python run_cross_trial_type_cca_analysis.py --data_file data.npz
```

#### With Region Aggregation and Custom Split
```bash
python run_cross_trial_type_cca_analysis.py \
    --data_file data.npz \
    --aggregate \
    --split_strategy cortical_subcortical
```

#### Compare Across Trial Types
```bash
python run_cross_trial_type_cca_analysis.py \
    --data_file data.npz \
    --compare_trials \
    --aggregate \
    --n_components 5 \
    --output_dir ./cca_results/
```

## Data Format

The scripts expect data in `.npz` format with the following structure:

```python
np.savez('data.npz',
    data=neural_data,          # Shape: (n_trials, n_regions) or (n_trials, n_timepoints, n_regions)
    region_labels=region_list,  # List of region names (e.g., ['MOp', 'MOs', ...])
    trial_types=trial_labels    # Optional: Array of trial type labels
)
```

### Example Data Creation

```python
import numpy as np

# Create synthetic data
n_trials = 100
n_regions = 13
region_labels = ['MOp', 'MOs', 'mPFC', 'ORB', 'ILM', 'OLF',
                'STR', 'STRv', 'MD', 'VALVM', 'LP', 'VPMPO', 'HY']

# Neural activity data
data = np.random.randn(n_trials, n_regions)

# Trial type labels (optional)
trial_types = np.array(['correct']*50 + ['error']*50)

# Save to file
np.savez('my_data.npz',
         data=data,
         region_labels=np.array(region_labels),
         trial_types=trial_types)
```

## Command-Line Arguments

### PCA Analysis

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_file` | str | Required | Path to input data file |
| `--aggregate` | flag | False | Enable region aggregation |
| `--aggregation_method` | str | 'mean' | Aggregation method (mean/sum/max) |
| `--n_components` | int | Auto | Number of principal components |
| `--compare_trials` | flag | False | Compare across trial types |
| `--output_dir` | str | './pca_results' | Output directory |
| `--no_standardize` | flag | False | Disable standardization |
| `--save_transformed` | flag | False | Save PC scores |
| `--verbose` | flag | False | Print detailed info |

### CCA Analysis

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_file` | str | Required | Path to input data file |
| `--aggregate` | flag | False | Enable region aggregation |
| `--aggregation_method` | str | 'mean' | Aggregation method (mean/sum/max) |
| `--n_components` | int | Auto | Number of canonical components |
| `--split_strategy` | str | 'cortical_subcortical' | Region splitting strategy |
| `--compare_trials` | flag | False | Compare across trial types |
| `--output_dir` | str | './cca_results' | Output directory |
| `--no_standardize` | flag | False | Disable standardization |
| `--save_transformed` | flag | False | Save canonical variates |
| `--verbose` | flag | False | Print detailed info |

### CCA Split Strategies

- **cortical_subcortical**: Split cortical regions from subcortical regions
- **motor_prefrontal**: Compare motor cortex with prefrontal cortex
- **first_half**: Simple split at the midpoint

## Output Files

### PCA Analysis Outputs

1. **pca_summary_[timestamp].txt**: Text summary of analysis results
2. **pca_results_[timestamp].png**: Comprehensive visualization
3. **variance_comparison_[timestamp].csv**: Variance comparison table (if comparing)
4. **pca_transformed_[timestamp].npz**: PC scores (if --save_transformed)

### CCA Analysis Outputs

1. **cca_summary_[timestamp].txt**: Text summary of analysis results
2. **cca_results_[timestamp].png**: Comprehensive visualization
3. **correlation_comparison_[timestamp].csv**: Correlation comparison (if comparing)
4. **cca_transformed_[timestamp].npz**: Canonical variates (if --save_transformed)

## Testing

Each analysis module includes built-in tests. Run them standalone:

```bash
# Test PCA module
python cross_trial_type_pca_analysis.py

# Test CCA module
python cross_trial_type_cca_analysis.py
```

## Python API

You can also import and use the modules in your own scripts:

```python
from cross_trial_type_pca_analysis import perform_pca_analysis, aggregate_brain_regions
from cross_trial_type_cca_analysis import perform_cca_analysis

# Load your data
data = ...
region_labels = ['MOp', 'MOs', ...]
trial_types = ...

# Perform PCA with aggregation
results = perform_pca_analysis(
    data,
    region_labels,
    trial_types,
    aggregate_regions=True,
    aggregation_method='mean'
)

# Access results
print(f"Variance explained: {results.get_variance_explained(3):.2%}")
print(f"Aggregated regions: {results.region_labels}")
```

## Notes

- If your data has a time dimension (3D: trials × timepoints × regions), it will be automatically averaged over time
- Region labels are case-sensitive and should match the predefined hierarchy
- Unknown regions will be kept as-is during aggregation
- All visualizations are saved as high-resolution PNG files (300 DPI)

## Support

For questions or issues, please refer to the main repository documentation.
