# EmpML Lab

A systematic machine learning experimentation framework for model development, evaluation, and comparison.

## Overview

EmpML Lab provides a structured workflow for ML experiments with cross-validation, automated tracking, statistical comparison, and hyperparameter optimization. It eliminates boilerplate code and ensures reproducible, comparable results across experiments.

## Key Features

- **Automated CV Evaluation**: Run experiments with configurable cross-validation strategies
- **Statistical Comparison**: Compare models using percentage thresholds or permutation tests
- **Experiment Tracking**: Automatic logging of results, predictions, and pipelines
- **Auto Mode**: Automatically update best model when improvements are found
- **Hyperparameter Optimization**: Grid and random search support
- **Feature Selection**: Permutation importance with recursive elimination
- **Baseline Suite**: Quick benchmarking with 10+ standard models
- **Checkpoint System**: Save and restore experiment state

## Installation

```bash
pip install empml
```

## Quick Start

```python
from empml.lab import Lab, EvalParams
from empml.base import DataDownloader, Metric, CVGenerator
from empml.pipelines import Pipeline
from empml.estimators import EstimatorWrapper
from sklearn.ensemble import RandomForestRegressor

# Setup
train_data = DataDownloader(source='data/train.csv')
metric = Metric(name='rmse')
cv = CVGenerator(n_folds=5, strategy='kfold')

# Initialize Lab
lab = Lab(
    train_downloader=train_data,
    metric=metric,
    cv_generator=cv,
    target='target_column',
    eval_params=EvalParams(n_folds_threshold=2, pct_threshold=0.01)
)

# Run experiment
pipeline = Pipeline([
    ('model', EstimatorWrapper(
        estimator=RandomForestRegressor(),
        features=['feature1', 'feature2'],
        target='target_column'
    ))
], name='rf_baseline')

lab.run_experiment(pipeline, compare_against=1, auto_mode=True)
```

## Core Concepts

### EvalParams

Configure how experiments are compared:

```python
# Percentage-based comparison
EvalParams(n_folds_threshold=2, pct_threshold=0.01)

# Statistical testing
EvalParams(n_folds_threshold=2, alpha=0.05, n_iters=200)
```

### Experiment Tracking

Lab automatically tracks:
- Cross-validation results per fold
- Overfitting metrics (train/valid gap)
- Predictions for each experiment
- Pipeline artifacts

Access results:
```python
lab.results  # Summary table
lab.results_details  # Fold-level details
```

### Auto Mode

Automatically update best experiment:

```python
lab.run_experiment(pipeline, auto_mode=True)
# If new experiment improves, best_experiment updates automatically
```

## Advanced Features

### Baseline Benchmarking

Test multiple models quickly:

```python
lab.run_base_experiments(
    features=['feat1', 'feat2', 'feat3'],
    problem_type='regression'  # or 'classification'
)
# Runs: Linear, KNN, SVM, RF, XGBoost, LightGBM, CatBoost, etc.
```

### Hyperparameter Optimization

```python
lab.hpo(
    features=['feat1', 'feat2'],
    params_list={
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    estimator=XGBRegressor,
    search_type='grid'  # or 'random'
)
```

### Feature Selection

```python
# Permutation feature importance
pfi = lab.permutation_feature_importance(
    pipeline=my_pipeline,
    features=['feat1', 'feat2', 'feat3'],
    n_iters=10
)

# Recursive feature elimination
selected_features = lab.recursive_permutation_feature_selection(
    estimator=RandomForestRegressor(),
    features=all_features,
    n_iters=5
)
```

### Statistical Comparison

```python
# Compute p-value between experiments
pvalue = lab.compute_pvalue(
    experiment_ids=(1, 2),
    n_iters=200
)
print(f"P-value: {pvalue}")
```

### Checkpoints

```python
# Save lab state
lab.save_check_point('after_baseline')

# Restore later
from empml.lab import restore_check_point
lab = restore_check_point('my_lab', 'after_baseline')
```

## Directory Structure

```
./lab_name/
├── pipelines/          # Serialized pipeline objects
├── predictions/        # Prediction parquet files
└── check_points/       # Lab checkpoints
```

## Example Workflow

```python
# 1. Initialize
lab = Lab(train_data, metric, cv, target='y', eval_params=params)

# 2. Baseline comparison
lab.run_base_experiments(features=feature_list)

# 3. Select best baseline
lab._set_best_experiment(experiment_id=5)

# 4. Hyperparameter tuning
lab.hpo(
    features=feature_list,
    params_list=param_grid,
    estimator=XGBRegressor,
    auto_mode=True  # Auto-update if improved
)

# 5. Feature selection
selected = lab.recursive_permutation_feature_selection(
    estimator=XGBRegressor(),
    features=feature_list
)

# 6. Final model with selected features
final_pipeline = Pipeline([...])
lab.run_experiment(final_pipeline, auto_mode=True)

# 7. Save checkpoint
lab.save_check_point('final_model')
```

## Results Analysis

```python
# View all experiments
print(lab.results)

# Get predictions from specific experiments
preds = lab.retrieve_predictions(
    experiment_ids=[1, 5, 10],
    extra_features=['feature1']
)

# Statistical comparison
lab._log_compare_experiments(experiment_ids=(5, 10))
```

## Requirements

- Python 3.10+
- polars
- numpy
- scikit-learn
- xgboost (optional)
- lightgbm (optional)
- catboost (optional)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a PR.

## Citation

If you use EmpML Lab in your research, please cite:

```bibtex
@software{empml_lab,
  title={EmpML Lab: A Framework for ML Experimentation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/empml}
}
```
