<p align="center">
  <img src="EmpiricML-logo.png", width = "250", height = "250", alt="EmpiricML Logo">
</p>

# EmpiricML

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

> **Science, not Alchemy.** Build reliable ML models on tabular data with the rigour of an empirical science.

EmpiricML is a Python framework designed to make building, testing, and tracking machine learning models on **tabular data** faster, easier, and most importantly, **robust**. Built on the shoulders of giants - **scikit-learn** and **Polars** - it brings the concept of a scientific "Laboratory" to your ML workflow.

---

## The Core Idea
The core idea behind the framework is that **Machine Learning is an empirical science**.
In empirical sciences, we design **experiments** to test hypotheses. To do that effectively, we need a controlled environment, i.e. a **Laboratory**.

**EmpiricML provides that Laboratory.**

Instead of scattered scripts and notebooks, the `Lab` class encapsulates everything required for a rigorous ML experiment:
*   **Data**: Training and testing datasets (handled efficiently via Polars).
*   **Protocol**: A defined Cross-Validation strategy.
*   **Measurement**: A specific Error or Performance Metric.
*   **Criteria**: Rules for statistical comparison to determine if Model A is *truly* better than Model B.

## Key Features

*   `Polars Integration`: Pipelines work with Polars LazyFrames for high-performance data processing.
*   `Automated CV Evaluation`: Every experiment is rigorously cross-validated.
*   `Statistical Comparison`: Don't guess. Use permutation tests and statistical thresholds to compare models.
*   `Automated Tracking`: Logs results, predictions, and pipeline configurations automatically.
*   `Early Stopping`: Aborts unpromising experiments early to save compute resources.
*   `Auto Mode`: Automatically tracks and persists the best-performing experiment.
*   `Hyperparameter Optimization`: Built-in support for Grid and Random search.
*   `Feature Selection`: Permutation importance with recursive elimination.
*   `Checkpointing`: Save/Restore your `Lab` state to pause and resume work seamlessly.


## Installation

```bash
pip install empiricml
```

## Quick Start

### 1. Initialize your Laboratory

First, define the environment for your experiments. This ensures all models are evaluated on the exact same data and metrics.

```python
import polars as pl
from empml.metrics import MAE
from empml.data import CSVDownloader
from empml.cv import KFold
from empml.lab import Lab, ComparisonCriteria

# Create the Lab
lab = Lab(
    name = 'house_prices_lab',
    # Data Loading
    train_downloader = CSVDownloader(path='train.csv', separator=','),
    test_downloader = CSVDownloader(path='test.csv', separator=','),
    
    # Target Variable
    target = 'price',
    
    # Evaluation Protocol
    metric = MAE(),
    minimize = True,
    cv_generator = KFold(n_splits=5, random_state=42),
    
    # Comparison Criteria
    comparison_criteria = ComparisonCriteria(n_folds_threshold=1, alpha=0.05, n_iters=200)
)
```

### 2. Define a Pipeline and Run an Experiment

EmpiricML pipelines combine Feature Engineering (Transformers) and Modeling (Estimators).

```python
from lightgbm import LGBMRegressor 
from empml.pipelines import Pipeline
from empml.transformers import Log1pFeatures
from empml.wrappers import SKlearnWrapper

# Define features to use
features = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms']

# Create a pipeline
pipe = Pipeline(
    steps = [
        # Feature Engineering: Apply Log1p to numerical features
        ('log_scale', Log1pFeatures(features=features, suffix='')),
        # Modeling: Wrap sklearn-compatible estimators
        ('model', SKlearnWrapper(
            estimator=LGBMRegressor(verbose=-1), 
            features=features, 
            target='price'
        ))
    ], 
    name = 'LGBM_Optimized', 
    description = 'LightGBM regressor with log-transformed features.'
)

# Run the experiment in the Lab
lab.run_experiment(pipeline=pipe)
```

### 3. Hyperparameter Optimization

EmpiricML simplifies hyperparameter tuning with built-in Grid and Random Search capabilities. This allows you to systematically explore different model configurations.

```python
from sklearn.tree import DecisionTreeRegressor

# Define the parameter grid
params = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Run Hyperparameter Optimization
# Note: Pass the estimator class (not an instance) to the hpo method
best_result_row = lab.hpo(
    features=features,
    params_list=params,
    estimator=DecisionTreeRegressor, 
    search_type='grid',              # Options: 'grid' or 'random'
    verbose=True
)
```

### 4. Accessing Experiment Results

All experiment tracking data, including results from single runs and HPO, is stored in the `lab.results` DataFrame. This Polars DataFrame contains metrics, execution times, and metadata for every experiment run in the session.

```python
# View all experiment results as a Polars DataFrame
lab.results

# Get the best performing experiment stats
lab.show_best_score()
```

## Project Structure

The library is organized into logical modules found in `src/empml`:

*   `lab`: The core `Lab` class management.
*   `pipelines`: Scikit-learn style pipelines compatible with Polars.
*   `wrappers`: Wrappers for ML algorithms (XGBoost, LightGBM, CatBoost, Sklearn, Pytorch).
*   `transformers`: Feature engineering blocks.
*   `metrics`: Performance metrics.
*   `data`: Tools for handling data loading and downloads.
*   `cv`: Cross-validation splitters.

## Contributing

Contributions are welcome! Please check out the issues or submit a PR.

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/new-feature`)
3.  Commit your changes (`git commit -m 'Add some new feature'`)
4.  Push to the branch (`git push origin feature/new-feature`)
5.  Open a Pull Request

## Citation

If you use EmpiricML in your research, please cite:

```bibtex
@software{EmpiricML,
  title={EmpiricML: A Python framework for building robust Machine Learning models on tabular data faster and easier},
  author={Pasquale Trani},
  year={2026},
  url={https://github.com/PasqualeTrani/EmpiricML}
}
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
