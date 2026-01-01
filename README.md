<p align="center">
  <img src="EmpiricML-logo.png", width = "250", height = "250", alt="EmpiricML Logo">
</p>

# EmpiricML

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)


EmpiricML is an open-source Python framework designed to bring the rigor of empirical science to the Machine Learning development process.
Are you tired of scattered Jupyter Notebooks and untracked experiments? EmpiricML provides a structured "Laboratory" environment to help you move from messy scripts to reproducible science.

## The Philosophy: ML as an Empirical Science
The core idea behind EmpiricML is that building a machine learning model is an iterative, scientific process. You form a hypothesis (e.g., "Adding these specific features will decrease the error"), and you must test it in a controlled environment.
EmpiricML provides that environment through the Lab class. It encapsulates everything needed for rigorous ML experimentation:

* Train and test data management
* Cross-validation strategies
* Evaluation metrics
* Standardized criteria for comparing models

## Key Features

### Experiment Tracking
Keep a detailed ledger of every run. EmpiricML automatically stores:

* Metric performance and overfitting percentages
* Training and inference latency
* Generated predictions for downstream analysis

### Polars-Native Pipelines
Performance is at the heart of EmpiricML. Unlike scikit-learn pipelines which are NumPy-based, EmpiricML transformations utilize Polars LazyFrames. This allows for lightning-fast, memory-efficient data handling even with large datasets.

### Automated Workflows
Stop writing boilerplate code for standard tasks. EmpiricML automates:

* Hyperparameter Optimization (HPO)
* Feature Importance calculation
* Automated Feature Selection

### Rigorous Model Comparison
Compare experiments with statistical confidence. Define comparison criteria in your Lab class based on:

Performance Thresholds: Does Model B outperform Model A by a significant margin?
Statistical Tests: Use built-in tests to ensure your improvements aren't just noise

EmpiricML can automatically update and store your "Best Model" based on these predefined rules.

### Fast ML Baselines
Go from zero to a leaderboard in seconds. With just a few lines of code, you can evaluate up to 10 baseline models (including LightGBM, XGBoost, Random Forest, MLP, and more) to establish a performance floor for your project.

### Early Stopping
Aborts unpromising experiments early to save compute resources.

### Checkpointing 

Save/Restore your `Lab` state to pause and resume work seamlessly.


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
    
    # Criteria for Comparing Models
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
*   `pipeline`: Scikit-learn style pipelines compatible with Polars.
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
