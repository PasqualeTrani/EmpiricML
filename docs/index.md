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