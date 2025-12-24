# EmpiricML

EmpiricML is a Python framework for building **robust** machine learning models on **tabular** data **faster** and **easier**.
It's built on top of two great libraries: scikit-learn and Polars.

## The Core Idea 

As the name suggests, the core idea behind the library is that **Machine Learning is an empirical science**. You have a hypothesis—for example, adding a feature to the model will increase performance according to some metric—and you need to test it. In empirical sciences, testing hypotheses requires designing **experiments**, which in turn requires a **laboratory** with all the equipment you need. **That's precisely the goal of EmpiricML: providing you with a laboratory containing all the equipment you need to run experiments for Machine Learning problems**. Specifically, this equipment includes:

- Train and test data 
- A cross-validation strategy to divide the training data into multiple train-validation samples for more reliable performance estimates
- An error or performance metric to quantify model quality
- Rules to establish when one model is clearly better than another

All these elements are stored in an instance of the **Lab** class, the main component of the framework. 

## Key Features of Lab Class and EmpiricML API

In addition to storing all the items you need to run experiments, the Lab class allows you to:

- **Automated CV Evaluation**: Run experiments with cross-validation and compute statistics like mean CV score, mean train time in seconds, etc.
- **Statistical Comparison**: Compare experiment results using percentage thresholds or permutation tests
- **Experiment Tracking**: Automatically log results, predictions, and pipelines related to an experiment
- **Early Stopping**: Stop experiments before final CV evaluation if partial results aren't promising
- **Auto Mode**: Automatically update the best experiment when improvements are found
- **Hyperparameter Optimization**: Grid and random search support
- **Feature Selection**: Permutation importance with recursive elimination
- **Baseline Suite**: Quick benchmarking with 10 standard models
- **Checkpoint System**: Save and restore a Lab instance to maintain the same conditions and preserve your experiment evaluations

There is a one-to-one relationship between EmpiricML experiments and pipelines.
EmpiricML Pipelines are combinations of transformers (data transformation classes) and estimators (algorithms like Random Forest, XGBoost, etc.), similar to sklearn Pipelines, but they work with Polars LazyFrames instead of numpy arrays. 
Every time you run an experiment, the pipeline related to that experiment is validated through cross-validation. 

## Installation

```bash
pip install empml
```

## Quick Start

```python
import polars as pl

from empml.pipelines import Pipeline
from empml.metrics import MAE
from empml.estimators import EstimatorWrapper 
from empml.data import CSVDownloader
from empml.cv import KFold
from empml.lab import Lab, EvalParams

# Initialize lab instance 
lab = Lab(
    train_downloader = CSVDownloader(path = 'train.csv', separator = ','),  
    metric = MAE(),
    cv_generator = KFold(n_splits = 5, random_state = 0),
    target = 'y',
    minimize = True, 
    name = 'mylab', 
    row_id = None, 
    eval_params = EvalParams(n_folds_threshold = 1, alpha = 0.05, n_iters = 200), 
    test_downloader = CSVDownloader(path = 'test.csv', separator = ','),
)
```

Here is a description of the arguments:
- **train_downloader**: Class for downloading the training data into the lab object as a Polars LazyFrame, accessible through the `.train` attribute
- **metric**: Class for computing the error or performance metric 
- **cv_generator**: Class for generating the cross-validation setting
- **target**: The column name of the target variable in the train and test data
- **minimize**: Boolean variable that establishes whether the metric should be minimized
- **name**: The name of the lab
- **row_id**: The name of the column that uniquely identifies a single row in the training data. Default value is None. If `row_id=None`, the column is automatically created 
- **eval_params**: Class containing the details for comparing two experiments (more details later)
- **test_downloader**: Class for downloading the test data into the lab object as a Polars LazyFrame, accessible through the `.test` attribute. Default value is None.


## How to Run an Experiment

```python
from lightgbm import LGBMRegressor 
from empml.pipelines import Pipeline
from empml.transformers import Log1pFeatures  # Class for creating new features by applying log1p transformation to existing ones
from empml.estimators import EstimatorWrapper # Transform a sklearn-like estimator to be compatible with Polars LazyFrames

features = [
    'feat_1', 'feat_2', 'feat_3'
]

pipe = Pipeline(steps = [
        ('log_scale', Log1pFeatures(features=features, new_features_suffix='')),
        ('estimator', EstimatorWrapper(estimator=LGBMRegressor(verbose=-1), features=features, target='y'))
    ], 
    name = 'lightgbm - log transform', 
    description = f'Base LightGBM with log transformation of the features. The features used are: {features}', 
)

lab.run_experiment(pipeline=pipe)
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

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