from typing import Any, Dict, List, Tuple
import uuid
import os
from datetime import datetime
import time 
import pickle 
from dataclasses import dataclass 

from sklearn.base import BaseEstimator as SKlearnEstimator

import polars as pl 
import numpy as np

from empml.base import (
    DataDownloader, 
    Metric, 
    CVGenerator
)

from empml.base import BaseTransformer
from empml.errors import RunExperimentConfigException
from empml.transformers import Identity
from empml.pipelines import Pipeline, eval_pipeline_cv, relative_performance, compare_results_stats
from empml.estimators import RegressorWrapper, ClassifierWrapper
from empml.utils import log_execution_time, log_step
from empml.lab_utils import (
    setup_row_id_column, 
    create_results_schema, 
    create_results_details_schema, 
    format_experiment_results, 
    format_experiment_details, 
    prepare_predictions_for_save, 
    log_performance_against, 
    retrieve_predictions_from_path, 
    generate_params_list, 
    generate_shuffle_preds, 
    compute_anomaly
)

# --- Logging Setup ---
import logging 
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass 
class EvalParams:
    n_folds_threshold: int
    pct_threshold: float | None = None
    alpha: float | None = None 
    n_iters: int | None = None
    
    def __post_init__(self):
        has_pct = self.pct_threshold is not None
        has_statistical = (self.alpha is not None) and (self.n_iters is not None)
        
        if not (has_pct or has_statistical):
            raise ValueError(
                "Must provide either 'pct_threshold' OR both 'alpha' and 'n_iters'"
            )
        
        if has_pct and has_statistical:
            raise ValueError(
                "Cannot provide both 'pct_threshold' and ('alpha', 'n_iters'). "
                "Choose one approach only."
            )
        
        self.has_pct = has_pct 
        self.has_statistical = has_statistical


# ANSI escape codes for colors in print and logging 
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
BOLD = '\033[1m'
RESET = '\033[0m'


# ------------------------------------------------------------------------------------------
# Lab Class
# ------------------------------------------------------------------------------------------

class Lab:
    # ------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------

    def __init__(
        self,
        train_downloader: DataDownloader,
        metric: Metric,
        cv_generator: CVGenerator,
        target: str,
        eval_params : EvalParams,
        minimize: bool = True,
        row_id: str | None = None,
        test_downloader: DataDownloader | None = None,
        name: str | None = None
    ):
        self.name = name or uuid.uuid1().hex[:8]
        self.metric = metric
        self.cv_generator = cv_generator
        self.target = target
        self.minimize = minimize

        self.train_downloader = train_downloader
        self.test_downloader = test_downloader
        
        self._setup_directories()
        self._load_data(train_downloader, test_downloader)
        self._setup_row_id(row_id)
        self._setup_results_tracking()
        
        self.cv_indexes = self.cv_generator.split(self.train, self.row_id)
        self.n_folds = len(self.cv_indexes)

        # eval experiment parameters
        self._set_eval_params(eval_params=eval_params)
                                               
        self.next_experiment_id = 1 
        self._set_best_experiment()

    def _setup_directories(self):
        """Create lab directory structure."""
        base = f'./{self.name}'
        os.makedirs(f'{base}/pipelines', exist_ok=True)
        os.makedirs(f'{base}/predictions', exist_ok=True)
        os.makedirs(f'{base}/check_points', exist_ok=True)

    def _load_data(self, train_downloader, test_downloader):
        """Load training and test data."""
        self.train = train_downloader.get_data()
        self.test = test_downloader.get_data() if test_downloader else None

    def _setup_row_id(self, row_id):
        """Setup row identifier column."""
        self.train, self.row_id = setup_row_id_column(self.train, row_id)

    def _setup_results_tracking(self):
        """Initialize Polars DataFrames for tracking experiments."""
        self.results = create_results_schema()
        self.results_details = create_results_details_schema()

    def _set_eval_params(self, eval_params : EvalParams):
        """Setup evaluation parameters for comparing experiments"""

        self.n_folds_threshold = eval_params.n_folds_threshold
        self.pct_threshold = eval_params.pct_threshold 
        self.alpha = eval_params.alpha
        self.n_iters = eval_params.n_iters

        # percentage mode
        if eval_params.has_pct:  
            self.eval_has_pct = True
            self.eval_has_statistical = False

        # statistical test mode
        else:  
            self.eval_has_pct = False
            self.eval_has_statistical = True


    def _set_best_experiment(self, experiment_id : int | None = None):
        self.best_experiment = experiment_id

    # ------------------------------------------------------------------------------------------
    # Experiments Metrics
    # ------------------------------------------------------------------------------------------

    def run_experiment(
        self,
        pipeline: Pipeline,
        eval_overfitting : bool = True, 
        store_preds : bool = True, 
        verbose : bool = True,
        compare_against: int | None = None, 
        auto_mode : bool = False
    ):
        """Run an experiment and save all the metrics."""
        
        # handling configuration errors for the experiment 
        if auto_mode and not(self.best_experiment): 
            raise RunExperimentConfigException(
                """
                Select a best experiment before using auto_mode to automatically update the best one.
                You can use the internal _set_best_experiment method. 
                """
            )
        
        if auto_mode:
            logging.info("Auto mode selected, thus compara_against parameter will be ignored and the current experiment will be compared against the best one.")
            compare_against = self.best_experiment # redefinition of compare_against
        
        
        eval = eval_pipeline_cv(
            pipeline=pipeline, 
            lz=self.train, 
            cv_indexes=self.cv_indexes, 
            row_id=self.row_id,
            metric=self.metric, 
            target=self.target, 
            minimize=self.minimize, 
            eval_overfitting=eval_overfitting, 
            store_preds=store_preds,
            verbose=verbose, 
            compare_df=self.results_details.filter(pl.col('experiment_id')==compare_against) if compare_against else pl.DataFrame(), 
            th_lower_performance_n_folds=self.n_folds_threshold
        )

        self._update_results_table(eval=eval, description=pipeline.description, name=pipeline.name)

        # Add new rows to details table regarding the experiment
        self._update_details_table(eval=eval)

        # Save pipeline
        self._save_pipeline(pipeline=pipeline)

        # Save predictions
        self._save_predictions(eval=eval)

        if compare_against and eval.shape[0] == self.n_folds:
            self._log_compare_experiments(experiment_ids=(compare_against, self.next_experiment_id))

            if auto_mode:
                self._update_best_experiment(experiment_ids=(compare_against, self.next_experiment_id))
                logging.info(F"{BLUE}{BOLD}BEST EXPERIMENT UPDATED. THE NEW BEST EXPERIMENT IS {self.best_experiment}{RESET}")

        elif eval.shape[0]<self.n_folds:
            logging.info(f"{BOLD}{RED}Experiment arrested since comparison showed no improvement over baseline.{RESET}")
        else:
            pass
        
        # Set new experiment_id for the next one
        self.next_experiment_id += 1

    def _update_results_table(self, eval: pl.DataFrame, description: str = '', name: str = ''):
        """Add row to results table regarding an experiment"""
        tmp = format_experiment_results(eval, self.next_experiment_id, eval.shape[0] == self.n_folds, description, name)
        self.results = pl.concat([
            self.results,
            tmp.select(self.results.columns)
        ], how='vertical_relaxed')

    def _update_details_table(self, eval: pl.DataFrame):
        """Add rows to results details table regarding an experiment"""
        tmp = format_experiment_details(eval, self.next_experiment_id)
        self.results_details = pl.concat([
            self.results_details,
            tmp.select(self.results_details.columns)
        ], how='vertical_relaxed')

    def _save_pipeline(self, pipeline: Pipeline):
        """Save pipeline used in an experiment"""
        pickle.dump(
            pipeline,
            open(f'./{self.name}/pipelines/pipeline_{self.next_experiment_id}.pkl', 'wb')
        )

    @log_execution_time
    def _save_predictions(self, eval: pl.DataFrame):
        """Save predictions created in an experiment to a parquet file"""
        preds = prepare_predictions_for_save(eval)
        preds.write_parquet(
            f'./{self.name}/predictions/predictions_{self.next_experiment_id}.parquet',
            compression='zstd',
            compression_level=22
        )

    def _update_best_experiment(self, experiment_ids : Tuple[int, int]):
        """Compare results between two experiments"""
        idx_a, idx_b = experiment_ids
        results_a = self.results_details.filter(pl.col('experiment_id')==idx_a)
        results_b = self.results_details.filter(pl.col('experiment_id')==idx_b)
        comparison = compare_results_stats(results_a, results_b, minimize = self.minimize)

        if self.eval_has_pct:
            c1 = (comparison['mean_cv_performance'] > self.pct_threshold)
            c2 = (comparison['n_folds_lower_performance']<=self.n_folds_threshold)
            if  c1 and c2:
                self.best_experiment = idx_b
                logging.info('')
            else: 
                pass

        else:
            pvalue = self.compute_pvalue(experiment_ids=experiment_ids, n_iters=self.n_iters)
            c1 = (comparison['mean_cv_performance'] > 0)
            c2 = (comparison['n_folds_lower_performance']<=self.n_folds_threshold)
            c3 = pvalue<self.alpha
            if  c1 and c2 and c3:
                self.best_experiment = idx_b
            else: 
                pass
        


    def _log_compare_experiments(self, experiment_ids : Tuple[int, int]):
        """Compare results between two experiments"""
        idx_a, idx_b = experiment_ids
        results_a = self.results_details.filter(pl.col('experiment_id')==idx_a)
        results_b = self.results_details.filter(pl.col('experiment_id')==idx_b)
        comparison = compare_results_stats(results_a, results_b, minimize = self.minimize)
        log_performance_against(comparison = comparison, n_folds_threshold = self.n_folds_threshold)

    # ------------------------------------------------------------------------------------------
    # MULTI-EXPERIMENTS
    # ------------------------------------------------------------------------------------------
    
    def multi_run_experiment(
        self,
        pipelines: List[Pipeline],
        eval_overfitting : bool = True, 
        store_preds : bool = True, 
        verbose : bool = True,
        compare_against: int | None = None, 
        auto_mode : bool = False
    ):
        """Run multiple experiments with a single command."""
        logging.info(f'{BOLD}{BLUE}Total Number of Experiments to run: {len(pipelines)}{RESET}')

        for i, pipeline in enumerate(pipelines):
            
            with log_step(f'{BOLD}{BLUE}Experiment {i+1} with name = {pipeline.name}{RESET}', verbose):

                self.run_experiment(
                    pipeline=pipeline, 
                    eval_overfitting=eval_overfitting, 
                    store_preds=store_preds,
                    verbose=verbose,
                    compare_against=compare_against, 
                    auto_mode=auto_mode
                )


    def run_base_experiments(
        self, 
        features: str, 
        preprocess_pipe : Pipeline | None = None, 
        eval_overfitting: bool = True, 
        store_preds: bool = True, 
        verbose: bool = True,
        compare_against: int | None = None,
        problem_type: str = 'regression'  # 'regression' or 'classification'
    ):
        """Run multiple experiments by using base estimators"""

        from sklearn.pipeline import Pipeline as SKlearnPipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Determine if classification or regression
        is_classification = problem_type.lower() == 'classification'
        
        if is_classification:
            # Classification imports
            from sklearn.linear_model import LogisticRegression as log_reg
            from sklearn.svm import SVC as svc
            from sklearn.neighbors import KNeighborsClassifier as knn
            from sklearn.ensemble import RandomForestClassifier as rf
            from lightgbm import LGBMClassifier as lgb
            from xgboost import XGBClassifier as xgb
            from catboost import CatBoostClassifier as ctb
            from sklearn.neural_network import MLPClassifier as mlp
            from sklearn.tree import DecisionTreeClassifier as dtree
            from sklearn.ensemble import HistGradientBoostingClassifier as hgb
            
            estimators = {
                'logistic_regression_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('clf', log_reg(max_iter=1000))
                ]),
                'knn_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('clf', knn())
                ]),
                'svm_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('clf', svc())
                ]),
                'random_forest_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('clf', rf(random_state=0, n_jobs=-1))
                ]),
                'decision_tree_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('clf', dtree())
                ]),
                'lightgbm_base': lgb(verbose=-1, random_state=0),
                'xgboost_base': xgb(verbosity=0, random_state=0),
                'catboost_base': ctb(verbose=0, random_state=0), 
                'hgb_base' : hgb(),
                'mlp_base' : SKlearnPipeline([
                    ('imputer', SimpleImputer()), 
                    ('scaler', StandardScaler()),
                    ('clf', mlp(hidden_layer_sizes = (64,32)))
                ]),
            }
            wrapper_class = ClassifierWrapper  
            
        else:
            # Regression imports
            from sklearn.linear_model import LinearRegression as lr
            from sklearn.svm import SVR as svr
            from sklearn.neighbors import KNeighborsRegressor as knn
            from sklearn.ensemble import RandomForestRegressor as rf
            from lightgbm import LGBMRegressor as lgb
            from xgboost import XGBRegressor as xgb
            from catboost import CatBoostRegressor as ctb
            from sklearn.neural_network import MLPRegressor as mlp
            from sklearn.tree import DecisionTreeRegressor as dtree
            from sklearn.ensemble import HistGradientBoostingRegressor as hgb
            
            estimators = {
                'linear_regression_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('reg', lr())
                ]),
                'knn_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('reg', knn())
                ]),
                'svm_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('reg', svr())
                ]),
                'random_forest_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('reg', rf(random_state=0, n_jobs=-1))
                ]),
                'decision_tree_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('reg', dtree())
                ]),
                'lightgbm_base': lgb(verbose=-1),
                'xgboost_base': xgb(verbosity=0),
                'catboost_base': ctb(verbose=0), 
                'hgb_base' : hgb(),
                'mlp_base' : SKlearnPipeline([
                    ('imputer', SimpleImputer()), 
                    ('scaler', StandardScaler()),
                    ('reg', mlp(hidden_layer_sizes = (64,32)))
                ]),
            }
            wrapper_class = RegressorWrapper

        # Create pipelines
        if preprocess_pipe:
            pipes = [
                Pipeline([
                    ('preprocess', preprocess_pipe),
                    ('model', wrapper_class(estimator=estimator, features=features, target=self.target))
                ], 
                name=name, 
                description=f'{name} with features = {features}'
                )
                for name, estimator in estimators.items()
            ]
        else:
            pipes = [
                Pipeline([
                    ('model', wrapper_class(estimator=estimator, features=features, target=self.target))
                ], 
                name=name, 
                description=f'{name} with features = {features}'
                )
                for name, estimator in estimators.items()
            ]


        self.multi_run_experiment(
            pipelines=pipes, 
            eval_overfitting=eval_overfitting, 
            store_preds=store_preds, 
            verbose=verbose, 
            compare_against=compare_against
        )


    # ------------------------------------------------------------------------------------------
    # HPO
    # ------------------------------------------------------------------------------------------

    def hpo(
        self, 
        features : List[str], 
        params_list : Dict[str, List[float | int | str]], 
        estimator : SKlearnEstimator, 
        preprocessor : Pipeline | BaseTransformer = Identity(), 
        eval_overfitting : bool = True, 
        store_preds : bool = True, 
        verbose : bool = True,
        compare_against: int | None = None, 
        search_type : str = 'grid', 
        num_samples : int = 64, 
        random_state : int = 0
    ):
        """Create an HPO procedure for a sklearn estimator"""
        
        pipelines = [
                Pipeline(steps=[
                ('preprocessor', preprocessor), 
                ('estimator', RegressorWrapper(estimator = estimator(**p), features=features, target=self.target))
            ], 
            name = f'{p}', 
            description = f'hpo with params = {p}'
            )
        for p in generate_params_list(
            params_list=params_list, 
            search_type=search_type, 
            num_samples=num_samples, 
            random_state=random_state
        )
        ]

        self.multi_run_experiment(
            pipelines=pipelines, 
            eval_overfitting=eval_overfitting, 
            store_preds=store_preds, 
            verbose=verbose, 
            compare_against=compare_against
        )


    # ------------------------------------------------------------------------------------------
    # RETRIEVE PIPELINE PREDICTIONS
    # ------------------------------------------------------------------------------------------

    def retrieve_predictions(self, experiment_ids = List[int], extra_features : List[str] = []) -> pl.LazyFrame:
        """
        Retrieve the predictions made by all the pipelines indicated into experiment_ids
        """
        base_preds = pl.concat([
                pl.LazyFrame(self.cv_indexes[j][1], schema=[self.row_id]).with_columns(pl.lit(j+1).alias('fold_number')) 
                for j in range(len(self.cv_indexes))
            ], 
            how = 'vertical_relaxed'
        )

        base_preds = base_preds.join(self.train.select([self.row_id, self.target] + extra_features), how = 'left', on = self.row_id)

        preds = (
            base_preds
            .with_columns(
                retrieve_predictions_from_path(lab_name=self.name, experiment_id=idx) 
                for idx in experiment_ids
            )
        )

        return preds
    

    def compute_pvalue(self, experiment_ids : Tuple[int, int], n_iters : int = 200) -> float:

        if not(isinstance(experiment_ids, tuple)):
            raise ValueError('experiment_ids should be a tuple containing two and only two experiment ids.')
        
        elif len(experiment_ids)!=2:
            raise ValueError('experiment_ids should be a tuple containing two and only two experiment ids.')
        
        else:
            preds = self.retrieve_predictions(experiment_ids=list(experiment_ids))
            idx_1, idx_2 = experiment_ids
            obs_anomaly = compute_anomaly(metric=self.metric, lf=preds, preds_1=f'preds_{idx_1}', preds_2=f'preds_{idx_2}', target=self.target)

            sim_anomaly = [
                compute_anomaly(
                    metric=self.metric, 
                    lf=generate_shuffle_preds(lf=preds, preds_1=f'preds_{idx_1}', preds_2=f'preds_{idx_2}', random_state=i), 
                    preds_1='shuffle_a', 
                    preds_2='shuffle_b',
                    target=self.target
                ) 
                for i in range(n_iters)
            ]

            r = (np.array(sim_anomaly) > obs_anomaly).sum() # number of times we found a result more "extreme" that the one we observed
            pvalue = (r+1)/(n_iters+1)  # empirical pvalue

            return pvalue
        
    def permutation_feature_importance(
            self, 
            pipeline : Pipeline, 
            features : List[str],
            n_iters : int = 5, 
            verbose : bool = True      
    ) -> pl.DataFrame:
        """Compute for every single features the permutation feature importance."""

        pfi_dfs = []
        for fold, (train_idx, valid_idx) in enumerate(self.cv_indexes):

            with log_step(f'Fold {fold+1}', verbose):

                train = self.train.filter(pl.col(self.row_id).is_in(train_idx))
                valid = self.train.filter(pl.col(self.row_id).is_in(valid_idx))

                pipeline.fit(train)

                valid = valid.with_columns(pl.Series(pipeline.predict(valid)).alias('base_preds'))

                pfi = {}
                for f in features:
                    shadow = (
                        valid
                        .with_columns(pl.Series(pipeline.predict(valid.with_columns(pl.col(f).sample(fraction=1, seed=j, shuffle=True)))).alias(f'shadow_{j}') for j in range(n_iters))
                    )

                    base_metric = self.metric.compute_metric(shadow, target='prob_1', preds='base_preds')
                    shadow_metric = np.array([self.metric.compute_metric(shadow, target=self.target, preds=f'shadow_{j}') for j in range(n_iters)]).mean()

                    pfi[f] = relative_performance(minimize = False, x1=base_metric, x2=shadow_metric)

                pfi_dfs.append(pl.DataFrame(pfi).with_columns(pl.lit(fold+1).alias('fold_number')))

        return pl.concat(pfi_dfs)
    

    def save_check_point(self, check_point_name : str | None = None) -> None:
        """Save check point of the lab"""

        check_point_name_ref = check_point_name if check_point_name else str(int(time.time()))
        pickle.dump(self, open(f'./{self.name}/check_points/{check_point_name_ref}.pkl', 'wb'))



def restore_check_point(lab_name : str, check_point_name : str) -> Lab:
    """Restore a lab checkpoint."""
    return pickle.load(open(f'./{lab_name}/check_points/{check_point_name}.pkl', 'rb'))