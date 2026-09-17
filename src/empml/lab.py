"""
Machine learning experimentation framework for systematic model evaluation and comparison.

Provides Lab class for running experiments with cross-validation, tracking results,
hyperparameter optimization, and statistical comparison of models.
"""

# --- Logging Setup ---
import logging
import time
import uuid
from dataclasses import dataclass

import polars as pl

from empml.artifacts import LabArtifacts
from empml.base import (
    BaseTransformer,
    CVGenerator,
    DataDownloader,
    Metric,
    SKlearnEstimator,
)
from empml.baselines import baseline_pipelines
from empml.comparison import (
    compare_experiments,
    is_improvement,
    permutation_pvalue,
    print_comparison_report,
    print_test_report,
)
from empml.errors import RunExperimentConfigException, RunExperimentOnTestException
from empml.feature_selection import (
    features_without_importance,
)
from empml.feature_selection import (
    permutation_feature_importance as compute_permutation_importance,
)
from empml.hpo import (
    build_hpo_pipelines,
    generate_params_list,
    select_hpo_best,
    validate_primary_metric_idx,
)
from empml.lab_utils import prepare_predictions_for_save, setup_row_id_column
from empml.pipeline import Pipeline, evaluate_cv, evaluate_fold
from empml.results import (
    MetricColumns,
    format_details_rows,
    format_results_row,
    results_details_schema,
    results_schema,
)
from empml.transformers import Identity
from empml.utils import BLUE, BOLD, RED, RESET, log_execution_time, log_step
from empml.wrappers import SKlearnWrapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


@dataclass
class ComparisonCriteria:
    """
    Statistical criteria for comparing experiment performance.

    Choose either percentage threshold OR statistical testing approach.
    """

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


def _minimize_per_metric(
    minimize: bool | list[bool], n_metrics: int, multi_metric: bool
) -> list[bool]:
    """Expand ``minimize`` to one flag per metric."""
    if not multi_metric:
        return [minimize] if isinstance(minimize, bool) else minimize
    if not isinstance(minimize, list):
        return [minimize] * n_metrics
    if len(minimize) != n_metrics:
        raise ValueError(f"len(minimize)={len(minimize)} != len(metric)={n_metrics}")
    return minimize


# ------------------------------------------------------------------------------------------
# Lab Class
# ------------------------------------------------------------------------------------------


class Lab:
    """
    Experimentation framework for ML model development and evaluation.

    Manages experiment lifecycle: data loading, CV splitting, pipeline execution,
    results tracking, and statistical comparison. Supports HPO and feature selection.
    """

    # Instance attributes are pickled into checkpoints. Keep their names stable and
    # derive helpers (columns, artifacts) from them instead of storing new state.

    # ------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------

    def __init__(
        self,
        train_downloader: DataDownloader,
        metric: Metric | list[Metric],
        cv_generator: CVGenerator,
        target: str,
        comparison_criteria: ComparisonCriteria,
        minimize: bool | list[bool] = True,
        row_id: str | None = None,
        test_downloader: DataDownloader | None = None,
        name: str | None = None,
    ):
        """
        Initialize Lab with data, evaluation metric(s), and CV strategy.

        Args:
            train_downloader: Source for training data
            metric: Single metric or list of metrics for evaluation
            cv_generator: Cross-validation splitting strategy
            target: Name of target column
            comparison_criteria: Criteria for experiment comparison
            minimize: Whether to minimize metric(s). Single bool or
                list of bools matching length of metric list.
            row_id: Column name for row identifier
            test_downloader: Optional test data source
            name: Lab identifier (auto-generated if None)
        """
        self.name = name or uuid.uuid1().hex[:8]
        self.cv_generator = cv_generator
        self.target = target

        self.train_downloader = train_downloader
        self.test_downloader = test_downloader

        # A single metric is stored as a one-element list; its columns stay unsuffixed.
        self._multi_metric = isinstance(metric, list)
        self.metrics: list[Metric] = metric if isinstance(metric, list) else [metric]
        self.n_metrics = len(self.metrics)
        self.metric = self.metrics[0]
        self.minimize_list = _minimize_per_metric(
            minimize, self.n_metrics, self._multi_metric
        )
        self.minimize = self.minimize_list[0]

        self._artifacts.create_directories()
        self.train = train_downloader.get_data()
        self.test = test_downloader.get_data() if test_downloader else None
        self.train, self.row_id = setup_row_id_column(self.train, row_id)
        self.results = results_schema(self._columns)
        self.results_details = results_details_schema(self._columns)

        self.cv_indexes = self.cv_generator.split(self.train, self.row_id)
        self.n_folds = len(self.cv_indexes)

        self.n_folds_threshold = comparison_criteria.n_folds_threshold
        self.pct_threshold = comparison_criteria.pct_threshold
        self.alpha = comparison_criteria.alpha
        self.n_iters = comparison_criteria.n_iters
        self.eval_has_pct = comparison_criteria.has_pct
        self.eval_has_statistical = not comparison_criteria.has_pct

        self.next_experiment_id = 1
        self._set_best_experiment()

    @property
    def _columns(self) -> MetricColumns:
        return MetricColumns(n_metrics=self.n_metrics, suffixed=self._multi_metric)

    @property
    def _artifacts(self) -> LabArtifacts:
        return LabArtifacts(self.name)

    def _set_best_experiment(self, experiment_id: int | None = None):
        """Set or clear best experiment tracker."""
        self.best_experiment = experiment_id

    def _details_of(self, experiment_id: int) -> pl.DataFrame:
        return self.results_details.filter(pl.col("experiment_id") == experiment_id)

    # ------------------------------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------------------------------

    def run_experiment(
        self,
        pipeline: Pipeline,
        eval_overfitting: bool = True,
        store_preds: bool = True,
        verbose: bool = True,
        compare_against: int | None = None,
        auto_mode: bool = False,
    ):
        """
        Execute pipeline evaluation with CV and track results.

        Args:
            pipeline: Pipeline to evaluate
            eval_overfitting: Whether to check train/valid gap
            store_preds: Whether to save predictions
            verbose: Enable detailed logging
            compare_against: Experiment ID to compare against
            auto_mode: Auto-update best experiment if improvement found
        """
        if auto_mode and not (self.best_experiment):
            raise RunExperimentConfigException(
                "Select a best experiment before using auto_mode."
            )

        if auto_mode:
            logging.info("Auto mode: comparing against current best.")
            compare_against = self.best_experiment

        eval = evaluate_cv(
            pipeline=pipeline,
            lz=self.train,
            cv_indexes=self.cv_indexes,
            row_id=self.row_id,
            metrics=self.metrics,
            target=self.target,
            minimize=self.minimize_list,
            columns=self._columns,
            eval_overfitting=eval_overfitting,
            store_preds=store_preds,
            verbose=verbose,
            compare_df=(self._details_of(compare_against) if compare_against else None),
            max_worse_folds=self.n_folds_threshold,
        )
        is_completed = eval.shape[0] == self.n_folds

        self._record_results(eval, pipeline, is_completed)
        self._artifacts.save_pipeline(pipeline, self.next_experiment_id)
        self._save_predictions(eval=eval)

        if compare_against and is_completed:
            comparisons = compare_experiments(
                self._details_of(compare_against),
                self._details_of(self.next_experiment_id),
                self.minimize_list,
                self._columns,
            )
            print_comparison_report(comparisons, self.n_folds_threshold, self._columns)

            if auto_mode:
                experiment_ids = (compare_against, self.next_experiment_id)
                if self._is_improvement(comparisons, experiment_ids):
                    self.best_experiment = self.next_experiment_id
                logging.info(
                    f"{BLUE}{BOLD}BEST EXPERIMENT UPDATED: {self.best_experiment}{RESET}"
                )

        elif not is_completed:
            logging.info(
                f"{BOLD}{RED}Experiment arrested: no improvement over baseline.{RESET}"
            )

        self.next_experiment_id += 1

    def _record_results(
        self, eval: pl.DataFrame, pipeline: Pipeline, is_completed: bool
    ) -> None:
        """Append the experiment's summary row and per-fold rows."""
        summary = format_results_row(
            eval,
            self.next_experiment_id,
            is_completed,
            self._columns,
            pipeline.description,
            pipeline.name,
        )
        details = format_details_rows(eval, self.next_experiment_id, self._columns)
        self.results = pl.concat(
            [self.results, summary.select(self.results.columns)],
            how="vertical_relaxed",
        )
        self.results_details = pl.concat(
            [self.results_details, details.select(self.results_details.columns)],
            how="vertical_relaxed",
        )

    def _validation_keys(self) -> list[pl.LazyFrame]:
        """Row IDs and fold numbers of each validation fold, in prediction order."""
        # Filtering preserves the source order used when each prediction was made.
        return [
            self.train.filter(pl.col(self.row_id).is_in(valid_idx))
            .select(self.row_id)
            .with_columns(pl.lit(fold_number).alias("fold_number"))
            for fold_number, (_, valid_idx) in enumerate(self.cv_indexes, 1)
        ]

    @log_execution_time
    def _save_predictions(self, eval: pl.DataFrame):
        """Save predictions as compressed parquet."""
        validation_keys = [keys.collect() for keys in self._validation_keys()]
        preds = prepare_predictions_for_save(eval, validation_keys, self.row_id)
        self._artifacts.save_predictions(preds, self.next_experiment_id)

    def _is_improvement(
        self, comparisons: list[dict], experiment_ids: tuple[int, int]
    ) -> bool:
        """Whether the second experiment beats the first under the comparison criteria."""
        if self.eval_has_pct:
            return is_improvement(
                comparisons, self.n_folds_threshold, self.pct_threshold
            )
        return is_improvement(
            comparisons,
            self.n_folds_threshold,
            min_improvement_pct=0,
            pvalues=self._pvalues(experiment_ids, self.n_iters),
            alpha=self.alpha,
        )

    def multi_run_experiment(
        self,
        pipelines: list[Pipeline],
        eval_overfitting: bool = True,
        store_preds: bool = True,
        verbose: bool = True,
        compare_against: int | None = None,
        auto_mode: bool = False,
    ):
        """Execute multiple experiments sequentially."""
        logging.info(f"{BOLD}{BLUE}Total experiments: {len(pipelines)}{RESET}")

        for i, pipeline in enumerate(pipelines):
            with log_step(
                f"{BOLD}{BLUE}Experiment {i + 1}: {pipeline.name}{RESET}", verbose
            ):
                self.run_experiment(
                    pipeline=pipeline,
                    eval_overfitting=eval_overfitting,
                    store_preds=store_preds,
                    verbose=verbose,
                    compare_against=compare_against,
                    auto_mode=auto_mode,
                )

    def run_base_experiments(
        self,
        features: str,
        preprocess_pipe: Pipeline | None = None,
        eval_overfitting: bool = True,
        store_preds: bool = True,
        verbose: bool = True,
        compare_against: int | None = None,
        problem_type: str = "regression",
    ):
        """
        Run suite of baseline models for quick benchmarking.

        Args:
            features: Feature columns to use
            preprocess_pipe: Optional preprocessing pipeline
            problem_type: 'regression' or 'classification'
        """
        self.multi_run_experiment(
            pipelines=baseline_pipelines(
                problem_type, features, self.target, preprocess_pipe
            ),
            eval_overfitting=eval_overfitting,
            store_preds=store_preds,
            verbose=verbose,
            compare_against=compare_against,
        )

    # ------------------------------------------------------------------------------------------
    # HPO
    # ------------------------------------------------------------------------------------------

    def hpo(
        self,
        features: list[str],
        params_list: dict[str, list[float | int | str]],
        estimator: SKlearnEstimator,
        preprocessor: Pipeline | BaseTransformer = Identity(),
        eval_overfitting: bool = True,
        store_preds: bool = True,
        verbose: bool = True,
        compare_against: int | None = None,
        search_type: str = "grid",
        num_samples: int = 64,
        random_state: int = 0,
        primary_metric_idx: int | str = "all",
    ):
        """
        Hyperparameter optimization via grid or random search.

        Args:
            features: Features to use in model
            params_list: Parameter grid/ranges
            estimator: sklearn-compatible estimator class
            preprocessor: Optional preprocessing step
            search_type: 'grid' or 'random'
            num_samples: Number of random samples (random search)
            primary_metric_idx: For multi-metric, which metric
                to sort by (0-indexed int) or 'all' to require
                improvement on every metric. Ignored for single
                metric.
        """
        if self._multi_metric:
            validate_primary_metric_idx(primary_metric_idx, self.n_metrics)

        params = generate_params_list(
            params_list=params_list,
            search_type=search_type,
            num_samples=num_samples,
            random_state=random_state,
        )
        pipelines = build_hpo_pipelines(
            estimator, params, features, self.target, preprocessor
        )

        self.multi_run_experiment(
            pipelines=pipelines,
            eval_overfitting=eval_overfitting,
            store_preds=store_preds,
            verbose=verbose,
            compare_against=compare_against,
        )

        hpo_results = select_hpo_best(
            results=self.results,
            n_experiments=len(pipelines),
            minimize=self.minimize_list,
            columns=self._columns,
            primary_metric_idx=primary_metric_idx,
            compare_against=compare_against,
        )
        if hpo_results is not None:
            hpo_results["best_pipeline"] = self.retrieve_pipeline(
                experiment_id=hpo_results["experiment_id"]
            )
        return hpo_results

    # ------------------------------------------------------------------------------------------
    # Predictions and statistical tests
    # ------------------------------------------------------------------------------------------

    def retrieve_predictions(
        self, experiment_ids=list[int], extra_features: list[str] = []
    ) -> pl.LazyFrame:
        """
        Load predictions from specified experiments.

        Returns LazyFrame with row_id, fold, target, and predictions from each experiment.
        """
        base_preds = pl.concat(
            self._validation_keys(), how="vertical_relaxed"
        ).with_row_index("_prediction_order")

        preds = base_preds.join(
            self.train.select([self.row_id, self.target] + extra_features),
            how="left",
            on=self.row_id,
        )
        for idx in experiment_ids:
            stored = self._artifacts.load_predictions(idx)
            prediction_col = f"preds_{idx}"
            if {self.row_id, "fold_number"}.issubset(stored.columns):
                preds = preds.join(
                    stored.lazy(),
                    how="left",
                    on=[self.row_id, "fold_number"],
                )
            else:
                # Legacy artifacts hold only predictions, in source-filter order.
                legacy = stored.with_row_index("_prediction_order").lazy()
                preds = preds.join(legacy, how="left", on="_prediction_order")
            if prediction_col not in preds.collect_schema().names():
                preds = preds.with_columns(pl.lit(None).alias(prediction_col))

        return preds.drop("_prediction_order")

    def compute_pvalue(
        self,
        experiment_ids: tuple[int, int],
        n_iters: int = 200,
        extra_features: list[str] = [],
    ) -> float | list[float]:
        """
        Compute permutation test p-value(s) comparing two experiments.

        Returns single float for single-metric, list of floats
        for multi-metric (one p-value per metric).
        """
        if not isinstance(experiment_ids, tuple) or len(experiment_ids) != 2:
            raise ValueError("experiment_ids must be tuple of two IDs.")

        pvalues = self._pvalues(experiment_ids, n_iters, extra_features)
        return pvalues if self._multi_metric else pvalues[0]

    def _pvalues(
        self,
        experiment_ids: tuple[int, int],
        n_iters: int,
        extra_features: list[str] = [],
    ) -> list[float]:
        """One permutation-test p-value per metric."""
        preds = self.retrieve_predictions(
            experiment_ids=list(experiment_ids),
            extra_features=extra_features,
        )
        idx_1, idx_2 = experiment_ids
        return [
            permutation_pvalue(
                metric, preds, f"preds_{idx_1}", f"preds_{idx_2}", self.target, n_iters
            )
            for metric in self.metrics
        ]

    # ------------------------------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------------------------------

    def permutation_feature_importance(
        self,
        pipeline: Pipeline,
        features: list[str],
        n_iters: int = 5,
        verbose: bool = True,
    ) -> pl.DataFrame:
        """
        Compute permutation feature importance for each feature.

        Measures performance drop when feature is randomly shuffled.
        Returns DataFrame with importance scores per fold.

        Note:
            For multi-metric Labs, uses the first metric only.
        """
        return compute_permutation_importance(
            pipeline=pipeline,
            data=self.train,
            cv_indexes=self.cv_indexes,
            row_id=self.row_id,
            features=features,
            metric=self.metric,
            target=self.target,
            minimize=self.minimize,
            n_iters=n_iters,
            verbose=verbose,
        )

    def recursive_permutation_feature_selection(
        self,
        estimator: SKlearnEstimator,
        features: list[str],
        preprocessor: Pipeline | BaseTransformer = Identity(),
        n_iters: int = 5,
        verbose: bool = True,
    ) -> list[str]:
        """
        Recursively eliminate features with negative importance.

        Returns list of selected features after iterative removal.
        """
        while True:
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "estimator",
                        SKlearnWrapper(
                            estimator=estimator, features=features, target=self.target
                        ),
                    ),
                ]
            )
            pfi = self.permutation_feature_importance(
                pipeline=pipeline, features=features, n_iters=n_iters, verbose=verbose
            )
            features_to_drop = features_without_importance(pfi)
            if not features_to_drop:
                logging.info("No features eliminated.")
                return features

            features = [f for f in features if f not in features_to_drop]
            logging.info(f"Features eliminated: {features_to_drop}")

    # ------------------------------------------------------------------------------------------
    # Test set, retrieval, and checkpoints
    # ------------------------------------------------------------------------------------------

    def run_experiment_on_test(
        self,
        experiment_id: int,
        eval_overfitting: bool = True,
        store_preds: bool = True,
        verbose: bool = True,
    ) -> dict[str, float | list[float]]:
        """Evaluate experiment pipeline on the test set.

        For multi-metric Labs, evaluates and prints stats
        for each metric independently.
        """
        if not self.test_downloader:
            raise RunExperimentOnTestException(
                "No test set detected. Please provide one "
                "through a downloader data class before "
                "final pipeline evaluation."
            )

        test_results = evaluate_fold(
            pipeline=self.retrieve_pipeline(experiment_id),
            train=self.train,
            valid=self.test,
            metrics=self.metrics,
            target=self.target,
            minimize=self.minimize_list,
            columns=self._columns,
            eval_overfitting=eval_overfitting,
            store_preds=store_preds,
            verbose=verbose,
        )
        cv_results = self.results.filter(pl.col("experiment_id") == experiment_id)
        print_test_report(cv_results, test_results, self.minimize_list, self._columns)
        return test_results

    def retrieve_pipeline(self, experiment_id: int) -> Pipeline:
        """Retrieve a pipeline related to an experiment"""
        return self._artifacts.load_pipeline(experiment_id)

    def show_best_score(self, metric_idx: int | None = None) -> pl.DataFrame:
        """Show stats for the experiment with the best score.

        Args:
            metric_idx: For multi-metric Labs, which metric
                to sort by (0-indexed). Defaults to first.
        """
        index = (metric_idx or 0) if self._multi_metric else 0
        return (
            self.results.filter(pl.col("is_completed"))
            .sort(
                self._columns.name("cv_mean_score", index + 1),
                descending=self.minimize_list[index],
            )
            .tail(1)
        )

    def save_check_point(self, check_point_name: str | None = None) -> None:
        """Serialize current lab state to disk."""
        self._artifacts.save_checkpoint(
            self, check_point_name if check_point_name else str(int(time.time()))
        )


def restore_check_point(lab_name: str, check_point_name: str) -> Lab:
    """Load saved lab state from checkpoint."""
    return LabArtifacts(lab_name).load_checkpoint(check_point_name)
