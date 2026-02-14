"""Tests for multi-metric support in Lab, pipeline, and lab_utils.

Covers:
- Schema creation for multi-metric results
- Formatting of multi-metric experiment results and details
- Multi-metric pipeline evaluation (single fold and CV)
- Multi-metric comparison between experiments
- Lab class initialization and dispatch logic
- HPO primary_metric_idx validation
- show_best_score with metric_idx
"""

import numpy as np
import polars as pl
import pytest

from empml.base import CVGenerator, DataDownloader
from empml.lab import ComparisonCriteria, Lab
from empml.lab_utils import (
    create_results_details_schema_multi,
    create_results_schema,
    create_results_schema_multi,
    format_experiment_details_multi,
    format_experiment_results_multi,
)
from empml.metrics import MAE, MSE
from empml.pipeline import (
    Pipeline,
    compare_results_stats_multi,
    compute_scores,
    eval_pipeline_single_fold_multi,
)
from empml.wrappers import SKlearnWrapper

# ------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------


class MockDownloader(DataDownloader):
    """Return a small dataset for testing."""

    def __init__(self, lf: pl.LazyFrame):
        self._lf = lf

    def get_data(self) -> pl.LazyFrame:
        return self._lf


class MockCVGenerator(CVGenerator):
    """Simple 2-fold CV split by row_id parity."""

    def split(self, lf, row_id):
        ids = lf.select(row_id).collect()[row_id].to_list()
        mid = len(ids) // 2
        fold_1_train = np.array(ids[mid:])
        fold_1_valid = np.array(ids[:mid])
        fold_2_train = np.array(ids[:mid])
        fold_2_valid = np.array(ids[mid:])
        return [
            (fold_1_train, fold_1_valid),
            (fold_2_train, fold_2_valid),
        ]


def _make_train_lf() -> pl.LazyFrame:
    """Create a small training LazyFrame."""
    np.random.seed(42)
    n = 20
    f1 = np.random.randn(n).tolist()
    f2 = np.random.randn(n).tolist()
    target = [f1[i] * 2 + f2[i] + np.random.randn() * 0.1 for i in range(n)]
    return pl.LazyFrame(
        {
            "f1": f1,
            "f2": f2,
            "target": target,
        }
    )


def _make_lab(
    multi_metric: bool = True,
    minimize: bool | list = True,
) -> Lab:
    """Create a Lab instance for testing."""
    lf = _make_train_lf()
    downloader = MockDownloader(lf)
    cv = MockCVGenerator()
    criteria = ComparisonCriteria(n_folds_threshold=1, pct_threshold=0.0)
    if multi_metric:
        metric = [MSE(), MAE()]
        if isinstance(minimize, bool):
            minimize = [minimize, minimize]
    else:
        metric = MSE()
    return Lab(
        train_downloader=downloader,
        metric=metric,
        cv_generator=cv,
        target="target",
        comparison_criteria=criteria,
        minimize=minimize,
    )


def _make_simple_pipeline(lab: Lab) -> Pipeline:
    """Create a simple linear regression pipeline."""
    from sklearn.linear_model import LinearRegression

    return Pipeline(
        steps=[
            (
                "model",
                SKlearnWrapper(
                    estimator=LinearRegression(),
                    features=["f1", "f2"],
                    target=lab.target,
                ),
            ),
        ],
        name="lr_test",
        description="test pipeline",
    )


# ------------------------------------------------------------------
# Unit tests: lab_utils.py
# ------------------------------------------------------------------


class TestMultiMetricSchemas:
    def test_create_results_schema_multi_columns(self):
        """Verify multi-metric results schema has correct columns."""
        schema = create_results_schema_multi(2)
        cols = schema.columns
        # Per-metric columns
        assert "cv_mean_score_1" in cols
        assert "cv_mean_score_2" in cols
        assert "train_mean_score_1" in cols
        assert "train_mean_score_2" in cols
        assert "mean_overfitting_pct_1" in cols
        assert "cv_std_score_1" in cols
        assert "cv_std_score_2" in cols
        # Shared columns
        assert "experiment_id" in cols
        assert "mean_train_time_s" in cols
        assert "is_completed" in cols
        assert "timestamp_utc" in cols

    def test_create_results_details_schema_multi_columns(self):
        """Verify multi-metric details schema has correct columns."""
        schema = create_results_details_schema_multi(2)
        cols = schema.columns
        assert "validation_score_1" in cols
        assert "validation_score_2" in cols
        assert "train_score_1" in cols
        assert "overfitting_pct_1" in cols
        assert "experiment_id" in cols
        assert "fold_number" in cols

    def test_single_metric_schema_unchanged(self):
        """Verify single-metric schema is not affected."""
        schema = create_results_schema()
        cols = schema.columns
        assert "cv_mean_score" in cols
        assert "cv_mean_score_1" not in cols

    def test_create_results_schema_multi_3_metrics(self):
        """Verify schema with 3 metrics has _1, _2, _3 columns."""
        schema = create_results_schema_multi(3)
        cols = schema.columns
        assert "cv_mean_score_3" in cols
        assert "cv_std_score_3" in cols


class TestMultiMetricFormatting:
    def test_format_experiment_results_multi(self):
        """Verify multi-metric results are aggregated correctly."""
        eval_df = pl.DataFrame(
            {
                "validation_score_1": [0.5, 0.6, 0.7],
                "validation_score_2": [1.0, 1.2, 0.8],
                "train_score_1": [0.3, 0.4, 0.5],
                "train_score_2": [0.7, 0.8, 0.6],
                "overfitting_1": [10.0, 8.0, 6.0],
                "overfitting_2": [5.0, 4.0, 3.0],
                "duration_train": [1.0, 1.1, 0.9],
                "duration_inf": [0.1, 0.12, 0.08],
                "preds": [[1.0], [2.0], [3.0]],
            }
        )
        result = format_experiment_results_multi(
            eval_df,
            experiment_id=1,
            is_completed=True,
            n_metrics=2,
            description="test",
            name="test",
        )
        assert result.height == 1
        assert "cv_mean_score_1" in result.columns
        assert "cv_mean_score_2" in result.columns
        assert "cv_std_score_1" in result.columns
        # Check mean computation
        assert abs(result["cv_mean_score_1"].item() - 0.6) < 1e-6
        assert abs(result["cv_mean_score_2"].item() - 1.0) < 1e-6

    def test_format_experiment_details_multi(self):
        """Verify multi-metric details are formatted correctly."""
        eval_df = pl.DataFrame(
            {
                "validation_score_1": [0.5, 0.6],
                "validation_score_2": [1.0, 1.2],
                "train_score_1": [0.3, 0.4],
                "train_score_2": [0.7, 0.8],
                "overfitting_1": [10.0, 8.0],
                "overfitting_2": [5.0, 4.0],
                "duration_train": [1.0, 1.1],
                "duration_inf": [0.1, 0.12],
                "preds": [[1.0], [2.0]],
            }
        )
        result = format_experiment_details_multi(
            eval_df,
            experiment_id=1,
            n_metrics=2,
        )
        assert result.height == 2
        assert "overfitting_pct_1" in result.columns
        assert "overfitting_pct_2" in result.columns
        assert "overfitting_1" not in result.columns
        # fold_number should be 1-indexed
        assert result["fold_number"].to_list() == [1, 2]


# ------------------------------------------------------------------
# Unit tests: pipeline.py
# ------------------------------------------------------------------


class TestMultiMetricPipeline:
    def test_compute_scores(self):
        """compute_scores returns a float per metric."""
        lf = pl.LazyFrame(
            {
                "target": [1.0, 2.0, 3.0],
            }
        )
        preds = np.array([1.1, 2.2, 2.8])
        scores = compute_scores(lf, preds, [MSE(), MAE()], "target")
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        # MSE should be different from MAE
        assert scores[0] != scores[1]

    def test_eval_single_fold_multi_keys(self):
        """eval_pipeline_single_fold_multi returns correct keys."""
        lab = _make_lab(multi_metric=True)
        pipe = _make_simple_pipeline(lab)

        train_idx, valid_idx = lab.cv_indexes[0]
        train = lab.train.filter(pl.col(lab.row_id).is_in(train_idx))
        valid = lab.train.filter(pl.col(lab.row_id).is_in(valid_idx))
        result = eval_pipeline_single_fold_multi(
            pipeline=pipe,
            train=train,
            valid=valid,
            metrics=[MSE(), MAE()],
            target="target",
            minimize=[True, True],
            verbose=False,
        )
        assert "validation_score_1" in result
        assert "validation_score_2" in result
        assert "train_score_1" in result
        assert "train_score_2" in result
        assert "overfitting_1" in result
        assert "overfitting_2" in result
        assert "preds" in result
        assert "duration_train" in result

    def test_compare_results_stats_multi(self):
        """compare_results_stats_multi returns one dict per metric."""
        a = pl.DataFrame(
            {
                "fold_number": [1, 2],
                "experiment_id": [1, 1],
                "validation_score_1": [0.5, 0.6],
                "train_score_1": [0.3, 0.4],
                "overfitting_pct_1": [10.0, 8.0],
                "validation_score_2": [1.0, 1.1],
                "train_score_2": [0.7, 0.8],
                "overfitting_pct_2": [5.0, 4.0],
            }
        )
        b = pl.DataFrame(
            {
                "fold_number": [1, 2],
                "experiment_id": [2, 2],
                "validation_score_1": [0.4, 0.5],
                "train_score_1": [0.2, 0.3],
                "overfitting_pct_1": [9.0, 7.0],
                "validation_score_2": [0.9, 1.0],
                "train_score_2": [0.6, 0.7],
                "overfitting_pct_2": [4.0, 3.0],
            }
        )
        comparisons = compare_results_stats_multi(
            a,
            b,
            minimize=[True, True],
            n_metrics=2,
        )
        assert len(comparisons) == 2
        for c in comparisons:
            assert "mean_cv_performance" in c
            assert "n_folds_lower_performance" in c
            assert "fold_performances" in c


# ------------------------------------------------------------------
# Integration tests: Lab class
# ------------------------------------------------------------------


class TestLabInit:
    def test_single_metric_unchanged(self):
        """Single metric Lab behaves exactly as before."""
        lab = _make_lab(multi_metric=False)
        assert lab._multi_metric is False
        assert lab.n_metrics == 1
        assert isinstance(lab.metric, MSE)
        assert "cv_mean_score" in lab.results.columns
        assert "cv_mean_score_1" not in lab.results.columns

    def test_multi_metric_init(self):
        """Multi metric Lab sets correct attributes."""
        lab = _make_lab(multi_metric=True)
        assert lab._multi_metric is True
        assert lab.n_metrics == 2
        assert len(lab.metrics) == 2
        assert len(lab.minimize_list) == 2
        assert "cv_mean_score_1" in lab.results.columns
        assert "cv_mean_score_2" in lab.results.columns

    def test_minimize_mismatch_raises(self):
        """Mismatched minimize list length raises ValueError."""
        lf = _make_train_lf()
        with pytest.raises(ValueError, match="len\\(minimize\\)"):
            Lab(
                train_downloader=MockDownloader(lf),
                metric=[MSE(), MAE()],
                cv_generator=MockCVGenerator(),
                target="target",
                comparison_criteria=ComparisonCriteria(
                    n_folds_threshold=1,
                    pct_threshold=0.0,
                ),
                minimize=[True, True, True],
            )

    def test_minimize_bool_broadcast(self):
        """Single bool minimize is broadcast to all metrics."""
        lab = _make_lab(multi_metric=True, minimize=True)
        assert lab.minimize_list == [True, True]


class TestLabRunExperiment:
    def test_run_experiment_multi_metric(self):
        """Run experiment with multi-metric updates results."""
        lab = _make_lab(multi_metric=True)
        pipe = _make_simple_pipeline(lab)
        lab.run_experiment(pipe, verbose=False)

        assert lab.results.height == 1
        row = lab.results.row(0, named=True)
        assert "cv_mean_score_1" in row
        assert "cv_mean_score_2" in row
        assert row["cv_mean_score_1"] is not None
        assert row["cv_mean_score_2"] is not None

    def test_run_experiment_single_unchanged(self):
        """Single metric run_experiment still works identically."""
        lab = _make_lab(multi_metric=False)
        pipe = _make_simple_pipeline(lab)
        lab.run_experiment(pipe, verbose=False)

        assert lab.results.height == 1
        row = lab.results.row(0, named=True)
        assert "cv_mean_score" in row
        assert row["cv_mean_score"] is not None

    def test_run_experiment_details_multi_metric(self):
        """Multi-metric details table has correct structure."""
        lab = _make_lab(multi_metric=True)
        pipe = _make_simple_pipeline(lab)
        lab.run_experiment(pipe, verbose=False)

        details = lab.results_details
        assert details.height == 2  # 2 folds
        assert "validation_score_1" in details.columns
        assert "validation_score_2" in details.columns
        assert "overfitting_pct_1" in details.columns

    def test_multi_experiment_comparison(self):
        """Two multi-metric experiments can be compared."""
        lab = _make_lab(multi_metric=True)
        pipe1 = _make_simple_pipeline(lab)
        pipe2 = _make_simple_pipeline(lab)
        pipe2.name = "lr_test_2"

        lab.run_experiment(pipe1, verbose=False)
        lab.run_experiment(pipe2, verbose=False, compare_against=1)

        assert lab.results.height == 2


class TestLabComparison:
    def test_best_experiment_all_better(self):
        """Best experiment updated when B better on all metrics."""
        lab = _make_lab(multi_metric=True)
        pipe = _make_simple_pipeline(lab)
        lab.run_experiment(pipe, verbose=False)
        lab._set_best_experiment(1)

        # Run same pipeline again (should be comparable)
        pipe2 = _make_simple_pipeline(lab)
        pipe2.name = "lr_2"
        lab.run_experiment(
            pipe2,
            verbose=False,
            compare_against=1,
            auto_mode=True,
        )
        # Best should still be set (either 1 or 2)
        assert lab.best_experiment is not None


class TestLabHPO:
    def test_hpo_primary_metric_idx_out_of_range(self):
        """Invalid primary_metric_idx raises ValueError."""
        lab = _make_lab(multi_metric=True)
        with pytest.raises(ValueError, match="out of range"):
            lab.hpo(
                features=["f1", "f2"],
                params_list={"n_jobs": [1]},
                estimator=_get_lr_class(),
                primary_metric_idx=5,
                verbose=False,
            )

    def test_hpo_primary_metric_idx_not_int(self):
        """Non-int, non-'all' primary_metric_idx raises."""
        lab = _make_lab(multi_metric=True)
        with pytest.raises(ValueError, match="must be int"):
            lab.hpo(
                features=["f1", "f2"],
                params_list={"n_jobs": [1]},
                estimator=_get_lr_class(),
                primary_metric_idx="first",
                verbose=False,
            )

    def test_hpo_primary_metric_idx_all_warns(self, caplog):
        """primary_metric_idx='all' emits a warning."""
        import logging

        lab = _make_lab(multi_metric=True)
        with caplog.at_level(logging.WARNING):
            lab.hpo(
                features=["f1", "f2"],
                params_list={"n_jobs": [1]},
                estimator=_get_lr_class(),
                primary_metric_idx="all",
                verbose=False,
            )
        assert "primary_metric_idx='all'" in caplog.text


class TestLabShowBestScore:
    def test_show_best_score_single(self):
        """show_best_score works for single metric."""
        lab = _make_lab(multi_metric=False)
        pipe = _make_simple_pipeline(lab)
        lab.run_experiment(pipe, verbose=False)
        result = lab.show_best_score()
        assert result.height == 1
        assert "cv_mean_score" in result.columns

    def test_show_best_score_multi_default(self):
        """show_best_score works for multi-metric with default."""
        lab = _make_lab(multi_metric=True)
        pipe = _make_simple_pipeline(lab)
        lab.run_experiment(pipe, verbose=False)
        result = lab.show_best_score()
        assert result.height == 1

    def test_show_best_score_multi_with_idx(self):
        """show_best_score with explicit metric_idx."""
        lab = _make_lab(multi_metric=True)
        pipe = _make_simple_pipeline(lab)
        lab.run_experiment(pipe, verbose=False)
        result = lab.show_best_score(metric_idx=1)
        assert result.height == 1


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_lr_class():
    """Return LinearRegression class for HPO tests."""
    from sklearn.linear_model import LinearRegression

    return LinearRegression
