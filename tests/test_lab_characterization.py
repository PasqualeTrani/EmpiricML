"""Characterization tests that pin the observable behavior of Lab end to end.

Each scenario serializes what a user can observe (results tables, returned
values, printed reports) and compares it with a golden snapshot recorded before
the Lab refactor. Timing columns and timestamps are excluded because they vary
between runs.

Regenerate the snapshots only for an intentional behavior change:
    EMPML_UPDATE_GOLDEN=1 pytest tests/test_lab_characterization.py
"""

import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from empml.base import CVGenerator, DataDownloader
from empml.errors import RunExperimentConfigException, RunExperimentOnTestException
from empml.lab import ComparisonCriteria, Lab, restore_check_point
from empml.metrics import MAE, MSE
from empml.pipeline import Pipeline
from empml.transformers import Identity
from empml.wrappers import SKlearnWrapper

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_PATH = FIXTURES / "lab_characterization.json"
LEGACY_CHECKPOINT = FIXTURES / "legacy_lab_checkpoint.pkl"
UPDATE_GOLDEN = os.environ.get("EMPML_UPDATE_GOLDEN") == "1"
VOLATILE_COLUMNS = {
    "timestamp_utc",
    "mean_train_time_s",
    "mean_inference_time_s",
    "duration_train",
    "duration_inf",
}


# ------------------------------------------------------------------
# Test doubles and builders
# ------------------------------------------------------------------


class FrameDownloader(DataDownloader):
    def __init__(self, lf: pl.LazyFrame):
        self.lf = lf

    def get_data(self) -> pl.LazyFrame:
        return self.lf


class HalvesCV(CVGenerator):
    """Two folds: first half and second half of the row IDs."""

    def split(self, lf, row_id):
        ids = lf.select(row_id).collect()[row_id].to_list()
        mid = len(ids) // 2
        return [
            (np.array(ids[mid:]), np.array(ids[:mid])),
            (np.array(ids[:mid]), np.array(ids[mid:])),
        ]


def _frame(seed: int, n: int = 24) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    target = 2 * f1 + f2 + rng.normal(scale=0.1, size=n)
    # f4 is constant, so permuting it never changes predictions.
    return pl.LazyFrame(
        {"f1": f1, "f2": f2, "f3": f3, "f4": np.zeros(n), "target": target}
    )


def _lab(multi: bool, with_test: bool = False, **criteria) -> Lab:
    criteria = criteria or {"n_folds_threshold": 1, "pct_threshold": 0.0}
    return Lab(
        train_downloader=FrameDownloader(_frame(0)),
        test_downloader=FrameDownloader(_frame(1, n=10)) if with_test else None,
        metric=[MSE(), MAE()] if multi else MSE(),
        minimize=[True, True] if multi else True,
        cv_generator=HalvesCV(),
        target="target",
        comparison_criteria=ComparisonCriteria(**criteria),
        name="lab",
    )


def _keyed(predictions: pl.LazyFrame) -> pl.DataFrame:
    # Streaming joins do not guarantee row order; alignment is by key.
    return predictions.collect().sort("fold_number", "row_id")


def _pipe(features: list[str], name: str = "lr", estimator=None) -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                SKlearnWrapper(
                    estimator=estimator or LinearRegression(),
                    features=features,
                    target="target",
                ),
            )
        ],
        name=name,
        description=f"{name} on {features}",
    )


# ------------------------------------------------------------------
# Snapshot helpers
# ------------------------------------------------------------------


def _plain(value):
    """Convert observed values into JSON-compatible data."""
    if isinstance(value, pl.LazyFrame):
        value = value.collect()
    if isinstance(value, pl.DataFrame):
        value = value.drop([c for c in value.columns if c in VOLATILE_COLUMNS])
        return {
            "schema": {k: str(v) for k, v in value.schema.items()},
            "data": _plain(value.to_dict(as_series=False)),
        }
    if isinstance(value, dict):
        return {
            str(k): ("<volatile>" if k in VOLATILE_COLUMNS else _plain(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return "NaN" if math.isnan(value) else value
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return repr(value)


def _assert_same(actual, expected, path="$"):
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert list(actual) == list(expected), path
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_same(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, path


@pytest.fixture(scope="module")
def golden():
    data = {} if UPDATE_GOLDEN else json.loads(GOLDEN_PATH.read_text())
    yield data
    if UPDATE_GOLDEN:
        FIXTURES.mkdir(exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(data, indent=1) + "\n")


@pytest.fixture
def snapshot(golden, request):
    def check(observed):
        key = request.node.name
        plain = _plain(observed)
        if UPDATE_GOLDEN:
            golden[key] = plain
        else:
            _assert_same(plain, golden[key])

    return check


MODES = pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])


# ------------------------------------------------------------------
# Experiments and comparisons
# ------------------------------------------------------------------


@MODES
def test_experiments_with_comparison_report(multi, snapshot, capsys):
    lab = _lab(multi)
    lab.run_experiment(_pipe(["f1"]), verbose=False)
    lab.run_experiment(_pipe(["f1", "f2"]), verbose=False, compare_against=1)
    lab.multi_run_experiment(
        [_pipe(["f2"], "a"), _pipe(["f1", "f3"], "b")], verbose=False
    )
    snapshot(
        {
            "results": lab.results,
            "details": lab.results_details,
            "stdout": capsys.readouterr().out,
            "next_id": lab.next_experiment_id,
            "files": sorted(p.name for p in Path("lab").rglob("*") if p.is_file()),
        }
    )


@MODES
def test_early_stopping_arrests_worse_experiment(multi, snapshot, caplog, capsys):
    lab = _lab(multi, n_folds_threshold=0, pct_threshold=0.0)
    lab.run_experiment(_pipe(["f1", "f2"]), verbose=False)
    with caplog.at_level("INFO"):
        lab.run_experiment(_pipe(["f3"]), verbose=False, compare_against=1)
    assert "Experiment arrested" in caplog.text
    snapshot(
        {
            "results": lab.results,
            "details": lab.results_details,
            "stdout": capsys.readouterr().out,
            "predictions": _keyed(lab.retrieve_predictions([1, 2])),
        }
    )


@MODES
def test_auto_mode_with_percentage_criterion(multi, snapshot, capsys):
    lab = _lab(multi, n_folds_threshold=0, pct_threshold=5.0)
    with pytest.raises(RunExperimentConfigException):
        lab.run_experiment(_pipe(["f1"]), verbose=False, auto_mode=True)
    lab.run_experiment(_pipe(["f1"]), verbose=False)
    lab._set_best_experiment(1)
    best = [lab.best_experiment]
    for features in (["f1", "f2"], ["f1", "f2", "f3"], ["f3"]):
        lab.run_experiment(_pipe(features), verbose=False, auto_mode=True)
        best.append(lab.best_experiment)
    snapshot({"best": best, "results": lab.results, "stdout": capsys.readouterr().out})


@MODES
def test_auto_mode_with_statistical_criterion(multi, snapshot, capsys):
    lab = _lab(multi, n_folds_threshold=1, alpha=0.2, n_iters=20)
    lab.run_experiment(_pipe(["f1"]), verbose=False)
    lab._set_best_experiment(1)
    best = []
    for features in (["f1", "f2"], ["f1", "f2", "f3"]):
        lab.run_experiment(_pipe(features), verbose=False, auto_mode=True)
        best.append(lab.best_experiment)
    snapshot(
        {
            "best": best,
            "pvalue": lab.compute_pvalue((1, 2), n_iters=15),
            "pvalue_type": type(lab.compute_pvalue((1, 3), n_iters=5)).__name__,
            "stdout": capsys.readouterr().out,
        }
    )
    with pytest.raises(ValueError):
        lab.compute_pvalue([1, 2])


@MODES
def test_retrieve_predictions_and_pipeline(multi, snapshot):
    lab = _lab(multi)
    lab.run_experiment(_pipe(["f1"]), verbose=False)
    lab.run_experiment(_pipe(["f2"]), verbose=False, store_preds=False)
    lab.run_experiment(_pipe(["f1", "f2"]), verbose=False, eval_overfitting=False)
    snapshot(
        {
            "predictions": _keyed(
                lab.retrieve_predictions([1, 2, 3], extra_features=["f1"])
            ),
            "pipeline": lab.retrieve_pipeline(3),
            "results": lab.results,
            "details": lab.results_details,
        }
    )


@MODES
def test_show_best_score(multi, snapshot):
    lab = _lab(multi)
    for features in (["f1"], ["f1", "f2"], ["f3"]):
        lab.run_experiment(_pipe(features), verbose=False)
    snapshot(
        {
            "default": lab.show_best_score(),
            "second": lab.show_best_score(metric_idx=1),
        }
    )


# ------------------------------------------------------------------
# Test-set evaluation
# ------------------------------------------------------------------


@MODES
def test_run_experiment_on_test(multi, snapshot, capsys):
    lab = _lab(multi, with_test=True)
    lab.run_experiment(_pipe(["f1", "f2"]), verbose=False)
    result = lab.run_experiment_on_test(1, verbose=False)
    no_overfit = lab.run_experiment_on_test(
        1, verbose=False, eval_overfitting=False, store_preds=False
    )
    snapshot(
        {
            "keys": list(result),
            "result": result,
            "no_overfit": no_overfit,
            "stdout": capsys.readouterr().out,
        }
    )


@MODES
def test_run_experiment_on_test_interval_upper_bound(multi, snapshot, capsys):
    """The test score equals mean + 2 * std exactly (zero std)."""
    lab = _lab(multi, with_test=True)
    lab.run_experiment(_pipe(["f1", "f2"]), verbose=False)
    score_key = "validation_score_1" if multi else "validation_score"
    score = lab.run_experiment_on_test(1, verbose=False)[score_key]
    capsys.readouterr()
    suffix = "_1" if multi else ""
    lab.results = lab.results.with_columns(
        pl.lit(score).alias(f"cv_mean_score{suffix}"),
        pl.lit(0.0).alias(f"cv_std_score{suffix}"),
    )
    lab.run_experiment_on_test(1, verbose=False)
    snapshot(capsys.readouterr().out)


def test_run_experiment_on_test_requires_test_set():
    lab = _lab(False)
    with pytest.raises(RunExperimentOnTestException):
        lab.run_experiment_on_test(1)


# ------------------------------------------------------------------
# HPO and baselines
# ------------------------------------------------------------------


@MODES
@pytest.mark.parametrize("search_type", ["grid", "random"])
def test_hpo(multi, search_type, snapshot):
    lab = _lab(multi)
    result = lab.hpo(
        features=["f1", "f2"],
        params_list={"alpha": [0.01, 1.0, 50.0], "fit_intercept": [True, False]},
        estimator=Ridge,
        search_type=search_type,
        num_samples=3,
        random_state=1,
        verbose=False,
    )
    snapshot({"result": result, "results": lab.results})


@pytest.mark.parametrize("primary_metric_idx", [0, 1, "all"])
@pytest.mark.parametrize(
    "with_baseline, alphas",
    [(False, [0.001, 5.0, 100.0]), (True, [0.001, 5.0, 100.0]), (True, [20.0, 90.0])],
    ids=["free", "baseline", "baseline-unbeaten"],
)
def test_hpo_multi_metric_selection(
    primary_metric_idx, with_baseline, alphas, snapshot
):
    lab = _lab(True)
    compare_against = None
    if with_baseline:
        lab.run_experiment(_pipe(["f1", "f2"]), verbose=False)
        compare_against = 1
    result = lab.hpo(
        features=["f1", "f2"],
        params_list={"alpha": alphas},
        estimator=Ridge,
        compare_against=compare_against,
        primary_metric_idx=primary_metric_idx,
        verbose=False,
    )
    snapshot({"result": result, "results": lab.results})


@pytest.mark.parametrize("problem_type", ["regression", "classification"])
@pytest.mark.parametrize("with_preprocess", [False, True], ids=["raw", "prep"])
def test_run_base_experiments_builds_catalog(
    problem_type, with_preprocess, snapshot, monkeypatch
):
    lab = _lab(False)
    calls = []
    monkeypatch.setattr(lab, "multi_run_experiment", lambda **kw: calls.append(kw))
    preprocess = Pipeline([("identity", Identity())]) if with_preprocess else None
    lab.run_base_experiments(
        features=["f1", "f2"],
        preprocess_pipe=preprocess,
        compare_against=None,
        problem_type=problem_type,
        verbose=False,
    )
    (call,) = calls
    pipelines = call.pop("pipelines")
    snapshot(
        {
            "kwargs": call,
            "pipelines": [
                {
                    "name": p.name,
                    "description": p.description,
                    "steps": [name for name, _ in p.steps],
                    "estimator": repr(p.steps[-1][1].estimator),
                    "preprocess": repr(p.steps[0][1]) if len(p) > 1 else None,
                }
                for p in pipelines
            ],
        }
    )


# ------------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------------


@MODES
def test_permutation_feature_importance_and_selection(multi, snapshot):
    lab = _lab(multi)
    importance = lab.permutation_feature_importance(
        _pipe(["f1", "f2", "f3", "f4"]),
        features=["f1", "f2", "f3", "f4"],
        n_iters=3,
        verbose=False,
    )
    selected = lab.recursive_permutation_feature_selection(
        estimator=LinearRegression(),
        features=["f1", "f2", "f3", "f4"],
        n_iters=3,
        verbose=False,
    )
    snapshot({"importance": importance, "selected": selected})


# ------------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------------


def _legacy_lab() -> Lab:
    lab = _lab(True, n_folds_threshold=1, pct_threshold=0.0)
    lab.run_experiment(_pipe(["f1"]), verbose=False)
    lab._set_best_experiment(1)
    return lab


@MODES
def test_checkpoint_round_trip(multi, snapshot):
    lab = _lab(multi)
    lab.run_experiment(_pipe(["f1"]), verbose=False)
    lab.save_check_point("cp")
    restored = restore_check_point("lab", "cp")
    assert sorted(vars(restored)) == sorted(vars(lab))
    restored.run_experiment(_pipe(["f1", "f2"]), verbose=False, compare_against=1)
    snapshot({"attributes": sorted(vars(restored)), "results": restored.results})


def test_checkpoint_written_before_refactor_still_works(snapshot, capsys):
    """Checkpoints pickled by the pre-refactor Lab must keep working."""
    # Never re-record this fixture after the refactor: it must stay a
    # pickle produced by the pre-refactor Lab class.
    if UPDATE_GOLDEN and not LEGACY_CHECKPOINT.exists():
        FIXTURES.mkdir(exist_ok=True)
        LEGACY_CHECKPOINT.write_bytes(pickle.dumps(_legacy_lab()))
    # Recreate the artifacts the legacy lab expects next to its checkpoint.
    _legacy_lab()
    Path("lab/check_points").mkdir(parents=True, exist_ok=True)
    Path("lab/check_points/legacy.pkl").write_bytes(LEGACY_CHECKPOINT.read_bytes())
    lab = restore_check_point("lab", "legacy")
    lab.run_experiment(
        _pipe(["f1", "f2"]), verbose=False, compare_against=1, auto_mode=True
    )
    snapshot(
        {
            "best": lab.best_experiment,
            "results": lab.results,
            "best_score": lab.show_best_score(metric_idx=1),
            "pipeline": lab.retrieve_pipeline(2),
            "predictions": _keyed(lab.retrieve_predictions([1, 2])),
            "stdout": capsys.readouterr().out,
        }
    )
