"""
Comparing two experiments.

Covers relative performance, fold-level comparison statistics, the rule that
decides whether a candidate beats a baseline, the permutation test, and the
printed comparison reports.
"""

import numpy as np
import polars as pl

from empml.base import Metric
from empml.results import MetricColumns
from empml.utils import BLUE, BOLD, GREEN, RED, RESET

# ------------------------------------------------------------------------------------------
# Comparison statistics
# ------------------------------------------------------------------------------------------


def relative_performance(minimize: bool, x1: float, x2: float) -> float | None:
    """
    Compute the relative performance of a pipeline with score x2 with respect to another of score x1 (reference).
    The same function can be used to compute overfitting.
    """
    if x1 is None or x2 is None or x1 == 0:
        return None

    if minimize:
        performance = round(((x1 - x2) / (x1)) * 100, 2)
    else:
        performance = round(((x2 - x1) / (x1)) * 100, 2)

    return performance


def compare_results_stats(
    results_a: pl.DataFrame, results_b: pl.DataFrame, minimize: bool
) -> dict[str, float | pl.DataFrame]:
    """Compute pipelines results stats, i.e. two different output of the eval_pipeline_cv function"""

    # build compare dataframe
    results_a = results_a.rename(
        {col: f"{col}_a" for col in results_a.columns if col != "fold_number"}
    )
    results_b = results_b.rename(
        {col: f"{col}_b" for col in results_b.columns if col != "fold_number"}
    )
    compare_df = results_a.join(results_b, how="left", on=["fold_number"])

    n_folds = compare_df.shape[0]

    # MAIN STATS
    # mean cv score performance
    mean_cv_performance = relative_performance(
        minimize=minimize,
        x1=compare_df["validation_score_a"].mean(),
        x2=compare_df["validation_score_b"].mean(),
    )

    # single fold validation performance
    fold_performances = compare_df.with_columns(
        pl.struct(["validation_score_a", "validation_score_b"])
        .map_elements(
            lambda x: relative_performance(
                minimize=minimize,
                x1=x["validation_score_a"],
                x2=x["validation_score_b"],
            )
        )
        .alias("relative_performance")
    ).select(["fold_number", "relative_performance"])

    # single fold overfittings - minimize = True
    fold_performances_overfitting = compare_df.with_columns(
        pl.struct(["overfitting_pct_a", "overfitting_pct_b"])
        .map_elements(
            lambda x: relative_performance(
                minimize=True, x1=x["overfitting_pct_a"], x2=x["overfitting_pct_b"]
            )
        )
        .alias("relative_performance_overfitting")
    ).select(["fold_number", "relative_performance_overfitting"])

    # mean overfitting - minimize always True
    mean_cv_performance_overfitting = relative_performance(
        minimize=True,
        x1=compare_df["overfitting_pct_a"].mean(),
        x2=compare_df["overfitting_pct_b"].mean(),
    )

    # std cv performance - minimize always True
    std_cv_performance = relative_performance(
        minimize=True,
        x1=compare_df["validation_score_a"].std(),
        x2=compare_df["validation_score_b"].std(),
    )

    n_folds_better_performance = fold_performances.filter(
        pl.col("relative_performance") > 0
    ).shape[0]  # for comparison on terminated experiments
    n_folds_lower_performance = fold_performances.filter(
        pl.col("relative_performance") <= 0
    ).shape[0]  # for interrupting the results evaluation prematurely

    return {
        # cv aggregate stats - float/int values
        "mean_cv_performance": mean_cv_performance,  # type: ignore[dict-item]
        "mean_cv_performance_overfitting": mean_cv_performance_overfitting,  # type: ignore[dict-item]
        "std_cv_performance": std_cv_performance,  # type: ignore[dict-item]
        "n_folds_better_performance": n_folds_better_performance,
        "n_folds_lower_performance": n_folds_lower_performance,
        "n_folds": n_folds,
        # single fold stats - they are polars dataframes
        "fold_performances": fold_performances,
        "fold_performances_overfitting": fold_performances_overfitting,
    }


def _single_metric_view(
    details: pl.DataFrame, columns: MetricColumns, index: int
) -> pl.DataFrame:
    """Per-fold details of one metric, with unsuffixed score columns."""
    renames = {
        columns.name(base, index): base
        for base in ("validation_score", "train_score", "overfitting_pct")
    }
    keep = ["fold_number", "experiment_id", *renames]
    return details.select([c for c in keep if c in details.columns]).rename(renames)


def compare_experiments(
    details_a: pl.DataFrame,
    details_b: pl.DataFrame,
    minimize: list[bool],
    columns: MetricColumns,
) -> list[dict[str, float | pl.DataFrame]]:
    """Compare experiment B with baseline A: one statistics dict per metric."""
    return [
        compare_results_stats(
            results_a=_single_metric_view(details_a, columns, i),
            results_b=_single_metric_view(details_b, columns, i),
            minimize=minimize[i - 1],
        )
        for i in columns.indices
    ]


def is_improvement(
    comparisons: list[dict],
    n_folds_threshold: int,
    min_improvement_pct: float,
    pvalues: list[float] | None = None,
    alpha: float | None = None,
) -> bool:
    """
    Decide whether experiment B beats baseline A.

    B must beat A on every metric. For each metric, the mean CV improvement must
    exceed ``min_improvement_pct``, B may be worse on at most
    ``n_folds_threshold`` folds, and, when p-values are given, the p-value must
    be below ``alpha``.
    """
    checked_pvalues: list[float | None] = (
        [None] * len(comparisons) if pvalues is None else list(pvalues)
    )
    return all(
        c["mean_cv_performance"] is not None
        and c["mean_cv_performance"] > min_improvement_pct
        and c["n_folds_lower_performance"] <= n_folds_threshold
        and (pvalue is None or pvalue < alpha)
        for c, pvalue in zip(comparisons, checked_pvalues, strict=True)
    )


# ------------------------------------------------------------------------------------------
# Permutation test
# ------------------------------------------------------------------------------------------


def generate_shuffle_preds(
    lf: pl.LazyFrame, preds_1: str, preds_2: str, random_state: int = 0
) -> pl.LazyFrame:
    """
    Create shuffled predictions for permutation testing in Lab.

    Lab can use permutation tests to assess whether performance differences
    between two experiments are statistically significant. This randomly swaps
    predictions between the two experiments to create null distribution.

    Args:
        lf: LazyFrame containing both sets of predictions
        preds_1: First experiment's prediction column
        preds_2: Second experiment's prediction column
        random_state: Random seed for reproducible permutations

    Returns:
        LazyFrame with 'shuffle_a' and 'shuffle_b' columns (randomly swapped)
    """
    transf_lz = (
        lf
        # Generate random binary mask (0 or 1) for each row
        .with_columns(
            rand_seq=(
                pl.int_range(0, pl.len()).sample(
                    fraction=1.0, with_replacement=True, seed=random_state
                )
                % 2
            )
        ).with_columns(
            # shuffle_a: takes preds_1 where mask=1, preds_2 where mask=0
            (
                (pl.col(preds_1) * pl.col("rand_seq"))
                + (pl.col(preds_2) * (1 - pl.col("rand_seq")))
            ).alias("shuffle_a"),
            # shuffle_b: inverse of shuffle_a
            (
                (pl.col(preds_2) * pl.col("rand_seq"))
                + (pl.col(preds_1) * (1 - pl.col("rand_seq")))
            ).alias("shuffle_b"),
        )
    )

    return transf_lz


def compute_anomaly(
    metric: Metric, lf: pl.LazyFrame, preds_1: str, preds_2: str, target: str
) -> float:
    """
    Compute test statistic for Lab's permutation testing.

    Calculates the absolute difference in metric scores between two experiments.
    Lab uses this as the test statistic when running permutation tests to assess
    significance of performance differences.

    Args:
        metric: Metric object with compute_metric method
        lf: LazyFrame containing predictions and target
        preds_1: First experiment's prediction column
        preds_2: Second experiment's prediction column
        target: Ground truth label column

    Returns:
        Absolute difference in metric scores (test statistic)
    """
    score_1 = metric.compute_metric(lf=lf, target=target, preds=preds_1)
    score_2 = metric.compute_metric(lf=lf, target=target, preds=preds_2)

    return abs(score_1 - score_2)


def permutation_pvalue(
    metric: Metric,
    preds: pl.LazyFrame,
    preds_1: str,
    preds_2: str,
    target: str,
    n_iters: int,
) -> float:
    """P-value of the observed score difference against ``n_iters`` random swaps."""
    observed = compute_anomaly(metric, preds, preds_1, preds_2, target)
    simulated = [
        compute_anomaly(
            metric=metric,
            lf=generate_shuffle_preds(preds, preds_1, preds_2, random_state=i),
            preds_1="shuffle_a",
            preds_2="shuffle_b",
            target=target,
        )
        for i in range(n_iters)
    ]
    n_exceeding = (np.array(simulated) > observed).sum()
    return (n_exceeding + 1) / (n_iters + 1)


# ------------------------------------------------------------------------------------------
# Reports
# ------------------------------------------------------------------------------------------


def format_log_performance(
    x: float | None, th: float, is_percentage: bool = True
) -> str:
    """
    Format performance metric with color coding for Lab's console output.

    Used by Lab when comparing experiments to highlight improvements (green)
    and regressions (red) in terminal output.

    Args:
        x: Performance value
        th: Threshold for determining good/bad performance
        is_percentage: Whether to append '%' symbol

    Returns:
        ANSI-colored string (green if x > th, red otherwise)
    """
    if x is None:
        return f"{BOLD}{BLUE}N/A{RESET}"

    percentage_str: str = "%" if is_percentage else ""

    if x > th:  # Improvement: green
        return f"{BOLD}{GREEN}{str(x)}{percentage_str}{RESET}"
    else:  # Regression: red
        return f"{BOLD}{RED}{str(x)}{percentage_str}{RESET}"


def log_performance_against(comparison: dict[str, float], n_folds_threshold: int):
    """
    Print comprehensive comparison report between two Lab experiments.

    Lab uses this to display detailed performance comparisons when evaluating
    whether a new experiment (B) improves upon a baseline experiment (A).

    Args:
        comparison: Dict with comparison metrics:
            - mean_cv_performance: Overall CV score difference
            - std_cv_performance: CV stability difference
            - fold_performances: Per-fold score differences
            - n_folds_better_performance: Count of folds where B beats A
            - mean_cv_performance_overfitting: Overall overfitting difference
            - fold_performances_overfitting: Per-fold overfitting differences
        n_folds_threshold: Minimum fold advantage needed to consider B better
    """
    print(
        f"\n{BOLD}{BLUE}Relative Performance Report Experiment B (Current) vs Experiment A (Chosen Baseline){RESET}"
    )
    print(f"""
    {BOLD}{BLUE}Note: positive performances like reduction of overfitting or increment/decrement of the metrics over the CV are indicated in {RESET}{BOLD}{GREEN}GREEN{RESET},
    {BOLD}{BLUE}while negative performances are indicated in {BOLD}{RED}RED{RESET}\n
    """)

    # Overall CV performance comparison
    print(
        f"Mean CV Score Experiment B vs A: {format_log_performance(comparison['mean_cv_performance'], 0)}"
    )
    print(
        f"Std CV Score Experiment B vs A: {format_log_performance(comparison['std_cv_performance'], 0)}\n"
    )

    # Per-fold breakdown
    for row in comparison["fold_performances"].iter_rows():
        print(
            f"Fold {row[0]} Score Performance Experiment B vs A: {format_log_performance(row[1], 0)}"
        )
    print(
        f"Number of Folds Experiment B is Better then A: {format_log_performance(comparison['n_folds_better_performance'], comparison['n_folds'] - n_folds_threshold - 1, is_percentage=False)}\n"
    )

    # Overfitting analysis
    print(
        f"Mean % Overfitting Score Experiment B vs A: {format_log_performance(comparison['mean_cv_performance_overfitting'], 0)}"
    )
    for row in comparison["fold_performances_overfitting"].iter_rows():
        print(
            f"\t - Fold {row[0]} % Overfitting Score Performance Experiment B vs A: {format_log_performance(row[1], 0)}"
        )


def log_performance_against_multi(
    comparisons: list[dict],
    n_folds_threshold: int,
    n_metrics: int,
) -> None:
    """
    Print comparison report for each metric independently.

    Iterates over comparison dicts and delegates per-metric
    logging to log_performance_against.

    Args:
        comparisons: List of comparison dicts, one per metric
        n_folds_threshold: Fold advantage threshold
        n_metrics: Number of metrics
    """
    for i, comparison in enumerate(comparisons, 1):
        print(f"\n{BOLD}{BLUE}{'=' * 60}\n  Metric {i}\n{'=' * 60}{RESET}")
        log_performance_against(
            comparison=comparison,
            n_folds_threshold=n_folds_threshold,
        )


def print_comparison_report(
    comparisons: list[dict], n_folds_threshold: int, columns: MetricColumns
) -> None:
    """Print the comparison report; multi-metric reports add a header per metric."""
    if columns.suffixed:
        log_performance_against_multi(comparisons, n_folds_threshold, columns.n_metrics)
    else:
        log_performance_against(comparisons[0], n_folds_threshold)


def print_test_report(
    cv_results: pl.DataFrame,
    test_results: dict,
    minimize: list[bool],
    columns: MetricColumns,
) -> None:
    """Print how test scores compare with the CV mean and its μ ± 2σ interval."""
    for i in columns.indices:
        cv_mean = cv_results[columns.name("cv_mean_score", i)].item()
        cv_std = cv_results[columns.name("cv_std_score", i)].item()
        test_score = test_results[columns.name("validation_score", i)]
        difference = format_log_performance(
            relative_performance(minimize[i - 1], cv_mean, test_score), 0
        )
        lower, upper = cv_mean - 2 * cv_std, cv_mean + 2 * cv_std

        if columns.suffixed:
            print(f"Metric {i} - CV vs Test: {difference}")
            if lower <= test_score <= upper:
                print(f"{BOLD}{GREEN} Metric {i}: test score within μ ± 2σ{RESET}")
            else:
                print(f"{BOLD}{RED} Metric {i}: test score NOT within μ ± 2σ{RESET}")
        else:
            print(f"Difference in performance CV vs Test: {difference}")
            # The single-metric report has always excluded the upper bound.
            if lower <= test_score < upper:
                print(f"{BOLD}{GREEN} The test score is between μ ± 2σ{RESET}")
            else:
                print(f"{BOLD}{RED} The test score is NOT between μ ± 2σ{RESET}")
