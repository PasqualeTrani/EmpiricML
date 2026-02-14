"""Tests for imputation transformers.

Covers: SimpleImputer, FillNulls.
"""

import polars as pl
import pytest

from empml.transformers import FillNulls, SimpleImputer

# ---------------------------------------------------------------------------
# TestSimpleImputer
# ---------------------------------------------------------------------------


class TestSimpleImputer:
    def test_mean_strategy(self, sample_lf):
        t = SimpleImputer(features=["f3"], strategy="mean")
        result = t.fit_transform(sample_lf).collect()
        # f3 non-null values: [1,3,5,6,8,9,10,11,13,14,16,17,18,20]
        expected_mean = sum([1, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 18, 20]) / 14
        # Index 1 was null -> now should be the mean
        assert result["f3"][1] == pytest.approx(expected_mean)
        # Index 3 was null -> same
        assert result["f3"][3] == pytest.approx(expected_mean)

    def test_median_strategy(self, sample_lf):
        t = SimpleImputer(features=["f3"], strategy="median")
        result = t.fit_transform(sample_lf).collect()
        # Median of f3 non-null values
        vals = sorted([1, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 18, 20])
        expected_median = (vals[6] + vals[7]) / 2  # 10.5
        assert result["f3"][1] == pytest.approx(expected_median)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="SimpleImputer strategy"):
            SimpleImputer(features=["f1"], strategy="mode")

    def test_no_nulls_unchanged(self, sample_lf):
        t = SimpleImputer(features=["f1"], strategy="mean")
        result = t.fit_transform(sample_lf).collect()
        original = sample_lf.collect()
        assert result["f1"].to_list() == pytest.approx(original["f1"].to_list())

    def test_all_nulls_default(self):
        lf = pl.LazyFrame({"f1": [None, None, None]}, schema={"f1": pl.Float64})
        t = SimpleImputer(features=["f1"], strategy="mean")
        result = t.fit_transform(lf).collect()
        # All null -> impute_value defaults to 0.0
        assert result["f1"].to_list() == pytest.approx([0.0, 0.0, 0.0])

    def test_multiple_features(self, sample_lf):
        t = SimpleImputer(features=["f1", "f3"], strategy="mean")
        result = t.fit_transform(sample_lf).collect()
        # f1 has no nulls -> unchanged
        original = sample_lf.collect()
        assert result["f1"].to_list() == pytest.approx(original["f1"].to_list())
        # f3 had nulls -> now imputed (no nulls left)
        assert result["f3"].null_count() == 0


# ---------------------------------------------------------------------------
# TestFillNulls
# ---------------------------------------------------------------------------


class TestFillNulls:
    def test_default_value(self, sample_lf):
        t = FillNulls(features=["f3"])
        result = t.fit_transform(sample_lf).collect()
        # Default value is -9999
        assert result["f3"][1] == pytest.approx(-9999)
        assert result["f3"][3] == pytest.approx(-9999)

    def test_custom_value(self, sample_lf):
        t = FillNulls(features=["f3"], value=0.0)
        result = t.fit_transform(sample_lf).collect()
        assert result["f3"][1] == pytest.approx(0.0)
        assert result["f3"][3] == pytest.approx(0.0)

    def test_no_nulls_unchanged(self, sample_lf):
        t = FillNulls(features=["f1"], value=-1.0)
        result = t.fit_transform(sample_lf).collect()
        original = sample_lf.collect()
        assert result["f1"].to_list() == pytest.approx(original["f1"].to_list())

    def test_nan_handling(self):
        lf = pl.LazyFrame({"f1": [1.0, float("nan"), 3.0]})
        t = FillNulls(features=["f1"], value=-1.0)
        result = t.fit_transform(lf).collect()
        assert result["f1"][1] == pytest.approx(-1.0)
