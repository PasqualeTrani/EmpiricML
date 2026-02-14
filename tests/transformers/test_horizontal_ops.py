"""Tests for horizontal/row-wise feature transformers.

Covers: Identity, AvgFeatures, MaxFeatures, MinFeatures, StdFeatures,
        MedianFeatures, ModuleFeatures, InteractionFeatures.
"""

import math

import polars as pl
import pytest

from empml.transformers import (
    AvgFeatures,
    Identity,
    InteractionFeatures,
    MaxFeatures,
    MedianFeatures,
    MinFeatures,
    ModuleFeatures,
    StdFeatures,
)

# ---------------------------------------------------------------------------
# TestIdentity
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_passthrough(self, sample_lf):
        t = Identity()
        result = t.fit_transform(sample_lf).collect()
        expected = sample_lf.collect()
        assert result.equals(expected)

    def test_columns_preserved(self, sample_lf):
        t = Identity()
        result = t.fit_transform(sample_lf).collect()
        assert result.columns == sample_lf.collect().columns

    def test_fit_returns_self(self, sample_lf):
        t = Identity()
        assert t.fit(sample_lf) is t


# ---------------------------------------------------------------------------
# TestAvgFeatures
# ---------------------------------------------------------------------------


class TestAvgFeatures:
    def test_basic_two_features(self, sample_lf):
        t = AvgFeatures(features=["f1", "f2"], new_feature="avg_f1_f2")
        result = t.fit_transform(sample_lf).collect()
        assert "avg_f1_f2" in result.columns
        # Row 0: mean(1.0, 20.0) = 10.5
        assert result["avg_f1_f2"][0] == pytest.approx(10.5)
        # Row 19: mean(20.0, 1.0) = 10.5
        assert result["avg_f1_f2"][19] == pytest.approx(10.5)

    def test_three_features_with_nulls(self, sample_lf):
        t = AvgFeatures(features=["f1", "f2", "f3"], new_feature="avg_all")
        result = t.fit_transform(sample_lf).collect()
        # Row 0: mean(1.0, 20.0, 1.0) = 7.333...
        assert result["avg_all"][0] == pytest.approx(22.0 / 3)
        # Row 1: f3=None, pl.mean_horizontal skips nulls -> mean(2.0, 19.0) = 10.5
        assert result["avg_all"][1] == pytest.approx(10.5)

    def test_new_feature_name(self, sample_lf):
        t = AvgFeatures(features=["f1", "f2"], new_feature="my_avg")
        result = t.fit_transform(sample_lf).collect()
        assert "my_avg" in result.columns

    def test_original_columns_preserved(self, sample_lf):
        t = AvgFeatures(features=["f1", "f2"], new_feature="avg")
        result = t.fit_transform(sample_lf).collect()
        assert "f1" in result.columns
        assert "f2" in result.columns


# ---------------------------------------------------------------------------
# TestMaxFeatures
# ---------------------------------------------------------------------------


class TestMaxFeatures:
    def test_basic(self, sample_lf):
        t = MaxFeatures(features=["f1", "f2"], new_feature="max_f1_f2")
        result = t.fit_transform(sample_lf).collect()
        assert "max_f1_f2" in result.columns
        # Row 0: max(1.0, 20.0) = 20.0
        assert result["max_f1_f2"][0] == pytest.approx(20.0)
        # Row 19: max(20.0, 1.0) = 20.0
        assert result["max_f1_f2"][19] == pytest.approx(20.0)

    def test_with_nulls(self, sample_lf):
        t = MaxFeatures(features=["f1", "f3"], new_feature="max_f1_f3")
        result = t.fit_transform(sample_lf).collect()
        # Row 1: f3=None -> max(2.0) = 2.0
        assert result["max_f1_f3"][1] == pytest.approx(2.0)

    def test_new_feature_name(self, sample_lf):
        t = MaxFeatures(features=["f1", "f2"], new_feature="the_max")
        result = t.fit_transform(sample_lf).collect()
        assert "the_max" in result.columns


# ---------------------------------------------------------------------------
# TestMinFeatures
# ---------------------------------------------------------------------------


class TestMinFeatures:
    def test_basic(self, sample_lf):
        t = MinFeatures(features=["f1", "f2"], new_feature="min_f1_f2")
        result = t.fit_transform(sample_lf).collect()
        assert "min_f1_f2" in result.columns
        # Row 0: min(1.0, 20.0) = 1.0
        assert result["min_f1_f2"][0] == pytest.approx(1.0)
        # Row 10: min(11.0, 10.0) = 10.0
        assert result["min_f1_f2"][10] == pytest.approx(10.0)

    def test_with_nulls(self, sample_lf):
        t = MinFeatures(features=["f1", "f3"], new_feature="min_f1_f3")
        result = t.fit_transform(sample_lf).collect()
        # Row 1: f3=None -> min(2.0) = 2.0
        assert result["min_f1_f3"][1] == pytest.approx(2.0)

    def test_new_feature_name(self, sample_lf):
        t = MinFeatures(features=["f1", "f2"], new_feature="the_min")
        result = t.fit_transform(sample_lf).collect()
        assert "the_min" in result.columns


# ---------------------------------------------------------------------------
# TestStdFeatures
# ---------------------------------------------------------------------------


class TestStdFeatures:
    def test_basic(self, sample_lf):
        t = StdFeatures(features=["f1", "f2"], new_feature="std_f1_f2")
        result = t.fit_transform(sample_lf).collect()
        assert "std_f1_f2" in result.columns
        # Row 0: std([1.0, 20.0]) - sample std = 13.435...
        assert result["std_f1_f2"][0] is not None
        assert result["std_f1_f2"][0] > 0

    def test_identical_values(self):
        lf = pl.LazyFrame({"f1": [5.0, 5.0, 5.0], "f2": [5.0, 5.0, 5.0]})
        t = StdFeatures(features=["f1", "f2"], new_feature="std_out")
        result = t.fit_transform(lf).collect()
        # std of [5.0, 5.0] = 0.0
        assert result["std_out"][0] == pytest.approx(0.0)

    def test_new_feature_name(self, sample_lf):
        t = StdFeatures(features=["f1", "f2"], new_feature="my_std")
        result = t.fit_transform(sample_lf).collect()
        assert "my_std" in result.columns


# ---------------------------------------------------------------------------
# TestMedianFeatures
# ---------------------------------------------------------------------------


class TestMedianFeatures:
    def test_basic_two_features(self, sample_lf):
        t = MedianFeatures(features=["f1", "f2"], new_feature="med_f1_f2")
        result = t.fit_transform(sample_lf).collect()
        assert "med_f1_f2" in result.columns
        # Row 0: median([1.0, 20.0]) = 10.5
        assert result["med_f1_f2"][0] == pytest.approx(10.5)

    def test_three_features(self, sample_lf):
        t = MedianFeatures(features=["f1", "f2", "f3"], new_feature="med_all")
        result = t.fit_transform(sample_lf).collect()
        # Row 0: median([1.0, 20.0, 1.0]) = 1.0
        assert result["med_all"][0] == pytest.approx(1.0)

    def test_new_feature_name(self, sample_lf):
        t = MedianFeatures(features=["f1", "f2"], new_feature="the_med")
        result = t.fit_transform(sample_lf).collect()
        assert "the_med" in result.columns


# ---------------------------------------------------------------------------
# TestModuleFeatures
# ---------------------------------------------------------------------------


class TestModuleFeatures:
    def test_basic(self, sample_lf):
        t = ModuleFeatures(features=("f1", "f2"), new_feature="mod_f1_f2")
        result = t.fit_transform(sample_lf).collect()
        assert "mod_f1_f2" in result.columns
        # Row 0: sqrt(1^2 + 20^2) = sqrt(401) ~ 20.025
        assert result["mod_f1_f2"][0] == pytest.approx(math.sqrt(401))

    def test_null_propagation(self, sample_lf):
        t = ModuleFeatures(features=("f1", "f3"), new_feature="mod_f1_f3")
        result = t.fit_transform(sample_lf).collect()
        # Row 1: f3=None -> null propagation
        assert result["mod_f1_f3"][1] is None

    def test_zero_values(self):
        lf = pl.LazyFrame({"a": [0.0], "b": [0.0]})
        t = ModuleFeatures(features=("a", "b"), new_feature="mod_ab")
        result = t.fit_transform(lf).collect()
        assert result["mod_ab"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestInteractionFeatures (migrated)
# ---------------------------------------------------------------------------


class TestInteractionFeatures:
    def test_basic(self, sample_lf):
        t = InteractionFeatures(feature_pairs=[("f1", "f2")])
        result = t.fit_transform(sample_lf).collect()
        assert "f1_x_f2" in result.columns
        # First row: 1.0 * 20.0 = 20.0
        assert result["f1_x_f2"][0] == pytest.approx(20.0)
        # Last row: 20.0 * 1.0 = 20.0
        assert result["f1_x_f2"][-1] == pytest.approx(20.0)

    def test_multiple_pairs(self, sample_lf):
        t = InteractionFeatures(feature_pairs=[("f1", "f2"), ("f1", "f3")])
        result = t.fit_transform(sample_lf).collect()
        assert "f1_x_f2" in result.columns
        assert "f1_x_f3" in result.columns

    def test_custom_separator(self, sample_lf):
        t = InteractionFeatures(feature_pairs=[("f1", "f2")], separator="_times_")
        result = t.fit_transform(sample_lf).collect()
        assert "f1_times_f2" in result.columns

    def test_null_propagation(self, sample_lf):
        t = InteractionFeatures(feature_pairs=[("f1", "f3")])
        result = t.fit_transform(sample_lf).collect()
        # f3 has None at index 1 -> product should be null
        assert result["f1_x_f3"][1] is None
