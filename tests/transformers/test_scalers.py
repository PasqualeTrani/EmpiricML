"""Tests for scaler transformers.

Covers: StandardScaler, MinMaxScaler, RobustScaler.
"""

import numpy as np
import polars as pl
import pytest

from empml.transformers import MinMaxScaler, RobustScaler, StandardScaler

# ---------------------------------------------------------------------------
# TestStandardScaler
# ---------------------------------------------------------------------------


class TestStandardScaler:
    def test_basic_scaling(self, sample_lf):
        t = StandardScaler(features=["f1"])
        result = t.fit_transform(sample_lf).collect()
        scaled = result["f1"].to_numpy()
        # Mean should be approximately 0
        assert np.mean(scaled) == pytest.approx(0.0, abs=1e-10)
        # Std should be approximately 1
        assert np.std(scaled, ddof=1) == pytest.approx(1.0, abs=0.1)

    def test_zero_std(self):
        lf = pl.LazyFrame({"f1": [5.0, 5.0, 5.0, 5.0, 5.0]})
        t = StandardScaler(features=["f1"])
        result = t.fit_transform(lf).collect()
        # All identical -> std=0 -> output 0.0
        assert result["f1"].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])

    def test_null_passthrough(self):
        lf = pl.LazyFrame({"f1": [1.0, None, 3.0, None, 5.0]})
        t = StandardScaler(features=["f1"], suffix="_scaled")
        result = t.fit_transform(lf).collect()
        # Nulls should remain null
        assert result["f1_scaled"][1] is None
        assert result["f1_scaled"][3] is None
        assert result["f1_scaled"][0] is not None

    def test_suffix(self, sample_lf):
        t = StandardScaler(features=["f1"], suffix="_z")
        result = t.fit_transform(sample_lf).collect()
        assert "f1_z" in result.columns
        # Original column should still be there
        assert "f1" in result.columns

    def test_multiple_features(self, sample_lf):
        t = StandardScaler(features=["f1", "f2"], suffix="_std")
        result = t.fit_transform(sample_lf).collect()
        assert "f1_std" in result.columns
        assert "f2_std" in result.columns

    def test_fit_transform_consistency(self, sample_lf):
        t1 = StandardScaler(features=["f1"], suffix="_s")
        r1 = t1.fit_transform(sample_lf).collect()

        t2 = StandardScaler(features=["f1"], suffix="_s")
        t2.fit(sample_lf)
        r2 = t2.transform(sample_lf).collect()

        assert r1["f1_s"].to_list() == pytest.approx(r2["f1_s"].to_list())


# ---------------------------------------------------------------------------
# TestMinMaxScaler
# ---------------------------------------------------------------------------


class TestMinMaxScaler:
    def test_basic_scaling(self, sample_lf):
        t = MinMaxScaler(features=["f1"])
        result = t.fit_transform(sample_lf).collect()
        scaled = result["f1"].to_numpy()
        assert np.min(scaled) == pytest.approx(0.0)
        assert np.max(scaled) == pytest.approx(1.0)

    def test_zero_range(self):
        lf = pl.LazyFrame({"f1": [5.0, 5.0, 5.0, 5.0, 5.0]})
        t = MinMaxScaler(features=["f1"])
        result = t.fit_transform(lf).collect()
        assert result["f1"].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])

    def test_null_passthrough(self):
        lf = pl.LazyFrame({"f1": [1.0, None, 3.0, None, 5.0]})
        t = MinMaxScaler(features=["f1"], suffix="_mm")
        result = t.fit_transform(lf).collect()
        assert result["f1_mm"][1] is None
        assert result["f1_mm"][3] is None

    def test_suffix(self, sample_lf):
        t = MinMaxScaler(features=["f1"], suffix="_mm")
        result = t.fit_transform(sample_lf).collect()
        assert "f1_mm" in result.columns
        assert "f1" in result.columns

    def test_unseen_extrapolation(self, sample_lf, unseen_lf):
        t = MinMaxScaler(features=["f1"])
        t.fit(sample_lf)
        result = t.transform(unseen_lf).collect()
        # f1=100.0 is outside [1, 20] -> scaled value > 1.0
        assert result["f1"][0] > 1.0


# ---------------------------------------------------------------------------
# TestRobustScaler (migrated)
# ---------------------------------------------------------------------------


class TestRobustScaler:
    def test_basic_scaling(self, sample_lf):
        t = RobustScaler(features=["f1"])
        result = t.fit_transform(sample_lf).collect()
        scaled = result["f1"].to_numpy()
        assert abs(np.median(scaled)) < 0.6

    def test_zero_iqr(self):
        lf = pl.LazyFrame({"f1": [5.0, 5.0, 5.0, 5.0, 5.0]})
        t = RobustScaler(features=["f1"])
        result = t.fit_transform(lf).collect()
        assert result["f1"].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])

    def test_null_passthrough(self):
        lf = pl.LazyFrame({"f1": [1.0, None, 3.0, None, 5.0]})
        t = RobustScaler(features=["f1"], suffix="_scaled")
        result = t.fit_transform(lf).collect()
        assert result["f1_scaled"][1] is None
        assert result["f1_scaled"][3] is None
        assert result["f1_scaled"][0] is not None
