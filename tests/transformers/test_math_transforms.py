"""Tests for mathematical feature transformers.

Covers: Log1pFeatures, Expm1Features, PowerFeatures, InverseFeatures.
"""

import math

import pytest
import polars as pl

from empml.transformers import (
    Log1pFeatures,
    Expm1Features,
    PowerFeatures,
    InverseFeatures,
)


# ---------------------------------------------------------------------------
# TestLog1pFeatures
# ---------------------------------------------------------------------------

class TestLog1pFeatures:

    def test_basic(self, sample_lf):
        t = Log1pFeatures(features=['f1'], suffix='_log')
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_log' in result.columns
        # f1=1.0 -> log(2) ~ 0.6931
        assert result['f1_log'][0] == pytest.approx(math.log(2.0))
        # f1=20.0 -> log(21)
        assert result['f1_log'][19] == pytest.approx(math.log(21.0))

    def test_zero_input(self):
        lf = pl.LazyFrame({'f1': [0.0]})
        t = Log1pFeatures(features=['f1'])
        result = t.fit_transform(lf).collect()
        # log(0+1) = log(1) = 0.0
        assert result['f1'][0] == pytest.approx(0.0)

    def test_null_propagation(self, sample_lf):
        t = Log1pFeatures(features=['f3'], suffix='_log')
        result = t.fit_transform(sample_lf).collect()
        # f3 has None at index 1
        assert result['f3_log'][1] is None
        # Non-null row should have a value
        assert result['f3_log'][0] is not None

    def test_suffix(self, sample_lf):
        t = Log1pFeatures(features=['f1'], suffix='_ln')
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_ln' in result.columns
        assert 'f1' in result.columns


# ---------------------------------------------------------------------------
# TestExpm1Features
# ---------------------------------------------------------------------------

class TestExpm1Features:

    def test_basic(self, sample_lf):
        t = Expm1Features(features=['f1'], suffix='_exp')
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_exp' in result.columns
        # f1=1.0 -> exp(1-1) = exp(0) = 1.0
        assert result['f1_exp'][0] == pytest.approx(1.0)
        # f1=2.0 -> exp(2-1) = exp(1) ~ 2.7183
        assert result['f1_exp'][1] == pytest.approx(math.exp(1.0))

    def test_zero_input(self):
        lf = pl.LazyFrame({'f1': [0.0]})
        t = Expm1Features(features=['f1'])
        result = t.fit_transform(lf).collect()
        # exp(0-1) = exp(-1) ~ 0.3679
        assert result['f1'][0] == pytest.approx(math.exp(-1.0))

    def test_null_propagation(self, sample_lf):
        t = Expm1Features(features=['f3'], suffix='_exp')
        result = t.fit_transform(sample_lf).collect()
        assert result['f3_exp'][1] is None
        assert result['f3_exp'][0] is not None

    def test_suffix(self, sample_lf):
        t = Expm1Features(features=['f1'], suffix='_e')
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_e' in result.columns


# ---------------------------------------------------------------------------
# TestPowerFeatures
# ---------------------------------------------------------------------------

class TestPowerFeatures:

    def test_default_power_2(self, sample_lf):
        t = PowerFeatures(features=['f1'], suffix='_pow')
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_pow' in result.columns
        # f1=3.0 -> 9.0
        assert result['f1_pow'][2] == pytest.approx(9.0)
        # f1=5.0 -> 25.0
        assert result['f1_pow'][4] == pytest.approx(25.0)

    def test_custom_power_3(self, sample_lf):
        t = PowerFeatures(features=['f1'], suffix='_p3', power=3)
        result = t.fit_transform(sample_lf).collect()
        # f1=2.0 -> 8.0
        assert result['f1_p3'][1] == pytest.approx(8.0)

    def test_fractional_power(self, sample_lf):
        t = PowerFeatures(features=['f1'], suffix='_sqrt', power=0.5)
        result = t.fit_transform(sample_lf).collect()
        # f1=4.0 -> 2.0
        assert result['f1_sqrt'][3] == pytest.approx(2.0)

    def test_null_propagation(self, sample_lf):
        t = PowerFeatures(features=['f3'], suffix='_pow')
        result = t.fit_transform(sample_lf).collect()
        assert result['f3_pow'][1] is None

    def test_suffix(self, sample_lf):
        t = PowerFeatures(features=['f1'], suffix='_sq')
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_sq' in result.columns
        assert 'f1' in result.columns


# ---------------------------------------------------------------------------
# TestInverseFeatures
# ---------------------------------------------------------------------------

class TestInverseFeatures:

    def test_basic(self, sample_lf):
        t = InverseFeatures(features=['f1'], suffix='_inv')
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_inv' in result.columns
        # f1=2.0 -> 0.5
        assert result['f1_inv'][1] == pytest.approx(0.5)
        # f1=5.0 -> 0.2
        assert result['f1_inv'][4] == pytest.approx(0.2)

    def test_zero_input(self):
        lf = pl.LazyFrame({'f1': [0.0]})
        t = InverseFeatures(features=['f1'])
        result = t.fit_transform(lf).collect()
        # 1/0 = inf
        assert result['f1'][0] == float('inf')

    def test_null_propagation(self, sample_lf):
        t = InverseFeatures(features=['f3'], suffix='_inv')
        result = t.fit_transform(sample_lf).collect()
        assert result['f3_inv'][1] is None

    def test_suffix(self, sample_lf):
        t = InverseFeatures(features=['f1'], suffix='_reciprocal')
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_reciprocal' in result.columns

    def test_negative_input(self):
        lf = pl.LazyFrame({'f1': [-2.0]})
        t = InverseFeatures(features=['f1'])
        result = t.fit_transform(lf).collect()
        assert result['f1'][0] == pytest.approx(-0.5)
