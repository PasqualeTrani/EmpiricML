"""Tests for target encoding transformers.

Covers: MeanTargetEncoder, StdTargetEncoder, MaxTargetEncoder,
        MinTargetEncoder, MedianTargetEncoder, KurtTargetEncoder,
        SkewTargetEncoder.
"""

import pytest
import polars as pl

from empml.transformers import (
    MeanTargetEncoder,
    StdTargetEncoder,
    MaxTargetEncoder,
    MinTargetEncoder,
    MedianTargetEncoder,
    KurtTargetEncoder,
    SkewTargetEncoder,
)


ALL_TARGET_ENCODERS = [
    MeanTargetEncoder,
    StdTargetEncoder,
    MaxTargetEncoder,
    MinTargetEncoder,
    MedianTargetEncoder,
    KurtTargetEncoder,
    SkewTargetEncoder,
]

DEFAULT_PREFIXES = {
    MeanTargetEncoder: 'mean_',
    StdTargetEncoder: 'std_',
    MaxTargetEncoder: 'max_',
    MinTargetEncoder: 'min_',
    MedianTargetEncoder: 'median_',
    KurtTargetEncoder: 'kurt_',
    SkewTargetEncoder: 'skew_',
}


# ---------------------------------------------------------------------------
# Shared parametrized tests
# ---------------------------------------------------------------------------

class TestTargetEncoderShared:

    @pytest.mark.parametrize("EncoderClass", ALL_TARGET_ENCODERS)
    def test_output_column_naming(self, target_lf, EncoderClass):
        prefix = DEFAULT_PREFIXES[EncoderClass]
        t = EncoderClass(features=['cat'], encoder_col='target')
        result = t.fit_transform(target_lf).collect()
        expected_col = f'{prefix}cat_encoded'
        assert expected_col in result.columns

    @pytest.mark.parametrize("EncoderClass", ALL_TARGET_ENCODERS)
    def test_replace_original(self, target_lf, EncoderClass):
        prefix = DEFAULT_PREFIXES[EncoderClass]
        t = EncoderClass(
            features=['cat'], encoder_col='target', replace_original=True
        )
        result = t.fit_transform(target_lf).collect()
        # Original 'cat' replaced with encoded values
        assert 'cat' in result.columns
        # No prefixed column
        expected_col = f'{prefix}cat_encoded'
        assert expected_col not in result.columns

    @pytest.mark.parametrize("EncoderClass", ALL_TARGET_ENCODERS)
    def test_unseen_category_fallback(
        self, target_lf, unseen_target_lf, EncoderClass
    ):
        prefix = DEFAULT_PREFIXES[EncoderClass]
        t = EncoderClass(features=['cat'], encoder_col='target')
        t.fit(target_lf)
        result = t.transform(unseen_target_lf).collect()
        col = f'{prefix}cat_encoded'
        # 'z' is unseen -> value == global fallback (not null)
        z_row = result.filter(pl.col('cat') == 'z')
        assert z_row[col][0] is not None
        assert z_row[col][0] == pytest.approx(t.global_encoded_val)

    @pytest.mark.parametrize("EncoderClass", ALL_TARGET_ENCODERS)
    def test_fit_returns_self(self, target_lf, EncoderClass):
        t = EncoderClass(features=['cat'], encoder_col='target')
        assert t.fit(target_lf) is t

    @pytest.mark.parametrize("EncoderClass", ALL_TARGET_ENCODERS)
    def test_custom_prefix_suffix(self, target_lf, EncoderClass):
        t = EncoderClass(
            features=['cat'], encoder_col='target',
            prefix='enc_', suffix='_val'
        )
        result = t.fit_transform(target_lf).collect()
        assert 'enc_cat_val' in result.columns


# ---------------------------------------------------------------------------
# Per-encoder aggregation correctness
# ---------------------------------------------------------------------------

class TestMeanTargetEncoder:

    def test_encoded_value(self, target_lf):
        t = MeanTargetEncoder(features=['cat'], encoder_col='target')
        result = t.fit_transform(target_lf).collect()
        col = 'mean_cat_encoded'
        # 'a' targets: indices 0,2,5,8,10,13,16,19 -> [10,12,11,13,14,15,9,16]
        expected = sum([10, 12, 11, 13, 14, 15, 9, 16]) / 8  # 12.5
        a_rows = result.filter(pl.col('cat') == 'a')
        assert a_rows[col][0] == pytest.approx(expected)


class TestStdTargetEncoder:

    def test_encoded_value(self, target_lf):
        t = StdTargetEncoder(features=['cat'], encoder_col='target')
        result = t.fit_transform(target_lf).collect()
        col = 'std_cat_encoded'
        a_rows = result.filter(pl.col('cat') == 'a')
        # Value should be a positive float (std of the 'a' group)
        assert a_rows[col][0] is not None
        assert a_rows[col][0] > 0


class TestMaxTargetEncoder:

    def test_encoded_value(self, target_lf):
        t = MaxTargetEncoder(features=['cat'], encoder_col='target')
        result = t.fit_transform(target_lf).collect()
        col = 'max_cat_encoded'
        # 'a' targets: [10,12,11,13,14,15,9,16] -> max=16.0
        a_rows = result.filter(pl.col('cat') == 'a')
        assert a_rows[col][0] == pytest.approx(16.0)


class TestMinTargetEncoder:

    def test_encoded_value(self, target_lf):
        t = MinTargetEncoder(features=['cat'], encoder_col='target')
        result = t.fit_transform(target_lf).collect()
        col = 'min_cat_encoded'
        # 'a' targets: [10,12,11,13,14,15,9,16] -> min=9.0
        a_rows = result.filter(pl.col('cat') == 'a')
        assert a_rows[col][0] == pytest.approx(9.0)


class TestMedianTargetEncoder:

    def test_encoded_value(self, target_lf):
        t = MedianTargetEncoder(features=['cat'], encoder_col='target')
        result = t.fit_transform(target_lf).collect()
        col = 'median_cat_encoded'
        # 'a' targets sorted: [9,10,11,12,13,14,15,16] -> median = (12+13)/2 = 12.5
        a_rows = result.filter(pl.col('cat') == 'a')
        assert a_rows[col][0] == pytest.approx(12.5)


class TestKurtTargetEncoder:

    def test_encoded_value(self, target_lf):
        t = KurtTargetEncoder(features=['cat'], encoder_col='target')
        result = t.fit_transform(target_lf).collect()
        col = 'kurt_cat_encoded'
        a_rows = result.filter(pl.col('cat') == 'a')
        # Kurtosis is numeric (can be negative for platykurtic distributions)
        assert a_rows[col][0] is not None


class TestSkewTargetEncoder:

    def test_encoded_value(self, target_lf):
        t = SkewTargetEncoder(features=['cat'], encoder_col='target')
        result = t.fit_transform(target_lf).collect()
        col = 'skew_cat_encoded'
        a_rows = result.filter(pl.col('cat') == 'a')
        # Skewness is numeric
        assert a_rows[col][0] is not None
