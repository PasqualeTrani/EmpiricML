"""Tests for categorical encoding transformers.

Covers: OrdinalEncoder, DummyEncoder, FrequencyEncoder.
"""

import pytest
import polars as pl

from empml.transformers import OrdinalEncoder, DummyEncoder, FrequencyEncoder


# ---------------------------------------------------------------------------
# TestOrdinalEncoder
# ---------------------------------------------------------------------------

class TestOrdinalEncoder:

    def test_basic_encoding(self, sample_lf):
        t = OrdinalEncoder(features=['cat'])
        result = t.fit_transform(sample_lf).collect()
        col = 'cat_ordinal_encoded'
        assert col in result.columns
        # Sorted unique: ['a', 'b', 'c'] -> 0, 1, 2
        a_rows = result.filter(pl.col('cat') == 'a')
        assert a_rows[col][0] == 0
        b_rows = result.filter(pl.col('cat') == 'b')
        assert b_rows[col][0] == 1
        c_rows = result.filter(pl.col('cat') == 'c')
        assert c_rows[col][0] == 2

    def test_unseen_category(self, sample_lf, unseen_lf):
        t = OrdinalEncoder(features=['cat'])
        t.fit(sample_lf)
        result = t.transform(unseen_lf).collect()
        col = 'cat_ordinal_encoded'
        # 'z' is unseen -> -9999
        z_rows = result.filter(pl.col('cat') == 'z')
        assert z_rows[col][0] == -9999

    def test_null_handling(self, sample_lf, unseen_lf):
        # Use a LazyFrame with a null value embedded
        lf = pl.LazyFrame({'cat': ['a', 'b', None, 'c', 'a']})
        t = OrdinalEncoder(features=['cat'])
        result = t.fit_transform(lf).collect()
        col = 'cat_ordinal_encoded'
        # Non-null rows should get ordinal values
        assert result.filter(pl.col('cat') == 'a')[col][0] == 0
        assert result.filter(pl.col('cat') == 'b')[col][0] == 1
        assert result.filter(pl.col('cat') == 'c')[col][0] == 2

    def test_replace_original(self, sample_lf):
        t = OrdinalEncoder(features=['cat'], replace_original=True)
        result = t.fit_transform(sample_lf).collect()
        # 'cat' should exist (replaced with ordinal integers)
        assert 'cat' in result.columns
        # No suffixed column
        assert 'cat_ordinal_encoded' not in result.columns
        # Value check
        a_rows = result.filter(pl.col('cat') == 0)
        assert len(a_rows) == 8  # 'a' appears 8 times

    def test_custom_suffix(self, sample_lf):
        t = OrdinalEncoder(features=['cat'], suffix='_ord')
        result = t.fit_transform(sample_lf).collect()
        assert 'cat_ord' in result.columns

    def test_multiple_features(self):
        lf = pl.LazyFrame({
            'cat1': ['x', 'y', 'z', 'x'],
            'cat2': ['m', 'n', 'm', 'n'],
        })
        t = OrdinalEncoder(features=['cat1', 'cat2'])
        result = t.fit_transform(lf).collect()
        assert 'cat1_ordinal_encoded' in result.columns
        assert 'cat2_ordinal_encoded' in result.columns


# ---------------------------------------------------------------------------
# TestDummyEncoder
# ---------------------------------------------------------------------------

class TestDummyEncoder:

    def test_basic_encoding(self, sample_lf):
        t = DummyEncoder(features=['cat'])
        result = t.fit_transform(sample_lf).collect()
        assert 'cat_dummy_a' in result.columns
        assert 'cat_dummy_b' in result.columns
        assert 'cat_dummy_c' in result.columns
        assert 'cat_dummy_null' in result.columns
        assert 'cat_dummy_unknown' in result.columns
        # 'a' rows should have cat_dummy_a=1, others=0
        a_row = result.filter(pl.col('cat') == 'a')
        assert a_row['cat_dummy_a'][0] == 1
        assert a_row['cat_dummy_b'][0] == 0
        assert a_row['cat_dummy_c'][0] == 0

    def test_unseen_category(self, sample_lf, unseen_lf):
        t = DummyEncoder(features=['cat'])
        t.fit(sample_lf)
        result = t.transform(unseen_lf).collect()
        # 'z' is unseen -> unknown=1, all known categories=0
        z_row = result.filter(pl.col('cat') == 'z')
        assert z_row['cat_dummy_unknown'][0] == 1
        assert z_row['cat_dummy_a'][0] == 0
        assert z_row['cat_dummy_b'][0] == 0
        assert z_row['cat_dummy_c'][0] == 0

    def test_null_handling(self):
        lf = pl.LazyFrame({'cat': ['a', None, 'b']})
        t = DummyEncoder(features=['cat'])
        result = t.fit_transform(lf).collect()
        # null row -> cat_dummy_null=1
        assert result['cat_dummy_null'][1] == 1
        # null == 'a' evaluates to null in Polars, cast to Int8 -> null
        # so category dummies are null (not 0) for null input rows
        assert result['cat_dummy_a'][1] is None
        assert result['cat_dummy_b'][1] is None

    def test_dtype_is_int8(self, sample_lf):
        t = DummyEncoder(features=['cat'])
        result = t.fit_transform(sample_lf).collect()
        for col in ['cat_dummy_a', 'cat_dummy_b', 'cat_dummy_c',
                     'cat_dummy_null', 'cat_dummy_unknown']:
            assert result[col].dtype == pl.Int8

    def test_multiple_features(self):
        lf = pl.LazyFrame({
            'cat1': ['a', 'b', 'a'],
            'cat2': ['x', 'y', 'x'],
        })
        t = DummyEncoder(features=['cat1', 'cat2'])
        result = t.fit_transform(lf).collect()
        assert 'cat1_dummy_a' in result.columns
        assert 'cat2_dummy_x' in result.columns


# ---------------------------------------------------------------------------
# TestFrequencyEncoder (migrated)
# ---------------------------------------------------------------------------

class TestFrequencyEncoder:

    def test_normalize_true(self, sample_lf):
        t = FrequencyEncoder(features=['cat'], normalize=True)
        result = t.fit_transform(sample_lf).collect()
        col = 'freq_cat_encoded'
        assert col in result.columns
        # 'a' appears 8 times out of 20
        a_rows = result.filter(pl.col('cat') == 'a')
        assert a_rows[col][0] == pytest.approx(8 / 20)

    def test_normalize_false(self, sample_lf):
        t = FrequencyEncoder(features=['cat'], normalize=False)
        result = t.fit_transform(sample_lf).collect()
        col = 'freq_cat_encoded'
        a_rows = result.filter(pl.col('cat') == 'a')
        assert a_rows[col][0] == 8

    def test_unseen_category(self, sample_lf, unseen_lf):
        t = FrequencyEncoder(features=['cat'], normalize=True)
        t.fit(sample_lf)
        result = t.transform(unseen_lf).collect()
        col = 'freq_cat_encoded'
        # 'z' is unseen -> should be 0
        z_rows = result.filter(pl.col('cat') == 'z')
        assert z_rows[col][0] == pytest.approx(0.0)

    def test_replace_original(self, sample_lf):
        t = FrequencyEncoder(
            features=['cat'], normalize=True, replace_original=True
        )
        result = t.fit_transform(sample_lf).collect()
        assert 'cat' in result.columns
        assert 'freq_cat_encoded' not in result.columns
