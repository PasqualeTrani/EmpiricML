"""Tests for time series transformers.

Covers: GenerateLags.
"""

import pytest
import polars as pl

from empml.transformers import GenerateLags


class TestGenerateLags:

    def test_single_lag(self, timeseries_lf):
        t = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='days', lag_min=1, lag_max=1,
        )
        result = t.fit_transform(timeseries_lf).collect()
        assert 'value_lag1days' in result.columns
        # Entity A, date 2024-01-02 (value=1.0) should have lag = 0.0 (2024-01-01)
        row = result.filter(
            (pl.col('entity_id') == 'A')
            & (pl.col('date') == pl.lit('2024-01-02').str.to_date())
        )
        assert row['value_lag1days'][0] == pytest.approx(0.0)

    def test_multiple_lags(self, timeseries_lf):
        t = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='days', lag_min=1, lag_max=3, lag_step=1,
        )
        result = t.fit_transform(timeseries_lf).collect()
        assert 'value_lag1days' in result.columns
        assert 'value_lag2days' in result.columns
        assert 'value_lag3days' in result.columns

    def test_lag_step(self, timeseries_lf):
        t = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='days', lag_min=1, lag_max=5, lag_step=2,
        )
        result = t.fit_transform(timeseries_lf).collect()
        assert 'value_lag1days' in result.columns
        assert 'value_lag3days' in result.columns
        assert 'value_lag5days' in result.columns
        assert 'value_lag2days' not in result.columns

    def test_null_for_missing_lags(self, timeseries_lf):
        t = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='days', lag_min=1, lag_max=1,
        )
        result = t.fit_transform(timeseries_lf).collect()
        # First row of entity A (2024-01-01): no data for 2023-12-31 -> null
        row = result.filter(
            (pl.col('entity_id') == 'A')
            & (pl.col('date') == pl.lit('2024-01-01').str.to_date())
        )
        assert row['value_lag1days'][0] is None

    def test_entity_isolation(self, timeseries_lf):
        t = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='days', lag_min=1, lag_max=1,
        )
        result = t.fit_transform(timeseries_lf).collect()
        # Entity B, date 2024-01-02: value=2.0, lag should be 0.0 (B's 2024-01-01)
        row = result.filter(
            (pl.col('entity_id') == 'B')
            & (pl.col('date') == pl.lit('2024-01-02').str.to_date())
        )
        assert row['value_lag1days'][0] == pytest.approx(0.0)

    def test_hourly_frequency(self, timeseries_datetime_lf):
        t = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='hours', lag_min=1, lag_max=2,
        )
        result = t.fit_transform(timeseries_datetime_lf).collect()
        assert 'value_lag1hours' in result.columns
        assert 'value_lag2hours' in result.columns

    def test_invalid_frequency(self):
        with pytest.raises(ValueError, match='lag_frequency'):
            GenerateLags(
                ts_index='entity_id', date_col='date', lag_col='value',
                lag_frequency='months',
            )

    def test_non_date_column_raises(self):
        lf = pl.LazyFrame({
            'entity_id': ['A', 'A'],
            'date': [1, 2],  # int, not date
            'value': [1.0, 2.0],
        })
        t = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='days',
        )
        with pytest.raises(TypeError, match="Date or Datetime"):
            t.fit(lf)

    def test_fit_transform_consistency(self, timeseries_lf):
        t1 = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='days', lag_min=1, lag_max=1,
        )
        r1 = t1.fit_transform(timeseries_lf).collect()

        t2 = GenerateLags(
            ts_index='entity_id', date_col='date', lag_col='value',
            lag_frequency='days', lag_min=1, lag_max=1,
        )
        t2.fit(timeseries_lf)
        r2 = t2.transform(timeseries_lf).collect()

        assert r1.columns == r2.columns
        assert r1.shape == r2.shape
