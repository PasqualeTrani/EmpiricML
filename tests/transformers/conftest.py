"""Fixtures specific to transformer tests."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest


@pytest.fixture
def target_lf() -> pl.LazyFrame:
    """LazyFrame with a numeric target column for target encoder tests.

    Categories 'a', 'b', 'c' have different target distributions.
    """
    return pl.LazyFrame(
        {
            "f1": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
            ],
            "cat": [
                "a",
                "b",
                "a",
                "c",
                "b",
                "a",
                "c",
                "b",
                "a",
                "c",
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "a",
                "c",
                "b",
                "a",
            ],
            "target": [
                10.0,
                5.0,
                12.0,
                2.0,
                6.0,
                11.0,
                3.0,
                7.0,
                13.0,
                1.0,
                14.0,
                4.0,
                2.5,
                15.0,
                5.5,
                3.5,
                9.0,
                4.0,
                8.0,
                16.0,
            ],
        }
    )


@pytest.fixture
def unseen_target_lf() -> pl.LazyFrame:
    """LazyFrame with unseen category for target encoder transform tests."""
    return pl.LazyFrame(
        {
            "f1": [100.0, 200.0],
            "cat": ["a", "z"],
            "target": [50.0, 60.0],
        }
    )


@pytest.fixture
def timeseries_lf() -> pl.LazyFrame:
    """LazyFrame for GenerateLags tests with Date column (daily)."""
    dates_a = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    dates_b = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    return pl.LazyFrame(
        {
            "entity_id": ["A"] * 10 + ["B"] * 10,
            "date": dates_a + dates_b,
            "value": [float(i) for i in range(10)] + [float(i * 2) for i in range(10)],
        }
    )


@pytest.fixture
def timeseries_datetime_lf() -> pl.LazyFrame:
    """LazyFrame with Datetime column (hourly) for GenerateLags tests."""
    dts_a = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(10)]
    dts_b = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(10)]
    return pl.LazyFrame(
        {
            "entity_id": ["A"] * 10 + ["B"] * 10,
            "date": dts_a + dts_b,
            "value": [float(i) for i in range(10)] + [float(i * 2) for i in range(10)],
        }
    )
