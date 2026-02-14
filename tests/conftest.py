"""Shared fixtures for the EmpiricML test suite."""

import polars as pl
import pytest


@pytest.fixture
def sample_lf() -> pl.LazyFrame:
    """LazyFrame with numeric, nullable, and categorical columns (~20 rows)."""
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
            "f2": [
                20.0,
                19.0,
                18.0,
                17.0,
                16.0,
                15.0,
                14.0,
                13.0,
                12.0,
                11.0,
                10.0,
                9.0,
                8.0,
                7.0,
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ],
            "f3": [
                1.0,
                None,
                3.0,
                None,
                5.0,
                6.0,
                None,
                8.0,
                9.0,
                10.0,
                11.0,
                None,
                13.0,
                14.0,
                None,
                16.0,
                17.0,
                18.0,
                None,
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
        }
    )


@pytest.fixture
def unseen_lf() -> pl.LazyFrame:
    """LazyFrame with an unseen category for testing encoder behavior."""
    return pl.LazyFrame(
        {
            "f1": [100.0, 200.0],
            "f2": [0.5, 0.5],
            "f3": [None, 50.0],
            "cat": ["a", "z"],  # 'z' is unseen
        }
    )
