"""EmpiricML: a framework for building robust tabular machine learning models."""

import polars as pl

# Every EmpiricML module relies on Polars' streaming engine for .collect().
# Python runs this package initializer before any submodule, so setting it once
# here applies to every import path.
pl.Config.set_engine_affinity(engine="streaming")
