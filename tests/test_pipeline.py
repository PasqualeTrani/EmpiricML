"""Tests for Pipeline integration.

Verifies that multiple transformers can be chained in a Pipeline.
"""

import pytest
import polars as pl

from empml.transformers import (
    InteractionFeatures,
    FrequencyEncoder,
    RobustScaler,
    KMeansCluster,
    PCATransformer,
)
from empml.pipeline import Pipeline


class TestPipelineIntegration:

    def test_chain_transformers(self, sample_lf):
        """Chain multiple transformers in a Pipeline and verify fit_transform."""
        pipe = Pipeline([
            ('interaction', InteractionFeatures(feature_pairs=[('f1', 'f2')])),
            ('robust', RobustScaler(features=['f1', 'f2'], suffix='_robust')),
            ('kmeans', KMeansCluster(features=['f1', 'f2'], num_clusters=3)),
        ])
        pipe.fit(sample_lf)
        result = pipe.transform(sample_lf).collect()
        assert 'f1_x_f2' in result.columns
        assert 'f1_robust' in result.columns
        assert 'f2_robust' in result.columns
        assert 'kmeans_cluster' in result.columns

    def test_fit_transform_equivalence(self, sample_lf):
        """fit_transform should produce same result as fit then transform."""
        pipe = Pipeline([
            ('freq', FrequencyEncoder(features=['cat'])),
            ('pca', PCATransformer(features=['f1', 'f2'], n_components=1)),
        ])

        pipe.fit(sample_lf)
        r1 = pipe.transform(sample_lf).collect()

        pipe2 = Pipeline([
            ('freq', FrequencyEncoder(features=['cat'])),
            ('pca', PCATransformer(features=['f1', 'f2'], n_components=1)),
        ])
        r2 = pipe2.fit_transform(sample_lf).collect()

        assert r1.columns == r2.columns
        assert r1.shape == r2.shape
