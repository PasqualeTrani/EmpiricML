"""Tests for sklearn-based transformers.

Covers: KMeansCluster, PCATransformer.
"""

import polars as pl

from empml.transformers import KMeansCluster, PCATransformer

# ---------------------------------------------------------------------------
# TestKMeansCluster (migrated)
# ---------------------------------------------------------------------------


class TestKMeansCluster:
    def test_basic_clustering(self, sample_lf):
        t = KMeansCluster(features=["f1", "f2"], num_clusters=3)
        result = t.fit_transform(sample_lf).collect()
        assert "kmeans_cluster" in result.columns
        assert result["kmeans_cluster"].dtype == pl.Int32

    def test_num_clusters(self, sample_lf):
        t = KMeansCluster(features=["f1", "f2"], num_clusters=3)
        result = t.fit_transform(sample_lf).collect()
        unique_labels = result["kmeans_cluster"].unique()
        assert len(unique_labels) <= 3

    def test_null_handling(self, sample_lf):
        t = KMeansCluster(features=["f1", "f3"], num_clusters=3)
        result = t.fit_transform(sample_lf).collect()
        assert result["kmeans_cluster"].null_count() == 0

    def test_reproducibility(self, sample_lf):
        t1 = KMeansCluster(features=["f1", "f2"], num_clusters=3, random_state=42)
        t2 = KMeansCluster(features=["f1", "f2"], num_clusters=3, random_state=42)
        r1 = t1.fit_transform(sample_lf).collect()["kmeans_cluster"].to_list()
        r2 = t2.fit_transform(sample_lf).collect()["kmeans_cluster"].to_list()
        assert r1 == r2

    def test_custom_feature_name(self, sample_lf):
        t = KMeansCluster(
            features=["f1", "f2"], num_clusters=3, new_feature="my_cluster"
        )
        result = t.fit_transform(sample_lf).collect()
        assert "my_cluster" in result.columns


# ---------------------------------------------------------------------------
# TestPCATransformer (migrated)
# ---------------------------------------------------------------------------


class TestPCATransformer:
    def test_basic_pca(self, sample_lf):
        t = PCATransformer(features=["f1", "f2"], n_components=2)
        result = t.fit_transform(sample_lf).collect()
        assert "pc_0" in result.columns
        assert "pc_1" in result.columns

    def test_n_components(self, sample_lf):
        t = PCATransformer(features=["f1", "f2"], n_components=1)
        result = t.fit_transform(sample_lf).collect()
        assert "pc_0" in result.columns
        assert "pc_1" not in result.columns

    def test_custom_prefix(self, sample_lf):
        t = PCATransformer(features=["f1", "f2"], n_components=2, prefix="pca_comp_")
        result = t.fit_transform(sample_lf).collect()
        assert "pca_comp_0" in result.columns
        assert "pca_comp_1" in result.columns

    def test_null_handling(self, sample_lf):
        t = PCATransformer(features=["f1", "f3"], n_components=2)
        result = t.fit_transform(sample_lf).collect()
        assert result["pc_0"].null_count() == 0
        assert result["pc_1"].null_count() == 0
