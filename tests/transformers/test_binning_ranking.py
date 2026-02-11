"""Tests for binning and ranking transformers.

Covers: QuantileBinning, RankFeatures.
"""

import pytest
import polars as pl
import numpy as np

from empml.transformers import QuantileBinning, RankFeatures


# ---------------------------------------------------------------------------
# TestQuantileBinning (migrated)
# ---------------------------------------------------------------------------

class TestQuantileBinning:

    def test_basic_bins(self, sample_lf):
        t = QuantileBinning(features=['f1'], num_bins=4)
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_qbin' in result.columns
        unique_bins = result['f1_qbin'].drop_nulls().unique()
        assert len(unique_bins) <= 4

    def test_with_labels(self, sample_lf):
        labels = ['low', 'med_low', 'med_high', 'high']
        t = QuantileBinning(features=['f1'], num_bins=4, labels=labels)
        result = t.fit_transform(sample_lf).collect()
        unique_vals = result['f1_qbin'].drop_nulls().unique().to_list()
        for v in unique_vals:
            assert v in labels

    def test_unseen_values(self, sample_lf, unseen_lf):
        t = QuantileBinning(features=['f1'], num_bins=4)
        t.fit(sample_lf)
        result = t.transform(unseen_lf).collect()
        # 100.0 and 200.0 are above training max -> should be in the highest bin
        assert result['f1_qbin'][0] is not None
        assert result['f1_qbin'][1] is not None

    def test_null_handling(self, sample_lf):
        t = QuantileBinning(features=['f3'], num_bins=4)
        result = t.fit_transform(sample_lf).collect()
        for i in [1, 3, 6, 11, 14, 18]:
            assert result['f3_qbin'][i] is None

    def test_identical_values(self):
        lf = pl.LazyFrame({'f1': [5.0, 5.0, 5.0, 5.0, 5.0]})
        t = QuantileBinning(features=['f1'], num_bins=4)
        result = t.fit_transform(lf).collect()
        assert result['f1_qbin'].unique().to_list() == [0]

    def test_invalid_labels_length(self):
        with pytest.raises(ValueError, match="labels length"):
            QuantileBinning(features=['f1'], num_bins=4, labels=['a', 'b'])


# ---------------------------------------------------------------------------
# TestRankFeatures (migrated)
# ---------------------------------------------------------------------------

class TestRankFeatures:

    def test_average_method(self, sample_lf):
        t = RankFeatures(features=['f1'])
        result = t.fit_transform(sample_lf).collect()
        assert 'f1_rank' in result.columns
        ranks = result['f1_rank'].to_numpy()
        assert ranks[0] == pytest.approx(0.0)
        assert ranks[-1] == pytest.approx(1.0)

    def test_min_method(self, sample_lf):
        t = RankFeatures(features=['f1'], method='min')
        result = t.fit_transform(sample_lf).collect()
        ranks = result['f1_rank'].to_numpy()
        assert ranks[0] == pytest.approx(0.0)
        assert ranks[-1] == pytest.approx(1.0)

    def test_dense_method(self, sample_lf):
        t = RankFeatures(features=['f1'], method='dense')
        result = t.fit_transform(sample_lf).collect()
        ranks = result['f1_rank'].to_numpy()
        assert ranks[0] == pytest.approx(0.0)
        assert ranks[-1] == pytest.approx(1.0)

    def test_null_handling(self, sample_lf):
        t = RankFeatures(features=['f3'])
        result = t.fit_transform(sample_lf).collect()
        assert np.isnan(result['f3_rank'][1])
        assert np.isnan(result['f3_rank'][3])

    def test_out_of_range(self, sample_lf, unseen_lf):
        t = RankFeatures(features=['f1'])
        t.fit(sample_lf)
        result = t.transform(unseen_lf).collect()
        ranks = result['f1_rank'].to_numpy()
        assert ranks[0] == pytest.approx(1.0)
        assert ranks[1] == pytest.approx(1.0)
