from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from empml.cv import KFold, TimeSeriesSplit, TrainTestSplit
from empml.data import (
    MSSQLDownloader,
    MySQLDownloader,
    OracleDownloader,
    PostgreSQLDownloader,
    RedshiftDownloader,
    SQLiteDownloader,
)
from empml.lab import Lab
from empml.lab_utils import prepare_predictions_for_save, setup_row_id_column
from empml.pipeline import relative_performance
from empml.transformers import (
    FrequencyEncoder,
    GenerateLags,
    KurtTargetEncoder,
    MaxTargetEncoder,
    MeanTargetEncoder,
    MedianTargetEncoder,
    MinMaxScaler,
    MinTargetEncoder,
    OrdinalEncoder,
    RobustScaler,
    SimpleImputer,
    SkewTargetEncoder,
    StandardScaler,
    StdTargetEncoder,
)
from empml.wrappers import TorchWrapper


def test_kfold_covers_uneven_input_once():
    data = pl.LazyFrame({"id": range(11)})

    splits = KFold(n_splits=3, random_state=7).split(data, "id")

    validation_ids = np.concatenate([valid for _, valid in splits])
    assert sorted(validation_ids.tolist()) == list(range(11))
    assert len(np.unique(validation_ids)) == 11
    for train, valid in splits:
        assert set(train).isdisjoint(valid)


def test_train_test_split_keeps_both_partitions_nonempty():
    train, test = TrainTestSplit(test_size=0.2, random_state=7).split(
        pl.LazyFrame({"id": [10, 11]}), "id"
    )[0]

    assert len(train) == len(test) == 1


@pytest.mark.parametrize(
    ("n_rows", "test_size", "expected_test_rows"),
    [(6, 0.25, 1), (7, 0.5, 3), (10, 0.25, 2)],
)
def test_train_test_split_keeps_floor_test_size(n_rows, test_size, expected_test_rows):
    train, test = TrainTestSplit(test_size=test_size, random_state=0).split(
        pl.LazyFrame({"id": range(n_rows)}), "id"
    )[0]

    assert len(test) == expected_test_rows
    assert len(train) == n_rows - expected_test_rows


def test_train_test_split_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least two rows"):
        TrainTestSplit().split(pl.LazyFrame({"id": [10]}), "id")


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)], pl.Date),
        (
            [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            pl.Datetime,
        ),
    ],
)
def test_time_series_split_accepts_native_temporal_columns(values, dtype):
    data = pl.LazyFrame(
        {"id": [1, 2, 3], "when": values}, schema={"id": pl.Int64, "when": dtype}
    )
    splitter = TimeSeriesSplit(
        [("2024-01-01", "2024-01-03", "2024-01-03", "2024-01-04")],
        "when",
    )

    train, valid = splitter.split(data, "id")[0]

    assert train.tolist() == [1, 2]
    assert valid.tolist() == [3]


def test_time_series_split_rejects_unsupported_date_type():
    data = pl.LazyFrame({"id": [1, 2], "when": [1, 2]})
    splitter = TimeSeriesSplit(
        [("2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03")],
        "when",
    )

    with pytest.raises(TypeError, match="Date, Datetime, or String"):
        splitter.split(data, "id")


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ([1, None, 3], "null"),
        ([1.0, float("nan"), 3.0], "null"),
        ([1, 1, 3], "unique"),
    ],
)
def test_explicit_row_ids_must_be_complete_and_unique(ids, message):
    with pytest.raises(ValueError, match=message):
        setup_row_id_column(pl.LazyFrame({"id": ids}), "id")


def test_explicit_row_id_must_exist():
    with pytest.raises(ValueError, match="not found"):
        setup_row_id_column(pl.LazyFrame({"value": [1]}), "id")


def test_prediction_artifact_contains_alignment_keys():
    evaluation = pl.DataFrame({"preds": [[20.0, 10.0], [30.0]]})
    keys = [
        pl.DataFrame({"id": [2, 1], "fold_number": [1, 1]}),
        pl.DataFrame({"id": [3], "fold_number": [2]}),
    ]

    result = prepare_predictions_for_save(evaluation, keys, "id")

    assert result.to_dict(as_series=False) == {
        "id": [2, 1, 3],
        "fold_number": [1, 1, 2],
        "preds": [20.0, 10.0, 30.0],
    }


def _prediction_lab(tmp_path: Path) -> Lab:
    lab = Lab.__new__(Lab)
    lab.name = str(tmp_path / "lab")
    lab.row_id = "id"
    lab.target = "target"
    lab.train = pl.LazyFrame({"id": [1, 2, 3, 4], "target": [10.0, 20.0, 30.0, 40.0]})
    lab.cv_indexes = [
        (np.array([3, 4]), np.array([2, 1])),
        (np.array([1, 2]), np.array([4, 3])),
    ]
    (tmp_path / "lab" / "predictions").mkdir(parents=True)
    return lab


def test_keyed_prediction_retrieval_aligns_shuffled_folds_and_partial_runs(tmp_path):
    lab = _prediction_lab(tmp_path)
    pl.DataFrame(
        {
            "id": [2, 1],
            "fold_number": [1, 1],
            "preds": [200.0, 100.0],
        }
    ).write_parquet(f"{lab.name}/predictions/predictions_1.parquet")

    result = lab.retrieve_predictions([1]).collect().sort("id")

    assert result["preds_1"].to_list() == [100.0, 200.0, None, None]


def test_legacy_prediction_retrieval_uses_source_filter_order(tmp_path):
    lab = _prediction_lab(tmp_path)
    pl.DataFrame({"preds": [100.0, 200.0, 300.0, 400.0]}).write_parquet(
        f"{lab.name}/predictions/predictions_1.parquet"
    )

    result = lab.retrieve_predictions([1]).collect().sort("id")

    assert result["preds_1"].to_list() == [100.0, 200.0, 300.0, 400.0]


def test_relative_performance_accepts_zero_candidate():
    assert relative_performance(minimize=True, x1=2.0, x2=0.0) == 100.0
    assert relative_performance(minimize=False, x1=2.0, x2=0.0) == -100.0


@pytest.mark.parametrize("candidate", [0.0, 1.0])
def test_relative_performance_returns_none_for_zero_reference(candidate):
    assert relative_performance(minimize=True, x1=0.0, x2=candidate) is None


def test_torch_wrapper_is_cloneable_without_resolving_defaults():
    wrapper = TorchWrapper(module=object, features=["x"], target="y")

    cloned = clone(wrapper)

    assert cloned.input_dim is None
    assert cloned.hidden_layers is None
    assert cloned.callbacks is None


def test_torch_wrapper_updates_nested_kwargs_and_forwards_generic_kwargs():
    wrapper = TorchWrapper(
        module=object,
        features=["x"],
        target="y",
        module__dropout=0.1,
        predict_nonlinearity="auto",
    )
    wrapper.estimator_ = object()
    wrapper.set_params(module__dropout=0.25, predict_nonlinearity=None)
    assert wrapper.estimator_ is None
    estimator = Mock()
    with (
        patch("empml.wrappers._check_torch_available"),
        patch(
            "empml.wrappers._check_skorch_available", return_value=(estimator, Mock())
        ),
    ):
        wrapper._create_estimator(np.array([1.0]))

    assert wrapper.kwargs["module__dropout"] == 0.25
    assert wrapper.estimator_ is estimator.return_value
    forwarded = estimator.call_args.kwargs
    assert forwarded["module__dropout"] == 0.25
    assert forwarded["predict_nonlinearity"] is None


def test_torch_wrapper_module_kwargs_override_architecture_defaults():
    wrapper = TorchWrapper(
        module=object, features=["x"], target="y", module__hidden_layers=[8]
    )
    estimator = Mock()
    with (
        patch("empml.wrappers._check_torch_available"),
        patch(
            "empml.wrappers._check_skorch_available", return_value=(estimator, Mock())
        ),
    ):
        wrapper._create_estimator(np.array([1.0]))

    assert estimator.call_args.kwargs["module__hidden_layers"] == [8]


def test_torch_wrapper_set_params_routes_constructor_params_to_attributes():
    wrapper = TorchWrapper(module=object, features=["x"], target="y")

    wrapper.set_params(lr=0.5, iterator_valid__shuffle=True, optimizer__momentum=0.9)

    assert wrapper.lr == 0.5
    assert wrapper.iterator_valid__shuffle is True
    assert wrapper.kwargs == {"optimizer__momentum": 0.9}
    assert clone(wrapper).get_params()["optimizer__momentum"] == 0.9


def test_imputer_ignores_nan_when_learning_statistic():
    data = pl.LazyFrame({"x": [1.0, float("nan"), 3.0, None]})

    result = SimpleImputer(["x"], strategy="mean").fit_transform(data).collect()

    assert result["x"].to_list() == pytest.approx([1.0, 2.0, 3.0, 2.0])


@pytest.mark.parametrize(
    "Encoder",
    [
        FrequencyEncoder,
        MeanTargetEncoder,
        StdTargetEncoder,
        MaxTargetEncoder,
        MinTargetEncoder,
        MedianTargetEncoder,
        KurtTargetEncoder,
        SkewTargetEncoder,
    ],
)
def test_encoders_match_learned_null_category(Encoder):
    train = pl.LazyFrame(
        {
            "cat": [None, None, None, None, "a", "a"],
            "target": [1.0, 2.0, 3.0, 8.0, 20.0, 30.0],
        }
    )
    transform = pl.LazyFrame({"cat": [None, "new"], "target": [0.0, 0.0]})
    encoder = (
        Encoder(features=["cat"], normalize=False)
        if Encoder is FrequencyEncoder
        else Encoder(features=["cat"], encoder_col="target")
    )

    result = encoder.fit(train).transform(transform).collect()
    output = (
        "freq_cat_encoded"
        if Encoder is FrequencyEncoder
        else encoder.target_encoder_dict["cat"].columns[1]
    )
    expected_null = (
        4.0
        if Encoder is FrequencyEncoder
        else encoder.target_encoder_dict["cat"]
        .filter(pl.col("cat").is_null())[output]
        .item()
    )

    assert result[output][0] == pytest.approx(expected_null)
    assert result[output][1] == pytest.approx(
        0.0 if Encoder is FrequencyEncoder else encoder.global_encoded_val
    )


@pytest.mark.parametrize(
    "make_transformer",
    [
        lambda: FrequencyEncoder(features=["cat"]),
        lambda: MeanTargetEncoder(features=["cat"], encoder_col="target"),
        lambda: StdTargetEncoder(features=["cat"], encoder_col="target"),
        lambda: MaxTargetEncoder(features=["cat"], encoder_col="target"),
        lambda: MinTargetEncoder(features=["cat"], encoder_col="target"),
        lambda: MedianTargetEncoder(features=["cat"], encoder_col="target"),
        lambda: KurtTargetEncoder(features=["cat"], encoder_col="target"),
        lambda: SkewTargetEncoder(features=["cat"], encoder_col="target"),
        lambda: OrdinalEncoder(features=["cat"]),
        lambda: GenerateLags(ts_index="cat", date_col="day", lag_col="target"),
    ],
)
def test_join_based_transformers_preserve_row_order(make_transformer):
    # Importing empml enables Polars' streaming engine, whose joins may reorder
    # rows; predictions are later attached to the untransformed frame by position.
    n_rows = 50_000
    rng = np.random.default_rng(0)
    data = pl.LazyFrame(
        {
            "row": range(n_rows),
            "cat": rng.choice(list("abcdefgh"), n_rows),
            "day": pl.date_range(
                date(2000, 1, 1),
                date(2000, 1, 1) + timedelta(days=n_rows - 1),
                eager=True,
            ),
            "target": rng.normal(size=n_rows),
        }
    )

    result = make_transformer().fit_transform(data).collect()

    assert result["row"].to_list() == list(range(n_rows))


@pytest.mark.parametrize("Scaler", [StandardScaler, MinMaxScaler, RobustScaler])
@pytest.mark.parametrize("values", [[5.0, None, 5.0], [None, None, None]])
def test_scalers_preserve_nulls_for_constant_and_all_null_features(Scaler, values):
    data = pl.LazyFrame({"x": values}, schema={"x": pl.Float64})

    result = Scaler(["x"]).fit_transform(data).collect()["x"].to_list()

    assert [value is None for value in result] == [value is None for value in values]
    assert all(value == 0.0 for value in result if value is not None)


@pytest.mark.parametrize(
    ("Downloader", "scheme", "port"),
    [
        (PostgreSQLDownloader, "postgresql", 5432),
        (MySQLDownloader, "mysql", 3306),
        (MSSQLDownloader, "mssql", 1433),
        (OracleDownloader, "oracle", 1521),
        (RedshiftDownloader, "redshift", 5439),
    ],
)
def test_database_name_uri_segments_encode_reserved_characters(
    Downloader, scheme, port
):
    downloader = Downloader("select 1", "host", "user name", "p@ss", "db/name?#", port)

    assert downloader.connection_uri == (
        f"{scheme}://user+name:p%40ss@host:{port}/db%2Fname%3F%23"
    )


def test_sqlite_path_encodes_reserved_characters():
    sqlite = SQLiteDownloader("select 1", "/tmp/a b?#.sqlite")

    assert sqlite.connection_uri == "sqlite:///tmp/a%20b%3F%23.sqlite"


def test_sqlite_downloader_reads_path_with_reserved_characters(tmp_path):
    pytest.importorskip("connectorx")
    import sqlite3

    path = tmp_path / "data dir" / "a b?#%20.sqlite"
    path.parent.mkdir()
    connection = sqlite3.connect(path)
    connection.execute("create table t (x integer)")
    connection.execute("insert into t values (7)")
    connection.commit()
    connection.close()

    data = SQLiteDownloader("select x from t", str(path)).get_data().collect()

    assert data["x"].to_list() == [7]
