"""Contracts for transformers that share one implementation.

These pin the instance attributes, repr, warnings, and output of the target
encoders and row-wise feature transformers. Attribute names and order matter:
``BaseTransformer.__repr__`` is built from them, HPO descriptions embed that
repr, and pickled pipelines restore them.
"""

import warnings

import polars as pl
import pytest

from empml.transformers import (
    AvgFeatures,
    KurtTargetEncoder,
    MaxFeatures,
    MaxTargetEncoder,
    MeanTargetEncoder,
    MedianFeatures,
    MedianTargetEncoder,
    MinFeatures,
    MinTargetEncoder,
    SkewTargetEncoder,
    StdFeatures,
    StdTargetEncoder,
)

TARGET_ENCODERS = [
    (MeanTargetEncoder, "mean_", "mean"),
    (StdTargetEncoder, "std_", "std"),
    (MaxTargetEncoder, "max_", "max"),
    (MinTargetEncoder, "min_", "min"),
    (MedianTargetEncoder, "median_", "median"),
    (KurtTargetEncoder, "kurt_", "kurtosis"),
    (SkewTargetEncoder, "skew_", "skew"),
]
ENCODER_IDS = [encoder.__name__ for encoder, _, _ in TARGET_ENCODERS]

DUPLICATE_NAMES_WARNING = (
    "prefix='' and suffix='' with replace_original=False would create duplicate "
    "column names. Setting replace_original=True automatically."
)
IGNORED_AFFIXES_WARNING = (
    "replace_original=True: prefix and suffix arguments are ignored. "
    "Encoded columns will use original column names."
)


def _empty_affixes_warning(prefix: str) -> str:
    return (
        "replace_original=True with prefix='' and suffix='' would cause errors. "
        f"Setting prefix='{prefix}' and suffix='_encoded' for internal processing."
    )


def _frame() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "cat": ["a", "b", "a", None, "b", "a", "c"],
            "y": [1.0, 4.0, 2.0, 8.0, 5.0, 9.0, 3.0],
        }
    )


def _collect_warnings(build):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder = build()
    return encoder, [(w.category, str(w.message)) for w in caught]


@pytest.mark.parametrize("Encoder, prefix, _", TARGET_ENCODERS, ids=ENCODER_IDS)
def test_target_encoder_attributes_and_repr(Encoder, prefix, _):
    encoder, caught = _collect_warnings(lambda: Encoder(["cat"], "y"))

    assert caught == []
    assert vars(encoder) == {
        "features": ["cat"],
        "encoder_col": "y",
        "prefix": prefix,
        "suffix": "_encoded",
        "replace_original": False,
    }
    assert repr(encoder) == (
        f"{Encoder.__name__}(features=['cat'], encoder_col='y', "
        f"prefix='{prefix}', suffix='_encoded', replace_original=False)"
    )

    encoder.fit(_frame())
    assert list(vars(encoder))[5:] == ["target_encoder_dict", "global_encoded_val"]


@pytest.mark.parametrize("Encoder, prefix, _", TARGET_ENCODERS, ids=ENCODER_IDS)
def test_target_encoder_empty_affixes_warn_and_replace(Encoder, prefix, _):
    encoder, caught = _collect_warnings(
        lambda: Encoder(["cat"], "y", prefix="", suffix="")
    )

    assert caught == [
        (UserWarning, DUPLICATE_NAMES_WARNING),
        (UserWarning, _empty_affixes_warning(prefix)),
    ]
    assert (encoder.prefix, encoder.suffix, encoder.replace_original) == (
        prefix,
        "_encoded",
        True,
    )


@pytest.mark.parametrize("Encoder, prefix, _", TARGET_ENCODERS, ids=ENCODER_IDS)
def test_target_encoder_ignored_affixes_warn(Encoder, prefix, _):
    _, caught = _collect_warnings(
        lambda: Encoder(["cat"], "y", prefix="p_", replace_original=True)
    )
    assert caught == [(UserWarning, IGNORED_AFFIXES_WARNING)]

    _, caught = _collect_warnings(lambda: Encoder(["cat"], "y", replace_original=True))
    assert caught == []


@pytest.mark.parametrize(
    "Encoder, prefix, aggregation", TARGET_ENCODERS, ids=ENCODER_IDS
)
@pytest.mark.parametrize("replace_original", [False, True])
def test_target_encoder_output(Encoder, prefix, aggregation, replace_original):
    lf = _frame()
    encoder = Encoder(["cat"], "y", replace_original=replace_original)
    result = encoder.fit_transform(lf).collect()

    y = pl.col("y")
    expected_values = (
        lf.with_columns(
            getattr(y, aggregation)().over("cat").alias("value"),
        )
        .collect()["value"]
        .fill_null(encoder.global_encoded_val)
    )
    global_value = lf.select(getattr(y, aggregation)()).collect().item()
    column = "cat" if replace_original else f"{prefix}cat_encoded"

    expected_global = 0.0 if global_value is None else global_value
    assert encoder.global_encoded_val == pytest.approx(expected_global, nan_ok=True)
    assert result.columns == (
        ["y", "cat"] if replace_original else ["cat", "y", column]
    )
    assert result[column].to_list() == pytest.approx(
        expected_values.to_list(), nan_ok=True
    )


def test_target_encoder_unseen_category_uses_global_value():
    encoder = MeanTargetEncoder(["cat"], "y").fit(_frame())
    unseen = pl.LazyFrame({"cat": ["zzz"], "y": [0.0]})

    assert encoder.transform(unseen).collect()["mean_cat_encoded"].to_list() == [
        pytest.approx(32 / 7)
    ]


HORIZONTAL = [
    # Nulls are ignored row-wise.
    (AvgFeatures, [2.0, 5.0]),
    (MaxFeatures, [3.0, 5.0]),
    (MinFeatures, [1.0, 5.0]),
    (StdFeatures, [1.0, 0.0]),
    (MedianFeatures, [2.0, 5.0]),
]


@pytest.mark.parametrize(
    "Transformer, expected", HORIZONTAL, ids=[t.__name__ for t, _ in HORIZONTAL]
)
def test_horizontal_features(Transformer, expected):
    transformer = Transformer(["a", "b", "c"], "out")
    lf = pl.LazyFrame(
        {"a": [1.0, 5.0], "b": [2.0, None], "c": [3.0, 5.0]},
    )

    assert vars(transformer) == {"features": ["a", "b", "c"], "new_feature": "out"}
    assert repr(transformer) == (
        f"{Transformer.__name__}(features=['a', 'b', 'c'], new_feature='out')"
    )
    assert transformer.fit(lf) is transformer
    result = transformer.transform(lf).collect()
    assert result.columns == ["a", "b", "c", "out"]
    assert result["out"].to_list() == pytest.approx(expected)
