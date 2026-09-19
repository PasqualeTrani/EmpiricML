"""
On-disk storage of a Lab's artifacts.

A Lab named ``name`` keeps its files under ``./<name>/`` relative to the working
directory: pickled pipelines, compressed prediction parquet files, and pickled
Lab checkpoints.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class LabArtifacts:
    """Paths and file I/O for one Lab's artifacts."""

    lab_name: str

    def create_directories(self) -> None:
        for folder in ("pipelines", "predictions", "check_points"):
            os.makedirs(f"./{self.lab_name}/{folder}", exist_ok=True)

    def _pipeline_path(self, experiment_id: int) -> str:
        return f"./{self.lab_name}/pipelines/pipeline_{experiment_id}.pkl"

    def _checkpoint_path(self, check_point_name: str) -> str:
        return f"./{self.lab_name}/check_points/{check_point_name}.pkl"

    def save_pipeline(self, pipeline: Any, experiment_id: int) -> None:
        with open(self._pipeline_path(experiment_id), "wb") as file:
            pickle.dump(pipeline, file)

    def load_pipeline(self, experiment_id: int) -> Any:
        with open(self._pipeline_path(experiment_id), "rb") as file:
            return pickle.load(file)

    def save_predictions(self, predictions: pl.DataFrame, experiment_id: int) -> None:
        predictions.write_parquet(
            f"./{self.lab_name}/predictions/predictions_{experiment_id}.parquet",
            compression="zstd",
            compression_level=22,
        )

    def load_predictions(self, experiment_id: int) -> pl.DataFrame:
        """
        Load stored predictions, renaming ``preds`` to ``preds_<experiment_id>``.

        Keyed artifacts also contain the row ID and ``fold_number``; legacy
        artifacts contain only ``preds``.
        """
        # Unlike the writers, this path has no "./" prefix. The two only differ for
        # absolute Lab names, where writes land under the working directory.
        predictions = pl.read_parquet(
            f"{self.lab_name}/predictions/predictions_{experiment_id}.parquet"
        )
        return predictions.rename({"preds": f"preds_{experiment_id}"})

    def save_checkpoint(self, lab: Any, check_point_name: str) -> None:
        with open(self._checkpoint_path(check_point_name), "wb") as file:
            pickle.dump(lab, file)

    def load_checkpoint(self, check_point_name: str) -> Any:
        with open(self._checkpoint_path(check_point_name), "rb") as file:
            return pickle.load(file)
