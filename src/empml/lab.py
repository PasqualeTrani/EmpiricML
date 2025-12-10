from typing import Any
import uuid
import os
from datetime import datetime
import pytz

import polars as pl

from empml.base import (
    DataDownloader, 
    Metric, 
    CVGenerator
)

from empml.pipelines import Pipeline, eval_pipeline_cv


class Lab:
    def __init__(
        self,
        train_downloader: DataDownloader,
        metric: Metric,
        cv_generator: CVGenerator,
        target: str,
        minimize: bool = True,
        row_id: str | None = None,
        test_downloader: DataDownloader | None = None,
        name: str | None = None
    ):
        self.name = name or uuid.uuid1().hex[:8]
        self.metric = metric
        self.cv_generator = cv_generator
        self.target = target
        self.minimize = minimize
        
        self._setup_directories()
        self._load_data(train_downloader, test_downloader)
        self._setup_row_id(row_id)
        self._setup_results_tracking()
        
        self.cv_indexes = self.cv_generator.split(self.train, self.row_id)
        self.next_experiment_id = 0 

    def _setup_directories(self):
        """Create lab directory structure."""
        base = f'./{self.name}'
        os.makedirs(f'{base}/pipelines', exist_ok=True)
        os.makedirs(f'{base}/predictions', exist_ok=True)

    def _load_data(self, train_downloader, test_downloader):
        """Load training and test data."""
        self.train = train_downloader.get_data()
        self.test = test_downloader.get_data() if test_downloader else None

    def _setup_row_id(self, row_id):
        """Setup row identifier column."""
        if row_id:
            self.row_id = row_id
        else:
            self.train = self.train.with_row_index().rename({'index': 'row_id'})
            self.row_id = 'row_id'

    def _setup_results_tracking(self):
        """Initialize Polars DataFrames for tracking experiments."""
        self.results = pl.DataFrame(
            schema={
                'experiment_id': pl.Int64,
                'description': pl.Utf8,
                'cv_mean_score': pl.Float64,
                'train_mean_score': pl.Float64,
                'mean_overfitting_pct': pl.Float64,
                'cv_std_score': pl.Float64,
                'mean_train_time_s': pl.Float64,
                'mean_inference_time_s': pl.Float64,
                'is_completed': pl.Boolean,
                'notes': pl.Utf8,
                'timestamp_utc': pl.Datetime,
            }
        )
        
        self.results_details = pl.DataFrame(
            schema={
                'detail_id': pl.Int64,
                'experiment_id': pl.Int64,
                'fold_number': pl.Int64,
                'validation_score': pl.Float64,
                'train_score': pl.Float64,
                'overfitting_pct': pl.Float64,
            }
        )


    def run_experiment(
            self,
            pipeline : Pipeline,
            description : str = '',
            notes : str = '',
            eval_overfitting : bool = True, 
            store_preds : bool = True, 
            verbose : bool = True
    ):
        
        eval = eval_pipeline_cv(
            pipeline = pipeline, 
            lz = self.train, 
            cv_indexes=self.cv_indexes,
            row_id = self.row_id,
            metric=self.metric, 
            target=self.target,
            minimize=self.minimize, 
            eval_overfitting=eval_overfitting, 
            store_preds=store_preds, 
            verbose=verbose
        )

        eval_df = pl.DataFrame(eval)

        results = (
            eval_df.drop('preds').mean()
            .rename({
                'validation_score' : 'cv_mean_score', 
                'train_score' : 'train_mean_score', 
                'overfitting' : 'mean_overfitting_pct', 
                'duration_train' : 'mean_train_time_s' ,
                'duration_inf' : 'mean_inference_time_s',
            })

            .with_columns(pl.lit(eval_df['validation_score'].std()).alias('cv_std_score'))

            .with_columns(
                pl.lit(self.next_experiment_id).alias('experiment_id'),
                pl.lit(description).alias('description'), 
                pl.lit(notes).alias('notes'), 
                pl.lit(True).alias('is_completed'), 
              #  pl.lit(self.name).alias('lab_name'),
                pl.lit(datetime.now(pytz.timezone('UTC'))).dt.replace_time_zone(None).alias('timestamp_utc')
            )
        )

        print(results.select([self.results.columns]).columns)
        print(self.results.columns)

        self.results = pl.concat([
            self.results,
            results.select(self.results.columns)
        ], how = 'vertical_relaxed')

        self.next_experiment_id+=1 



        