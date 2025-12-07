from typing import Any

from empml.data import DataDownloader
from empml.metrics import Metric
from empml.cv import CVGenerator


class Lab: 
    def __init__(
        self, 
        train_downloader : DataDownloader, 
        metric : Metric, 
        cv_generator : CVGenerator, 
        target : str, 
        minimize : bool = True, 
        row_id : str | Any = None,   # unique identifier of the rows 
        test_downloader : DataDownloader | Any = None
    ):
        # load data
        self.train = train_downloader.get_data()

        if test_downloader!=None:
            self.test = test_downloader.get_data()
        else:
            self.test = None

        self.metric = metric
        self.cv_generator = cv_generator
        self.target = target
        self.minimize = minimize

        # row_id
        if row_id!=None:
            self.row_id = row_id
        else:
            # if no row_id is specified, create one 
            self.train=self.train.with_row_index().rename({'index' : 'row_id'})
            self.row_id = 'row_id'

        # setting cv indexes 
        self.cv_indexes = self.cv_generator.split(self.train, self.row_id)