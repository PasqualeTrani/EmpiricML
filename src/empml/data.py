# standard import libraries 
from pathlib import Path

# wranglers 
import polars as pl  # type: ignore

# internal imports
from empml.utils import log_execution_time
from empml.base import DataDownloader # base class 


# ------------------------------------------------------------------------------------------
# Implementations of the DataDownloader base class
# ------------------------------------------------------------------------------------------

class CSVDownloader(DataDownloader):
    """Class for reading a CSV file and returns a Polars LazyFrame."""
    def __init__(self, path : str, separator : str = ';'):
        self.path = path
        self.separator = separator

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.scan_csv(self.path, separator = self.separator) 
    
class ParquetDownloader(DataDownloader):
    """Class for reading a Parquet file and returns a Polars LazyFrame."""
    def __init__(self, path : str):
        self.path = path
        
    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path) 