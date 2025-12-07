# standard import libraries 
from dataclasses import dataclass 
from abc import ABC, abstractmethod
from pathlib import Path

# wranglers 
import polars as pl 

# internal imports
from empml.utils import log_execution_time


# ------------------------------------------------------------------------------------------
# DEFINITION OF THE ABSTRACT CLASS 
# ------------------------------------------------------------------------------------------

class DataDownloader(ABC):
    """Abstract class for downloading data into Polars LazyFrames."""
    @abstractmethod
    def get_data(self) -> pl.LazyFrame:
        pass


# ------------------------------------------------------------------------------------------
# IMPLEMENTATIONS 
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