# standard import libraries 
from dataclasses import dataclass 
from abc import ABC, abstractmethod
from pathlib import Path

# wranglers 
import polars as pl 


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
    def __init__(self, path : Path):
        self.path = path

    def get_data(self, separator : str = ',') -> pl.LazyFrame:
        return pl.scan_csv(self.path, separator = separator) 
    
class ParquetDownloader(DataDownloader):
    """Class for reading a Parquet file and returns a Polars LazyFrame."""
    def __init__(self, path : Path):
        self.path = path

    def get_data(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path) 