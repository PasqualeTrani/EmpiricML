# standard import libraries 
from pathlib import Path

# wranglers 
import polars as pl  

# internal imports
from empml.utils import log_execution_time
from empml.base import DataDownloader # base class 

# streaming engine as the default for .collect()
pl.Config.set_engine_affinity(engine='streaming')

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
    
class ExcelDownloader(DataDownloader):
    """Class for reading an Excel file and returns a Polars LazyFrame."""
    def __init__(self, path : str, sheet_name : str | None = None):
        self.path = path
        self.sheet_name = sheet_name
        
    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.read_excel(self.path, sheet_name = self.sheet_name).lazy()