# empml.data

| Object | Description |
| :--- | :--- |
| `CSVDownloader` | Class for reading a CSV file and returning a Polars LazyFrame. |
| `ParquetDownloader` | Class for reading a Parquet file and returning a Polars LazyFrame. |
| `ExcelDownloader` | Class for reading an Excel file and returning a Polars LazyFrame. |

## CSVDownloader
Class for reading a CSV file and returning a Polars LazyFrame.

### Methods

```python
def __init__(self, path : str, separator : str = ';'):
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.scan_csv(self.path, separator = self.separator)
```

## ParquetDownloader
Class for reading a Parquet file and returning a Polars LazyFrame.

### Methods

```python
def __init__(self, path : str):
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.scan_parquet(self.path)
```

## ExcelDownloader
Class for reading an Excel file and returning a Polars LazyFrame.

### Methods

```python
def __init__(self, path : str, sheet_name : str | None = None):
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.read_excel(self.path, sheet_name = self.sheet_name).lazy()
```
