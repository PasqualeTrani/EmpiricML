# empml.data

| Object | Description |
| :--- | :--- |
| `CSVDownloader` | Class for reading a CSV file and returning a Polars LazyFrame. |
| `ParquetDownloader` | Class for reading a Parquet file and returning a Polars LazyFrame. |
| `ExcelDownloader` | Class for reading an Excel file and returning a Polars LazyFrame. |
| `SQLDownloader` | Class for reading data from any SQL database via connection URI. |
| `PostgreSQLDownloader` | Class for reading data from PostgreSQL. |
| `MySQLDownloader` | Class for reading data from MySQL. |
| `MSSQLDownloader` | Class for reading data from Microsoft SQL Server. |
| `SQLiteDownloader` | Class for reading data from a SQLite database. |
| `OracleDownloader` | Class for reading data from Oracle Database. |
| `RedshiftDownloader` | Class for reading data from Amazon Redshift. |
| `BigQueryDownloader` | Class for reading data from Google BigQuery. |
| `SnowflakeDownloader` | Class for reading data from Snowflake. |
| `DatabricksDownloader` | Class for reading data from Databricks SQL. |

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

## SQLDownloader
Class for reading data from any SQL database via connection URI.

Uses connectorx under the hood (`pip install connectorx`).
Supported URI schemes: `postgresql://`, `mysql://`, `mssql://`, `sqlite://`, `oracle://`, and more.

### Methods

```python
def __init__(self, query: str, connection_uri: str):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    connection_uri : str
        Full connection URI (e.g., 'postgresql://user:pass@host:5432/db').
    """
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.read_database_uri(self.query, self.connection_uri).lazy()
```

## PostgreSQLDownloader
Class for reading data from PostgreSQL and returning a Polars LazyFrame.

Requires connectorx (`pip install connectorx`).

### Methods

```python
def __init__(self, query: str, host: str, user: str, password: str, database: str, port: int = 5432):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    host : str
        PostgreSQL server hostname.
    user : str
        Database username.
    password : str
        Database password.
    database : str
        Database name.
    port : int
        Server port (default: 5432).
    """
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.read_database_uri(self.query, self.connection_uri).lazy()
```

## MySQLDownloader
Class for reading data from MySQL and returning a Polars LazyFrame.

Requires connectorx (`pip install connectorx`).

### Methods

```python
def __init__(self, query: str, host: str, user: str, password: str, database: str, port: int = 3306):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    host : str
        MySQL server hostname.
    user : str
        Database username.
    password : str
        Database password.
    database : str
        Database name.
    port : int
        Server port (default: 3306).
    """
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.read_database_uri(self.query, self.connection_uri).lazy()
```

## MSSQLDownloader
Class for reading data from Microsoft SQL Server and returning a Polars LazyFrame.

Also works with Azure SQL Database and Azure Synapse Analytics since they use the same protocol.
Requires connectorx (`pip install connectorx`).

### Methods

```python
def __init__(self, query: str, host: str, user: str, password: str, database: str, port: int = 1433):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    host : str
        SQL Server hostname.
    user : str
        Database username.
    password : str
        Database password.
    database : str
        Database name.
    port : int
        Server port (default: 1433).
    """
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.read_database_uri(self.query, self.connection_uri).lazy()
```

## SQLiteDownloader
Class for reading data from a SQLite database and returning a Polars LazyFrame.

Requires connectorx (`pip install connectorx`).

### Methods

```python
def __init__(self, query: str, path: str):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    path : str
        Path to the SQLite database file.
    """
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.read_database_uri(self.query, self.connection_uri).lazy()
```

## OracleDownloader
Class for reading data from Oracle Database and returning a Polars LazyFrame.

Requires connectorx (`pip install connectorx`).

### Methods

```python
def __init__(self, query: str, host: str, user: str, password: str, database: str, port: int = 1521):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    host : str
        Oracle server hostname.
    user : str
        Database username.
    password : str
        Database password.
    database : str
        Database name (service name).
    port : int
        Server port (default: 1521).
    """
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.read_database_uri(self.query, self.connection_uri).lazy()
```

## RedshiftDownloader
Class for reading data from Amazon Redshift and returning a Polars LazyFrame.

Requires connectorx (`pip install connectorx`).

### Methods

```python
def __init__(self, query: str, host: str, user: str, password: str, database: str, port: int = 5439):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    host : str
        Redshift cluster endpoint.
    user : str
        Database username.
    password : str
        Database password.
    database : str
        Database name.
    port : int
        Server port (default: 5439).
    """
    pass

def get_data(self) -> pl.LazyFrame:
    return pl.read_database_uri(self.query, self.connection_uri).lazy()
```

## BigQueryDownloader
Class for reading data from Google BigQuery and returning a Polars LazyFrame.

Requires google-cloud-bigquery (`pip install google-cloud-bigquery`).

### Methods

```python
def __init__(self, query: str, project_id: str, credentials_path: str | None = None):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    project_id : str
        Google Cloud project ID.
    credentials_path : str | None
        Path to service account JSON credentials file.
        If None, uses Application Default Credentials.
    """
    pass

def get_data(self) -> pl.LazyFrame:
    # Uses google.cloud.bigquery client and Arrow for efficient transfer
    ...
```

## SnowflakeDownloader
Class for reading data from Snowflake and returning a Polars LazyFrame.

Requires snowflake-connector-python (`pip install snowflake-connector-python`).

### Methods

```python
def __init__(self, query: str, account: str, user: str, password: str, warehouse: str, database: str, schema: str):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    account : str
        Snowflake account identifier.
    user : str
        Snowflake username.
    password : str
        Snowflake password.
    warehouse : str
        Snowflake warehouse name.
    database : str
        Snowflake database name.
    schema : str
        Snowflake schema name.
    """
    pass

def get_data(self) -> pl.LazyFrame:
    # Uses snowflake-connector-python and Arrow for efficient transfer
    ...
```

## DatabricksDownloader
Class for reading data from Databricks SQL and returning a Polars LazyFrame.

Requires databricks-sql-connector (`pip install databricks-sql-connector`).

### Methods

```python
def __init__(self, query: str, server_hostname: str, http_path: str, access_token: str):
    """
    Parameters:
    -----------
    query : str
        SQL query to execute.
    server_hostname : str
        Databricks workspace server hostname.
    http_path : str
        HTTP path for the SQL warehouse or cluster.
    access_token : str
        Databricks personal access token.
    """
    pass

def get_data(self) -> pl.LazyFrame:
    # Uses databricks-sql-connector and Arrow for efficient transfer
    ...
```
