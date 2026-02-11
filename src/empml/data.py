# standard import libraries
from pathlib import Path
from urllib.parse import quote_plus

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


# ------------------------------------------------------------------------------------------
# SQL Database downloaders (connectorx-based)
# ------------------------------------------------------------------------------------------

class SQLDownloader(DataDownloader):
    """Class for reading data from any SQL database via connection URI.

    Uses connectorx under the hood (pip install connectorx).
    Supported URI schemes: postgresql://, mysql://, mssql://,
    sqlite://, oracle://, and more.
    """

    def __init__(
        self,
        query: str,
        connection_uri: str,
    ):
        self.query = query
        self.connection_uri = connection_uri

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.read_database_uri(
            self.query, self.connection_uri
        ).lazy()


class PostgreSQLDownloader(DataDownloader):
    """Class for reading data from PostgreSQL
    and returns a Polars LazyFrame.

    Requires connectorx (pip install connectorx).
    """

    def __init__(
        self,
        query: str,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 5432,
    ):
        self.query = query
        self.connection_uri = (
            f"postgresql://{quote_plus(user)}:"
            f"{quote_plus(password)}@{host}:{port}"
            f"/{database}"
        )

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.read_database_uri(
            self.query, self.connection_uri
        ).lazy()


class MySQLDownloader(DataDownloader):
    """Class for reading data from MySQL
    and returns a Polars LazyFrame.

    Requires connectorx (pip install connectorx).
    """

    def __init__(
        self,
        query: str,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
    ):
        self.query = query
        self.connection_uri = (
            f"mysql://{quote_plus(user)}:"
            f"{quote_plus(password)}@{host}:{port}"
            f"/{database}"
        )

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.read_database_uri(
            self.query, self.connection_uri
        ).lazy()


class MSSQLDownloader(DataDownloader):
    """Class for reading data from Microsoft SQL Server
    and returns a Polars LazyFrame.

    Also works with Azure SQL Database and Azure Synapse
    Analytics since they use the same protocol.
    Requires connectorx (pip install connectorx).
    """

    def __init__(
        self,
        query: str,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 1433,
    ):
        self.query = query
        self.connection_uri = (
            f"mssql://{quote_plus(user)}:"
            f"{quote_plus(password)}@{host}:{port}"
            f"/{database}"
        )

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.read_database_uri(
            self.query, self.connection_uri
        ).lazy()


class SQLiteDownloader(DataDownloader):
    """Class for reading data from a SQLite database
    and returns a Polars LazyFrame.

    Requires connectorx (pip install connectorx).
    """

    def __init__(self, query: str, path: str):
        self.query = query
        self.connection_uri = f"sqlite://{path}"

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.read_database_uri(
            self.query, self.connection_uri
        ).lazy()


class OracleDownloader(DataDownloader):
    """Class for reading data from Oracle Database
    and returns a Polars LazyFrame.

    Requires connectorx (pip install connectorx).
    """

    def __init__(
        self,
        query: str,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 1521,
    ):
        self.query = query
        self.connection_uri = (
            f"oracle://{quote_plus(user)}:"
            f"{quote_plus(password)}@{host}:{port}"
            f"/{database}"
        )

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.read_database_uri(
            self.query, self.connection_uri
        ).lazy()


# ------------------------------------------------------------------------------------------
# Cloud Data Warehouse downloaders
# ------------------------------------------------------------------------------------------

class RedshiftDownloader(DataDownloader):
    """Class for reading data from Amazon Redshift
    and returns a Polars LazyFrame.

    Requires connectorx (pip install connectorx).
    """

    def __init__(
        self,
        query: str,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 5439,
    ):
        self.query = query
        self.connection_uri = (
            f"redshift://{quote_plus(user)}:"
            f"{quote_plus(password)}@{host}:{port}"
            f"/{database}"
        )

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        return pl.read_database_uri(
            self.query, self.connection_uri
        ).lazy()


class BigQueryDownloader(DataDownloader):
    """Class for reading data from Google BigQuery
    and returns a Polars LazyFrame.

    Requires google-cloud-bigquery
    (pip install google-cloud-bigquery).
    """

    def __init__(
        self,
        query: str,
        project_id: str,
        credentials_path: str | None = None,
    ):
        self.query = query
        self.project_id = project_id
        self.credentials_path = credentials_path

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery is required for "
                "BigQueryDownloader. Install it with: "
                "pip install google-cloud-bigquery"
            )

        if self.credentials_path:
            client = (
                bigquery.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project_id,
                )
            )
        else:
            client = bigquery.Client(
                project=self.project_id
            )

        arrow_table = client.query(self.query).to_arrow()
        return pl.from_arrow(arrow_table).lazy()


class SnowflakeDownloader(DataDownloader):
    """Class for reading data from Snowflake
    and returns a Polars LazyFrame.

    Requires snowflake-connector-python
    (pip install snowflake-connector-python).
    """

    def __init__(
        self,
        query: str,
        account: str,
        user: str,
        password: str,
        warehouse: str,
        database: str,
        schema: str,
    ):
        self.query = query
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "snowflake-connector-python is required "
                "for SnowflakeDownloader. Install it with:"
                " pip install snowflake-connector-python"
            )

        conn = snowflake.connector.connect(
            account=self.account,
            user=self.user,
            password=self.password,
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema,
        )
        try:
            cursor = conn.cursor()
            cursor.execute(self.query)
            arrow_table = cursor.fetch_arrow_all()
        finally:
            conn.close()

        return pl.from_arrow(arrow_table).lazy()


class DatabricksDownloader(DataDownloader):
    """Class for reading data from Databricks SQL
    and returns a Polars LazyFrame.

    Requires databricks-sql-connector
    (pip install databricks-sql-connector).
    """

    def __init__(
        self,
        query: str,
        server_hostname: str,
        http_path: str,
        access_token: str,
    ):
        self.query = query
        self.server_hostname = server_hostname
        self.http_path = http_path
        self.access_token = access_token

    @log_execution_time
    def get_data(self) -> pl.LazyFrame:
        try:
            from databricks import sql as databricks_sql
        except ImportError:
            raise ImportError(
                "databricks-sql-connector is required "
                "for DatabricksDownloader. Install it "
                "with: pip install databricks-sql-connector"
            )

        conn = databricks_sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token,
        )
        try:
            cursor = conn.cursor()
            cursor.execute(self.query)
            arrow_table = cursor.fetchall_arrow()
        finally:
            conn.close()

        return pl.from_arrow(arrow_table).lazy()