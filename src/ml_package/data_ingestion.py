import os
import tarfile

import mlflow
import pandas as pd
from six.moves import urllib

from ml_package.config.logging_config import get_logger

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
logger = get_logger("DataIngestion", "data_ingestion.log")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Downloads and extracts the housing dataset.

    This function downloads a housing dataset from a given URL and extracts it
    to the specified directory.

    Parameters
    ----------
    housing_url : str, optional
        The URL from which to download the housing dataset. Defaults to `HOUSING_URL`.

    housing_path : str, optional
        The local directory path where the dataset should be saved and extracted.
        Defaults to `HOUSING_PATH`.

    Returns
    -------
    bool
        Returns `True` if the dataset is successfully downloaded and extracted,
        otherwise returns `False`.

    Raises
    ------
    FileNotFoundError
        If the specified dataset URL is incorrect or unavailable.
    PermissionError
        If the function lacks permission to write to the target directory.
    Exception
        Catches any other unexpected errors during the data ingestion process.

    Examples
    --------
    >>> success = fetch_housing_data()
    >>> print(success)
    True
    """
    with mlflow.start_run(run_name="Fetch Housing Data", nested=True):
        mlflow.log_param("housing_url", housing_url)
        mlflow.log_param("housing_path", housing_path)

        try:
            logger.info("Data Ingestion started")
            os.makedirs(housing_path, exist_ok=True)
            tgz_path = os.path.join(housing_path, "housing.tgz")
            urllib.request.urlretrieve(housing_url, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=housing_path)
            housing_tgz.close()
            logger.info(f"Data downloaded and saved to {HOUSING_PATH}..")
            return True
        except FileNotFoundError as e:
            logger.error(
                f"File not found. Check if the dataset exists: {e}", exc_info=True
            )
            return False
        except PermissionError as e:
            logger.error(f"Permission denied while accessing files: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred during data ingestion! {e}",
                exc_info=True,
            )
            return False


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load housing data from a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the housing data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the housing data.

    Examples
    --------
    >>> df = load_housing_data("datasets/housing/housing.csv")
    >>> print(df.head())
    """
    with mlflow.start_run(run_name="Load Housing Data", nested=True):
        mlflow.log_param("housing_path", housing_path)
        try:
            csv_path = os.path.join(housing_path, "housing.csv")
            df = pd.read_csv(csv_path)
            mlflow.log_metric("rows", df.shape[0])
            mlflow.log_metric("columns", df.shape[1])
            return df
        except Exception as e:
            logger.error(f"Problem while loading housing data: {e}", exc_info=True)
            return None
