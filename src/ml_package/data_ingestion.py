import os
import tarfile

import pandas as pd
from six.moves import urllib

from ml_package.config.logging_config import get_logger

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
logger = get_logger("DataIngestion", "data_ingestion.log")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    try:
        """Downloads and extracts the housing dataset."""
        logger.info("Data Ingestion started")
        logger.info("Downloads and extracts the housing dataset.")
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        logger.info(f"extracting housing data from {tgz_path}..")
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        logger.info(f"Data downloaded and saved to {HOUSING_PATH}..")
    except FileNotFoundError as e:
        logger.error("File not found. Check if the dataset exists.", e, exc_info=True)
    except PermissionError as e:
        logger.error("Permission denied while accessing files.", e, exc_info=True)
    except Exception as e:
        logger.critical(
            "An unexpected error occurred during data ingestion!", e, exc_info=True
        )


def load_housing_data(housing_path=HOUSING_PATH):
    try:
        csv_path = os.path.join(housing_path, "housing.csv")
        # loading the data and returing
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Problem while loading housing data", e, exc_info=True)
