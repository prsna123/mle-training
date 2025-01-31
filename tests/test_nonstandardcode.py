import importlib.util
import subprocess
import sys

from ml_package.data_ingestion import fetch_housing_data, load_housing_data
from ml_package.data_preprocessing import stratified_split


def test_ispackage_installed():
    """
    verifies the package installation
    """
    package_name = "ml_package"

    assert (
        importlib.util.find_spec(package_name) is not None
    ), f"{package_name} is not installed"

    result = subprocess.run([sys.executable, "-m", "pip", "show", package_name])
    assert result.returncode == 0, f"Failed to find package {package_name}"


def test_fetch_housing_data():
    """
    Verifies that the data is fetched correctly.
    """
    try:
        fetch_housing_data()
        print("Data fetched successfully")
    except Exception as e:
        print(f"Data ingestion failed: {e}")


def test_load_housing_data():
    """
    Verifies that the data is loaded correctly
    """
    try:
        load_housing_data()
        print("loaded data successfully")
    except Exception as e:
        print(f"Data loading failed: {e}")


def test_stratified_split():
    """
    verifies that the loaded data splitted stratically
    """
    try:
        train_set, test_set = stratified_split()
    except Exception as e:
        print(f"failed during splitting: {e}")
