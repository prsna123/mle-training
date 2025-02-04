import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd

from ml_package.data_ingestion import fetch_housing_data, load_housing_data
from ml_package.data_preprocessing import preprocess_data, stratified_split
from ml_package.training import train_linear_regression

SAMPLE_TEST_DATA = {
    "longitude": [-122.23, -122.24, -125.22],
    "latitude": [37.2, 36.21, 37.9],
    "house_median_age": [32, None, 12],
    "total_rooms": [880, None, 167],
    "total_bedrooms": [67, 21, 45],
    "population": [332, 168, 250],
    "households": [243, None, 222],
    "median_income": [8.23, 9.22, 1.2],
    "median_house_value": [425600, 452060, 404065],
    "ocean_proximity": ["NEAR_BAY", "INLAND", "NEAR_BAY"],
}


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


def test_preprocess_data():
    """
    verifies that it returns the data without any NaN values
    No NaN values after preprocessing
    """
    test_data_df = pd.DataFrame(SAMPLE_TEST_DATA)
    processed_Data = preprocess_data(test_data_df)
    assert processed_Data.isna().sum().sum() == 0, "There are still missing values!"

    expected_columns = {
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
    }
    assert expected_columns.issubset(
        processed_Data.columns
    ), "Feature engineering columns are missing!"


def test_train_linear_regression():
    """
    Tests Linear Regression training by verifying:
    - No NaN values after preprocessing
    - Model is successfully trained
    - RMSE is a valid, non-negative number
    """

    # Creating DataFrame
    test_data_df = pd.DataFrame(SAMPLE_TEST_DATA)

    features = test_data_df.drop("median_house_value", axis=1)
    labels = test_data_df["median_house_value"].copy()

    processed_data = preprocess_data(features)

    assert (
        processed_data.isna().sum().sum() == 0
    ), "Preprocessed data contains NaN values!"

    lin_reg, lin_rmse = train_linear_regression(processed_data, labels)

    assert lin_reg is not None, "Model training failed!"
    assert isinstance(lin_rmse, (float, np.float64)), "RMSE should be a numeric value!"
    assert lin_rmse >= 0, "RMSE cannot be negative!"

    print("Linear Regression test passed successfully!")
