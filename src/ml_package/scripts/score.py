import argparse

import numpy as np
import pandas as pd

from ml_package.training import train_linear_regression, train_random_forest


def score(input_file):
    """Loads preprocessed data, trains models, and prints results."""
    housing_prepared = pd.read_csv(input_file)

    # Here, you need the labels (median_house_value), so ensure it's available
    housing_labels = np.random.randint(
        100000, 500000, housing_prepared.shape[0]
    )  # Dummy labels for now

    lin_reg, lin_rmse = train_linear_regression(housing_prepared, housing_labels)
    train_random_forest(housing_prepared, housing_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models on Housing Data")
    parser.add_argument(
        "input_file", type=str, help="Path to preprocessed housing data CSV file"
    )
    args = parser.parse_args()
    score(args.input_file)
