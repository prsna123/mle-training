import argparse

import pandas as pd

from ml_package.data_preprocessing import preprocess_data, stratified_split


def preprocess_and_save(input_file, output_file):
    """Loads, preprocesses, and saves the processed housing data."""
    print(f"Loading data from {input_file}...")
    housing = pd.read_csv(input_file)

    print("Splitting data into train and test sets...")
    train_set, _ = stratified_split(housing)

    print("Preprocessing training data...")
    housing_prepared = preprocess_data(train_set.drop("median_house_value", axis=1))

    print(f"Saving preprocessed data to {output_file}...")
    housing_prepared.to_csv(output_file, index=False)
    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Housing Data")
    parser.add_argument(
        "input_file", type=str, help="Path to raw housing data CSV file"
    )
    parser.add_argument("output_file", type=str, help="Path to save processed data")
    args = parser.parse_args()
    preprocess_and_save(args.input_file, args.output_file)
