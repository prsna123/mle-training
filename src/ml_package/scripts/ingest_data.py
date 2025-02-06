import argparse
import os

from ml_package.data_ingestion import fetch_housing_data  # Import from existing module


def ingest_data(output_folder):
    """Fetches housing data and saves it to the specified folder."""
    print(f"Fetching housing data and saving to {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    fetch_housing_data(
        housing_url="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz",
        housing_path=output_folder,
    )  # Call function to fetch data
    print(f"Data downloaded and saved to {output_folder}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Housing Data")
    parser.add_argument(
        "output_folder", type=str, help="Directory to save the housing data"
    )
    args = parser.parse_args()
    ingest_data(args.output_folder)
