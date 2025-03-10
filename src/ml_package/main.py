import mlflow

from ml_package.data_ingestion import fetch_housing_data, load_housing_data
from ml_package.data_preprocessing import preprocess_data, stratified_split
from ml_package.training import train_linear_regression, train_random_forest


def main():
    """
    Runs the entire housing prediction pipeline under a single parent MLflow run with nested child runs.
    """
    with mlflow.start_run(run_name="Housing Prediction Pipeline"):
        print("Fetching housing data...")
        fetch_housing_data()

        print("Loading housing data...")
        housing = load_housing_data()

    if housing is not None:
        print("Splitting data into train and test sets...")
        with mlflow.start_run(run_name="Data Preprocessing", nested=True):
            train_set, test_set = stratified_split(housing)
            print("Preprocessing training data...")
            housing_prepared = preprocess_data(
                train_set.drop("median_house_value", axis=1)
            )
            housing_labels = train_set["median_house_value"].copy()

        print("Training Linear Regression model...")
        with mlflow.start_run(run_name="Linear Regression Training", nested=True):
            lin_reg, lin_rmse = train_linear_regression(
                housing_prepared, housing_labels
            )
            print(f"Linear Regression RMSE: {lin_rmse}")

        print("Training Random Forest model...")
        with mlflow.start_run(run_name="Random Forest Training", nested=True):
            best_forest_model = train_random_forest(housing_prepared, housing_labels)

        print(best_forest_model.__format__)
        print("Models trained successfully!")
    else:
        print("Data ingestion failed. Pipeline aborted.")


if __name__ == "__main__":
    main()
