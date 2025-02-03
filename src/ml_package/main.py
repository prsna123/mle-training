from .data_ingestion import fetch_housing_data, load_housing_data
from .data_preprocessing import preprocess_data, stratified_split
from .training import train_linear_regression, train_random_forest


def main():
    print("Fetching housing data...")
    fetch_housing_data()

    print("Loading housing data...")
    housing = load_housing_data()

    print("Splitting data into train and test sets...")
    train_set, test_set = stratified_split(housing)

    print("Preprocessing training data...")
    housing_prepared = preprocess_data(train_set.drop("median_house_value", axis=1))
    housing_labels = train_set["median_house_value"].copy()

    print("Training Linear Regression model...")
    lin_reg, lin_rmse = train_linear_regression(housing_prepared, housing_labels)
    print(f"Linear Regression RMSE: {lin_rmse}")

    print("Training Random Forest model...")
    best_forest_model = train_random_forest(housing_prepared, housing_labels)

    print(best_forest_model.__format__)
    print("Models trained successfully!")


if __name__ == "__main__":
    main()
