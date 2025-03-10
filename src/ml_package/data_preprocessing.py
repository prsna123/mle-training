import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from ml_package.config.logging_config import get_logger

logger = get_logger("DataPreprocessing", "data_preprocessing.log")

logger.info("Loads, preprocesses, and saves the processed housing data.")


def stratified_split(housing):
    """
    Performs stratified splitting of the dataset based on income categories.
    """
    with mlflow.start_run(run_name="Stratified Split", nested=True):
        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )
        try:
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            logger.info("Splitting data into train and test sets...")
        except Exception as e:
            logger.error("Error while stratified split of data", e, exc_info=True)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        mlflow.log_metric("train_set_size", len(strat_train_set))
        mlflow.log_metric("test_set_size", len(strat_test_set))

        return strat_train_set, strat_test_set


def preprocess_data(housing):
    """
    Preprocesses the housing dataset by handling missing values and creating new features.
    """
    with mlflow.start_run(run_name="Preprocess Data", nested=True):
        logger.info("Preprocessing training data..")
        imputer = SimpleImputer(strategy="median")
        try:
            housing_num = housing.drop("ocean_proximity", axis=1)
        except Exception as e:
            logger.error("ocean_proximity column is not found", e, exc_info=True)
            return None  # Return None when error

        imputer.fit(housing_num)
        X = imputer.transform(housing_num)

        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
        housing_tr["rooms_per_household"] = (
            housing_tr["total_rooms"] / housing_tr["households"]
        )
        housing_tr["bedrooms_per_room"] = (
            housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
        )
        housing_tr["population_per_household"] = (
            housing_tr["population"] / housing_tr["households"]
        )

        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
        logger.info("Preprocessing completed")
        return housing_prepared
