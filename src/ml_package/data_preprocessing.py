import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from ml_package.config.logging_config import get_logger

logger = get_logger("DataPreprocessing", "data_preprocessing.log")

logger.info("Loads, preprocesses, and saves the processed housing data.")


def stratified_split(housing):
    """
    Performs stratified split based on income categories.
    Reason: Need the data with composed way.
    Eg: For 100 data's we need to be equally distributed with train and test data
    """
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[
            0.0,
            1.5,
            3.0,
            4.5,
            6.0,
            np.inf,
        ],  # splitting the data in multiple bins [like 0.0 to 1.5 ]
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    logger.info("Splitting data into train and test sets...")
    # splitted the equally distributed data based on income category
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def preprocess_data(housing):
    """Handles missing values and feature engineering."""
    logger.info("Preprocessing training data..")
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop(
        "ocean_proximity", axis=1
    )  # dropping this column since it require only integer

    # responsible for calculating the median of the columns which have missing values
    imputer.fit(housing_num)

    # Filling those missing with median values
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    # Creating new columns
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    # Just fetching the ocean_proximity column from housing data frame
    housing_cat = housing[["ocean_proximity"]]

    # filling the values with true or false based on the value i.e: Inland, Nearbay
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    logger.info("Preprocessing completed")
    return housing_prepared
