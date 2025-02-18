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

    The dataset is split into training and testing sets in a way that maintains
    the same proportion of income categories in both sets. This ensures a
    balanced distribution of data.

    Parameters
    ----------
    housing : pandas.DataFrame
        The housing dataset containing a `median_income` column.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - `strat_train_set` : The training dataset after stratified splitting.
        - `strat_test_set` : The testing dataset after stratified splitting.

    Raises
    ------
    Exception
        If an error occurs during stratified splitting.

    Examples
    --------
    >>> train_set, test_set = stratified_split(housing_df)
    >>> print(len(train_set), len(test_set))
    16512 4128
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
    try:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        logger.info("Splitting data into train and test sets...")
        # splitted the equally distributed data based on income category
    except Exception as e:
        logger.error("Error while startified split of data", e, exc_info=True)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def preprocess_data(housing):
    """
    Preprocesses the housing dataset by handling missing values and creating new features.

    This function:
    - Handles missing values using the median strategy.
    - Performs feature engineering by creating new ratio-based features.
    - Encodes categorical variables using one-hot encoding.

    Parameters
    ----------
    housing : pandas.DataFrame
        The housing dataset containing both numerical and categorical features.

    Returns
    -------
    pandas.DataFrame
        The processed dataset with missing values handled, new features added,
        and categorical variables one-hot encoded.

    Raises
    ------
    Exception
        If the 'ocean_proximity' column is not found in the dataset.

    Examples
    --------
    >>> housing_prepared = preprocess_data(housing_df)
    >>> housing_prepared.head()
       longitude  latitude  housing_median_age  ...  rooms_per_household  bedrooms_per_room  population_per_household
    0 -122.23    37.88     41                  ...  5.445713              0.146590           2.555556
    1 -122.22    37.86     21                  ...  6.263570              0.190476           2.109842
    """
    logger.info("Preprocessing training data..")
    imputer = SimpleImputer(strategy="median")
    try:
        housing_num = housing.drop(
            "ocean_proximity", axis=1
        )  # dropping this column since it require only integer
    except Exception as e:
        logger.error("ocean_proximity column is not found", e, exc_info=True)

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
