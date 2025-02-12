import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from ml_package.config.logging_config import get_logger

logger = get_logger("Training data", "model_training.log")
logger.info("Loads preprocessed data, trains models, and prints results.")


def train_linear_regression(housing_prepared, housing_labels):
    """
    Trains and evaluates a Linear Regression model on the given dataset.

    This function:
    - Fits a Linear Regression model to the training data.
    - Predicts housing prices using the trained model.
    - Computes the Root Mean Squared Error (RMSE) to evaluate model performance.

    Parameters
    ----------
    housing_prepared : pandas.DataFrame or numpy.ndarray
        The processed dataset with features ready for training.

    housing_labels : pandas.Series or numpy.ndarray
        The target variable (actual housing prices).

    Returns
    -------
    tuple
        A tuple containing:
        - lin_reg (sklearn.linear_model.LinearRegression): The trained model.
        - lin_rmse (float): The RMSE value indicating model performance.

    Raises
    ------
    Exception
        If training fails due to an unexpected error.

    Examples
    --------
    >>> model, rmse = train_linear_regression(housing_prepared, housing_labels)
    >>> print(f"RMSE: {rmse}")
    """
    try:
        """Trains and evaluates a Linear Regression model."""
        logger.info("Training Linear Regressing model started")
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        housing_predictions = lin_reg.predict(housing_prepared)
        lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
        logger.info(f"Linear Regression RMSE: {lin_rmse}")

        return lin_reg, lin_rmse
    except Exception as e:
        logger.error("Failed to train the model in linear regression", e, exc_info=True)


def train_random_forest(housing_prepared, housing_labels):
    """
    Trains a Random Forest model with hyperparameter tuning using RandomizedSearchCV.

    This function:
    - Defines a hyperparameter search space for `n_estimators` and `max_features`.
    - Uses `RandomizedSearchCV` to find the best combination of hyperparameters.
    - Trains a Random Forest model on the given dataset.
    - Returns the best-performing model.

    Parameters
    ----------
    housing_prepared : pandas.DataFrame or numpy.ndarray
        The processed dataset with features ready for training.

    housing_labels : pandas.Series or numpy.ndarray
        The target variable (actual housing prices).

    Returns
    -------
    RandomForestRegressor
        The best-trained Random Forest model selected by hyperparameter tuning.

    Raises
    ------
    Exception
        If training fails due to an unexpected error.

    Examples
    --------
    >>> best_model = train_random_forest(housing_prepared, housing_labels)
    >>> print(best_model)
    """
    try:
        """Trains a Random Forest model with hyperparameter tuning."""
        logger.info("Training Random Forest model started")

        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(housing_prepared, housing_labels)
        logger.info("Best Random Forest model trained")

        return rnd_search.best_estimator_

    except Exception as e:
        logger.error("Failed to train the model in random forest ", e, exc_info=True)
