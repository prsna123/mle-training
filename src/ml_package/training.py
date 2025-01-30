import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


def train_linear_regression(housing_prepared, housing_labels):
    """Trains and evaluates a Linear Regression model."""
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))

    return lin_reg, lin_rmse


def train_random_forest(housing_prepared, housing_labels):
    """Trains a Random Forest model with hyperparameter tuning."""
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

    return rnd_search.best_estimator_
