import joblib
import numpy as np
import pandas as pd
from data_ingestion import load_housing_data, stratified_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load data
housing = load_housing_data()
_, test_set = stratified_split(housing)

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

# Preprocess test data
imputer = SimpleImputer(strategy="median")
X_test_num = X_test.drop("ocean_proximity", axis=1)
imputer.fit(X_test_num)
X_test_prepared = imputer.transform(X_test_num)

X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

# Load trained model
best_model = joblib.load("best_model.pkl")

# Make predictions
final_predictions = best_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(f"Final RMSE: {final_rmse}")
