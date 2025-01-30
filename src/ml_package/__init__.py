from .data_ingestion import fetch_and_load_data
from .scoring import evaluate_model
from .training import train_model


def run_pipeline():
    print("Fetching and loading data...")
    housing, strat_train_set, strat_test_set = fetch_and_load_data()

    print("Training the model...")
    final_model, imputer = train_model(strat_train_set)

    print("Evaluating the model...")
    final_rmse = evaluate_model(final_model, imputer, strat_test_set)

    print(f"Final RMSE on Test Data: {final_rmse}")
    return final_rmse
