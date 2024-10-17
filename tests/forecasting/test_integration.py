import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from forecasting_library.forecasting.forecasting_api import ForecastingAPI
from forecasting_library.forecasting.preprocessing import preprocess_data


def test_full_pipeline_workflow_w_grid_search():
    data = pd.DataFrame(
        {
            "target": np.arange(100),
            "datetime": pd.date_range("2024-01-01", periods=100, freq="h"),
            "feature1": np.arange(100),
            "feature2": np.arange(100, 200),
        }
    )

    # Create a preprocessing pipeline
    lagged_features = [c for c in data if c != "CAISO_system_load"]
    data_preprocessed = preprocess_data(
        data,
        datetime_column="datetime",
        lagged_features=lagged_features,
        feature_lags=[1, 2],
        target_col="target",
        target_lags=[2, 3],
    )

    # Initialize the ForecastingAPI
    forecasting_api = ForecastingAPI(model_type="xgboost")

    # Train the pipeline and perform backtesting
    param_grid = {
        "n_estimators": [10, 12],
        "max_depth": [2, 3],
        "learning_rate": [0.1, 0.2],
        "subsample": [0.2, 0.3],
    }

    X = data_preprocessed.drop(columns=["target"])
    y = data_preprocessed["target"]

    X_train, X_test = X.iloc[:-12], X.iloc[-12:]
    y_train, y_test = y.iloc[:-12], y.iloc[-12:]

    forecasting_api.train_pipeline(X_train, y_train, param_grid=param_grid, n_splits=3)
    predictions = forecasting_api.forecast(X_test)[-10:]

    assert (
        forecasting_api.grid_search_result.best_estimator_ is not None
    ), "model best_estimator has been set"
    assert isinstance(
        forecasting_api.grid_search_result, GridSearchCV
    ), "forecast api returns grid_search_result"
    assert len(predictions) == 10, "The forecasted output does not match the expected length."
