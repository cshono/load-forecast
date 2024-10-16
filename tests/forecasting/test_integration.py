import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from forecasting_library.forecasting.forecasting_api import ForecastingAPI
from forecasting_library.forecasting.preprocessing import create_preprocessing_pipeline


def test_full_pipeline_workflow_w_grid_search():
    X = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=100, freq="h"),
            "feature1": np.arange(100),
            "feature2": np.arange(100, 200),
        }
    )
    y = pd.Series(np.arange(100))

    # Create a preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline(
        numeric_cols=["feature1", "feature2"],
        categorical_cols=["month", "hour", "day_of_week"],
        datetime_column="datetime",
    )

    # Initialize the ForecastingAPI
    forecasting_api = ForecastingAPI(
        model_type="xgboost", preprocessing_pipeline=preprocessing_pipeline
    )

    # Train the pipeline and perform backtesting
    param_grid = {
        "n_estimators": [10, 12],
        "max_depth": [2, 3],
        "learning_rate": [0.1, 0.2],
        "subsample": [0.2, 0.3],
    }

    forecasting_api.train_pipeline(X, y, param_grid=param_grid, test_size=0.2)
    predictions = forecasting_api.forecast(X.iloc[-10:])

    assert (
        forecasting_api.grid_search_result.best_estimator_ is not None
    ), "model best_estimator has been set"
    assert isinstance(
        forecasting_api.grid_search_result, GridSearchCV
    ), "forecast api returns grid_search_result"
    assert len(predictions) == 10, "The forecasted output does not match the expected length."
