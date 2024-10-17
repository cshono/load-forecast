import numpy as np
import pandas as pd

from forecasting_library.forecasting.forecast_model import ForecastModel


def test_backtest_error_calculation():
    data = pd.DataFrame(
        {
            "target": np.arange(50),
            "feature1": np.arange(50),
            "feature2": np.arange(50, 100),
        }
    )
    target_col = "target"
    X = data.drop(columns=[target_col])
    y = data[target_col]
    forecast_run = ForecastModel(model_type="random_forest")
    forecast_run.train(X, y)

    backtest_results, backtest_preds = forecast_run.backtest_model(X, y, n_splits=2, test_size=10)
    assert len(backtest_results) > 0, "Backtesting did not calculate metrics for each window."
    assert all(
        all(isinstance(metric, float) for metric in result.values()) for result in backtest_results
    ), "Backtesting metrics are not floats."
