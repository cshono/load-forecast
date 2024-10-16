import pandas as pd
import numpy as np
from forecasting_library.forecasting.forecasting_api import ForecastingAPI
from forecasting_library.forecasting.evaluation import backtest_model

def test_backtest_error_calculation():
    X = pd.DataFrame({'feature1': np.arange(50), 'feature2': np.arange(50, 100)})
    y = pd.Series(np.arange(50))
    forecasting_api = ForecastingAPI(model_type='random_forest')
    forecasting_api.train_pipeline(X, y)
    
    backtest_results = backtest_model(forecasting_api.model, X, y, n_splits=2, test_size=10)
    assert len(backtest_results) > 0, "Backtesting did not calculate metrics for each window."
    assert all(all(isinstance(metric, float) for metric in result.values()) for result in backtest_results), "Backtesting metrics are not floats."
