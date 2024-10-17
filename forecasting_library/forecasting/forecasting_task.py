from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from forecasting_library.forecasting.forecast_model import ForecastModel
from forecasting_library.forecasting.preprocessing import preprocess_data


@dataclass
class ForecastingTaskData:
    data: pd.DataFrame
    datetime_column: str
    lagged_features: list
    feature_lags: list
    target_col: str
    target_lags: list


class ForecastingTask:
    def __init__(self, task_data: ForecastingTaskData):
        self.task_data = task_data

    def preprocess_data(self) -> None:
        self.data_preprocessed = preprocess_data(
            data=self.task_data.data,
            datetime_column=self.task_data.datetime_column,
            lagged_features=self.task_data.lagged_features,
            feature_lags=self.task_data.feature_lags,
            target_col=self.task_data.target_col,
            target_lags=self.task_data.target_lags,
        )

    def split_train_test(self, split_date: Optional[str] = None, test_size: int = 24) -> None:
        self.X = self.data_preprocessed.drop(columns=[self.task_data.target_col])
        self.y = self.data_preprocessed[self.task_data.target_col]

        if split_date:
            self.X_train = self.X.loc[self.X.index < split_date]
            self.X_test = self.X.loc[self.X.index >= split_date]
            self.y_train = self.y.loc[self.y.index < split_date]
            self.y_test = self.y.loc[self.y.index >= split_date]
        else:
            self.X_train, self.X_test = self.X.iloc[:-test_size], self.X.iloc[-test_size:]
            self.y_train, self.y_test = self.y.iloc[:-test_size], self.y.iloc[-test_size:]

    def set_model(self, model_type: str) -> None:
        self.model = ForecastModel(model_type)

    def train_model(self, param_grid: Optional[dict] = None, n_splits: int = 5) -> None:
        self.model.train(
            X_train=self.X_train, y_train=self.y_train, param_grid=param_grid, n_splits=n_splits
        )

    def forecast(self, X: pd.DataFrame) -> None:
        y_forecast = self.model.forecast(X)
        y_forecast = pd.Series(y_forecast, index=X.index)
        self.y_forecast = y_forecast

    def backtest_model(self, X: pd.DataFrame, y: pd.Series, n_splits: int, test_size: int) -> None:
        results, preds = self.model.backtest_model(X, y, n_splits, test_size)
        self.backtest_results = results
        n_preds = len(preds)
        self.y_backtest = pd.Series(preds, index=y[-n_preds:].index)

    def plot_actual_v_pred(self, y_actual: pd.Series, y_pred: pd.Series, title: str = "") -> None:
        plt.figure(figsize=(10, 4))
        plt.plot(y_actual, label="actual", alpha=0.7)
        plt.plot(y_pred, label="pred", alpha=0.7)
        plt.ylabel("System Load (MW)")
        plt.title(title)
        plt.legend()
