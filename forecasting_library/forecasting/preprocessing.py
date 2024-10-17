from typing import Any, List, Self, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column: str = "index"):
        """
        Custom transformer to add time-based features.

        Args:
            datetime_column (str): The name of the datetime column.
        """
        self.datetime_column = datetime_column

    def fit(self, X: Any, y: Any = None) -> Self:
        # No fitting required for adding time-based features
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.datetime_column == "index":
            X[self.datetime_column] = X.index
        X["hour"] = X[self.datetime_column].dt.hour
        X["day_of_week"] = X[self.datetime_column].dt.dayofweek
        X["month"] = X[self.datetime_column].dt.month
        return X.drop(columns=[self.datetime_column])


class LagFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, lagged_features: List[str], lags: List[int]):
        self.lagged_features = lagged_features
        self.lags = lags

    def fit(self, X: Any, y: Any = None) -> Self:
        # No fitting is necessary for lagging
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure input is a DataFrame
        X_lagged = X.copy()
        for feature in self.lagged_features:
            if feature not in X:
                continue
            for lag in self.lags:
                X_lagged[f"{feature}_lag_{lag}"] = X[feature].shift(lag)
        # Drop rows with NaN values introduced by lagging
        X_lagged = X_lagged.dropna()
        return X_lagged


def preprocess_data(
    data: pd.DataFrame,
    datetime_column: str,
    lagged_features: list,
    feature_lags: list,
    target_col: str,
    target_lags: list,
) -> pd.DataFrame:
    """Preprocess the data by adding time features and lag features."""
    # Add time-based features
    time_feature_adder = TimeFeaturesAdder(datetime_column=datetime_column)
    data_with_time_features = time_feature_adder.transform(data)

    # Add lag features
    lag_feature_adder = LagFeatureAdder(lagged_features=lagged_features, lags=feature_lags)
    data_with_lags = lag_feature_adder.transform(data_with_time_features)

    # Add lag target
    lag_target_adder = LagFeatureAdder(lagged_features=[target_col], lags=target_lags)
    data_preprocessed = lag_target_adder.transform(data_with_lags)

    return data_preprocessed
