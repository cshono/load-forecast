from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from forecasting_library.forecasting.evaluation import backtest_model


class ForecastingAPI:
    def __init__(
        self,
        model_type: str = "linear",
        preprocessing_pipeline: Optional[Pipeline] = None,
        **model_params: Any,
    ) -> None:
        """
        Initialize the forecasting API.

        Args:
            model_type (str): Type of the forecasting model (e.g., 'linear', 'random_forest').
            preprocessing_pipeline (Pipeline, optional): Custom preprocessing pipeline.
            model_params: Additional parameters for the model.
        """
        self.model_type = model_type
        self.model = self._initialize_model(model_type, **model_params)
        self.preprocessing_pipeline = preprocessing_pipeline
        self.grid_search_result = None

    def _initialize_model(self, model_type: str, **model_params: Any) -> BaseEstimator:
        """
        Initialize the model based on the specified type.
        """
        if model_type == "linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression(**model_params)
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**model_params)
        if model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(**model_params)
        if model_type == "xgboost":
            return xgb.XGBRegressor(**model_params)
        raise ValueError(f"Unsupported model type: {model_type}")

    def split_train_test(
        self, data: pd.DataFrame, target_col: str, split_date: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """_summary_

        Args:
            data (pd.DataFrame): dataset
            target_col (str): target column name
            split_date (_type_): used to split train/test according to index value
        """
        data_train = data.loc[data.index < split_date]
        data_test = data.loc[data.index >= split_date]

        X_train = data_train.drop(columns=[target_col])
        X_test = data_test.drop(columns=[target_col])
        y_train = data_train[target_col]
        y_test = data_train[target_col]

        return X_train, X_test, y_train, y_test

    def train_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[dict] = None,
        n_splits: int = 5,
        test_size: float = 0.2,
        scoring: str = "neg_mean_absolute_error",
    ) -> None:
        """
        Train the forecasting model using the complete training pipeline.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            param_grid (dict, optional): Hyperparameter grid for tuning.
            n_splits (int): Number of splits for time series cross-validation.
            test_size (float): Fraction of data for the test set.
            scoring (str): Scoring metric for evaluation.
        """
        # Step 1: Train-Test Split
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # Step 2: Preprocess the Data (if a preprocessing pipeline is provided)
        if self.preprocessing_pipeline is not None:
            X_train = self.preprocessing_pipeline.fit_transform(X_train)
            X_test = self.preprocessing_pipeline.transform(X_test)

        # Step 3: Hyperparameter Tuning (if param_grid is provided)
        if param_grid is not None:
            self.train_with_grid_search(
                X_train, y_train, param_grid, n_splits=n_splits, scoring=scoring
            )
        else:
            # Train the model directly without hyperparameter tuning
            self.model.fit(X_train, y_train)

    def train_with_grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict,
        n_splits: int = 5,
        scoring: str = "neg_mean_absolute_error",
    ) -> None:
        """
        Perform grid search with time series cross-validation for hyperparameter tuning.
        """
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        grid_search = GridSearchCV(
            self.model, param_grid, cv=tscv, scoring=scoring, n_jobs=-1, verbose=1
        )
        grid_search.fit(X, y)
        self.grid_search_result = grid_search
        self.model = grid_search.best_estimator_
        print(f"Best hyperparameters: {grid_search.best_params_}")

    def forecast(self, X: pd.DataFrame) -> Sequence:
        """
        Generate forecasts for the given data.
        """
        if self.preprocessing_pipeline is not None:
            X = self.preprocessing_pipeline.transform(X)
        return self.model.predict(X)

    def backtest_model(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int, test_size: int
    ) -> List[Dict[str, float]]:
        return backtest_model(self.model, X, y, n_splits, test_size)
