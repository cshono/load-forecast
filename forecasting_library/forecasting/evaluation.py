from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit


def evaluate(y_test, y_pred):
    """runs evaluation metrics returned in a dict

    Returns:
        dict: evaluation metrics (MAE, MAPE, RMSE)
    """
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse}


def backtest_model(model, X, y, n_splits, test_size):
    """
    Perform backtesting with a rolling window approach.

    Args:
        model: The trained forecasting model.
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        initial_train_size (int): Number of samples for the initial training set.
        test_size (int): Number of samples for each test window.
        step_size (int): The step size for moving the window forward.

    Returns:
        list: A list of mean absolute errors for each backtesting window.
    """
    results = []
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    for train_index, test_index in tscv.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Generate predictions
        y_pred = model.predict(X_test)

        # Evaluate the performance
        result = evaluate(y_test, y_pred)
        results.append(result)

    return results
