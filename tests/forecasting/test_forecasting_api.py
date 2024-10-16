import pandas as pd

from forecasting_library.forecasting.forecasting_api import ForecastingAPI


def test_model_initialization():
    for model_type in ["linear", "random_forest", "gradient_boosting", "xgboost"]:
        forecasting_api = ForecastingAPI(model_type=model_type)
        assert (
            forecasting_api.model is not None
        ), f"{model_type} model did not initialize correctly."


def test_model_training():
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([1, 2, 3])
    forecasting_api = ForecastingAPI(model_type="random_forest")
    forecasting_api.train_pipeline(X, y)
    assert forecasting_api.model is not None, "Model did not train correctly."


def test_forecast_output_shape():
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.Series([1, 2, 3])
    X_test = pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]})

    forecasting_api = ForecastingAPI(model_type="random_forest")
    forecasting_api.train_pipeline(X_train, y_train)
    predictions = forecasting_api.forecast(X_test)

    assert len(predictions) == len(X_test), "Forecast output length does not match input length."
