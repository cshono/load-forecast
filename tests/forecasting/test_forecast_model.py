import pandas as pd

from forecasting_library.forecasting.forecast_model import ForecastModel


def generate_train_data():
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
        }
    )
    y = pd.Series([7, 8, 9])
    return X, y


def test_model_initialization():
    for model_type in ["linear", "random_forest", "gradient_boosting", "xgboost"]:
        forecast_run = ForecastModel(model_type=model_type)
        assert forecast_run.model is not None, f"{model_type} model did not initialize correctly."


def test_model_training():
    X, y = generate_train_data()
    forecast_run = ForecastModel(model_type="random_forest")
    forecast_run.train(X, y)
    assert forecast_run.model is not None, "Model did not train correctly."


def test_forecast_output_shape():
    X, y = generate_train_data()
    X_test = pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]})

    forecast_run = ForecastModel(model_type="random_forest")
    forecast_run.train(X, y)
    predictions = forecast_run.forecast(X_test)

    assert len(predictions) == len(X_test), "Forecast output length does not match input length."
