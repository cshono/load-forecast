import numpy as np
import pandas as pd

from forecasting_library.forecasting.preprocessing import TimeFeaturesAdder


def generate_test_data():
    return pd.DataFrame(
        {
            "target": [5, 6, 7, 8],
            "feature1": [1, 2, np.nan, 4],
            "feature2": [np.nan, 2, 3, 4],
            "datetime": pd.to_datetime(pd.date_range("2024-01-01", periods=4, freq="H")),
        }
    )


def generate_test_data_w_dt_index():
    return generate_test_data().set_index("datetime")


def test_add_time_features():
    data = generate_test_data()
    pipeline = TimeFeaturesAdder(datetime_column="datetime")
    transformed_data = pipeline.fit_transform(data)
    expected_columns = ["hour", "day_of_week", "month"]
    assert all(
        col in transformed_data.columns for col in expected_columns
    ), "Time features not added correctly."


def test_add_time_features_from_index():
    data = generate_test_data_w_dt_index()
    pipeline = TimeFeaturesAdder(datetime_column="index")
    transformed_data = pipeline.fit_transform(data)
    expected_columns = ["hour", "day_of_week", "month"]
    assert all(
        col in transformed_data.columns for col in expected_columns
    ), "Time features not added correctly."
