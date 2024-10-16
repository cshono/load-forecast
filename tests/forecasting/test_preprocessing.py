import numpy as np
import pandas as pd

from forecasting_library.forecasting.preprocessing import (
    TimeFeaturesAdder,
    create_preprocessing_pipeline,
)


def generate_test_data():
    return pd.DataFrame(
        {
            "feature1": [1, 2, np.nan, 4],
            "feature2": [np.nan, 2, 3, 4],
            "datetime": pd.to_datetime(pd.date_range("2024-01-01", periods=4, freq="H")),
        }
    )


def generate_test_data_w_dt_index():
    return generate_test_data().set_index("datetime")


def test_imputer_handles_missing_values():
    data_with_nans = generate_test_data()
    pipeline = create_preprocessing_pipeline(
        numeric_cols=["feature1", "feature2"], categorical_cols=["hour"], datetime_column="datetime"
    )
    transformed_data = pipeline.fit_transform(data_with_nans)
    assert not np.any(
        np.isnan(transformed_data)
    ), "Imputer did not handle missing values correctly."


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


def test_scaling():
    data = generate_test_data()
    pipeline = create_preprocessing_pipeline(
        numeric_cols=["feature1", "feature2"], categorical_cols=["hour"], datetime_column="datetime"
    )
    transformed_data = pipeline.fit_transform(data)
    assert np.allclose(
        np.min(transformed_data, axis=0), 0, atol=1e-7
    ), "Data min is not scaled to 0 after scaling."
    assert np.allclose(
        np.max(transformed_data, axis=0), 1, atol=1e-7
    ), "Data max is not scaled to 1 after scaling."
