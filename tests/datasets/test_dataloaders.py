import pandas as pd

from forecasting_library.datasets import load_caiso


def test_caiso_dataset():
    caiso_df = load_caiso()

    assert isinstance(caiso_df.index, pd.DatetimeIndex)
    assert set(caiso_df.columns) == {
        "CAISO_system_load",
        "temp_forecast_dayahead_bakersfield",
        "temp_forecast_dayahead_los_angeles",
        "temp_forecast_dayahead_san_francisco",
        "dewpoint_forecast_dayahead_bakersfield",
        "dewpoint_forecast_dayahead_los_angeles",
        "dewpoint_forecast_dayahead_san_francisco",
    }
    assert len(caiso_df) == 31367
