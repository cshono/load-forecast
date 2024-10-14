from pathlib import Path

import pandas as pd


def load_caiso_data() -> pd.DataFrame:
    caiso_filepath = Path(__file__).parent / "data" / "caiso_system_load.csv"
    caiso_df = pd.read_csv(caiso_filepath)
    caiso_df["interval_start_time"] = pd.to_datetime(
        caiso_df["interval_start_time"], utc=True
    ).dt.tz_convert("US/Pacific")
    caiso_df = caiso_df.set_index("interval_start_time").sort_index()
    return caiso_df
