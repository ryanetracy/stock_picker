
"""
utility functions
    build_forecast_dates
"""

from datetime import timedelta
import pandas as pd

def build_forecast_dates(
        last_date,
        horizon_days: int,
        skip_weekends: bool = True
) -> pd.DataFrame:
    """create a dataframe of dates based on `horizon_days`.

    Args:
        last_date (_type_): either a string date or datetime object.
        horizon_days (int): number of days (rows) to create dataframe of.
        skip_weekends (bool, optional): should weekends be skipped. defaults to
            True.

    Returns:
        pd.DataFrame: pandas dataframe of dates.
    """
    if not isinstance(last_date, (pd.Timestamp, )):
        last_date = pd.Timestamp(last_date)

    forecast_dates = []
    current_date = last_date 

    for _ in range(horizon_days):
        current_date = current_date + timedelta(days=1)

        if skip_weekends:
            while current_date.weekday() >= 5:
                current_date = current_date + timedelta(days=1)

        forecast_dates.append(current_date)

    return pd.DataFrame({"date": forecast_dates})
