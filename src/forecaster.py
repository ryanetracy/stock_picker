
"""
contains a class for forecasting the future value from the trained XGBoost model.
"""

import polars as pl
import xgboost as xgb
from datetime import timedelta

from src.load_data import load_stocks
from src.data_etl import *


class XGBStockForecaster:
    """
    use the trained XGBoost model to make a forecast over a specified interval.
    """
    def __init__(
        self, model: xgb.XGBRegressor, feature_cols: list, label: str
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.label = label

    def _predict_from_features(self, df_feat: pl.DataFrame) -> float:
        """make a prediction based on the passed in features.

        Args:
            df_feat (pl.DataFrame): features table on which the model was
                trained.

        Returns:
            float: single predicted value.
        """
        row_pd = df_feat.select(self.feature_cols).tail(1).to_pandas()
        preds = self.model.predict(row_pd)
        return float(preds[0])

    def forecast_horizon(self, df_raw: pl.DataFrame, days: int) -> pl.DataFrame:
        """run a forecast on the full horizon indciated by `days`.

        Args:
            df_raw (pl.DataFrame): raw `yfinance` stock dataframe.
            days (int): number of days to forecast for.

        Returns:
            pl.DataFrame: table with a date and a predicted value column.
        """
        df_current = df_raw.clone()

        forecast_dates = []
        forecast_values = []

        for _ in range(days):
            df_feat = prep_data_frame(df_current)
            pred = self._predict_from_features(df_feat)
            last_date = df_current["date"][-1]
            next_date = last_date + timedelta(days=1)

            while next_date.weekday() >= 5:
                next_date = next_date + timedelta(days=1)

            forecast_dates.append(next_date)
            forecast_values.append(pred)

            last_row = df_current.tail(1)

            date_dtype = df_current.schema["date"]

            new_row = last_row.with_columns(
                pl.lit(next_date).cast(date_dtype).alias("date"),
                pl.lit(pred).alias(f"{self.label}")
            )

            df_current = df_current.vstack(new_row)

        return pl.DataFrame(
            {
                "date": forecast_dates,
                f"pred_{self.label}": forecast_values
            }
        )
