
import polars as pl
import xgboost as xgb
from datetime import timedelta

from src.load_data import load_stocks
from src.data_etl import *


class XGBStockForecaster:
    def __init__(
        self, model: xgb.XGBRegressor, feature_cols: list, label: str
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.label = label

    def _predict_from_features(self, df_feat: pl.DataFrame) -> float:
        row_pd = df_feat.select(self.feature_cols).tail(1).to_pandas()
        preds = self.model.predict(row_pd)
        return float(preds[0])

    def forecast_horizon(self, df_raw: pl.DataFrame, days: int) -> pl.DataFrame:
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
