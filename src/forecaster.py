
import polars as pl
import xgboost as xgb
from datetime import timedelta

from src.load_data import load_stocks
from src.data_etl import *

# df_raw = load_stocks(["AAPL"], "2025-11-01")

class XGBStockForecaster:
    def __init__(
        self, model: xgb.XGBRegressor, label: str
    ):
        self.model = model
        self.label = label

    def _last_row_features(self, df_feat: pl.DataFrame) -> pl.DataFrame:
        row = df_feat[-1].drop("target", "ticker", "date")

        return row

    def forecast_next(self, df: pl.DataFrame) -> float:
        row = df[-1].drop(self.label, "ticker", "date")
        preds = self.model.predict(row.to_pandas())[0]

        return float(preds[0])

    def forecast_horizon(self, df_raw: pl.DataFrame, days: int) -> list[float]:
        df_feat = prep_data_frame(df_raw)
        forecasts = []

        for _ in range(days):
            pred = self.forecast_next(df_feat)
            forecasts.append(pred)

            next_date = df_raw["date"][-1] + timedelta(days=1)

            df_raw = df_raw.vstack(
                pl.DataFrame(
                    {
                        "date": [next_date],
                        "ticker": [df_raw["ticker"][-1]],
                        "open": [df_raw["open"][-1]],
                        "close": [pred]
                    }
                )
            )

            df_feat = prep_data_frame(df_raw)

        return forecasts
