
import polars as pl
from datetime import datetime
from typing import Tuple

from src.load_data import load_stocks
from src.train_model import train_model
from src.forecaster import XGBStockForecaster

def train_and_forecast(
    ticker: str,
    start_date: str,
    end_date: str,
    cutoff: datetime,
    horizon_days: int,
    label: str = "close",
    n_estimators: int = 200,
    learning_rate: float = 0.05
) -> Tuple[pl.DataFrame, float]:
    stocks = [ticker]

    model, mse, df_feat, feature_cols = train_model(
        stocks=stocks,
        start_date=start_date,
        end_date=end_date,
        cutoff=cutoff,
        label=label,
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )

    df_raw = load_stocks(stocks, start_date, end_date)

    forecaster = XGBStockForecaster(model, feature_cols, label=label)
    forecasts_df = forecaster.forecast_horizon(df_raw, days=horizon_days)

    return forecasts_df, mse
