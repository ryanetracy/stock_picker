
"""
full pipeline for loading, transforming, training, and forecasting data for the
XGBoost model.
"""

import polars as pl
from datetime import datetime
from typing import Tuple

from src.load_data import load_stocks
from src.xgboost_model.train_xgboost_model import train_xgb_model
from src.xgboost_model.xgboost_forecaster import XGBStockForecaster

def train_and_forecast_xgb(
    ticker: str,
    start_date: str,
    end_date: str,
    cutoff: datetime,
    horizon_days: int,
    label: str = "close",
    n_estimators: int = 200,
    learning_rate: float = 0.05
) -> Tuple[pl.DataFrame, float]:
    """full pipeline for XGBoost model.

    Args:
        ticker (str): stock ticker to predict.
        start_date (str): when to start the training data.
        end_date (str): last day of the training data.
        cutoff (datetime): datetime object for train-test split.
        horizon_days (int): how many days in the future to forecast.
        label (str, optional): which value to predict. defaults to "close".
        n_estimators (int, optional): XGBoost `n_estimators` hyperparameter.
            defaults to 200.
        learning_rate (float, optional): XGBoost `learning_rate` hyperparameter.
            defaults to 0.05.

    Returns:
        Tuple[pl.DataFrame, float]: dataframe of predicted values per date and
            the RMSE from training.
    """
    stocks = [ticker]

    model, rmse, df_feat, feature_cols = train_xgb_model(
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

    return forecasts_df, rmse
