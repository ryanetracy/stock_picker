"""
full pipeline for loading, transforming, training, and forecasting data for the
XGBoost model.
"""
from __future__ import annotations 

import pandas as pd
import polars as pl
from datetime import datetime
from typing import Tuple, Dict, List, Optional 
import numpy as np

from src.load_data import load_stocks
from src.utils import build_forecast_dates
from src.xgboost_model.train_xgboost_model import train_xgb_model
from src.xgboost_model.xgboost_forecaster import XGBStockForecaster, XGBExpiryForecaster

def train_and_forecast_xgb_options(
    ticker: str,
    start_date: str,
    end_date: str,
    cutoff: datetime,
    label_col: str = "close",
    horizons: List[int] = [10, 21, 40],
    quantiles: Optional[List[float]] = [0.1, 0.5, 0.9],
    label_mode: str = "log_return",
    n_estimators: int = 400,
    learning_rate: float = 0.03,
) -> Tuple[pl.DataFrame, Dict[int, float], Dict[int, float]]:
    """full pipeline for forecasting options.

    for each horizon, train model(s) and forecast expiry distribution. returns
    combined forecast table (rows = horizons) and RMSEs per horizon.

    Args:
        ticker (str): stock ticker to predict.
        start_date (str): when to start the training data.
        end_date (str): last day of the training data.
        cutoff (datetime): datetime object for train-test split.
        label_col (str, optional): which value to predict. defaults to "close".
        horizons (List[int], optional): horizon days to forecast. 
            defaults to [10, 21, 40].
        quantiles (Optional[List[float]], optional): quantile distributions to
            model on. defaults to [0.1, 0.5, 0.9].
        label_mode (str, optional): how to transform the label. defaults to
            "log_return".
        n_estimators (int, optional): XGBoost `n_estimators` hyperparameter.
            defaults to 400.
        learning_rate (float, optional): XGBoost `learning_rate` hyperparameter. 
            defaults to 0.03.

    Returns:
        Tuple[pl.DataFrame, Dict[int, float], Dict[int, float]]: dataframe of p
            redicted values per horizon and the RMSE from training and the 
            improvements.
    """
    stocks = [ticker]
    df_raw = load_stocks(stocks, start_date, end_date, use_polars=True)

    forecasts = []
    rmses: Dict[int, float] = {}
    improvements: Dict[int, float] = {}

    for h in horizons:
        models, rmse, df_feat, feature_cols = train_xgb_model(
            stocks=stocks,
            start_date=start_date,
            end_date=end_date,
            cutoff=cutoff,
            label_col=label_col,
            horizon=h,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            quantiles=quantiles,
            label_mode=label_mode
        )
        rmses[h] = rmse 

        y_test = (
            df_feat
            .filter(pl.col("date") >= cutoff)
            .select("label")
            .to_numpy()
            .ravel()
        )

        rmse_zero = (
            float(np.sqrt(np.mean(np.square(y_test)))) 
            if len(y_test) 
            else np.nan
        )

        improvement = (
            float(1.0 - (rmse / rmse_zero))
            if np.isfinite(rmse_zero) and rmse_zero > 0
            else np.nan
        )
        improvements[h] = improvement

        forecaster = XGBExpiryForecaster(
            models=models, feature_cols=feature_cols, label_mode=label_mode
        )
        fc = forecaster.forecast_expiry(
            df_raw=df_raw, horizon=h, price_col=label_col
        )
        forecasts.append(fc)

    return pl.concat(forecasts, how="vertical"), rmses, improvements


def train_and_forecast_xgb(
    ticker: str,
    start_date: str,
    end_date: str,
    cutoff: datetime,
    horizon_days: int,
    label_col: str = "close",
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
        label_col (str, optional): which value to predict. defaults to "close".
        n_estimators (int, optional): XGBoost `n_estimators` hyperparameter.
            defaults to 200.
        learning_rate (float, optional): XGBoost `learning_rate` hyperparameter.
            defaults to 0.05.

    Returns:
        Tuple[pl.DataFrame, float]: dataframe of predicted values per date and
            the RMSE from training.
    """
    stocks = [ticker]

    df_raw = load_stocks(stocks, start_date, end_date, use_polars=True)
    last_date = df_raw.select("date").max().item()
    future_dates = build_forecast_dates(last_date, horizon_days)

    preds = []
    rmses = []

    for h in range(1, horizon_days + 1):
        model, rmse, df_feat, feature_cols = train_xgb_model(
            stocks=stocks,
            start_date=start_date,
            end_date=end_date,
            cutoff=cutoff,
            label_col=label_col,
            horizon=h,
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )

        forecaster = XGBStockForecaster(model, feature_cols, label_col=label_col)
        pred = forecaster.predict_one(df_raw)

        preds.append(pred)
        rmses.append(rmse)

    rmse_out = float(sum(rmses) / len(rmses))

    df_out = pl.from_pandas(
        pd.DataFrame({"date": future_dates, f"pred_{label_col}": preds})
    )

    return df_out, rmse_out
