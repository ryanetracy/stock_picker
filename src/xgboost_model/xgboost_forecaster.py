"""
contains a class for forecasting the future value from the trained XGBoost model.
"""
from __future__ import annotations 

import pandas as pd
import polars as pl
import xgboost as xgb
from datetime import timedelta, datetime
from typing import Dict, List 

from src.xgboost_model.xgboost_etl import prep_data_frame
from src.utils import build_trading_future_dates


class XGBExpiryForecaster:
    """
    forecast a horizon-ahead return distribution (or point) from the latest 
    features.
    """
    def __init__(
        self,
        models: Dict[str, xgb.XGBRegressor],
        feature_cols: List[str],
        label_mode: str = "log_return"
    ):
        self.models = models
        self.feature_cols = feature_cols
        self.label_mode = label_mode

    def _latest_feature_row(self, df_raw: pl.DataFrame) -> pd.DataFrame:
        df_feat = prep_data_frame(df_raw)
        return df_feat.select(self.feature_cols).tail(1).to_pandas()

    def _return_to_price(self, s0: float, r: float) -> float:
        if self.label_mode == "log_return":
            return float(s0 * (2.718281828459045 ** r))
        return float(s0 * (1.0 + r))

    def forecast_expiry(
        self,
        df_raw: pl.DataFrame,
        horizon: int,
        price_col: str = "close"
    ) -> pl.DataFrame:
        """forecast to expiry at +horizon trading days

        returns 1-row df with date + predicted return quantiles + price quantiles.

        Args:
            df_raw (pl.DataFrame): raw `yfinance` stock data.
            horizon (int): forecast horizon trading days.
            price_col (str, optional): the initial column off of which the label
                was built. defaults to 'close'. 

        Returns:
            pl.DataFrame: table with dates and predicted values.
        """
        last_date = df_raw["date"][-1]

        if isinstance(last_date, datetime):
            last_dt = last_date
        else:
            last_dt = datetime.combine(last_date, datetime.min.time())

        expiry_date = build_trading_future_dates(last_dt, horizon)[-1]

        s0 = float(df_raw[price_col][-1])
        X = self._latest_feature_row(df_raw)

        out = {"date": [expiry_date], "spot": [s0], "horizon": [horizon]}

        for name, m in self.models.items():
            rhat = float(m.predict(X)[0])
            out[f"pred_{name}_ret"] = [rhat]
            out[f"pred_{name}_px"] = [self._return_to_price(s0, rhat)]

        return pl.from_pandas(pd.DataFrame(out))


class XGBStockForecaster:
    """
    use the trained XGBoost model to make a forecast over a specified interval.
    """
    def __init__(
        self, model: xgb.XGBRegressor, feature_cols: list, label_col: str
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.label_col = label_col

    def predict_one(self, df_raw: pl.DataFrame) -> float:
        """predict using the most recent row of features built from `df_raw`"""
        df_feat = prep_data_frame(df_raw)
        row_pd = df_feat.select(self.feature_cols).tail(1).to_pandas()
        pred = self.model.predict(row_pd)
        return float(pred[0])
