
import pandas as pd 
import polars as pl 
from prophet import Prophet 
from datetime import datetime
from typing import Tuple
from sklearn.metrics import root_mean_squared_error

import warnings 
warnings.filterwarnings("ignore")

from src.prophet_model.prophet_etl import build_prophet_df
from src.utils import build_forecast_dates
from src.model_preprocess import split_prophet_df


class ProphetForecaster:
    """wrapper class to train and forecast Prophet model."""

    def __init__(self, base_params: dict | None = None):
        self.base_params = base_params or {"interval_width": 0.95}
        self.model = None
        self.reg_cols = [] 

    def train_model(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        cutoff: datetime,
        label: str = "close",
        indexes: list[str] | None = None
    ) -> Tuple[float, pd.DataFrame]:
        """build data, train and evaluate Prophet model.

        data includes the stock label (defaults to close), index fund values,
        returns, RSI (7, 14, 21 days), and lag features of label (1, 7, 30 days)
        to predict future label values.

        Args:
            stock (list): stock to be predicted.
            start_date (str): start of historical price data.
            end_date (str): end of historical price data.
            cutoff (datetime): date to split for train/eval.
            label (str, optional): value to predict. defaults to "close".
            indexes (list[str] | None, optional): specific index funds to add to
                feature space. defaults to None.

        Returns:
            Tuple[float, pd.DataFrame]: RMSE value for evaluation and eval
                dataframe.
        """
        stock = [ticker]

        df = build_prophet_df(stock, start_date, end_date, label, indexes)
        df_train, df_eval = split_prophet_df(df, cutoff)

        self.reg_cols = [c for c in df.columns if c not in ["ds", "y"]]

        pr_model = Prophet(**self.base_params)
        for col in self.reg_cols:
            pr_model.add_regressor(col)

        pr_model.fit(df_train[["ds", "y"] + self.reg_cols])
        self.model = pr_model

        df_eval_future = df_eval[["ds"] + self.reg_cols].copy()
        forecast_eval = self.model.predict(df_eval_future)

        df_forecast = forecast_eval[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        df_eval_out = df_eval[["ds", "y"]]
        df_eval_out = df_eval_out.merge(df_forecast, on="ds", how="inner")

        rmse = root_mean_squared_error(df_eval["y"], df_eval_out["yhat"])

        return rmse, df_eval_out

    def _build_future_regressors(
            self, df_full: pd.DataFrame, future_dates: pd.DataFrame
        ) -> pd.DataFrame:
        """stand-in function to make a dataframe for future predictions.

        will need to be added to if this were to ever go live for the sake of
        adding future regressors (and not just dates).

        Args:
            df_full (pd.DataFrame): full features dataframe.
            future_dates (pd.DataFrame): dataframe of days in the future.

        Returns:
            pd.DataFrame: single row containing date and regressors for making
                the forecast.
        """
        last_row = df_full.sort_values("ds").iloc[-1]
        date_list = future_dates["date"].tolist()

        rows = []

        for d in date_list:
            row = {"ds": d}
            for col in self.reg_cols:
                row[col] = last_row[col]
            rows.append(row)

        return pd.DataFrame(rows)

    def make_prediction(
        self,
        df_full: pd.DataFrame,
        horizon_days: int,
        label: str = "close"
    ) -> pl.DataFrame:
        """generate predictions `horizon_days` in the future.

        returns a polars dataframe for in-line displays.

        Args:
            df_full (pd.DataFrame): full features dataframe.
            horizon_days (int): number of days to forecast into the future.
            label (str, optional): specific value to forecast. defaults to 
            "close".

        Returns:
            pl.DataFrame: polars dataframe containing future dates, predicted
                values, and upper/lower bound CIs (95%).
        """
        assert self.model is not None

        last_date = df_full["ds"].max()

        future_dates = build_forecast_dates(
            last_date,
            horizon_days=horizon_days,
            skip_weekends=True
        )

        future_regs = self._build_future_regressors(df_full, future_dates)

        future_df = future_regs[["ds"] + self.reg_cols]

        forecast = self.model.predict(future_df)
        out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        df_out = pl.from_pandas(out)

        return df_out.select(
            pl.col("ds").alias("date"),
            pl.col("yhat").alias(f"pred_{label}"),
            pl.col("yhat_lower").alias("lower_bound"),
            pl.col("yhat_upper").alias("upper_bound")
        )
