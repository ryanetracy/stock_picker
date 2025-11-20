
"""
full pipeline for loading the data, training, and forecasting with SARIMAX.
"""

import polars as pl
import pandas as pd
import pmdarima as pm 
from sklearn.metrics import root_mean_squared_error
from datetime import datetime, timedelta
from typing import Tuple

from src.load_data import load_stocks
from src.data_etl import *
from src.model_preprocess import split_ar_on_cutoff


def fit_sarimax(
    ticker: str,
    start_date: str,
    end_date: str,
    label: str,
    cutoff: datetime,
    days: int,
    eval_mode: bool = True
):
    """fit the actual SARIMAX model.

    has two modes: eval and forecast. eval mode uses the train-eval split to 
    gauge how accurate the forecasts are (using RMSE). forecast mode uses the
    full dataset to train and make a forecast.

    Args:
        ticker (str): stock ticker to predict
        start_date (str): when to start the training data.
        end_date (str): last day of the training data.
        label (str): which value to predict. defaults to "close".
        cutoff (datetime): datetime object for train-test split.
        days (int): how many days in the future to forecast.
        eval_mode (bool, optional): whether to run the model as an evaluation of
        performance or as a full forecast. defaults to True (i.e., evaluate the
        model performance).

    Returns:
        _type_: output depends on `eval_mode`. returns a float (RMSE) if 
        `eval_mode` is True, or a table (date, predicted value) if `eval_mode` 
        is False.
    """
    stock = [ticker]

    df_raw = load_stocks(stock, start_date, end_date)
    df_idx = build_df_with_indices(df_raw, label, start_date, end_date)

    feats = [
        "date",
        "dow",
        "month",
        "mon_or_fri",
        "volume",
        "SPY_close",
        "SPY_volume",
        "QQQ_close",
        "QQQ_volume",
        "IWM_close",
        "IWM_volume",
        "VXX_close",
        "VXX_volume",
        "UUP_close",
        "UUP_volume",
        "HYG_close",
        "HYG_volume",
        "LQD_close",
        "LQD_volume"
    ]

    exog_cols = [feat for feat in feats if feat != "date"]

    _sarima_hyperparams = {
        "start_p": 1,
        "start_q": 1,
        "test": "adf",
        "max_p": 3,
        "max_q": 3,
        "m": 5,
        "start_P": 0,
        "seasonal": True,
        "d": None,
        "D": None,
        "trace": False,
        "error_action": "ignore",
        "suppress_warnings": True,
        "stepwise": True
    }

    if eval_mode:
        df_train, df_eval = split_ar_on_cutoff(df_idx, cutoff, "close", feats)

        sarimax_model = pm.auto_arima(
            df_train[[label]],
            exogenous=df_train[exog_cols],
            **_sarima_hyperparams
        )

        fitted, confint = sarimax_model.predict(
            n_periods=len(df_eval),
            return_conf_int=True,
            exogenous=df_eval[exog_cols]
        )

        fitted = pd.DataFrame(fitted, columns=["pred"]).reset_index(drop=True)
        df_eval["pred"] = fitted["pred"]
        rmse = root_mean_squared_error(df_eval[["close"]], df_eval[["pred"]])

        return rmse
    else:
        df = df_idx.to_pandas()

        sarimax_model = pm.auto_arima(
            df[[label]],
            exogenous=df[exog_cols],
            **_sarima_hyperparams
        )

        fitted, confint = sarimax_model.predict(
            n_periods=days,
            return_conf_int=True,
            exogenous=df[exog_cols]
        )
        fitted = pd.DataFrame(fitted, columns=[f"pred_{label}"]).reset_index(
            drop=True
        )
        ci_series = pd.DataFrame(
            confint, columns=["lower_bound", "upper_bound"]
        )

        forecast_dates = []
        last_date = df["date"].iloc[-1]

        for d in range(days):
            next_date = last_date + timedelta(days=d)

            while next_date.weekday() >= 5:
                next_date = next_date + timedelta(days=d)

            forecast_dates.append(next_date)

        df_out = pd.DataFrame({"date": forecast_dates})

        df_out[f"pred_{label}"] = fitted[f"pred_{label}"]
        df_out["lower_bound"] = ci_series["lower_bound"]
        df_out["upper_bound"] = ci_series["upper_bound"]

        return df_out.sort_values(by="date", ascending=True)

def sarimax_wrapper(
    ticker: str,
    start_date: str,
    end_date: str,
    label: str,
    cutoff: datetime,
    days: int
) -> Tuple[pl.DataFrame, float]:
    """wrapper to run both versions of `fit_sarimax`.

    gets training results (`eval_mode == True`) and forecast results (`eval_mode
    == False`).

    Args:
        ticker (str): stock ticker to predict
        start_date (str): when to start the training data.
        end_date (str): last day of the training data.
        label (str): which value to predict. defaults to "close".
        cutoff (datetime): datetime object for train-test split.
        days (int): how many days in the future to forecast.

    Returns:
        Tuple[pl.DataFrame, float]: table of predictions (date, predicted value)
        and the RMSE from training.
    """
    rmse = fit_sarimax(
        ticker,
        start_date,
        end_date,
        label,
        cutoff,
        days,
        eval_mode=True
    )

    forecasts = fit_sarimax(
        ticker,
        start_date,
        end_date,
        label,
        cutoff,
        days,
        eval_mode=False
    )

    return pl.from_pandas(forecasts), rmse
