
"""
set of functions to process `yfinance` data for the SARIMAX model.

pulls code from xgboost model (`build_dataset`) and uses it to add indexes to 
the dataframe. 
"""

import polars as pl
from datetime import datetime, date, timedelta
import yfinance as yf
from typing import Tuple

from src.load_data import load_stocks
from src.xgboost_model.xgboost_etl import build_dataset

def build_df_with_indices(
    df: pl.DataFrame, label: str, start: str, end: str
) -> pl.DataFrame:
    """process raw dataframe and add indices.

    this is originally intended to bolster the SARIMAX model by adding broad
    exogenous variables to the features space.

    Args:
        df (pl.DataFrame): raw `yfiance` stock dataframe.
        label (str): 'close' or 'move' price indicator.
        start (str): start date for index pulling from `yfinance`.
        end (str): end date for index pulling from `yfinance`.

    Returns:
        pl.DataFrame: features data with lagged columns and added index values.
    """
    indices_list = ["SPY", "QQQ", "IWM", "VXX", "UUP", "HYG", "LQD"]
    df_idx_out = None

    for idx in indices_list:
        idx_cl = idx.replace("^", "")

        df_idx = load_stocks([idx], start, end).select(
            pl.col("date"),
            pl.col("close").alias(f"{idx_cl}_close"),
            pl.col("volume").alias(f"{idx_cl}_volume")
        )

        if df_idx_out is None:
            df_idx_out = df_idx
        else:
            df_idx_out = df_idx_out.join(df_idx, on=["date"], how="inner")

    df_ticker = build_dataset(df, label)

    df_out = df_ticker.join(df_idx_out, on=["date"], how="inner")

    return df_out
