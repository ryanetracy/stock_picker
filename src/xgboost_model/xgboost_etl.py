"""
set of functions to process `yfinance` data for the XGBoost model.

adds lagged features and time indicators to build the full feature space for
each model.
"""
from __future__ import annotations 

import polars as pl
from datetime import datetime, date, timedelta
import yfinance as yf
from typing import Tuple, Literal

from src.load_data import load_stocks


def prep_columns(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """prep the columns pulled from `yfinance` into a clean dataframe

    adds a 'move' column to indicate overall daily change; time-lapse columns
    lagged over 1, 7, 30 days; and rolling mean and SD values over 7 days.

    Args:
        df (pl.DataFrame): raw `yfinance` dataframe.
        col (str): which column (choose between 'close', 'open') to compute the
            lag features for.

    Returns:
        pl.DataFrame: full dataframe with lagged features of chosen column.
    """
    if col == "move":
        df = df.with_columns(
            (pl.col("close") - pl.col("open")).alias(col)
        )

    df_out =  (
        df.select(
            [
                "date",
                "ticker",
                "volume",
                col
            ]
        )
        .sort([pl.col("ticker"), pl.col("date")], descending=False)
        .with_columns(
            pl.col(col).shift(1).over("ticker").alias(f"prev1_{col}"),
            pl.col(col).shift(7).over("ticker").alias(f"prev7_{col}"),
            pl.col(col).shift(30).over("ticker").alias(f"prev30_{col}"),
        )
        .with_columns(
            pl.col(col)
            .rolling_mean(window_size=7, min_samples=2)
            .shift(1)
            .over("ticker")
            .alias(f"{col}_rolling_mean_7")
        )
        .with_columns(
            pl.col(col)
            .rolling_std(window_size=7, min_samples=2)
            .shift(1)
            .over("ticker")
            .alias(f"{col}_rolling_std_7")
        )
    )

    return df_out


def prep_data_frame(df: pl.DataFrame) -> pl.DataFrame:
    """prepare lag columns for all of 'open', 'close', and 'move'.

    Args:
        df (pl.DataFrame): raw `yfinance` stock dataframe.

    Returns:
        pl.DataFrame: processed stock data with lag columns for all price
            indicators.
    """
    markers = ["open", "close", "move"]
    df_out = None

    for marker in markers:
        df_prep = prep_columns(df, marker)
        if df_out is None:
            df_out = df_prep
        else:
            df_out = df_prep.join(
                df_out, on=["date", "ticker", "volume"], how="inner"
            )

    return (
        df_out.with_columns(
            pl.col("date").dt.weekday().alias("dow")
        )
        .with_columns(
            pl.col("date").dt.month().alias("month")
        )
        .with_columns(
            pl.when(pl.col("dow").is_in([0, 4]))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("mon_or_fri")
        )
    )

LabelMode = Literal["log_return", "simple_return"]

def build_dataset(
    df: pl.DataFrame, 
    label_col: str = "close", 
    horizon: int = 21, 
    label_mode: LabelMode = "log_return"
) -> pl.DataFrame:
    """wrapper for `prep_data_frame`.

    Args:
        df (pl.DataFrame): raw `yfinance` stock dataframe.
        label (str, optional): which of 'close' or 'move' to process. defaults
            to "close".
        horizon (int, optional): length of forecast horizon. must be >= 1.
        label_mode (Literal): 

    Raises:
        ValueError: only accepts `horizon` values >= 1.
        ValueError: only accepts 'close' or 'move' or 'open'.
        ValueError: only accepts 'log_return' or 'simple_return'.

    Returns:
        pl.DataFrame: fully processed `yfinance` data with nulls removed.
    """
    if label_col not in {"close", "open", "move"}:
        raise ValueError("`label_col` must be one of ['close', 'open', 'move']")

    df_feat = prep_data_frame(df)

    if horizon < 1:
        raise ValueError("`horizon` must be >= 1")

    if label_col == "move":
        base = pl.col("move")
    else:
        base = pl.col(label_col)

    future = base.shift(-horizon)

    if label_mode == "log_return":
        df_feat = df_feat.with_columns(
            (future.log() - base.log()).alias("label")
        )
    elif label_mode == "simple_return":
        df_feat = df_feat.with_columns(((future / base) - 1.0).alias("label"))
    else:
        raise ValueError(
            "`labl_mode` must be one of ['log_return', 'simple_return']"
        )

    return df_feat.drop_nulls()
