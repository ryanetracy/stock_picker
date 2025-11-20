
"""
set of functions to process `yfinance` data, adding lagged features and time
indicators to build the full feature space for each model.
"""

import polars as pl
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

def build_dataset(df: pl.DataFrame, label: str = "close") -> pl.DataFrame:
    """wrapper for `prep_data_frame`.

    Args:
        df (pl.DataFrame): raw `yfinance` stock dataframe.
        label (str, optional): which of 'close' or 'move' to process. Defaults
        to "close".

    Raises:
        ValueError: only accepts 'close' or 'move'.

    Returns:
        pl.DataFrame: fully processed `yfinance` data with nulls removed.
    """
    df_feat = prep_data_frame(df)

    if label == "close":
        df_feat = df_feat.with_columns(pl.col("close").shift(-1).alias("label"))
    elif label == "move":
        df_feat = df_feat.with_columns(pl.col("move").shift(-1).alias("label"))
    else:
        raise ValueError("label must be one of ['close', 'move']")

    return df_feat.drop_nulls()

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
