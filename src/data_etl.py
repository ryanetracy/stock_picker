
"""
set of functions to process `yfinance` data, adding lagged features and time
indicators to build the full feature space for each model.
"""

import polars as pl
from datetime import datetime, date, timedelta
import yfinance as yf
from typing import Tuple

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

class GetSectorETF:
    """simple class to infer sector ETFs"""
    def __init__(
        self,
        indexes: list,
        label: str
    ):
        self.SECTOR_TO_ETF = {
            "Technology": "XLK",
            "Communication Services": "XLC",
            "Financial Services": "XLF",
            "Energy": "XLE",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Industrials": "XLI",
            "Healthcare": "XLV",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
        }

        self.indexes = indexes

        if label not in ["open", "close", "move"]:
            raise ValueError("label must be one of ['open', 'close', 'move]")
        else:
            self.label = label

    def extract_stock_info(self, stock_df: pl.DataFrame) -> Tuple[str, str, str]:
        """extract the ticker, earliest, and latest data from `stock_df`.

        Args:
            stock_df (pl.DataFrame): stock dataframe loaded from `yfinance`.

        Returns:
            Tuple[str, str, str]: ticker, start_date, end_date values
        """
        tickers = [stock_df.select("ticker").unique().item()]

        date_df = (
            stock_df
            .select("date")
            .unique()
            .sort(by="date", descending=True)
        )

        min_date = date_df.select("date").tail(1).item().strftime("%Y-%m-%d")
        max_date = date_df.select("date").head(1).item().strftime("%Y-%m-%d")

        return tickers, min_date, max_date

    def infer_sector_etfs(self, stock_df: pl.DataFrame) -> pl.DataFrame:
        """using tickers, extracts the close and volume of sector ETFs.

        Args:
            stock_df (pl.DataFrame): stock dataframe loaded from `yfinance`.

        Returns:
            pl.DataFrame: dataframe of dates, `label` and volume values for
                relevant ETFs for the stock's ticker, as well as `label` and 
                volume values for the passed-in indexes.
        """
        tickers, start_date, end_date = self.extract_stock_info(stock_df)

        etfs = set()

        for t in tickers:
            info = yf.Ticker(t).info
            sector = info.get("sector")
            if sector in self.SECTOR_TO_ETF:
                etfs.add(self.SECTOR_TO_ETF[sector])

        ticker_list = list(etfs)
        ticker_list += self.indexes
        df_out = None

        for ticker in ticker_list:
            df_temp = load_stocks([ticker], start_date, end_date).select(
                pl.col("date"),
                pl.col("volume").alias(f"{ticker}_volume"),
                pl.col(self.label).alias(f"{ticker}_{self.label}")
            )

            if df_out is None:
                df_out = df_temp 
            else:
                df_out = df_out.join(df_temp, on=["date"], how="inner")

        return df_out

    def build_etf_df(self, stock_df: pl.DataFrame) -> pl.DataFrame:
        """combine ETF values with stock dataframe

        Args:
            stock_df (pl.DataFrame): stock dataframe loaded from `yfinance`.

        Returns:
            pl.DataFrame: combined dataframe of `stock_df` with ETF values.
        """
        df_etf = self.infer_sector_etfs(stock_df)

        return stock_df.join(df_etf, on=["date"], how="inner")
