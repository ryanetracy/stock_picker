
"""
set of functions to process `yfinance` data for the Prophet model.

focuses on pulling indexes and computing returns and volatility to add these as
regressors in Prophet.
"""

import yfinance as yf 
import polars as pl 
import pandas as pd
from typing import Tuple

from src.load_data import load_stocks 


class GetSectorETF:
    """simple class to infer sector ETFs"""
    def __init__(
        self,
        indexes: list,
        label: str,
        target_ticker: str
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
        self.target_ticker = target_ticker

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
        """using tickers, extracts the close values of sector ETFs.

        Args:
            stock_df (pl.DataFrame): stock dataframe loaded from `yfinance`.

        Returns:
            pl.DataFrame: dataframe of dates, `label` values for relevant ETFs 
                for the stock's ticker, as well as `label` and values for the
                passed-in indexes.
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
                pl.col(self.label).alias(f"{ticker}")
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

        return (
            stock_df.select(
                pl.col("date"), pl.col(self.label).alias(self.target_ticker)
            )
            .join(
                df_etf, on=["date"], how="inner"
            )
        )

def compute_returns(df: pl.DataFrame) -> pl.DataFrame:
    """compute the daily return rate.

    Args:
        df (pl.DataFrame): stock dataframe loaded from `yfinance` and processed
            through `GetSectorETF`.

    Returns:
        pl.DataFrame: polars dataframe with daily returns for a target stock and
            desired indexes.
    """
    tickers = [c for c in df.columns if c != "date"]

    for ticker in tickers:
        df = df.with_columns(
            pl.col(ticker).pct_change().alias(f"{ticker}_return")
        )

    return df

def compute_return_volatility(df: pl.DataFrame, window: int) -> pd.DataFrame:
    """compute the rolling volatility (SD) of target stock and indexes.

    returns a pandas dataframe for use in prepping data for Prophet.

    Args:
        df (pl.DataFrame): polars dataframe with target stock and index prices
            and returns.
        window (int): rolling window (in days) to compute volatility.

    Returns:
        pd.DataFrame: pandas dataframe with prices, returns, and return 
            volatility.
    """
    tickers = [c for c in df.columns if c.endswith("_return")]

    for ticker in tickers:
        df = df.with_columns(
            pl.col(ticker)
            .rolling_std(window_size=window, min_samples=1)
            .alias(f"{ticker}_return_rolling_std_{window}")
        )

    return df.to_pandas()

def compute_rsi(df: pd.DataFrame, ticker: str, period: int) -> pd.Series:
    """add a column for RSI over a period window.

    Args:
        df (pd.DataFrame): pandas dataframe with target stock PRICES.
        ticker (str): stock to compute RSI for.
        period (int): window (days).

    Returns:
        pd.Series: pandas series to add to `df` as a column containing RSI
            values.
    """
    prices = pd.to_numeric(df[ticker], errors="coerce")
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def compute_price_volatility(
    df: pd.DataFrame, ticker: str, period: int
) -> pd.Series:
    """add price volatility columns.

    Args:
        df (pd.DataFrame): pandas dataframe with price data.
        ticker (str): stock to compute volatility for.
        period (int): window (days).

    Returns:
        pd.Series: column indicating lagged prices.
    """
    return df[ticker].shift(period)

def build_prophet_df(
    stocks: list,
    start: str,
    end: str,
    label: str,
    indexes: list
) -> pd.DataFrame:
    """run through suite of functions to build the final dataset for Prophet.

    Args:
        stocks (list): _description_
        start (str): _description_
        end (str): _description_
        label (str): _description_
        indexes (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    tkr = stocks[0]

    df_raw = load_stocks(stocks, start, end)
    etfs = GetSectorETF(indexes=indexes, label=label, target_ticker=tkr)
    df_etf = etfs.build_etf_df(df_raw)

    df_ret = compute_returns(df_etf)
    df_vol = compute_return_volatility(df_ret, 10)

    df_vol["rsi_7"] = compute_rsi(df_vol, tkr, 7)
    df_vol["rsi_14"] = compute_rsi(df_vol, tkr, 14)
    df_vol["rsi_21"] = compute_rsi(df_vol, tkr, 21)
    df_vol["prev7_close"] = compute_price_volatility(df_vol, tkr, 1)
    df_vol["prev14_close"] = compute_price_volatility(df_vol, tkr, 7)
    df_vol["prev30_close"] = compute_price_volatility(df_vol, tkr, 30)

    return df_vol.rename(
        columns={"date": "ds", f"{tkr}": "y"}
    ).dropna()
