
"""
function written to easily load and process stock data from `yfinance`.
"""

import pandas as pd
import yfinance as yf
import polars as pl

def load_stocks(stocks: list, start: str, end: str, use_polars: bool = True):
    """load stock data from `yfinance`.

    stocks are loaded singularly, from start to end date. option to return a 
    polars dataframe or a pandas dataframe.

    Args:
        stocks (list): single-item list of stock tockers.
        start (str): first historical date.
        end (str): last historical date (up to today).
        use_polars (bool, optional): whether to return a polars dataframe or a
            pandas dataframe. defaults to true.

    Raises:
        ValueError: raised if users enter more than 1 ticker

    Returns:
        DataFrame: polars or pandas dataframe, depending on the value of
        `use_polars`.
    """
    if len(stocks) > 1:
        raise ValueError("can only do one stock forecast at a time")
    df = yf.download(stocks, start, end)
    df.index = pd.to_datetime(df.index)
    df.columns = (
        pd.MultiIndex.from_tuples(df.columns) 
        if not isinstance(df.columns, pd.MultiIndex) else df.columns
    )
    df.columns = df.columns.set_names(["Field", "Ticker"])
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")

    df_out = (
        df.swaplevel("Field", "Ticker", axis=1)
        .sort_index(axis=1)
        .stack("Ticker", future_stack=True)
        .reset_index()
    )

    df_out = df_out.rename(columns=str.lower)

    return pl.from_pandas(df_out) if use_polars else df_out
