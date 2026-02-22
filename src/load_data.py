"""
function written to easily load and process stock data from `yfinance`.
"""
import pandas as pd
import yfinance as yf
import polars as pl
from datetime import date, datetime 

from src.utils import trading_days_between

def load_stocks(
    stocks: str | list[str], start: str, end: str, use_polars: bool = True
) -> pl.DataFrame | pd.DataFrame:
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
    if isinstance(stocks, str):
        stocks = [stocks]

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


def get_option_chain(
    ticker: str, expiry: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """return the chain of options available for a given ticker.

    Args:
        ticker (str): ticker symbol to retrieve options for.
        expiry (str): date of expiry.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: list of expiry datasets for calls/puts.
    """
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls["type"] = "call" 
    puts["type"] = "put"
    return calls, puts 


def list_expiries(ticker: str) -> list[str]:
    """simple list of all expiries for a stock ticker."""
    return list(yf.Ticker(ticker).options)


def pick_expiries_for_horizons(
    ticker: str,
    horizons_td: list[int],
    asof: date | None = None 
) -> dict[int, str]:
    """map each horizon to the closest available listed expiry.

    Args:
        ticker (str): ticker symbol to retrieve options for.
        horizons_td (list[int]): list of horizons being forecasted.
        asof (date | None, optional): 'as of' run date. defaults to None.

    Returns:
        dict[int, str]: mapping dictionary of horizons and closest dates of
            expiry.
    """
    if asof is None:
        asof = date.today()

    exp_strs = list_expiries(ticker)
    exp_dates = [datetime.strptime(s, "%Y-%m-%d").date() for s in exp_strs]

    mapping = {}
    for h in horizons_td:
        best = None 
        best_dist = None 

        for d, s in zip(exp_dates, exp_strs):
            td = trading_days_between(asof, d)
            dist = abs(td - h)
            if best is None or dist < best_dist:
                best, best_dist = s, dist 
        mapping[h] = best 

    return mapping
