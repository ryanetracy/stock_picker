import pandas as pd
import yfinance as yf
import polars as pl

def load_stocks(stocks: list, start: str, end: str, use_polars: bool = True):
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
