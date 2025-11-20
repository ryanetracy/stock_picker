
from datetime import datetime
import polars as pl
import pandas as pd
from typing import Tuple

def train_test_split_cutoff(
    df: pl.DataFrame, cutoff: datetime, label: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("date") < cutoff)
    test = df.filter(pl.col("date") >= cutoff)

    X_train, X_test = (
        train.drop(label, "ticker", "date"),
        test.drop(label, "ticker", "date")
    )
    y_train, y_test = train[label], test[label]

    return X_train, X_test, y_train, y_test

def split_ar_on_cutoff(
    df: pl.DataFrame, cutoff: datetime, label: str, exog_feats: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.filter(pl.col("date") < cutoff)
    eval = df.filter(pl.col("date") >= cutoff)

    train_pd = train.to_pandas()
    eval_pd = eval.to_pandas()

    chg_cols = [
        # f"prev1_{label}",
        # f"prev7_{label}",
        # f"prev30_{label}",
        # f"{label}_rolling_mean_7",
        f"{label}_rolling_std_7"
    ]

    cols_list = [label] + exog_feats + chg_cols

    train_pd = train_pd[cols_list]
    eval_pd = eval_pd[cols_list]

    return train_pd, eval_pd
