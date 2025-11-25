
""" 
handle the creation of train-test splits for each model. not all are the same.
split for xgboost is traditional (X,y train/test tables), but for forecasting
models the split is a train/eval split without a test dataframe.

each split is done based on a cutoff date to only allow training on past data 
and testing/eval on future data.
"""

from datetime import datetime
import polars as pl
import pandas as pd
from typing import Tuple

def train_test_split_cutoff(
    df: pl.DataFrame, cutoff: datetime, label: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """do a train-test split based on a cutoff value.

    training data is all data prior to the cutoff, testing data is all data on
    or after the cutoff.

    Args:
        df (pl.DataFrame): dataframe to do the split on.
        cutoff (datetime): datetime object indicating the split date.
        label (str): label column to indicate which is 'y'.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]: gives the
            "traditional" X_train, X_test, y_train, y_test output (akin to
            sklearn).
    """
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
    """split autoregressive data on a cutoff value.

    this doesn't return unique tables for X and y and is specifically designed
    for SARIMAX. can also be used for training Prophet. a pandas dataframe is
    returned (not polars) for use in the forecasting models.

    Args:
        df (pl.DataFrame): dataframe to do the split on.
        cutoff (datetime): datetime object indicating the split date.
        label (str): label column to indicate which is the endogenous value.
        exog_feats (list): list of exogenous features for SARIMAX.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train/eval dataframes returned as
            pandas dataframes.
    """
    train = df.filter(pl.col("date") < cutoff)
    eval = df.filter(pl.col("date") >= cutoff)

    train_pd = train.to_pandas()
    eval_pd = eval.to_pandas()

    chg_cols = [f"{label}_rolling_std_7"]

    cols_list = [label] + exog_feats + chg_cols

    train_pd = train_pd[cols_list]
    eval_pd = eval_pd[cols_list]

    return train_pd, eval_pd

def split_prophet_df(
        df: pd.DataFrame, cutoff: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """train/eval split for Prophet model.

    Args:
        df (pd.DataFrame): pandas dataframe set up for Prophet.
        cutoff (datetime): datetime object indicating the split date.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train/eval dataframes returned as
            pandas dataframes.
    """
    df_train = df[df["ds"] < cutoff]
    df_eval = df[df["ds"] >= cutoff]

    return df_train, df_eval
