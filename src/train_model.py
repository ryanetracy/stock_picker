
"""
contains a wrapper function for loading the data and training the XGBoost model.
forecasting is not done here, only model training.
"""

import polars as pl
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from datetime import datetime
from typing import Tuple

from src.load_data import load_stocks
from src.data_etl import *
from src.model_preprocess import train_test_split_cutoff


def train_xgb_model(
    stocks: list,
    start_date: str,
    end_date: str,
    cutoff: datetime,
    label: str,
    n_estimators: int,
    learning_rate: float,
) -> Tuple[xgb.XGBRegressor, float, pl.DataFrame, list]:
    """load raw data, preprocess, and train XGBoost model.

    Args:
        stocks (list): single-item list of stock tickers.
        start_date (str): when to start the dataframe.
        end_date (str): final date of the dataframe.
        cutoff (datetime): cutoff datetime object for train/test splits.
        label (str): label column (y).
        n_estimators (int): XGBoost `n_estimators` hyperparameter.
        learning_rate (float): XGBoost `learning_rate` hyperparameter.

    Raises:
        ValueError: cannot process more than one stock at a time.

    Returns:
        Tuple[xgb.XGBRegressor, float, pl.DataFrame, list]: XGBoost regression
        model, RMSE value, full dataframe with features, features list.
    """
    if len(stocks) > 1:
        raise ValueError("can only do one stock forecast at a time")

    df_raw = load_stocks(
        stocks=stocks,
        start=start_date,
        end=end_date,
        use_polars=True
    )

    df_feat = build_dataset(df=df_raw, label=label)

    X_train, X_test, y_train, y_test = train_test_split_cutoff(
        df=df_feat, cutoff=cutoff, label="label"
    )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)

    feature_cols = X_train.to_pandas().columns.tolist()

    return model, rmse, df_feat, feature_cols
