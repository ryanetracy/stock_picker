"""
contains a wrapper function for loading the data and training the XGBoost model.
forecasting is not done here, only model training.
"""
from __future__ import annotations

import polars as pl
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from datetime import datetime
from typing import Tuple, Dict, List, Optional

from src.load_data import load_stocks
from src.model_preprocess import train_test_split_cutoff
from src.xgboost_model.xgboost_etl import build_dataset


def _make_base_params(n_estimators: int, learning_rate: float) -> dict:
    return dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        objective="reg:squarederror"
    )


def _fit_point_model(X_train, y_train, params: dict) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model 


def _fit_quantile_model(
    X_train, y_train, params: dict, alpha: float
) -> xgb.XGBRegressor:
    qparams = dict(params)
    qparams["objective"] = "reg:quantileerror" 
    qparams["quantile_alpha"] = alpha 
    model = xgb.XGBRegressor(**qparams)
    model.fit(X_train, y_train)
    return model 


def train_xgb_model(
    stocks: List[str],
    start_date: str,
    end_date: str,
    cutoff: datetime,
    label_col: str,
    horizon: int,
    n_estimators: int,
    learning_rate: float,
    quantiles: Optional[List[float]] = None,
    label_mode: str = "log_return"
) -> Tuple[Dict[str, xgb.XGBRegressor], float, pl.DataFrame, List[str]]:
    """load raw data, preprocess, and train XGBoost model.

    trains either a point model (no quantiles) or a quantiles bundle (list of 
    floats).

    Args:
        stocks (list): single-item list of stock tickers.
        start_date (str): when to start the dataframe.
        end_date (str): final date of the dataframe.
        cutoff (datetime): cutoff datetime object for train/test splits.
        label_col (str): label column (y).
        horizon (int): forecast horizon.
        n_estimators (int): XGBoost `n_estimators` hyperparameter.
        learning_rate (float): XGBoost `learning_rate` hyperparameter.
        quantiles (Optional[List[float]]): list of quantiles to train models on.
        label_mode: whether the label is a log or a simple return.

    Raises:
        ValueError: cannot process more than one stock at a time.

    Returns:
        Tuple[Dict[str, xgb.XGBRegressor], float, pl.DataFrame, list]: XGBoost 
        regression model, RMSE value, full dataframe with features, features
        list.
    """
    if len(stocks) > 1:
        raise ValueError("can only do one stock forecast at a time")

    df_raw = load_stocks(
        stocks=stocks, start=start_date, end=end_date, use_polars=True
    )
    df_feat = build_dataset(
        df=df_raw, label_col=label_col, horizon=horizon, label_mode=label_mode
    )
    X_train, X_test, y_train, y_test = train_test_split_cutoff(
        df=df_feat, cutoff=cutoff, label_col="label"
    )
    feature_cols = X_train.to_pandas().columns.tolist()

    base_params = _make_base_params(
        n_estimators=n_estimators, learning_rate=learning_rate
    )

    models: Dict[str, xgb.XGBRegressor] = {}

    if not quantiles:
        m = _fit_point_model(X_train, y_train, base_params)
        preds = m.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        models["point"] = m 
        return models, rmse, df_feat, feature_cols

    for q in quantiles:
        key = f"q{int(round(q*100))}"
        try:
            models[key] = _fit_quantile_model(
                X_train, y_train, base_params, alpha=q
            )
        except TypeError as e:
            raise TypeError(
                "xgboost installed doesn't support sklearn quantile params "
                "('reg:quantileerror' / 'quantile_alpha'). "
                "upgrade xgboost or use the point model + residual bands."
            ) from e 

    if "q50" not in models:
        mid = sorted(quantiles)[len(quantiles)//2]
        mid_key = f"q{int(round(mid*100))}"
    else:
        mid_key = "q50" 

    preds = models[mid_key].predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)

    return models, rmse, df_feat, feature_cols

