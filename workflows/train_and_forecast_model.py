
"""
script for getting all model results.
"""

import pandas as pd 
import polars as pl 
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")

from src.xgboost_model.xgboost_pipeline import train_and_forecast_xgb
from src.sarimax_model.train_predict_sarimax_model import *
from src.prophet_model.prophet_model_pipeline import ProphetForecaster
from src.prophet_model.prophet_etl import build_prophet_df

ticker = "MSFT"
horizon = 10

end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - relativedelta(years=10)).strftime("%Y-%m-%d")
cutoff = (datetime.today() - relativedelta(months=2))

label = "close"

print(f"\n\nforecasting '{ticker}' prices over the next {horizon} days")
print(
    f"training models from {start_date} to {end_date}, splitting on {cutoff.strftime("%Y-%m-%d")}\n\n"
)

timer_xgb_start = timer()

xgb_forecasts, xgb_rmse = train_and_forecast_xgb(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    cutoff=cutoff,
    horizon_days=horizon,
    label=label,
    n_estimators=200,
    learning_rate=0.05
)
timer_xgb_end = timer() - timer_xgb_start

print(f"\nXGBoost test RMSE on holdout: {xgb_rmse:.4f}\n")
print(f"XGBoost run duration: {timer_xgb_end:.5f} seconds\n\n")
print(xgb_forecasts)
print("\n\n")

timer_smax_start = timer()

smax_forecasts, smax_rmse = sarimax_wrapper(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    label=label,
    cutoff=cutoff,
    horizon_days=horizon,    
)
timer_smax_end = timer() - timer_smax_start

print(f"\nSARIMAX test RMSE on holdout: {smax_rmse:.4f}\n")
print(f"SARIMAX run duration: {timer_smax_end:.5f} seconds\n\n")
print(smax_forecasts)
print("\n\n")


timer_prph_start = timer()

idxs = ["VXX", "QQQ", "SPY", "IWM"]

forecaster = ProphetForecaster()
prph_rmse, eval_df = forecaster.train_model(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    cutoff=cutoff,
    label=label,
    indexes=idxs
)
df_full = build_prophet_df(
    stocks=[ticker], start=start_date, end=end_date, label=label, indexes=idxs
)
prph_forecasts = forecaster.make_prediction(
    df_full=df_full, horizon_days=horizon
)
timer_prph_end = timer() - timer_prph_start

print(f"\nProphet test RMSE on holdout: {prph_rmse:.4f}\n")
print(f"Prophet run duration: {timer_prph_end:.5f} seconds\n\n")
print(prph_forecasts)
print("\n\n")
