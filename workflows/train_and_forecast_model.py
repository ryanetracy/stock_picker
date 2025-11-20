
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")

from src.pipeline import train_and_forecast_xgb
from src.fit_sarimax_model import *

end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - relativedelta(years=5)).strftime("%Y-%m-%d")
cutoff = (datetime.today() - relativedelta(months=2))

ticker = "AAPL"

timer_xgb_start = timer()

xgb_forecasts, xgb_mse = train_and_forecast_xgb(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    cutoff=cutoff,
    horizon_days=10,
    label="close",
    n_estimators=200,
    learning_rate=0.05
)
timer_xgb_end = timer() - timer_xgb_start

print(f"\nXGBoost test MSE on holdout: {xgb_mse:.4f}\n")
print(f"XGBoost run duration: {timer_xgb_end:.5f} seconds\n\n")
print(xgb_forecasts)
print("\n\n")

timer_smax_start = timer()

smax_forecasts, smax_mse = sarimax_wrapper(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    label="close",
    cutoff=cutoff,
    days=10,    
)
timer_smax_end = timer() - timer_smax_start

print(f"\nSARIMAX test MSE on holdout: {smax_mse:.4f}\n")
print(f"SARIMAX run duration: {timer_smax_end:.5f} seconds\n\n")
print(smax_forecasts)
print("\n\n")
