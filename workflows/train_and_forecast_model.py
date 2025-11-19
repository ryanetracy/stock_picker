
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")

from src.pipeline import train_and_forecast

end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - relativedelta(years=5)).strftime("%Y-%m-%d")
cutoff = (datetime.today() - relativedelta(months=2))

ticker = "NICE"

forecasts, mse = train_and_forecast(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    cutoff=cutoff,
    horizon_days=10,
    label="close",
    n_estimators=200,
    learning_rate=0.05
)

print(f"\ntest MSE on holdout: {mse:.4f}\n\n")
print(forecasts)
