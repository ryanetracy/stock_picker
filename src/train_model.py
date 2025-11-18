
import polars as pl
import argparse
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from datetime import datetime

from src.load_data import load_stocks
from src.data_etl import prep_columns
from src.model_preprocess import train_test_split_cutoff

# ### parameters (use argparse module)

# # model hyperparameters
# DEFAULT_NUM_ESTIMATORS = 100
# DEFAULT_LEARNING_RATE = 0.01

# parser = argparse.ArgumentParser(
#     description="model hyperparameters"
# )

# parser.add_argument(
#     "-NUM_ESTIMATORS",
#     type=int,
#     default=DEFAULT_NUM_ESTIMATORS,
#     help="number of estimators"
# )
# parser.add_argument(
#     "-LEARNING_RATE",
#     type=float,
#     default=DEFAULT_LEARNING_RATE,
#     help="how fast the model learns"
# )

# # data parameters
# DEFAULT_LABEL = "move"

# parser.add_argument(
#     "-START_DATE",
#     type=str,
#     default=None,
#     help="data training start date"
# )

# parser.add_argument(
#     "-END_DATE",
#     type=str,
#     default=None,
#     help="data training end date"
# )

# parser.add_argument(
#     "-STOCKS",
#     type=list,
#     default=None,
#     help="stocks to forecast"
# )

# parser.add_argument(
#     "-LABEL",
#     type=str,
#     default=DEFAULT_LABEL,
#     help="one of 'move', 'open', 'close'; which of these values to forecast"
# )

# # cutoff value
# parser.add_argument(
#     "-CUTOFF",
#     type=datetime,
#     default=None,
#     help="cutoff value for train/test split"
# )

# # create args
# args = parser.parse_args()

# NUM_ESTIMATORS = args.NUM_ESTIMATORS
# LEARNING_RATE = args.LEARNING_RATE
# STOCKS = args.STOCKS
# START_DATE = args.START_DATE
# END_DATE = args.END_DATE
# LABEL = args.LABEL
# CUTOFF = args.CUTOFF

# NOW build the model
def train_model(
    stocks: list,
    start_date: str,
    end_date: str,
    cutoff: datetime,
    label: str,
    n_estimators: int,
    learning_rate: float,
):
    df_raw = load_stocks(
        stocks=stocks,
        start=start_date,
        end=end_date,
        use_polars=True
    )

    df_etl = prep_columns(df=df_raw, col=label)

    X_train, X_test, y_train, y_test = train_test_split_cutoff(
        df=df_etl, cutoff=cutoff, label=label
    )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return model, mse
