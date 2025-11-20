
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from datetime import datetime

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
):
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
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    feature_cols = X_train.to_pandas().columns.tolist()

    return model, mse, df_feat, feature_cols
