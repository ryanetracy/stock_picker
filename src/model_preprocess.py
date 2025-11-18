
from datetime import datetime
import polars as pl
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
