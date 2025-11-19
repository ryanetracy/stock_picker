
import polars as pl

def prep_columns(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col == "move":
        df = df.with_columns(
            (pl.col("close") - pl.col("open")).alias(col)
        )

    df_out =  (
        df.select(
            [
                "date",
                "ticker",
                col
            ]
        )
        .sort([pl.col("ticker"), pl.col("date")], descending=False)
        .with_columns(
            pl.col(col).shift(1).over("ticker").alias(f"prev1_{col}"),
            pl.col(col).shift(7).over("ticker").alias(f"prev7_{col}"),
            pl.col(col).shift(30).over("ticker").alias(f"prev30_{col}"),
        )
        .with_columns(
            pl.col(col)
            .rolling_mean(window_size=7, min_samples=2)
            .shift(1)
            .over("ticker")
            .alias(f"{col}_rolling_mean_7")
        )
        .with_columns(
            pl.col(col)
            .rolling_std(window_size=7, min_samples=2)
            .shift(1)
            .over("ticker")
            .alias(f"{col}_rolling_std_7")
        )
    )

    return df_out


def prep_data_frame(df):
    markers = ["open", "close", "move"]
    df_out = None

    for marker in markers:
        df_prep = prep_columns(df, marker)
        if df_out is None:
            df_out = df_prep
        else:
            df_out = df_prep.join(df_out, on=["date", "ticker"], how="inner")

    return (
        df_out.with_columns(
            pl.col("date").dt.weekday().alias("dow")
        )
        .with_columns(
            pl.col("date").dt.month().alias("month")
        )
        .with_columns(
            pl.when(pl.col("dow").is_in([0, 4]))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("mon_or_fri")
        )
    )

def build_dataset(df: pl.DataFrame, label: str = "close") -> pl.DataFrame:
    df_feat = prep_data_frame(df)

    if label == "close":
        df_feat = df_feat.with_columns(pl.col("close").shift(-1).alias("label"))
    elif label == "move":
        df_feat = df_feat.with_columns(pl.col("move").shift(-1).alias("label"))
    else:
        raise ValueError("label must be one of ['close', 'move']")

    return df_feat.drop_nulls()
