
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
