
"""
create the forecasts from the trained LSTM model.
"""

import numpy as np 
import pandas as pd
import torch 
from datetime import timedelta 
from typing import List 
from sklearn.preprocessing import StandardScaler

class LSTMAutoregressiveForecaster:
    """generate future dates and use the model to forecast prices."""
    def __init__(
        self,
        model: torch.nn.Module,
        scaler: StandardScaler,
        feat_cols: list,
        lookback: int,
        label: str,
        use_delta: bool = False
    ):
        self.model = model 
        self.model.eval()
        self.scaler = scaler 
        self.feat_cols = feat_cols 
        self.lookback = lookback 
        self.label = label 
        self.use_delta = use_delta 

    def _build_initial_window(self, df_feat: pd.DataFrame) -> torch.Tensor:
        """_summary_

        Args:
            df_feat (pd.DataFrame): _description_

        Returns:
            torch.Tensor: _description_
        """
        df_feat = df_feat.sort_values("date")
        tail = df_feat.iloc[-self.lookback :].copy()

        feats_scaled = self.scaler.transform(tail[self.feat_cols])
        window = torch.tensor(feats_scaled, dtype=torch.float32).unsqueeze(0)

        return window

    def _generate_future_dates(
        self, last_date: pd.Timestamp, horizon_days: int
    ) -> List[pd.Timestamp]:
        """_summary_

        Args:
            last_date (pd.Timestamp): _description_
            horizon_days (int): _description_

        Returns:
            List[pd.Timestamp]: _description_
        """
        future_dates = []
        current = last_date 

        while len(future_dates) < horizon_days:
            current = current + timedelta(days=1)
            if current.weekday() < 5:
                future_dates.append(current)

        return future_dates

    def forecast_horizon(
        self,
        df_feat: pd.DataFrame,
        horizon_days: int = 10
    ) -> pd.DataFrame:
        """_summary_

        Args:
            df_feat (pd.DataFrame): _description_
            horizon_days (int, optional): _description_. Defaults to 10.

        Returns:
            pd.DataFrame: _description_
        """
        df_feat =  df_feat.sort_values("date").reset_index(drop=True).copy()

        last_date = pd.to_datetime(df_feat["date"].iloc[-1])
        last_close = float(df_feat["close"].iloc[-1])

        future_dates = self._generate_future_dates(last_date, horizon_days)
        window = self._build_initial_window(df_feat)

        preds_close = []
        preds_raw = []

        last_feat_row = df_feat[self.feat_cols].iloc[-1].values.astype("float32")

        for _ in range(horizon_days):
            with torch.inference_mode():
                pred_raw = self.model(window).squeeze(-1).item()

            preds_raw.append(pred_raw)

            if self.use_delta:
                last_close = last_close + pred_raw 
            else:
                last_close = pred_raw 

            preds_close.append(last_close)

            next_feat = last_feat_row.copy() 
            if "close" in self.feat_cols:
                close_idx = self.feat_cols.index("close")
                next_feat[close_idx] = last_close 

            next_feat_scaled = self.scaler.transform(next_feat.reshape(1, -1))
            next_feat_tensor = torch.tensor(
                next_feat_scaled, dtype=torch.float32
            ).unsqueeze(0)

            window = torch.cat(
                [window[:, 1:, :], next_feat_tensor], dim=1
            )

            last_feat_row = next_feat 

        return pd.DataFrame(
            {
                "date": future_dates,
                "pred_raw": preds_raw,
                "pred_close": preds_close
            }
        )
