
"""
set of functions used for preprocessing data for torch LSTM modeling. includes
code for scaling and setting up sequences within a train/test split.
"""

import torch
import pandas as pd 
import numpy as np 
from typing import Tuple
from sklearn.preprocessing import StandardScaler


def get_train_size(n_samples: int, train_proportion: float = 0.7) -> int:
    """determine the size of the training sample.

    Args:
        n_samples (int): size of the full dataset.
        train_proportion (float, optional): proportion of the sample to save for
            training. defaults to 0.7.

    Returns:
        int: integer denoting cutoff value for train/test split.
    """
    return int(n_samples * train_proportion)

def scale_features(
    df: pd.DataFrame, feat_cols: list, train_proportion: float = 0.7
) -> Tuple[pd.DataFrame, StandardScaler]:
    """perform scaling (fit on train).

    Args:
        df (pd.DataFrame): pandas dataframe of unscaled features.
        feat_cols (list): list of columns to scale.
        train_proportion (float, optional): proportion of the sample to save for
            training. defaults to 0.7.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: dataframe with scaled features and
            scaler fit on training sample.
    """
    n = len(df)
    train_n = int(n * train_proportion)

    scaler = StandardScaler()
    scaler.fit(df[feat_cols].iloc[:train_n])

    scaled_feats = scaler.transform(df[feat_cols])
    df_scaled = df.copy()
    df_scaled[feat_cols] = scaled_feats

    return df_scaled, scaler

def make_lstm_sequences(
        df: pd.DataFrame,
        feat_cols: list,
        label: str,
        train_proportion: float = 0.7,
        lookback: int = 30
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """make a sequence of `lookback` days for LSTM modeling.

    Args:
        df (pd.DataFrame): (scaled) pandas dataframe.
        feat_cols (list): list of features.
        label (str): label column name.
        train_proportion (float, optional): proportion of the sample to save for
            training. defaults to 0.7.
        lookback (int, optional): size of lookback window for sequencing. 
            defaults to 30.

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: X_train,
            X_test, y_train, y_test returned as torch tensors.
    """
    X_list, y_list = [], []
    values = df[feat_cols + [label]].values.astype("float32")

    for i in range(lookback, len(values)):
        X_list.append(values[i-lookback:i, :-1])
        y_list.append(values[i, -1])

    X = torch.tensor(np.stack(X_list))
    y = torch.tensor(np.array(y_list))

    train_size = get_train_size(len(X), train_proportion) ## stick with this for now, incorporate cutoff later (i like that approach..)

    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test
