
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
    # n = len(df)
    # train_n = int(n * train_proportion)

    train_n = get_train_size(len(df), train_proportion)

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
        lookback: int = 30,
        use_delta: bool = False
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
        use_delta (bool, optional): convert label to a delta, centering it and
            allowing the model to predict more signal relative to raw label. 
            defaults to false

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: X_train,
            X_test, y_train, y_test returned as torch tensors.
    """
    if use_delta:
        df[f"delta_{label}"] = df[label].diff()
        df  = df.dropna()
        label = f"delta_{label}"

    X_list, y_list = [], []
    values = df[feat_cols + [label]].values.astype("float32")

    for i in range(lookback, len(values)):
        X_list.append(values[i-lookback:i, :-1])
        y_list.append(values[i, -1])

    X = torch.tensor(np.stack(X_list))
    y = torch.tensor(np.array(y_list))

    train_size = get_train_size(len(X), train_proportion)

    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test

def train_test_split_single_feature(
    df: pd.DataFrame, label: str, train_proportion: float = 0.7
) -> Tuple[np.array, np.array]:
    """split training and testing data from pandas dataframe.

    Args:
        df (pd.DataFrame): pandas dataframe.
        label (str): label column name.
        train_proportion (float, optional): proportion of the sample to save for
            training. defaults to 0.7.

    Returns:
        Tuple[np.array, np.array]: numpy arrays of training and testing data.
    """
    timeseries = df[[label]].values.astype("float32")
    train_size = get_train_size(len(df), train_proportion)
    train, test = timeseries[:train_size], timeseries[train_size:]

    return train, test

def make_lstm_sequences_single_feature(
    data: np.array, lookback: int = 30
) -> Tuple[list, list]:
    """_summary_

    Args:
        data (np.array): numpy array of timeseries data.
        lookback (int, optional): size of lookback window for sequencing. 
            defaults to 30.

    Returns:
        Tuple[list, list]: list of arrays of timeseries sequences.
    """
    X, y = [], []

    for i in range(len(data) - lookback):
        feature = data[i: i + lookback]
        target = data[i + 1: i + lookback + 1]
        X.append(feature)
        y.append(target)

    return X, y 

def process_single_feature_data(
    df: pd.DataFrame,
    label: str,
    train_proportion: float = 0.7,
    lookback: str = 30,
    use_delta: bool = False
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """_summary_

    Args:
        df (pd.DataFrame): pandas dataframe of timeseries data.
        label (str): label column name.
        train_proportion (float, optional): proportion of the sample to save for
            training. defaults to 0.7.
        lookback (str, optional): size of lookback window for sequencing. 
            defaults to 30.
        use_delta (bool, optional): convert label to a delta, centering it and
            allowing the model to predict more signal relative to raw label. 
            defaults to false

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: X_train,
            X_test, y_train, y_test returned as torch tensors.
    """
    if use_delta:
        df[f"delta_{label}"] = df[label].diff()
        df = df.dropna()
        label = f"delta_{label}"

    train, test = train_test_split_single_feature(
        df=df, label=label, train_proportion=train_proportion
    )

    X_train, y_train = make_lstm_sequences_single_feature(
        data=train, lookback=lookback
    )
    X_test, y_test = make_lstm_sequences_single_feature(
        data=test, lookback=lookback
    )

    return (
        torch.tensor(X_train),
        torch.tensor(X_test),
        torch.tensor(y_train),
        torch.tensor(y_test)
    )
