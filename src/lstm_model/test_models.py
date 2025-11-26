
"""
these are models used to run experiments on the ~best~ NN approach to the
forecasting problem. each is a marginal shift built on the previous model, going
from:
1. a baseline LSTM with an MLP cap 
2. a bidirectional LSTM 
3. an attentive LSTM (weighting the timestamps) 
4. a GRU (gated recurrent unit)
"""
import torch 
from torch import nn


class LSTMStockForecaster(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units: int = 64,
        num_layers: int = 1,
        lstm_dropout: float = 0.2,
        mlp_hidden: int = 32,
        mlp_dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_units, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.mlp(last_hidden)


class LSTMBiDirStockForecaster(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        lstm_dropout: float = 0.2
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0.0
        )

        self.linear = nn.Linear(hidden_units * self.num_directions, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.linear(last_hidden)

class AttentiveLSTMStockForecaster(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units: int = 64,
        num_layers: int = 1,
        lstm_dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0
        )

        self.attn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )

        self.linear = nn.Linear(hidden_units, 1)

    def forward(self, x):
        out, (hh_n, c_n) = self.lstm(x)

        scores = self.attn(out)
        weights = torch.softmax(scores, dim=1)

        context = (weights * out).sum(dim=1)

        return self.linear(context)

class GRUStockForecaster(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.linear = nn.Linear(hidden_units, 1)

    def forward(self, x):
        out, h_n = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.linear(last_hidden)
