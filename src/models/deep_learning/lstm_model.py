"""
lstm_model.py — 2 Katmanlı LSTM Modeli (PyTorch)
Hiperparametreler configs/experiments.yaml → models.lstm bölümünden okunur.
"""

import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, n_features: int, config: dict):
        super().__init__()
        cfg         = config["models"]["lstm"]
        hidden_size = cfg["hidden_size"]
        num_layers  = cfg["num_layers"]
        dropout     = cfg["dropout"]

        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # raw logit — BCEWithLogitsLoss sigmoid'i iceriyor


def build_lstm(n_features: int, config: dict) -> LSTMModel:
    return LSTMModel(n_features, config)
