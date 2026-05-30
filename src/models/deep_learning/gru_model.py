"""
gru_model.py — 2 Katmanlı GRU Modeli (PyTorch)
Hiperparametreler configs/experiments.yaml → models.gru bölümünden okunur.
"""

import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(self, n_features: int, config: dict):
        super().__init__()
        cfg         = config["models"]["gru"]
        hidden_size = cfg["hidden_size"]
        num_layers  = cfg["num_layers"]
        dropout     = cfg["dropout"]

        self.gru = nn.GRU(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])   # raw logit — BCEWithLogitsLoss sigmoid'i iceriyor


def build_gru(n_features: int, config: dict) -> GRUModel:
    return GRUModel(n_features, config)
