"""
cnn1d_model.py — 1D-CNN Modeli (PyTorch)
3 × (Conv1d + BatchNorm + ReLU) → Global Average Pooling → FC
Hiperparametreler configs/experiments.yaml → models.cnn1d bölümünden okunur.
"""

import torch.nn as nn


class CNN1DModel(nn.Module):

    def __init__(self, n_features: int, config: dict):
        super().__init__()
        cfg         = config["models"]["cnn1d"]
        filters     = cfg["num_filters"]       # [32, 64, 128]
        kernel_size = cfg["kernel_size"]        # 3
        dropout     = cfg["dropout"]
        pad         = kernel_size // 2          # 'same' padding

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_features, filters[0], kernel_size, padding=pad),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(),
            nn.Conv1d(filters[0], filters[1], kernel_size, padding=pad),
            nn.BatchNorm1d(filters[1]),
            nn.ReLU(),
            nn.Conv1d(filters[1], filters[2], kernel_size, padding=pad),
            nn.BatchNorm1d(filters[2]),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(filters[2], 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features) — conv icin transpose
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        x = x.mean(dim=2)        # Global Average Pooling
        x = self.dropout(x)
        return self.fc(x)        # raw logit — BCEWithLogitsLoss sigmoid'i iceriyor


def build_cnn1d(n_features: int, config: dict) -> CNN1DModel:
    return CNN1DModel(n_features, config)
