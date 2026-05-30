"""
deep_learning/__init__.py — Shared Training & Inference Loop
LSTM, GRU, 1D-CNN için ortak egitim dongusu.
"""

import time
import numpy as np
import torch
import torch.nn as nn


def _compute_pos_weight(train_loader, device):
    """
    Train loader uzerinden pos_weight hesaplar.
    pos_weight = n_negatives / n_positives
    Sinif dengesizligini (class imbalance) telafi eder.
    BATADAL gibi az anomalili veri setlerinde kritik.
    """
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.numpy().tolist())
    all_labels = np.array(all_labels)
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    if n_pos == 0:
        return torch.tensor([1.0], device=device)
    weight = n_neg / n_pos
    return torch.tensor([weight], device=device)


def train_model(model, train_loader, val_loader, config, device=None):
    """
    Tum DL modeller icin ortak egitim dongusu.
    Early stopping: val_loss patience=5 (config'den).
    pos_weight: train verisinden otomatik hesaplanir (class imbalance fix).

    Returns:
        (model, history_dict, training_time_seconds)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_weight = _compute_pos_weight(train_loader, device)
    # BCEWithLogitsLoss = sigmoid + BCELoss, pos_weight ile imbalance duzeltilir
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    patience   = config["training"]["early_stopping"]["patience"]
    max_epochs = config["training"]["epochs"]

    model.to(device)
    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None
    history          = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for _ in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(1)  # raw logit, sigmoid yok
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits    = model(X_batch).squeeze(1)
                val_loss += criterion(logits, y_batch).item()
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, time.time() - t0


def predict_model(model, loader, device=None) -> tuple:
    """
    Test loader uzerinde binary tahmin uretir.
    Model raw logit dondurdugu icin sigmoid uygulanir, sonra 0.5 esigi.

    Returns:
        (predictions: list[int], true_labels: list[int])
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    preds, labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device)).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds.extend((probs >= 0.5).astype(int).tolist())
            labels.extend(y_batch.numpy().astype(int).tolist())

    return preds, labels
