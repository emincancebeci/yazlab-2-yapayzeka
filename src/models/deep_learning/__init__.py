"""
deep_learning/__init__.py — Shared Training & Inference Loop
LSTM, GRU, 1D-CNN için ortak eğitim döngüsü.
"""

import time
import torch
import torch.nn as nn


def train_model(model, train_loader, val_loader, config, device=None):
    """
    Tüm DL modeller için ortak eğitim döngüsü.
    Early stopping: val_loss patience=5 (config'den).

    Returns:
        (model, history_dict, training_time_seconds)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    criterion = nn.BCELoss()
    patience  = config["training"]["early_stopping"]["patience"]
    max_epochs = config["training"]["epochs"]

    model.to(device)
    best_val_loss   = float("inf")
    patience_counter = 0
    best_state      = None
    history         = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for _ in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(1), y_batch)
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
                val_loss += criterion(model(X_batch).squeeze(1), y_batch).item()
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
    Test loader üzerinde binary tahmin üretir.

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
            out = model(X_batch.to(device)).squeeze(1).cpu().numpy()
            preds.extend((out >= 0.5).astype(int).tolist())
            labels.extend(y_batch.numpy().astype(int).tolist())

    return preds, labels
