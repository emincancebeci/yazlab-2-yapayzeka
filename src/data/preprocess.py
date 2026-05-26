"""
preprocess.py — Veri Ön İşleme Pipeline'ı

Kurallar (Data Leakage Önleme):
- MinMaxScaler yalnızca train verisi üzerinde fit edilir
- PCA yalnızca train verisi üzerinde fit edilir
- Val ve test setine yalnızca .transform() uygulanır
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class Preprocessor:
    """
    Tek bir Preprocessor nesnesi train'de fit edilir,
    val/test'e transform uygulanır. Data leakage olmaz.
    """

    def __init__(self, config: dict):
        self.config = config
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=config["preprocessing"]["pca_components"])
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Yalnızca TRAIN verisi üzerinde çağrılır."""
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Val ve Test verisi için — fit yapılmaz."""
        if not self._fitted:
            raise RuntimeError("Preprocessor henüz fit edilmedi. Önce fit_transform() çağırın.")
        return self.scaler.transform(X)

    def fit_transform_pca(self, X: np.ndarray) -> np.ndarray:
        """Automata için PCA — yalnızca TRAIN üzerinde fit edilir."""
        X_scaled = self.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca  # shape: (n_samples, 1) — PC1

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """Val/Test için PCA transform — fit yapılmaz."""
        X_scaled = self.transform(X)
        return self.pca.transform(X_scaled)

    def add_gaussian_noise(self, X: np.ndarray, std: float) -> np.ndarray:
        """Robustness deneyi için Gaussian gürültü ekler."""
        noise = np.random.normal(0, std, X.shape)
        return X + noise


def handle_missing(df: pd.DataFrame, strategy: str = "ffill") -> pd.DataFrame:
    """Eksik veri işleme."""
    if strategy == "ffill":
        return df.ffill().bfill()
    elif strategy == "bfill":
        return df.bfill().ffill()
    elif strategy == "drop":
        return df.dropna()
    else:
        raise ValueError(f"Bilinmeyen strateji: {strategy}")


def sliding_window(data: np.ndarray, window_size: int, step: int = 1):
    """
    Zaman serisi üzerinde kayan pencere oluşturur.

    Args:
        data: (n_samples,) veya (n_samples, n_features)
        window_size: pencere boyutu (config'den okunur)
        step: adım boyutu (varsayılan 1)

    Returns:
        windows: (n_windows, window_size) veya (n_windows, window_size, n_features)
        indices: her pencerenin son indeksi (etiket hizalaması için)
    """
    windows = []
    indices = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i: i + window_size])
        indices.append(i + window_size - 1)
    return np.array(windows), np.array(indices)
