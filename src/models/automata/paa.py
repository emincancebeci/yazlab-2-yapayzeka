"""
paa.py — Piecewise Aggregate Approximation
Zaman serisi penceresini n_segments eşit parçaya böler, her parçanın ortalamasını döner.
window_size = n_segments → configs/experiments.yaml'dan okunur.
"""

import numpy as np


def paa(series: np.ndarray, n_segments: int) -> np.ndarray:
    """
    Args:
        series   : 1D array, uzunluk L
        n_segments: PAA segment sayısı (= window_size)
    Returns:
        (n_segments,) — her segmentin ortalaması
    """
    n = len(series)
    result = np.empty(n_segments, dtype=float)
    for i in range(n_segments):
        start = int(i * n / n_segments)
        end   = int((i + 1) * n / n_segments)
        if start == end:          # segment uzunluğu < 1 olmasın
            end = start + 1
        result[i] = np.mean(series[start:end])
    return result
