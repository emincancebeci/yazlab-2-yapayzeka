"""
sax.py — Symbolic Aggregate approXimation
PAA değerlerini Gaussian dağılımdan türetilen breakpoint'lere göre harflere çevirir.
alphabet_size configs/experiments.yaml'dan okunur.
"""

import numpy as np
from scipy.stats import norm

from .paa import paa


def get_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    Standart normal dağılımdan eşit-alan kırılma noktaları hesaplar.
    alphabet_size=3 → [-0.431, 0.431] gibi alphabet_size-1 değer döner.
    """
    quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    return norm.ppf(quantiles)


def paa_to_sax(paa_values: np.ndarray, alphabet_size: int,
               breakpoints: np.ndarray = None) -> str:
    """
    PAA değer dizisini SAX string'ine çevirir.

    Args:
        paa_values  : (n_segments,) PAA çıktısı
        alphabet_size: alfabe boyutu (3–6)
        breakpoints : önceden hesaplanmışsa ilet (data leakage önleme)
    Returns:
        Örn: "aab", "bcca"
    """
    if breakpoints is None:
        breakpoints = get_breakpoints(alphabet_size)
    alphabet = [chr(ord('a') + i) for i in range(alphabet_size)]
    letters = [alphabet[int(np.searchsorted(breakpoints, v))] for v in paa_values]
    return ''.join(letters)


def series_to_sax(series: np.ndarray, n_segments: int, alphabet_size: int,
                  breakpoints: np.ndarray = None) -> str:
    """PAA → SAX pipeline'ı tek adımda uygular."""
    paa_vals = paa(series, n_segments)
    return paa_to_sax(paa_vals, alphabet_size, breakpoints)
