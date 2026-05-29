"""
automata.py — Probabilistic Automata
- P(Si → Sj) = (count(i→j) + alpha) / (total_out_i + alpha * |V|)   [Laplace smoothing]
- path_probability = Π P(Si → Si+1)
- Karar: path_prob < anomaly_threshold → ANOMALY
"""

import numpy as np
from collections import defaultdict

from .paa import paa
from .sax import get_breakpoints, paa_to_sax
from .sax_dict import SAXDictionary
from .levenshtein import find_nearest


class ProbabilisticAutomata:

    def __init__(self, config: dict, window_size: int = None, alphabet_size: int = None):
        """
        window_size / alphabet_size: sweep deneyleri için override edilebilir,
        aksi halde config["fixed"] değerleri kullanılır.
        """
        self.window_size   = window_size   or config["fixed"]["window_size"]
        self.alphabet_size = alphabet_size or config["fixed"]["alphabet_size"]
        self.threshold     = config["automata"]["anomaly_threshold"]
        self.alpha         = config["automata"]["smoothing_alpha"]   # Laplace α

        self.breakpoints = get_breakpoints(self.alphabet_size)
        self.sax_dict    = SAXDictionary()

        # {src_pattern: {dst_pattern: count}}
        self._counts: dict = defaultdict(lambda: defaultdict(int))
        # {src_pattern: {dst_pattern: probability}}
        self._probs: dict  = {}
        self._fitted       = False

    # ------------------------------------------------------------------
    # Yardımcı
    # ------------------------------------------------------------------

    def _window_to_pattern(self, window: np.ndarray) -> str:
        paa_vals = paa(window, self.window_size)
        return paa_to_sax(paa_vals, self.alphabet_size, self.breakpoints)

    def _resolve(self, pattern: str):
        """Unseen ise Levenshtein ile en yakına eşle."""
        if self.sax_dict.is_seen(pattern):
            return pattern, False, None, 0
        nearest, dist = find_nearest(pattern, self.sax_dict.all_patterns())
        return nearest, True, nearest, dist

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------

    def fit(self, series: np.ndarray) -> None:
        """
        series: 1D PCA (PC1) eğitim serisi.
        Geçiş sayılarını hesaplar, SAX sözlüğünü doldurur, olasılıkları normalize eder.
        """
        n = len(series)
        patterns = []
        for i in range(n - self.window_size + 1):
            pat = self._window_to_pattern(series[i: i + self.window_size])
            patterns.append(pat)
            self.sax_dict.add(pat)

        for t in range(len(patterns) - 1):
            self._counts[patterns[t]][patterns[t + 1]] += 1

        self._normalize()
        self._fitted = True

    def _normalize(self) -> None:
        """Laplace smoothing uygulayarak olasılıkları hesaplar."""
        vocab = self.sax_dict.all_patterns()
        v = len(vocab)
        self._probs = {}
        for src in vocab:
            dst_counts = self._counts.get(src, {})
            total = sum(dst_counts.values()) + self.alpha * v
            self._probs[src] = {}
            for dst in vocab:
                cnt = dst_counts.get(dst, 0)
                self._probs[src][dst] = (cnt + self.alpha) / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------

    def predict_sequence(self, series: np.ndarray) -> list:
        """
        Test serisi üzerinde pencere pencere karar ve açıklama üretir.

        Returns:
            list[dict] — her pencere için explainability çıktısı
        """
        if not self._fitted:
            raise RuntimeError("Automata fit edilmedi. Önce fit() çağırın.")

        n        = len(series)
        results  = []
        prev     = None
        path_log = []   # geçiş olasılıklarının log'u (path probability için)

        for t, i in enumerate(range(n - self.window_size + 1)):
            raw = self._window_to_pattern(series[i: i + self.window_size])
            resolved, is_unseen, mapped_to, lev_dist = self._resolve(raw)

            trans_prob = None
            if prev is not None and prev in self._probs:
                trans_prob = self._probs[prev].get(resolved, 0.0)
                path_log.append(trans_prob)

            # Path probability: geçişe kadar biriken çarpım
            path_prob = float(np.prod(path_log)) if path_log else 1.0

            if trans_prob is None:
                # İlk pencere: yalnızca seen/unseen bazlı karar
                decision = "anomaly" if is_unseen else "normal"
            else:
                decision = "anomaly" if trans_prob < self.threshold else "normal"

            results.append({
                "time_step"      : t,
                "state"          : prev,
                "pattern"        : raw,
                "status"         : "unseen" if is_unseen else "seen",
                "mapped_to"      : mapped_to,
                "levenshtein_dist": lev_dist,
                "transition_prob": trans_prob,
                "path_probability": path_prob,
                "decision"       : decision,
            })
            prev = resolved

        return results

    def predict(self, series: np.ndarray) -> np.ndarray:
        """
        Binary tahmin: 0 = normal, 1 = anomaly.
        Uzunluk: len(series) - window_size + 1
        """
        return np.array(
            [1 if r["decision"] == "anomaly" else 0
             for r in self.predict_sequence(series)],
            dtype=int,
        )

    # ------------------------------------------------------------------
    # Erişimciler (görselleştirme / explainability için)
    # ------------------------------------------------------------------

    @property
    def transition_matrix(self) -> dict:
        return self._probs

    @property
    def states(self) -> list:
        return self.sax_dict.all_patterns()

    @property
    def n_states(self) -> int:
        return len(self.sax_dict)
