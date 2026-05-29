"""
test_sax.py — SAX Dönüşümü Birim Testleri
Bilinen giriş → beklenen SAX sembolü üretiliyor mu?
Çalıştır: py -m pytest tests/test_sax.py -v
"""
import pytest
import numpy as np
from src.models.automata.sax import get_breakpoints, paa_to_sax, series_to_sax


class TestBreakpoints:

    def test_alphabet3_count(self):
        bp = get_breakpoints(3)
        assert len(bp) == 2

    def test_alphabet3_symmetric(self):
        bp = get_breakpoints(3)
        assert abs(bp[0] + bp[1]) < 1e-9   # simetrik sıfır etrafında

    def test_alphabet4_count(self):
        bp = get_breakpoints(4)
        assert len(bp) == 3

    def test_breakpoints_sorted(self):
        for a in [3, 4, 5, 6]:
            bp = get_breakpoints(a)
            assert list(bp) == sorted(bp)


class TestPAAtoSAX:

    def test_all_same_returns_same_letter(self):
        bp = get_breakpoints(3)
        # tüm değerler 0 → 'b' (orta bölge)
        result = paa_to_sax(np.zeros(4), 3, bp)
        assert all(c == result[0] for c in result)

    def test_alphabet_size_respected(self):
        for a in [3, 4, 5, 6]:
            bp = get_breakpoints(a)
            letters = set(paa_to_sax(np.linspace(-2, 2, a), a, bp))
            assert letters.issubset(set(chr(ord('a') + i) for i in range(a)))

    def test_output_length_equals_input(self):
        bp = get_breakpoints(3)
        assert len(paa_to_sax(np.array([0.1, 0.2, 0.3, 0.4]), 3, bp)) == 4

    def test_very_large_value_maps_to_last_letter(self):
        bp = get_breakpoints(3)
        result = paa_to_sax(np.array([100.0]), 3, bp)
        assert result == "c"

    def test_very_small_value_maps_to_first_letter(self):
        bp = get_breakpoints(3)
        result = paa_to_sax(np.array([-100.0]), 3, bp)
        assert result == "a"
