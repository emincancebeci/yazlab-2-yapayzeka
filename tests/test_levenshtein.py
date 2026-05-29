"""
test_levenshtein.py — Levenshtein (Unseen Pattern) Birim Testleri
ZORUNLU: Bu testlerin tamamı geçmelidir.
Çalıştır: py -m pytest tests/test_levenshtein.py -v
"""
import pytest
from src.models.automata.levenshtein import levenshtein_distance, find_nearest


class TestLevenshteinDistance:

    def test_identical_strings(self):
        assert levenshtein_distance("abc", "abc") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_substitution(self):
        assert levenshtein_distance("abc", "adc") == 1

    def test_single_insertion(self):
        assert levenshtein_distance("ab", "abc") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("abc", "ab") == 1

    def test_full_replacement(self):
        assert levenshtein_distance("aaa", "bbb") == 3

    def test_symmetry(self):
        assert levenshtein_distance("aab", "bca") == levenshtein_distance("bca", "aab")

    def test_sax_window4(self):
        assert levenshtein_distance("aabb", "aabb") == 0
        assert levenshtein_distance("aabb", "abbb") == 1


class TestFindNearest:

    def test_exact_match_returns_zero_dist(self):
        nearest, dist = find_nearest("abc", ["abc", "bca", "aab"])
        assert nearest == "abc"
        assert dist == 0

    def test_nearest_is_one_edit_away(self):
        nearest, dist = find_nearest("adc", ["abc", "bca", "aab"])
        assert nearest == "abc"
        assert dist == 1

    def test_empty_vocabulary_raises(self):
        with pytest.raises(ValueError):
            find_nearest("abc", [])

    def test_consistent_tie_breaking(self):
        n1, d1 = find_nearest("ab", ["aa", "bb"])
        n2, d2 = find_nearest("ab", ["aa", "bb"])
        assert n1 == n2 and d1 == d2 == 1
