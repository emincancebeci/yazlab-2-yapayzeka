"""
levenshtein.py — Edit Distance Tabanlı Unseen Pattern Eşleştirme
Unseen bir pattern için SAX sözlüğündeki en yakın pattern'ı bulur.
Unit testler: tests/test_levenshtein.py
"""


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    İki string arasındaki minimum edit (Levenshtein) mesafesini O(n) bellek ile hesaplar.
    """
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def find_nearest(pattern: str, known_patterns: list) -> tuple:
    """
    SAX sözlüğünden en yakın pattern'ı döner.

    Returns:
        (nearest_pattern: str, distance: int)
    Raises:
        ValueError: sözlük boşsa
    """
    if not known_patterns:
        raise ValueError("SAX sözlüğü boş — fit() çağrılmadan predict() kullanılamaz.")
    best, best_dist = None, float('inf')
    for p in known_patterns:
        d = levenshtein_distance(pattern, p)
        if d < best_dist:
            best_dist = d
            best = p
    return best, best_dist
