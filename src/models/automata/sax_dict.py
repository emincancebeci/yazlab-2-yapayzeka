"""
sax_dict.py — SAX Sözlüğü (Unseen Pattern Tespiti)
Eğitim verisinden elde edilen tüm SAX pattern'larını saklar.
Test sırasında sözlükte bulunmayan pattern → Unseen olarak işaretlenir.
"""


class SAXDictionary:
    """
    Train aşamasında .add() ile doldurulur.
    Test aşamasında .is_seen() ile unseen kontrolü yapılır.
    """

    def __init__(self):
        self._patterns: set = set()

    def add(self, pattern: str) -> None:
        self._patterns.add(pattern)

    def is_seen(self, pattern: str) -> bool:
        return pattern in self._patterns

    def all_patterns(self) -> list:
        return sorted(self._patterns)

    def __len__(self) -> int:
        return len(self._patterns)

    def __repr__(self) -> str:
        return f"SAXDictionary({len(self._patterns)} patterns)"
