### Tablo 1: Model Performansi ve Stabilitesi (Ortalama F1-score +/- Standart Sapma)

| Model | SKAB F1 | BATADAL F1 |
|-------|---------|------------|
| LSTM | 0.8529 ± 0.0048 | 0.0383 ± 0.0239 |
| GRU | 0.8591 ± 0.0038 | 0.0338 ± 0.0093 |
| 1D-CNN | 0.8529 ± 0.0033 | 0.0324 ± 0.0162 |
| Automata | 0.0171 ± 0.0000 | 0.1449 ± 0.0000 |

> *5 farkli random seed [42, 123, 2026, 7, 999] ile elde edilen ortalama ve standart sapma.*
> *SKAB icin GroupKFold (k=5) fold ortalamasi alinmistir.*
