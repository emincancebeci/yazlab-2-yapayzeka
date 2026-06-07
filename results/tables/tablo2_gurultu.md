### Tablo 2: Gurultu Etkisi ve Unseen Senaryo Analizi

| Model | Orijinal F1 | Gurultulu F1 | F1 Dususu | Unseen Det. Rate | Unseen Map. Acc. |
|-------|-------------|--------------|-----------|-----------------|-----------------|
| LSTM | 0.0338 | 0.0326 | +0.0012 | N/A | N/A |
| GRU | 0.0799 | 0.0831 | -0.0032 | N/A | N/A |
| 1D-CNN | 0.0000 | 0.0394 | -0.0394 | N/A | N/A |
| Automata | 0.1449 | 0.1606 | -0.0157 | 1.0000 | 0.2000 |

> *Gurultu: Gaussian noise (std=0.1) BATADAL test setine eklenmistir.*
> *Unseen: 5/833 pattern (0.60%) egitim SAX sozlugunde bulunmamistir.*
