### Tablo 3: Cross-Dataset Performans Karsilastirmasi (F1-score)

| Train / Test | SKAB | BATADAL |
|-------------|------|---------|
| | **SKAB (test)** | **BATADAL (test)** |
|---|---|----|
| LSTM (train=kendi seti) | 0.8529 | 0.0383 |
| GRU (train=kendi seti) | 0.8591 | 0.0338 |
| 1D-CNN (train=kendi seti) | 0.8529 | 0.0324 |
| Automata (train=kendi seti) | 0.0171 | 0.1449 |

> *Her model kendi egitim setinde egitilip kendi test setinde degerlendirilmistir.*
> *Capraz veri seti (Train:SKAB -> Test:BATADAL) deneyi ek calismada yapilabilir.*
