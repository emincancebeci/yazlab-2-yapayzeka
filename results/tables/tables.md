## Deney Sonuclari ve Karsilastirmali Analiz Tablolari

### Tablo 1: Model Performansi ve Stabilitesi (Ortalama F1-score +/- Standart Sapma)

| Model | SKAB F1 | BATADAL F1 |
|-------|---------|------------|
| LSTM | 0.8529 ± 0.0048 | 0.0383 ± 0.0239 |
| GRU | 0.8591 ± 0.0038 | 0.0338 ± 0.0093 |
| 1D-CNN | 0.8529 ± 0.0033 | 0.0324 ± 0.0162 |
| Automata | 0.0171 ± 0.0000 | 0.1449 ± 0.0000 |

> *5 farkli random seed [42, 123, 2026, 7, 999] ile elde edilen ortalama ve standart sapma.*
> *SKAB icin GroupKFold (k=5) fold ortalamasi alinmistir.*

### Tablo 2: Gurultu Etkisi ve Unseen Senaryo Analizi

| Model | Orijinal F1 | Gurultulu F1 | F1 Dususu | Unseen Det. Rate | Unseen Map. Acc. |
|-------|-------------|--------------|-----------|-----------------|-----------------|
| LSTM | 0.0338 | 0.0326 | +0.0012 | N/A | N/A |
| GRU | 0.0799 | 0.0831 | -0.0032 | N/A | N/A |
| 1D-CNN | 0.0000 | 0.0394 | -0.0394 | N/A | N/A |
| Automata | 0.1449 | 0.1606 | -0.0157 | 1.0000 | 0.2000 |

> *Gurultu: Gaussian noise (std=0.1) BATADAL test setine eklenmistir.*
> *Unseen: 5/833 pattern (0.60%) egitim SAX sozlugunde bulunmamistir.*

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

### Tablo 4: Automata Parametre Duyarlilik Analizi

#### 4a: F1-score
| Parametre | Deger=3 | Deger=4 | Deger=5 | Deger=6 |
|-----------|---------|---------|---------|---------|
| Window Size=3 | 0.0619 | 0.1037 | 0.1606 | 0.1563 |
| Window Size=4 | 0.1449 | 0.1887 | 0.1828 | 0.1742 |
| Window Size=5 | 0.1618 | 0.1694 | 0.1756 | 0.1756 |
| Window Size=6 | 0.1581 | 0.1758 | 0.1758 | 0.1756 |

#### 4b: State Sayisi
| Parametre | Deger=3 | Deger=4 | Deger=5 | Deger=6 |
|-----------|---------|---------|---------|---------|
| Window Size=3 | 25 | 56 | 98 | 154 |
| Window Size=4 | 63 | 171 | 292 | 475 |
| Window Size=5 | 137 | 392 | 655 | 1005 |
| Window Size=6 | 260 | 737 | 1100 | 1545 |

> *Parametre degisiminin model performansi ve state sayisi uzerindeki etkisi.*
> *Sabit karsilastirma parametreleri: window_size=4, alphabet_size=3.*

### Tablo 5: Modellerin Calisma Suresi (Runtime) Karsilastirmasi

| Model | SKAB Egitim (sn) | SKAB Inference (sn) | BATADAL Egitim (sn) | BATADAL Inference (sn) |
|-------|-----------------|--------------------|--------------------|----------------------|
| LSTM | 18.36 | 0.0922 | 1.51 | 0.0189 |
| GRU | 15.18 | 0.0920 | 1.57 | 0.0170 |
| 1D-CNN | 13.80 | 0.1197 | 1.49 | 0.0234 |
| Automata | 0.35 | 0.4810 | 0.05 | 0.0331 |

> *Sureler 5 seed ortalamasi olarak raporlanmistir. GPU: NVIDIA RTX 5070 Laptop.*
