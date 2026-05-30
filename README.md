# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

**Grup 15** — SKAB + BATADAL Veri Setleri  
**Ders:** Yazılım Geliştirme — 2. Proje  

---

## İçindekiler
1. [Proje Hakkında](#1-proje-hakkında)
2. [Kurulum](#2-kurulum)
3. [Kullanım](#3-kullanım)
4. [Proje Yapısı](#4-proje-yapısı)
5. [Deneysel Sonuçlar](#5-deneysel-sonuçlar)
6. [Açıklanabilirlik Modülü](#6-açıklanabilirlik-modülü)
7. [İstatistiksel Analiz](#7-istatistiksel-analiz)

---

## 1. Proje Hakkında

Bu proje, zaman serisi anomali tespitinde iki farklı modelleme paradigmasını karşılaştırmaktadır:

- **Black-box modeller:** LSTM, GRU, 1D-CNN (Derin Öğrenme - PyTorch)
- **Explainable model:** Probabilistic Automata (PAA → SAX → Geçiş Matrisi)

### Araştırma Sorusu
Farklı modelleme yaklaşımları, farklı veri koşulları altında nasıl davranmaktadır ve bu davranışlar istatistiksel olarak anlamlı mıdır?

### Veri Setleri
| Veri Seti | Kaynak | Özellik Sayısı | Anomali Oranı | Bölme Stratejisi |
|-----------|--------|---------------|---------------|-----------------|
| **SKAB** | valve1 + valve2 klasörleri birleştirildi | 8 sensör | ~%35 | GroupKFold (k=5, grup: dosya adı) |
| **BATADAL** | Training Dataset 2 (dataset04) | 43 özellik | ~%5 | Temporal %60/%20/%20 |

---

## 2. Kurulum

```bash
# Sanal ortam
python -m venv venv
.\venv\Scripts\activate  # Windows

# PyTorch (CUDA 12.8 - RTX 5070+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Diğer bağımlılıklar
pip install -r requirements.txt
```

### Veri Setleri
```
data/raw/skab/valve1/   ← SKAB valve1 klasöründeki tüm CSV'ler
data/raw/skab/valve2/   ← SKAB valve2 klasöründeki tüm CSV'ler
data/raw/batadal/BATADAL_dataset04.csv
```
- **SKAB:** https://github.com/waico/SKAB
- **BATADAL:** https://www.batadal.net/

---

## 3. Kullanım

```bash
# Tüm deneyleri çalıştır (SKAB + BATADAL + Noise + Unseen + Sweep)
python src/pipeline.py --config configs/experiments.yaml

# Tabloları üret
python src/evaluate/table_generator.py

# Görselleri üret
python src/evaluate/visualizer.py

# Unit testleri çalıştır (29 test)
pytest tests/ -v
```

---

## 4. Proje Yapısı

```
yazlab-2-2/
├── configs/experiments.yaml      # Tüm parametreler (hard-coded değer YOK)
├── data/
│   ├── raw/skab/                 # SKAB valve1 + valve2
│   └── raw/batadal/              # BATADAL Training Dataset 2
├── src/
│   ├── data/
│   │   ├── preprocess.py         # MinMaxScaler, PCA (train-only fit), sliding window
│   │   └── loader.py             # SKAB GroupKFold | BATADAL temporal split
│   ├── models/
│   │   ├── deep_learning/        # LSTM, GRU, 1D-CNN + BCEWithLogitsLoss(pos_weight)
│   │   └── automata/             # PAA, SAX, sax_dict, automata, levenshtein
│   ├── explainability/
│   │   └── explainer.py          # [SYSTEM DECISION] metin + JSON çıktısı
│   ├── evaluate/
│   │   ├── metrics.py            # Accuracy, Precision, Recall, F1
│   │   ├── statistical.py        # Wilcoxon, McNemar
│   │   ├── table_generator.py    # Tablo 1-5 otomatik üretimi
│   │   └── visualizer.py         # 5 zorunlu görsel
│   └── pipeline.py               # Ana çalıştırma betiği
├── tests/                         # 29 unit test (Levenshtein, SAX, Automata)
└── results/
    ├── results.json               # Ham deney çıktıları
    ├── tables/                    # Markdown tablolar
    └── figures/                   # Görseller
```

---

## 5. Deneysel Sonuçlar

### Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)

*5 farklı random seed [42, 123, 2026, 7, 999] ile elde edilen ortalama ve standart sapma.*  
*SKAB için GroupKFold (k=5) fold ortalaması alınmıştır.*

| Model | SKAB F1 | BATADAL F1 |
|-------|---------|------------|
| LSTM | 0.8529 ± 0.0048 | 0.0383 ± 0.0239 |
| GRU | 0.8591 ± 0.0038 | 0.0338 ± 0.0093 |
| 1D-CNN | 0.8529 ± 0.0033 | 0.0324 ± 0.0162 |
| Automata | 0.0171 ± 0.0000 | 0.1449 ± 0.0000 |

**Bulgular:**
- SKAB'da DL modelleri ~0.85 F1 ile yüksek performans sergilerken, Automata 0.017 ile düşük kalmıştır. Bunun temel nedeni, SKAB'ın çok değişkenli yapısı nedeniyle PCA ile tek boyuta indirgenmesinin bilgi kaybına yol açmasıdır.
- BATADAL'da ise Automata (0.145), tüm DL modellerinden (~0.03) daha iyi performans göstermiştir. BATADAL'ın düşük anomali oranı (%5) DL modellerini olumsuz etkilemiştir.

---

### Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi

*Gürültü: Gaussian noise (std=0.1) BATADAL test setine eklenmiştir.*  
*Unseen: SAX sözlüğünde bulunmayan pattern sayısı ve Levenshtein eşleştirme doğruluğu.*

| Model | Orijinal F1 | Gürültülü F1 | F1 Değişimi | Unseen Det. Rate | Unseen Map. Acc. |
|-------|-------------|--------------|-------------|-----------------|-----------------|
| LSTM | 0.0338 | 0.0326 | -0.0012 | N/A | N/A |
| GRU | 0.0799 | 0.0831 | +0.0032 | N/A | N/A |
| 1D-CNN | 0.0000 | 0.0394 | +0.0394 | N/A | N/A |
| Automata | 0.1449 | 0.1606 | +0.0157 | 1.0000 | 0.2000 |

*5/833 pattern (%0.60) eğitim SAX sözlüğünde bulunmamıştır (Unseen). Levenshtein ile en yakın pattern bulunarak eşleştirilmiştir.*

---

### Tablo 3: Cross-Dataset Performans Karşılaştırması (F1-score)

*Her model kendi eğitim setinde eğitilip kendi test setinde değerlendirilmiştir.*

| Model | SKAB (test) | BATADAL (test) |
|-------|-------------|----------------|
| LSTM | 0.8529 | 0.0383 |
| GRU | 0.8591 | 0.0338 |
| 1D-CNN | 0.8529 | 0.0324 |
| Automata | 0.0171 | 0.1449 |

---

### Tablo 4: Automata Parametre Duyarlılık Analizi

#### 4a: F1-score (BATADAL test seti)

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| **3** | 0.0619 | 0.1037 | 0.1606 | 0.1563 |
| **4** | 0.1449 | 0.1887 | 0.1828 | 0.1742 |
| **5** | 0.1618 | 0.1694 | 0.1756 | 0.1756 |
| **6** | 0.1581 | 0.1758 | 0.1758 | 0.1756 |

#### 4b: State Sayısı

| Window Size \ Alphabet Size | 3 | 4 | 5 | 6 |
|-----------------------------|---|---|---|---|
| **3** | 25 | 56 | 98 | 154 |
| **4** | 63 | 171 | 292 | 475 |
| **5** | 137 | 392 | 655 | 1005 |
| **6** | 260 | 737 | 1100 | 1545 |

*En iyi F1 (0.1887): window_size=4, alphabet_size=4. State sayısı parametre büyüdükçe dramatik biçimde artmaktadır.*

---

### Tablo 5: Modellerin Çalışma Süresi (Runtime)

*GPU: NVIDIA RTX 5070 Laptop. Süreler 5 seed ortalamasıdır.*

| Model | SKAB Eğitim (sn) | SKAB Inference (sn) | BATADAL Eğitim (sn) | BATADAL Inference (sn) |
|-------|-----------------|--------------------|--------------------|----------------------|
| LSTM | 18.36 | 0.0922 | 1.51 | 0.0189 |
| GRU | 15.18 | 0.0920 | 1.57 | 0.0170 |
| 1D-CNN | 13.80 | 0.1197 | 1.49 | 0.0234 |
| Automata | 0.35 | 0.4810 | 0.05 | 0.0331 |

*Automata eğitim süresi DL modellerinden ~50x daha hızlıdır. Inference süresi ise path probability hesabı nedeniyle biraz daha uzundur.*

---

## 6. Açıklanabilirlik Modülü

Automata modelinin her kararı için üretilen örnek çıktı:

```
[SYSTEM DECISION]
Time Step: t = 5
Previous State: "aab"
Incoming Pattern: "adc"
Status: Unseen
Nearest Pattern: "abc" (distance = 1)
Transitions:
  aab -> abc : 0.72
  abc -> bcc : 0.15
Path Probability: 0.72 * 0.15 = 0.108
Decision: Low probability path detected
Result: ANOMALY
Confidence Score: 0.108 (Low)
```

JSON formatı:
```json
{
  "time_step": 5,
  "state": "aab",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "probability": 0.108,
  "decision": "anomaly"
}
```

---

## 7. İstatistiksel Analiz

> 🔄 Wilcoxon ve McNemar testleri ekleniyor.

- **K-Fold Cross Validation** (k=5, GroupKFold — SKAB için)
- **Wilcoxon Testi:** DL modelleri vs Automata istatistiksel anlamlılık karşılaştırması
- **McNemar Testi:** İkili karar matrislerinin karşılaştırması
- Seeds: [42, 123, 2026, 7, 999]
