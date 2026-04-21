# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

**Grup 15** — SWAT + WADI Veri Setleri  
**Ders:** Yazılım Geliştirme — 2. Proje  

---

## İçindekiler
1. [Proje Hakkında](#1-proje-hakkında)
2. [Kurulum](#2-kurulum)
3. [Kullanım](#3-kullanım)
4. [Proje Yapısı](#4-proje-yapısı)
5. [Deneysel Sonuçlar](#5-deneysel-sonuçlar)
6. [Açıklanabilirlik Modülü](#6-açıklanabilirlik-modülü)
7. [İstatistiksel Analiz](#7-i̇statistiksel-analiz)

---

## 1. Proje Hakkında

Bu proje, zaman serisi anomali tespitinde iki farklı modelleme paradigmasını karşılaştırmaktadır:

- **Black-box modeller:** LSTM, GRU, 1D-CNN (Derin Öğrenme)
- **Explainable model:** Probabilistic Automata (PAA → SAX → Geçiş Matrisi)

### Araştırma Sorusu
Farklı modelleme yaklaşımları, farklı veri koşulları altında nasıl davranmaktadır ve bu davranışlar istatistiksel olarak anlamlı mıdır?

---

## 2. Kurulum

```bash
# Sanal ortam
python -m venv venv
.\venv\Scripts\activate  # Windows

# Bağımlılıklar
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Veri Setleri
Veri setlerini aşağıdaki dizinlere yerleştirin:
```
data/raw/swat/
data/raw/wadi/
```

---

## 3. Kullanım

```bash
# Tüm deneyleri çalıştır
python src/pipeline.py --config configs/experiments.yaml

# Testleri çalıştır
pytest tests/ -v
```

---

## 4. Proje Yapısı

```
yazlab-2-2/
├── configs/experiments.yaml     # Tüm parametreler
├── data/                        # Ham ve işlenmiş veriler
├── src/
│   ├── data/                    # Ön işleme ve yükleme
│   ├── models/
│   │   ├── deep_learning/       # LSTM, GRU, 1D-CNN
│   │   └── automata/            # PAA, SAX, Automata
│   ├── explainability/          # Açıklanabilirlik modülü
│   ├── evaluate/                # Metrikler, tablolar, görseller
│   └── pipeline.py              # Ana çalıştırma betiği
├── tests/                       # Unit testler
└── results/                     # Deney çıktıları
```

---

## 5. Deneysel Sonuçlar

> 🔄 Bu bölüm deneyler tamamlandıkça güncellenecektir.

### Tablo 1: Model Performansı ve Stabilitesi (F1 ± Std, 5 Seed)

| Model | SWAT | WADI |
|-------|------|------|
| LSTM | — | — |
| GRU | — | — |
| 1D-CNN | — | — |
| Automata | — | — |

### Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi

| Model | Orijinal F1 | Gürültülü F1 | Det. Rate | Map. Acc. |
|-------|------------|-------------|-----------|-----------|
| LSTM | — | — | — | — |
| GRU | — | — | — | — |
| 1D-CNN | — | — | — | — |
| Automata | — | — | — | — |

### Tablo 3: Cross-Dataset Genellenebilirlik Matrisi

| Train / Test | SWAT | WADI |
|-------------|------|------|
| Train: SWAT | — | — |
| Train: WADI | — | — |

### Tablo 4: Automata Parametre Duyarlılık Analizi (F1)

| Parametre | Değer=3 | Değer=4 | Değer=5 | Değer=6 |
|-----------|---------|---------|---------|---------|
| Window Size | — | — | — | — |
| Alphabet Size | — | — | — | — |

### Tablo 5: Çalışma Süreleri

| Model | Training Time (sn) | Inference Time (sn) |
|-------|-------------------|---------------------|
| LSTM | — | — |
| GRU | — | — |
| 1D-CNN | — | — |
| Automata | — | — |

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

---

## 7. İstatistiksel Analiz

> 🔄 Bu bölüm deneyler tamamlandıkça güncellenecektir.

- **K-Fold Cross Validation** (k=5)
- **Wilcoxon Testi:** DL modelleri vs Automata karşılaştırması
- **McNemar Testi:** İkili karar matrislerinin karşılaştırması
