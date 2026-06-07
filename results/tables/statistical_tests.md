## Istatistiksel Analiz Sonuclari

### Wilcoxon Signed-Rank Testi

H0: Iki model arasinda istatistiksel olarak anlamli fark yoktur.
Anlamlilik esigi: alpha = 0.05

#### SKAB

| Model A | Model B | Istatistik | p-degeri | Anlamli? | Not |
|---------|---------|-----------|----------|----------|-----|
| LSTM | GRU | 0.0000 | 0.062500 | Hayir | p>=0.05: H0 reddedilemedi |
| LSTM | 1D-CNN | 6.0000 | 0.812500 | Hayir | p>=0.05: H0 reddedilemedi |
| LSTM | Automata | 0.0000 | 0.062500 | Hayir | p>=0.05: H0 reddedilemedi |
| GRU | 1D-CNN | 0.0000 | 0.062500 | Hayir | p>=0.05: H0 reddedilemedi |
| GRU | Automata | 0.0000 | 0.062500 | Hayir | p>=0.05: H0 reddedilemedi |
| 1D-CNN | Automata | 0.0000 | 0.062500 | Hayir | p>=0.05: H0 reddedilemedi |

#### BATADAL

| Model A | Model B | Istatistik | p-degeri | Anlamli? | Not |
|---------|---------|-----------|----------|----------|-----|
| LSTM | GRU | 7.0000 | 1.000000 | Hayir | p>=0.05: H0 reddedilemedi |
| LSTM | 1D-CNN | 4.0000 | 0.437500 | Hayir | p>=0.05: H0 reddedilemedi |
| LSTM | Automata | 0.0000 | 0.062500 | Hayir | p>=0.05: H0 reddedilemedi |
| GRU | 1D-CNN | 6.0000 | 0.812500 | Hayir | p>=0.05: H0 reddedilemedi |
| GRU | Automata | 0.0000 | 0.062500 | Hayir | p>=0.05: H0 reddedilemedi |
| 1D-CNN | Automata | 0.0000 | 0.062500 | Hayir | p>=0.05: H0 reddedilemedi |

### McNemar Testi

H0: Iki modelin hata oranlari istatistiksel olarak esittir.

| Model A | Model B | n01 | n10 | Istatistik | p-degeri | Anlamli? |
|---------|---------|-----|-----|-----------|----------|----------|
| LSTM | GRU | 71 | 44 | 5.8783 | 0.015329 | **Evet** |
| LSTM | 1D-CNN | 72 | 53 | 2.5920 | 0.107405 | Hayir |
| LSTM | Automata | 75 | 64 | 0.7194 | 0.396333 | Hayir |
| GRU | 1D-CNN | 53 | 61 | 0.4298 | 0.512075 | Hayir |
| GRU | Automata | 55 | 71 | 1.7857 | 0.181449 | Hayir |
| 1D-CNN | Automata | 61 | 69 | 0.3769 | 0.539255 | Hayir |
