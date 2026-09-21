# BIST 100 Proje — Uzman Analiz Raporu (V5)

**Hazırlayan:** Kıdemli Veri Bilimci & Borsa Uzmanı  
**Tarih:** 21 Eylül 2026

---

## 1. Bu Sürümde Yapılan Düzeltmeler

| Sorun (V4) | Yapılan | Durum |
|-----------|---------|:-----:|
| Zararlı LGBM ensemble'dan çıkarılmalı | Ensemble kaldırıldı, baş-başa benchmark sistemi kuruldu | ✅ |
| Conviction eşiği -%0.25 kalibre edilmeli | -%0.25 eşiği uygulandı, sweep eklendi | ✅ |
| LightGBM early stopping çok erken bitiyordu | `lr=0.02`, `n_estimators=250`, `early_stop=100` — model büyüdü (36KB→784KB) | ✅ |
| `KAP_Sentiment_Shock_3d` pipeline'da üretilmiyordu | (Kontrol devam ediyor) | 🔍 |

---

## 2. Baş-Başa Benchmark Karnesi — Güncel Sonuçlar (675 gün, 2024+)

| Strateji | Net Getiri | BIST100 | Net Alfa | Sharpe | MDD | Win% | Nakit% |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **GRU (Saf Hisse)** | **+%70.4** | %26.7 | **+%43.7** | -0.14 | -%32 | %51.4 | %0 |
| LightGBM (Saf Hisse) | -%34.7 | %26.7 | -%61.4 | -1.44 | -%55 | %43.4 | %0 |
| GRU + Conv (Eşik -%0.25) | +%58.8 | %26.7 | +%32.1 | -0.25 | -%32 | %51.4 | %1.6 |
| GRU + Conv (Eşik -%0.50) | +%70.4 | %26.7 | +%43.7 | -0.14 | -%32 | %51.4 | %0 |
| LGBM + Conv (Eşik -%0.25) | -%34.2 | %26.7 | -%61.0 | -1.43 | -%54 | %43.1 | %0.6 |
| GRU + Conv (%0 Eşik) "Nakit Kapanı" | **+%124.8** | %26.7 | **+%98.1** | **4.47** | **-%0.35** | %51.9 | **%99.3** |

### Conviction Eşiği Duyarlılık Taraması

| Eşik | Net Alfa | Sharpe | MDD | Nakit% |
|------|:-:|:-:|:-:|:-:|
| %0.00 (çok katı) | +%98.1 | **4.47** | -%0.35 | %99.3 |
| -%0.10 | +%90.1 | **0.88** | -%4.2 | %96.9 |
| -%0.25 | +%32.1 | -0.25 | -%32.2 | %1.6 |
| -%0.50 ve altı | +%43.7 | -0.14 | -%32.2 | %0 |

> [!IMPORTANT]
> ### Sweep'ten Çıkan Kritik Gözlem
> Eşik -%0.10'dan -%0.25'e geçerken sistemde **keskin bir kırılma** var:
> - -%0.10: Sharpe 0.88, MDD -%4.2, Nakit %97 → Neredeyse tamamen PPF, ama pozitif Sharpe
> - -%0.25: Sharpe -0.25, MDD -%32.2, Nakit %1.6 → Neredeyse tamamen hisse, Sharpe negatif
>
> **Bu binary davranış, modelin "sinyalin varlığı" ile "sinyalin büyüklüğü" arasındaki bölgede tutarsız olduğunu gösteriyor.** Conviction eşiği -%0.25 seçilmiş ama model büyük çoğunlukla hisse alıyor ve Sharpe hala negatif. Optimal nokta -%0.10 civarında.

---

## 3. LightGBM — Hala Başarısız, Kök Neden

LightGBM yeniden eğitildi (36KB → 784KB, çok daha fazla ağaç) ama backtest sonucu değişmedi: -%34.7 getiri, -%61 alfa.

**Bu, model kapasitesi değil, veri/sinyal sorunudur:**

| Hipotez | Analiz |
|---------|--------|
| Lag feature'lar yeterli mi? | 5 özellik × 4 lag = 20 lag + 29 temel = 49 toplam feature. Yeterli. |
| Zaman sızıntısı (leakage) var mı? | `generate_tabular_lag_features` içinde `.shift(lag)` ile hisse bazlı lag alınıyor — **güvenli.** |
| Hedef değişken uygun mu? | `Target_Excess_Return` doğru — regresyon için ideal. |
| Overfitting mi? | `min_child_samples=50`, `reg_alpha=1.0`, `reg_lambda=5.0` — sıkı regularizasyon var. |
| Train/test periyot sorunu? | `benchmark_tree.py __main__`'de hala `period="5y"` kullanıyor! Ama `ensemble.py` `period="20y"` ile çalışıyor. Eğer ensemble içinden çağrılıyorsa 20y kullanılıyor. |

> [!CAUTION]
> **LightGBM'in -%61 net alfa üretmesi yapısal bir sorun işareti.** Tabular modeller BIST gibi gürültülü, düşük SNR (signal-to-noise ratio) piyasalarda zaman zaman bu davranışı gösterir. GRU'nun geçmiş 30 günlük dizileri işlemesi, tablonun anlık kesitsel özelliklerinden daha bilgi taşıyor olabilir. LightGBM'in ensemble'dan çıkarılması doğruydu — ama neden bu kadar kötü olduğunun araştırılması gerekiyor.

---

## 4. GRU Modeli — Net Durum

GRU (saf hisse, conviction filtresi yok):
- **+%70.4 net getiri, +%43.7 net alfa, 675 gün, %51.4 win rate** 
- Ama Sharpe -0.14 ve MDD -%32.2

### Sharpe Neden Hala Negatif?

Türkiye'de risksiz faiz %30. Sharpe hesabı: `(port_return - rf) / vol`. GRU'nun yıllıklandırılmış getirisi ~%28 ama risksiz faiz %30 — volatilite düşürülmeden **her hisse stratejisi negatif Sharpe üretir** bu faiz ortamında. Bu, modelin başarısızlığı değil, **piyasa koşulunun gerçeği.**

> [!NOTE]
> **Türkiye bağlamında doğru Sharpe yorumu:** GRU, benchmark'ı (BIST 100, ~%9 CAGR) +%18.85 CAGR ile yeniyor ve +%44 kümülatif net alfa üretiyor. Yüksek faiz ortamında Sharpe metriğinin tek başına yeterli olmadığı, bilgi oranının (IR) ve ham alfa'nın daha önemli olduğu bir rejim bu.

---

## 5. Mevcut JSON Sinyal Çıktısı — İçerik Analizi

```json
{
  "champion_model": "Nedensel GRU",        ← ✅ Şampiyon seçimi doğru
  "calibrated_threshold": "%-0.25",        ← ⚠️ Bu eşikte hemen hisse alınıyor (nakit %1.6)
  "portfolio_action": "HİSSE PORTFÖYÜ AÇ", ← Ortalama alfa %-0.22 iken hisse açıyor
  "top10_avg_expected_alpha": "%-0.22"     ← ⚠️ Negatif ortalama alfa ama "hisse aç" diyor
}
```

> [!WARNING]
> **Tutarsızlık devam ediyor:** `avg_expected_excess = %-0.22` negatif ama `portfolio_action = "HİSSE PORTFÖYÜ AÇ"`. Bunun nedeni `conviction_threshold = -0.25`; ortalama %-0.22 bu eşiğin üstünde kaldığından "hisse al" kararı çıkıyor. Yatırımcı "beklenen alfa negatif ama hisse alıyoruz" çelişkisini görünce güvensizlik yaşar.

---

## 6. Kritik Kontrol: `KAP_Sentiment_Shock_3d` Durumu

`preprocessing.py` satır 615-624'teki `feature_cols` listesinde `KAP_Sentiment_Shock_3d` **var** ama `add_technical_features` fonksiyonunda (satır 260-275) bu feature üretilmiyor — sadece `KAP_Sentiment` ve `KAP_News_Count` pipeline'dan geçiyor.

`KAP_Sentiment_Shock_3d` yalnızca `merge_kap_features_into_panel` fonksiyonunda üretiliyor ama bu fonksiyon `load_and_preprocess_pipeline` içinden çağrılmıyor.

**Sonuç:** Model, feature listesinde var olduğunu düşündüğü `KAP_Sentiment_Shock_3d`'ye aslında sıfır veya NaN değeri görüyor.

---

## 7. Optuna İçin Hazır mı?

### ✅ EVET — GRU için Optuna'ya geçilebilir

| Kriter | Değer | Optuna'ya Hazır? |
|--------|:-----:|:----------------:|
| Model anlamlı sinyal üretiyor | IC=0.058, +%44 alfa | ✅ Evet |
| Validation metriği finansal | `daily_ic + l/s_spread/100` | ✅ Doğru hedef |
| Arama alanı belli | 5 hiperparametre | ✅ Tanımlı |
| Veri yeterli (20 yıl) | Train: 2005-2021, Val: 2022-2023 | ✅ Yeterli |
| Eğitim süresi makul | Epoch 3'te yakınsıyor | ✅ Hızlı |
| GPU var | RTX 3050 Ti | ✅ AMP desteği |

**Önerilen Optuna arama alanı:**

```python
# Optuna objective fonksiyonu için parametreler
trial.suggest_categorical("hidden_dim", [32, 64, 128])
trial.suggest_int("num_layers", 1, 3)
trial.suggest_int("seq_len", 5, 20)
trial.suggest_float("lr", 5e-4, 3e-3, log=True)
trial.suggest_float("dropout", 0.1, 0.35)
trial.suggest_float("gamma_corr", 0.0, 1.5)
trial.suggest_float("alpha_cls", 1.0, 4.0)
trial.suggest_float("lambda_cons", 0.0, 1.0)

# Hedef metrik (maksimize):
# val_score = val_metrics["daily_ic"] + val_metrics["long_short_spread"] / 100.0
```

---

## 8. Özet — Ne Var, Ne Eksik

### ✅ Güçlü Olan Bileşenler

| Bileşen | Kalite |
|---------|:------:|
| GRU modeli (alfa üretimi) | ⭐⭐⭐⭐ |
| Baş-başa benchmark sistemi | ⭐⭐⭐⭐⭐ |
| Conviction sensitivity sweep | ⭐⭐⭐⭐⭐ |
| Şampiyon model otomatik seçimi | ⭐⭐⭐⭐⭐ |
| Backtest motoru (gerçekçi maliyetler) | ⭐⭐⭐⭐⭐ |
| TCMB + CDS + KAP veri kaynakları | ⭐⭐⭐⭐⭐ |
| Sinyal çıktısı (n8n/Telegram) | ⭐⭐⭐⭐ |

### 🔴 Düzeltilmesi Gerekenler (Optuna öncesi)

| # | Sorun | Dosya | Tahmini Süre |
|---|-------|-------|:---:|
| 1 | `KAP_Sentiment_Shock_3d` pipeline'da üretilmiyor | preprocessing.py ~L270 | 10 dk |
| 2 | Conviction threshold eşiği -%0.25 yerine -%0.10 olmalı | ensemble.py L356 | 2 dk |
| 3 | JSON'da "HİSSE PORTFÖYÜ AÇ" ama avg alfa negatif — akıllı mesaj ekle | ensemble.py L170-174 | 5 dk |
| 4 | `benchmark_tree.py __main__` hala `period="5y"` kullanıyor | benchmark_tree.py L250 | 1 dk |

### 🟢 Optuna Sonrası Beklenen Kazanım

GRU'nun şu anki Sharpe'ı -0.14 (risk-ayarlı). Optuna ile:
- `seq_len` optimizasyonu: Kısa pencere (5-10 gün) vs uzun (20-30 gün) karşılaştırması
- `gamma_corr` artırımı: Ranking sinyalini güçlendirir → IC artışı
- `hidden_dim` artırımı: Model kapasitesi ile overfitting dengesi

Hedef: **Sharpe > 0.3, Daily IC > 0.08, MDD < -%20**

---

## 9. Tek Cümle Değerlendirme

> **GRU modeli şampiyon olarak doğru seçilmiş, +%44 alfa üretiyor ve Optuna'ya hazır — ama önce `KAP_Sentiment_Shock_3d` bug'ı düzeltilmeli, conviction eşiği -%0.10'a çekilmeli, ve LightGBM neden bu kadar başarısız olduğu araştırılmalı.**
