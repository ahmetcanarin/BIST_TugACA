from extraction import tum_hisseleri_cek, son_gun_ozet_tablosu, endeks_verisini_cek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

# 1. extraction.py'daki fonksiyonu çağırarak verileri çek
ham_veri = tum_hisseleri_cek(period="1mo", interval="1d")

# 2. Özet tabloyu oluştur
ozet_df = son_gun_ozet_tablosu(ham_veri)

# 3. Endeks verisini çek (1 aylık zaman serisi)
endeks_df = endeks_verisini_cek(period="1mo")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles / Summary #####################")
    print(dataframe.describe([0.05, 0.25, 0.75, 0.95]).T)

print("=== Ham Özet Tablo ===")
check_df(ham_veri)



# ---------------------------------------------------------------------------
# 4. ENDEKS (XU100) ZAMAN SERİSİ GÖRSELLEŞTİRMESİ
#    (X Ekseni: Tarih, Y Ekseni: Değişkenler)
# ---------------------------------------------------------------------------
#
# plot_df = endeks_df.copy()
#
# # Date zaten DatetimeIndex olduğu için doğrudan kullanılır
# plot_df.index = pd.to_datetime(plot_df.index)
# plot_df.index.name = "Date"
#
#
# def plot_time_series(data, columns=None, figsize=(12, 6)):
#     """
#     Belirtilen sütunların zaman serisi grafiğini tek bir figürde çizer.
#     Volume (Hacim) fiyat ölçeğini bozmaması için varsayılan olarak hariç tutulur.
#     """
#     if columns is None:
#         columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in data.columns]
#         if not columns:
#             columns = [c for c in data.columns if str(c).lower() not in ['volume', 'ticker']]
#
#     fig, ax = plt.subplots(figsize=figsize)
#     for column in columns:
#         if column in data.columns:
#             ax.plot(data.index, data[column], label=str(column), marker='o', markersize=3)
#
#     ax.set_title('BIST 100 (XU100) Zaman Serisi Görselleştirmesi', fontsize=14)
#     ax.set_xlabel('Tarih', fontsize=11)
#     ax.set_ylabel('Fiyat / Endeks Değeri', fontsize=11)
#     fig.autofmt_xdate()
#     ax.legend(loc='best')
#     ax.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_individual_series(data, figsize=(12, 4)):
#     """
#     DataFrame'deki her sayısal sütun için ayrı ayrı zaman serisi grafiği çizer.
#     """
#     numeric_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in data.columns]
#     for column in numeric_cols:
#         fig, ax = plt.subplots(figsize=figsize)
#         ax.plot(data.index, data[column], label=str(column), color='royalblue', marker='.', markersize=4)
#         ax.set_title(f'BIST 100 (XU100) - {column}', fontsize=12)
#         ax.set_xlabel('Tarih', fontsize=10)
#         ax.set_ylabel(str(column), fontsize=10)
#         fig.autofmt_xdate()
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.legend(loc='best')
#         plt.tight_layout()
#         plt.show()
#
#
# # Fonksiyonları çağırma
# plot_time_series(plot_df)
# plot_individual_series(plot_df)
