"""
BIST 100 — Merkezi Yapay Zeka Çıkarım & Karar Destek Motoru (Inference Engine)

Bu modül:
1. PyTorch Çift Hedefli Şampiyon GRU modelini (Alfa Getiri + Yükseliş Sınıflandırması) yükler.
2. PyTorch Volatilite & Rejim GRU modelini (Oynaklık % + Sıkışma/Normal/Patlama) yükler.
3. Canlı piyasa verisinden 34 durağan ve ölçekten bağımsız özniteliği dinamik türetir.
4. Temel bilanço analitiği, takas/para akışı ve KAP duygu analizini birleştirir.
5. Hem Streamlit arayüzü hem de REST API / Telegram botları için TEK MERKEZİ çıkarım sağlar.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import json
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from model import BISTDualTargetModel
from model_volatility import BISTVolatilityModel
from preprocessing import _compute_rsi
from fundamental_engine import analyze_company_fundamentals
from custody_engine import analyze_custody_and_money_flow


class BISTInferenceEngine:
    """
    Tüm BIST modellerini ve analitik motorlarını tek çatıda toplayan Singleton çıkarım motoru.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BISTInferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        dual_model_path: str = "models/bist_dual_model_best.pt",
        vol_model_path: str = "models/bist_volatility_gru.pt",
        device: Optional[str] = None
    ):
        if self._initialized:
            return

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dual_model_path = dual_model_path
        self.vol_model_path = vol_model_path if os.path.exists(vol_model_path) else "models/bist_volatility_model.pt"

        self.dual_model: Optional[BISTDualTargetModel] = None
        self.vol_model: Optional[BISTVolatilityModel] = None
        self.dual_feature_cols: List[str] = []
        self.vol_feature_cols: List[str] = []
        self.dual_seq_len: int = 15
        self.vol_seq_len: int = 30

        self._load_models()
        self._initialized = True

    def _load_models(self):
        """Modelleri bellek üzerine yükler."""
        # 1. Çift Hedefli GRU Modeli (Return + Direction)
        if os.path.exists(self.dual_model_path):
            try:
                ckpt = torch.load(self.dual_model_path, map_location=self.device, weights_only=False)
                self.dual_feature_cols = ckpt.get("feature_cols", [])
                nf = len(self.dual_feature_cols) if self.dual_feature_cols else 34
                self.dual_seq_len = ckpt.get("seq_len", 15)
                h_dim = ckpt.get("hidden_dim", 64)
                n_layers = ckpt.get("num_layers", 2)
                dropout = ckpt.get("dropout", 0.20)

                self.dual_model = BISTDualTargetModel(
                    num_features=nf,
                    hidden_dim=h_dim,
                    num_layers=n_layers,
                    dropout=dropout
                ).to(self.device)
                self.dual_model.load_state_dict(ckpt["model_state_dict"])
                self.dual_model.eval()
                print(f"[InferenceEngine] Çift Hedefli GRU yüklendi (seq_len={self.dual_seq_len}, num_features={nf}).")
            except Exception as e:
                print(f"[InferenceEngine] Çift Hedefli GRU yüklenirken hata: {e}")
                self.dual_model = None

        # 2. Volatilite Modeli
        if os.path.exists(self.vol_model_path):
            try:
                ckpt = torch.load(self.vol_model_path, map_location=self.device, weights_only=False)
                self.vol_feature_cols = ckpt.get("feature_cols", [])
                nf = ckpt.get("num_features", len(self.vol_feature_cols) if self.vol_feature_cols else 34)
                self.vol_seq_len = 30

                self.vol_model = BISTVolatilityModel(num_features=nf).to(self.device)
                self.vol_model.load_state_dict(ckpt["model_state_dict"])
                self.vol_model.eval()
                print(f"[InferenceEngine] Volatilite GRU yüklendi (seq_len={self.vol_seq_len}, num_features={nf}).")
            except Exception as e:
                print(f"[InferenceEngine] Volatilite GRU yüklenirken hata: {e}")
                self.vol_model = None

    @staticmethod
    def extract_features_for_ticker(
        df_stock: pd.DataFrame,
        feature_cols: List[str],
        seq_len: int = 30
    ) -> torch.Tensor:
        """
        Canlı hisse DataFrame'inden modelin beklediği 3D Tensörü ((1, seq_len, num_features)) üretir.
        """
        if df_stock.empty or len(df_stock) < 5 or not feature_cols:
            return torch.zeros(1, seq_len, len(feature_cols) if feature_cols else 34, dtype=torch.float32)

        df = df_stock.copy().sort_values("Date").reset_index(drop=True)
        close_s = df["Close"]
        high_s = df["High"]
        low_s = df["Low"]
        open_s = df["Open"]
        vol_s = df["Volume"]

        feat_dict = {}

        # 1. Getiriler & Oranlar
        log_ret = np.log(close_s / close_s.shift(1)).fillna(0.0)
        feat_dict["Log_Return"] = log_ret
        feat_dict["HL_Spread"] = ((high_s - low_s) / (close_s + 1e-7)).fillna(0.0)
        feat_dict["CO_Return"] = ((close_s - open_s) / (open_s + 1e-7)).fillna(0.0)

        # 2. Hacim Dinamikleri
        log_vol = np.log1p(vol_s)
        feat_dict["Log_Volume"] = log_vol
        feat_dict["Volume_Change"] = log_vol.diff().fillna(0.0)

        # 3. Hareketli Ortalamalar
        sma10 = close_s.rolling(10, min_periods=2).mean()
        sma30 = close_s.rolling(30, min_periods=3).mean()
        feat_dict["SMA10_Ratio"] = ((close_s / (sma10 + 1e-7)) - 1.0).fillna(0.0)
        feat_dict["SMA30_Ratio"] = ((close_s / (sma30 + 1e-7)) - 1.0).fillna(0.0)

        # 4. Volatilite
        feat_dict["Volatility_20"] = log_ret.rolling(20, min_periods=3).std().fillna(0.0)

        # 5. RSI Norm [-1, 1]
        raw_rsi = _compute_rsi(close_s, window=14)
        feat_dict["RSI_Norm"] = ((raw_rsi - 50.0) / 50.0).fillna(0.0)

        # 6. MFI Norm [-1, 1]
        tp = (high_s + low_s + close_s) / 3.0
        rmf = tp * vol_s
        pos_mf = (rmf.where(tp > tp.shift(1), 0.0)).rolling(14, min_periods=3).sum()
        neg_mf = (rmf.where(tp < tp.shift(1), 0.0)).rolling(14, min_periods=3).sum()
        raw_mfi = 100.0 - (100.0 / (1.0 + (pos_mf / (neg_mf + 1e-7))))
        feat_dict["MFI_Norm"] = ((raw_mfi.fillna(50.0) - 50.0) / 50.0).fillna(0.0)

        # 7. OBV Norm
        obv = (np.sign(close_s.diff()).fillna(0.0) * vol_s).cumsum()
        obv_std = obv.rolling(30, min_periods=5).std().replace(0, 1.0)
        feat_dict["OBV_Norm"] = (((obv - obv.rolling(30, min_periods=5).mean()) / obv_std).fillna(0.0).clip(-3.0, 3.0))

        # 8. RS_5d
        feat_dict["RS_5d"] = close_s.pct_change(5).fillna(0.0)

        # 9. MACD Norm
        ema12 = close_s.ewm(span=12, adjust=False).mean()
        ema26 = close_s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        feat_dict["MACD_Norm"] = (((macd_hist / (close_s + 1e-7)) * 10.0).fillna(0.0).clip(-2.0, 2.0))

        # 10. Custody Flow Proxy
        vol_mean = vol_s.rolling(20, min_periods=3).mean()
        vol_std = vol_s.rolling(20, min_periods=3).std().replace(0, 1.0)
        vol_z = ((vol_s - vol_mean) / (vol_std + 1e-7)).fillna(0.0).clip(-3.0, 3.0)
        feat_dict["Custody_Flow_Proxy"] = (((feat_dict["MFI_Norm"] + feat_dict["OBV_Norm"] + vol_z) / 3.0).fillna(0.0).clip(-2.0, 2.0))

        # Kalan makro ve sistem değişkenleri için güvenli 0.0 dolguları
        for col in feature_cols:
            if col not in feat_dict:
                feat_dict[col] = pd.Series(0.0, index=df.index)

        df_feat = pd.DataFrame({col: feat_dict[col].values for col in feature_cols}).fillna(0.0)

        if len(df_feat) >= seq_len:
            seq_data = df_feat.iloc[-seq_len:].values
        else:
            pad_len = seq_len - len(df_feat)
            pad = np.tile(df_feat.iloc[0].values, (pad_len, 1))
            seq_data = np.vstack([pad, df_feat.values])

        tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0)
        return tensor

    def fetch_stock_data(self, ticker: str, period: str = "120d") -> pd.DataFrame:
        """Hisse verisini yfinance üzerinden çeker."""
        ticker_sym = ticker.upper() if ticker.endswith(".IS") else ticker.upper() + ".IS"
        df = yf.download(ticker_sym, period=period, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker_sym, axis=1, level=1) if ticker_sym in df.columns.levels[1] else df.iloc[:, :6]
        df = df.reset_index()
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])

        # Eksik veya seans dışı NaN satırlarını filtrele
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"])
        for col in ["High", "Low", "Open"]:
            if col in df.columns and "Close" in df.columns:
                df[col] = df[col].fillna(df["Close"])

        return df.sort_values("Date").reset_index(drop=True)

    def predict_single_ticker(
        self,
        ticker: str,
        df_stock: Optional[pd.DataFrame] = None,
        include_fundamentals: bool = True,
        include_custody: bool = True
    ) -> Dict[str, Any]:
        """
        Belirtilen hisse senedi için 360° tam yapay zeka ve kantitatif analiz üretir.
        """
        clean_ticker = ticker.replace(".IS", "").upper()
        ticker_sym = clean_ticker + ".IS"

        if df_stock is None or df_stock.empty:
            df_stock = self.fetch_stock_data(clean_ticker, period="120d")

        if df_stock.empty or len(df_stock) < 5:
            return {"error": f"'{ticker}' için piyasa verisi bulunamadı.", "ticker": clean_ticker}

        # Eksik Close satırlarını temizle
        if "Close" in df_stock.columns:
            df_stock = df_stock.dropna(subset=["Close"])
        for col in ["High", "Low", "Open"]:
            if col in df_stock.columns and "Close" in df_stock.columns:
                df_stock[col] = df_stock[col].fillna(df_stock["Close"])

        latest_row = df_stock.iloc[-1]
        prev_row = df_stock.iloc[-2] if len(df_stock) > 1 else latest_row
        close_price = float(latest_row["Close"]) if pd.notna(latest_row.get("Close")) else 0.0
        prev_close = float(prev_row["Close"]) if pd.notna(prev_row.get("Close")) and float(prev_row["Close"]) > 0 else close_price
        daily_change_pct = ((close_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        volume = int(latest_row["Volume"]) if pd.notna(latest_row.get("Volume")) else 0

        log_ret = np.log(df_stock["Close"] / df_stock["Close"].shift(1)).fillna(0.0)
        hist_vol_20 = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100.0)
        rsi_14 = float(_compute_rsi(df_stock["Close"], window=14).iloc[-1])

        # 1. Çift Hedefli GRU Model Tahmini
        pred_return_pct = 0.0
        up_prob_pct = 50.0
        conviction_signal = "🟡 NÖTR / İZLE"

        if self.dual_model is not None:
            try:
                x_dual = self.extract_features_for_ticker(
                    df_stock,
                    feature_cols=self.dual_feature_cols,
                    seq_len=self.dual_seq_len
                ).to(self.device)

                with torch.no_grad():
                    p_reg, p_cls = self.dual_model(x_dual)
                    pred_return_pct = float(p_reg.item())
                    up_prob_pct = float(torch.sigmoid(p_cls).item() * 100.0)

                if up_prob_pct >= 60.0 and pred_return_pct > 0:
                    conviction_signal = "🟢 GÜÇLÜ AL"
                elif up_prob_pct >= 52.0:
                    conviction_signal = "🟢 AL / BİRİKTİR"
                elif up_prob_pct <= 40.0 and pred_return_pct < 0:
                    conviction_signal = "🔴 GÜÇLÜ SAT"
                elif up_prob_pct <= 48.0:
                    conviction_signal = "🔴 AZALT / SAT"
                else:
                    conviction_signal = "🟡 NÖTR / İZLE"
            except Exception as e:
                print(f"[InferenceEngine] Çift hedefli çıkarım hatası ({ticker}): {e}")

        # 2. Volatilite & Rejim GRU Model Tahmini
        pred_volatility_pct = hist_vol_20
        regime_label = "🟡 Normal"

        if self.vol_model is not None:
            try:
                x_vol = self.extract_features_for_ticker(
                    df_stock,
                    feature_cols=self.vol_feature_cols,
                    seq_len=self.vol_seq_len
                ).to(self.device)

                with torch.no_grad():
                    pv, pr = self.vol_model(x_vol)
                    if pv.item() > 0:
                        pred_volatility_pct = float(pv.item())
                    ri = int(pr.argmax(dim=1).item())
                    regimes = ["🟢 Sıkışma (Düşük)", "🟡 Normal", "🔴 Patlama (Yüksek)"]
                    regime_label = regimes[ri]
            except Exception as e:
                print(f"[InferenceEngine] Volatilite çıkarım hatası ({ticker}): {e}")

        # 3. Temel Analiz & Mali Yapı (Opsiyonel)
        fund_res = {}
        if include_fundamentals:
            try:
                fund_res = analyze_company_fundamentals(clean_ticker)
            except Exception as e:
                fund_res = {"error": str(e)}

        # 4. Takas & Para Akışı (Opsiyonel)
        cust_res = {}
        if include_custody:
            try:
                cust_res = analyze_custody_and_money_flow(clean_ticker)
            except Exception as e:
                cust_res = {"error": str(e)}

        return {
            "ticker": clean_ticker,
            "full_symbol": ticker_sym,
            "price": round(close_price, 2),
            "daily_change_pct": round(daily_change_pct, 2),
            "volume": volume,
            "rsi_14": round(rsi_14, 1),
            "realized_volatility_20": round(hist_vol_20, 1),
            "ai_predictions": {
                "expected_excess_return_pct": round(pred_return_pct, 2),
                "up_probability_pct": round(up_prob_pct, 1),
                "conviction_signal": conviction_signal,
                "predicted_volatility_pct": round(pred_volatility_pct, 1),
                "volatility_regime": regime_label,
            },
            "fundamental": fund_res,
            "custody": cust_res
        }


# Global Singleton instance
_engine_instance: Optional[BISTInferenceEngine] = None


def get_inference_engine() -> BISTInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = BISTInferenceEngine()
    return _engine_instance


def predict_single_ticker(ticker: str) -> Dict[str, Any]:
    """Hızlı bağımsız tahmin fonksiyonu."""
    engine = get_inference_engine()
    return engine.predict_single_ticker(ticker)


if __name__ == "__main__":
    ticker_test = "THYAO"
    print(f"\n[Test] {ticker_test} için inference_engine çalıştırılıyor...")
    res = predict_single_ticker(ticker_test)
    print("\n--- ÇIKIŞ SONUCU ---")
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
