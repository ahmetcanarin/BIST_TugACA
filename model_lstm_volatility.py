"""
BIST 100 LSTM Volatilite Tahmin Modeli (GRU ile Kıyaslama için)

Bu modül, mevcut GRU Volatilite modeliyle aynı mimari yapıyı kullanır
ancak GRU yerine LSTM katmanlarını içerir:
1. LSTM + Temporal Self-Attention ile gelecek 20 günlük volatilite tahmini (Regresyon)
2. 3 Sınıflı Volatilite Rejimi tahmini (Sıkışma/Normal/Patlama)
3. GRU ile başarı oranı karşılaştırması için birebir aynı çıktı arayüzü
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """30 günlük geçmiş zaman adımlarını dinamik olarak ağırlıklandıran Temporal Attention."""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.projection(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class BISTVolatilityLSTMModel(nn.Module):
    """
    BIST 100 LSTM Volatilite & Rejim Tahmin Modeli.
    GRU modeliyle birebir aynı çıktı arayüzü kullanır (karşılaştırma için).

    Outputs:
        pred_vol: Tahmin edilen 20 günlük volatilite (%) (Regresyon)
        pred_regime_logits: 3 Sınıflı Volatilite Rejimi Logitleri
    """
    def __init__(
        self,
        num_features: int = 34,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.20,
        dense_dim: int = 64
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(num_features)

        # LSTM - GRU yerine LSTM kullanılıyor
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = TemporalSelfAttention(input_dim=hidden_dim, hidden_dim=32)

        combined_dim = hidden_dim * 2
        self.shared_dense = nn.Sequential(
            nn.Linear(combined_dim, dense_dim),
            nn.LayerNorm(dense_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 1. Regresyon Başlığı: Gelecek Volatilite (%)
        self.vol_head = nn.Sequential(
            nn.Linear(dense_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Volatilite negatif olamaz
        )

        # 2. Sınıflandırma Başlığı: 3 Sınıflı Volatilite Rejimi
        self.regime_head = nn.Sequential(
            nn.Linear(dense_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.input_norm(x)
        lstm_out, (h_n, c_n) = self.lstm(x_norm)

        last_step = lstm_out[:, -1, :]
        context, _ = self.attention(lstm_out)

        combined = torch.cat([last_step, context], dim=-1)
        shared_feat = self.shared_dense(combined)

        pred_vol = self.vol_head(shared_feat).squeeze(-1)
        pred_regime_logits = self.regime_head(shared_feat)

        return pred_vol, pred_regime_logits
