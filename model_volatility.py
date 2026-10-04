"""
BIST 100 Derin Öğrenme Volatilite Tahmin Modeli (Volatility Neural Architecture)

Bu modül, zaman serisi pencerelerinden (Batch, Seq_Len=30, Features)
hisse senedinin gelecek 20 günlük gerçekleşecek yıllanmış volatilitesini (Regresyon)
ve Volatilite Rejimini (Sıkışma/Patlama, Normal, Yüksek Volatilite) eşzamanlı tahminleyen
Causal GRU + Temporal Self-Attention mimarisini içerir.
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


class BISTVolatilityModel(nn.Module):
    """
    BIST 100 Derin Öğrenme Volatilite & Rejim Tahmin Modeli.
    Outputs:
        pred_vol: Tahmin edilen 20 günlük volatilite (%) (Regresyon)
        pred_regime_logits: 3 Sınıflı Volatilite Rejimi Logitleri (0: Düşük/Sıkışma, 1: Normal, 2: Yüksek/Patlama)
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

        self.gru = nn.GRU(
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
        gru_out, _ = self.gru(x_norm)

        last_step = gru_out[:, -1, :]
        context, _ = self.attention(gru_out)

        combined = torch.cat([last_step, context], dim=-1)
        shared_feat = self.shared_dense(combined)

        pred_vol = self.vol_head(shared_feat).squeeze(-1)
        pred_regime_logits = self.regime_head(shared_feat)

        return pred_vol, pred_regime_logits
