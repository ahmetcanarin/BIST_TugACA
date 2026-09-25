"""
BIST 100 Çok Hedefli (Dual-Target) Derin Öğrenme Model Mimarisi

Bu modül, zaman serisi pencerelerinden (Batch, Seq_Len=30, Features=12)
hem T+1 getiri yüzdesini (Regresyon) hem de yükseliş yönünü (Sınıflandırma)
eşzamanlı tahminleyen nedensel (Causal Unidirectional) GRU + Temporal Attention
mimarisini içerir.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """
    30 günlük geçmiş zaman adımlarını dinamik olarak ağırlıklandıran
    Temporal Attention modülü.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (Batch, Seq_Len, input_dim)
        scores = self.projection(x)  # (Batch, Seq_Len, 1)
        weights = F.softmax(scores, dim=1)  # (Batch, Seq_Len, 1)
        context = torch.sum(weights * x, dim=1)  # (Batch, input_dim)
        return context, weights


class BISTDualTargetModel(nn.Module):
    """
    BIST 100 Pay Piyasası için Çok Görevli (Multi-Task) Nedensel Derin Öğrenme Modeli.
    Desteklenen Hücre Tipleri: 'gru', 'lstm', 'rnn' (Vanilla Elman RNN).

    Girdi:
        x: (Batch_Size, seq_len, num_features)
    Çıktılar:
        pred_return: Beklenen getiri / göreceli alfa skoru (Regresyon) -> (Batch_Size,)
        pred_direction_logit: Yükseliş / endeksi yenme olasılığı logiti (Sınıflandırma) -> (Batch_Size,)
    """
    def __init__(
        self,
        num_features: int = 39,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.40,
        dense_dim: int = 32,
        cell_type: str = "gru"
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()

        # 1. Giriş Normalizasyonu
        self.input_norm = nn.LayerNorm(num_features)

        # 2. Nedensel Zamansal Gövde (RNN / LSTM / GRU)
        # Gelecekten geçmişe sızıntıyı önlemek için kesinlikle tek yönlü (bidirectional=False)
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=num_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:  # Varsayılan: gru
            self.rnn = nn.GRU(
                input_size=num_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0.0
            )
        self.gru = self.rnn  # Geriye dönük uyumluluk referansı

        # 3. Temporal Attention Katmanı
        self.attention = TemporalSelfAttention(input_dim=hidden_dim, hidden_dim=16)

        # 4. Son Adım (t) ve Zamansal Bağlam Birleşimi
        # last_step (hidden_dim) + context (hidden_dim) -> hidden_dim * 2
        combined_dim = hidden_dim * 2

        # 5. Paylaşılan Temsil Katmanı (Shared Latent Representation)
        self.shared_dense = nn.Sequential(
            nn.Linear(combined_dim, dense_dim),
            nn.LayerNorm(dense_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 6. Çift Başlık (Dual Output Heads)
        # Regresyon Başlığı: Beklenen Getiri (Target_Return %)
        self.reg_head = nn.Sequential(
            nn.Linear(dense_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        # Sınıflandırma Başlığı: Yön Tahmini Logiti (Target_Direction)
        self.cls_head = nn.Sequential(
            nn.Linear(dense_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        :param x: (B, seq_len, num_features)
        :param return_attention: True ise attention ağırlıklarını da döner (XAI / Yorumlanabilirlik)
        :return: (pred_return, pred_direction_logit) veya (pred_return, pred_direction_logit, attn_weights)
        """
        x_norm = self.input_norm(x)
        rnn_out, _ = self.rnn(x_norm)  # (B, seq_len, hidden_dim)

        # Son adım (en güncel gün t)
        last_step = rnn_out[:, -1, :]  # (B, hidden_dim)

        # Zamansal attention bağlamı
        context, attn_weights = self.attention(rnn_out)  # context: (B, hidden_dim)

        # Son adım ile bağlamı birleştir
        combined = torch.cat([last_step, context], dim=-1)  # (B, hidden_dim * 2)

        # Paylaşılan özellik katmanı
        shared_feat = self.shared_dense(combined)  # (B, dense_dim)

        # Çift Başlık Tahminleri
        pred_return = self.reg_head(shared_feat).squeeze(-1)  # (B,)
        pred_direction_logit = self.cls_head(shared_feat).squeeze(-1)  # (B,)

        if return_attention:
            return pred_return, pred_direction_logit, attn_weights
        return pred_return, pred_direction_logit
