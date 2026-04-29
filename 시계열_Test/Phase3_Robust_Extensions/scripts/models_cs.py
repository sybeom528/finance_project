"""Phase 3 — Cross-Sectional LSTM 모델.

종목별 학습 (Phase 1.5 v8) → Cross-sectional 학습 (Phase 3)

본 모듈의 핵심 차이점
--------------------
[Phase 1.5 의 LSTMRegressor (종목별)]
    - 74 종목 × 92 fold = 6,808 개 독립 모델
    - 종목당 학습 sample 약 441 개/fold
    - Sample/Parameter ratio: 0.10:1 (over-fitting 위험 ↑)
    - 종목 특화 패턴 학습

[Phase 3 의 CrossSectionalLSTMRegressor (cross-sectional)]
    - 92 fold × 1 = 92 개 모델 (또는 1 개)
    - fold 당 학습 sample = 441 × 종목수 (74 배 ↑)
    - Sample/Parameter ratio: 7.2:1 (74 배 향상)
    - 시장 공통 패턴 + 종목 차별화 (ticker embedding)

학술 근거
---------
- Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning"
  *Review of Financial Studies, 33*(5), 2223-2273
- Chen, Pelger, Zhu (2024) "Deep Learning in Asset Pricing"
  *Management Science*

핵심 구조
---------
Input:
    x: (B, T, F)         — features (rv_d, rv_w, rv_m, vix_log)
    ticker_ids: (B,)      — 종목 ID (LongTensor)

Forward:
    1. ticker_emb = embedding(ticker_ids)      # (B, embed_dim)
    2. ticker_emb_seq = expand to (B, T, embed_dim)
    3. x_aug = concat(x, ticker_emb_seq)        # (B, T, F + embed_dim)
    4. lstm_out = LSTM(x_aug)                   # (B, T, hidden)
    5. y_pred = Linear(lstm_out[:, -1, :])      # (B,)

Output:
    y_pred: (B,) — 각 (종목, 시점) 의 log-RV 예측
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossSectionalLSTMRegressor(nn.Module):
    """Cross-sectional LSTM with ticker embedding.

    Phase 1.5 의 종목별 LSTM 과 달리, 모든 종목이 동일한 LSTM weights 를 공유.
    종목 특이성은 ticker embedding 을 input feature 에 concat 하여 표현.

    Parameters
    ----------
    input_size : int, default 4
        시계열 feature 차원 (rv_d, rv_w, rv_m, vix_log).
    hidden_size : int, default 64
        LSTM 은닉 차원. Phase 1.5 의 32 보다 크게 설정 (cross-sectional 은 sample 풍부).
    num_layers : int, default 2
        LSTM 레이어 수. cross-sectional 환경에서 깊이 ↑ 가능.
    dropout : float, default 0.3
        LSTM 내부 dropout (num_layers > 1 시).
    n_tickers : int, default 200
        고유 종목 수 (universe 의 unique ticker).
    embedding_dim : int, default 8
        Ticker embedding 차원. 작은 값 (4-16) 권장.
    output_dropout : float, default 0.2
        FC layer 직전 dropout.

    Attributes
    ----------
    ticker_embedding : nn.Embedding (n_tickers, embedding_dim)
    lstm : nn.LSTM (input_size + embedding_dim → hidden_size)
    fc : nn.Linear (hidden_size → 1)

    Notes
    -----
    파라미터 수 추정:
        embedding: 200 × 8 = 1,600
        LSTM (2 layer): 약 60,000
        FC: 65
        총 약 62,000 (Phase 1.5 의 4,513 의 14 배)

    그러나 sample 도 74 배 ↑ → ratio 개선:
        Phase 1.5: 4,513 / 441 = 10.2:1 ratio
        Cross-sec: 62,000 / 32,634 = 1.9:1 ratio (5 배 향상)

    더 robust 한 학습 가능.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_tickers: int = 200,
        embedding_dim: int = 8,
        output_dropout: float = 0.2,
        embedding_init_std: float = 0.01,    # ⭐ embedding 초기화 std
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_tickers = n_tickers
        self.embedding_dim = embedding_dim

        # Ticker embedding (종목 ID → embedding vector)
        self.ticker_embedding = nn.Embedding(n_tickers, embedding_dim)
        # ⭐ Embedding 초기화 (학술 권장: 작은 std)
        nn.init.normal_(self.ticker_embedding.weight, mean=0.0, std=embedding_init_std)

        # LSTM (input + ticker embedding)
        self.lstm = nn.LSTM(
            input_size=input_size + embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output dropout + FC
        self.output_dropout = nn.Dropout(output_dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F)
            Time series features. B = batch_size, T = seq_len, F = input_size.
        ticker_ids : torch.Tensor, shape (B,)
            Ticker integer IDs (0 ~ n_tickers - 1).
            Must be convertible to LongTensor (auto-converted).

        Returns
        -------
        y_pred : torch.Tensor, shape (B,)
            Predicted log-RV (forward 21d).

        Raises
        ------
        ValueError
            ticker_ids 가 [0, n_tickers) 범위 밖일 때.
        """
        # ⭐ dtype 검증 (auto-cast to long)
        if ticker_ids.dtype != torch.long:
            ticker_ids = ticker_ids.long()

        # ⭐ 범위 검증 (학습 안전성)
        if ticker_ids.numel() > 0:
            if ticker_ids.max().item() >= self.n_tickers or ticker_ids.min().item() < 0:
                raise ValueError(
                    f'ticker_ids 범위 오류: min={ticker_ids.min().item()}, '
                    f'max={ticker_ids.max().item()}, n_tickers={self.n_tickers}'
                )

        B, T, _ = x.shape

        # 1. Ticker embedding lookup
        ticker_emb = self.ticker_embedding(ticker_ids)        # (B, embed_dim)

        # 2. Expand to seq_len dimension
        ticker_emb_seq = ticker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, embed_dim)

        # 3. Concat with time series features
        x_aug = torch.cat([x, ticker_emb_seq], dim=-1)         # (B, T, F + embed_dim)

        # 4. LSTM forward
        lstm_out, _ = self.lstm(x_aug)                          # (B, T, hidden)

        # 5. Take last timestep + FC
        last_hidden = lstm_out[:, -1, :]                       # (B, hidden)
        last_hidden = self.output_dropout(last_hidden)
        y_pred = self.fc(last_hidden).squeeze(-1)              # (B,)

        return y_pred

    def num_parameters(self) -> int:
        """모델의 총 학습 가능한 파라미터 수."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_ticker_embedding_norm(self) -> torch.Tensor:
        """Ticker embedding 의 L2 norm 분포 (학습 진단용).

        학습 진행 시 norm 이 너무 커지면 ticker overfitting 가능성.
        """
        with torch.no_grad():
            return self.ticker_embedding.weight.norm(dim=1)


# =============================================================================
# Cross-Sectional v4 best 하이퍼파라미터 (Phase 3-1 권장)
# =============================================================================
CS_V4_BEST_CONFIG: dict = {
    'input_size': 4,           # rv_d, rv_w, rv_m, vix_log
    'hidden_size': 64,         # Phase 1.5 의 32 보다 크게 (sample 풍부)
    'num_layers': 2,           # 2 layer (cross-sectional 환경)
    'dropout': 0.3,
    'embedding_dim': 8,        # ticker embedding 차원
    'output_dropout': 0.2,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'loss_type': 'mse',
    'huber_delta': 0.01,
    'max_epochs': 50,
    'early_stop_patience': 10,
    'lr_patience': 5,
    'lr_factor': 0.5,
    'batch_size': 256,         # Phase 1.5 의 64 보다 크게 (대규모 sample)
    'is_len': 1250,
    'seq_len': 63,
    'embargo': 63,
    'oos_len': 21,
    'step': 21,
    'window': 21,
    'har_w': 5,
    'har_m': 22,
}
