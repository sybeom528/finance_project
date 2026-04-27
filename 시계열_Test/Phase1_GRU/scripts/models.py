"""Phase 1 — GRU 회귀 모델 정의 모듈.

공개 인터페이스
--------------
GRURegressor                   — PyTorch nn.Module 서브클래스 (시퀀스 → scalar)
count_parameters(model) → int   — 학습 가능 파라미터 수 집계 (보조 유틸)

주요 설계 결정 (학습자료_주의사항.md 참고)
----------------------------------------
1. **batch_first=True 기본값** (§3.3, 02_setting_A_daily21.ipynb §5·§7 명시 요구)
   DataLoader 출력이 ``(B, T, F)`` 이므로, nn.GRU 를 ``batch_first=True`` 로 맞춘다.
   ``output[:, -1, :]`` (마지막 시점 hidden 추출) 은 ``batch_first=True`` 기준
   축 인덱스이다. False 로 전환 시 ``output[-1, :, :]`` 으로 분기된다.

2. **num_layers=1 dropout 함정** (§4.4)
   PyTorch ``nn.GRU`` 의 ``dropout`` 인자는 ``num_layers > 1`` 일 때만 효과를
   가지며, 1층 구조에서 사용하면 warning 이 발생한다.
   본 모듈은 1층일 때 GRU dropout 인자를 0.0 으로 넘기고, 별도
   ``nn.Dropout`` 을 Linear 헤드 앞에 삽입한다.

3. **마지막 시점 hidden 추출** (vs 평균 풀링)
   GRU 는 순방향으로 과거 정보를 누적하므로 마지막 hidden state 가 전체
   시퀀스를 요약한다. 수익률 예측에서 마지막 시점 정보가 가장 최근·유효한
   신호이므로 ``output[:, -1, :]`` 선택.

4. **LSTM 대비 파라미터 수 약 25% 감소**
   GRU 는 리셋(reset)·업데이트(update) 게이트 2개만 사용하므로 LSTM 의
   4게이트(input·forget·cell·output) 대비 파라미터가 적다.
   소규모 훈련셋(n_train ≈ 168 @ IS=231, seq_len=63)에서 과적합 완화 효과를 기대한다.
   (Phase 1 결과분석5.md §6 Option C 근거 — "GRU 용량 감소로 과적합 완화")

5. **forget gate bias 없음 (LSTM 과의 차이)**
   GRU 는 forget gate 가 없으므로 LSTMRegressor 의 ``forget_gate_bias_init``
   인자가 존재하지 않는다. 게이트 편향은 PyTorch 기본값으로 초기화된다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRURegressor(nn.Module):
    """GRU 기반 회귀 모델 (시퀀스 → scalar 예측).

    Parameters
    ----------
    input_size : int
        입력 피처 수. 단변량이면 1, 다변량이면 feature 개수.
    hidden_size : int, default 64
        GRU hidden state 차원.
        LSTM(hidden=32) 과 파라미터 수를 맞추려면 hidden=42 수준이나,
        비교 실험 목적상 기본값은 64 로 설정한다.
    num_layers : int, default 1
        GRU stacking 층수. 2 이상이면 GRU 자체 dropout 사용, 1 이면 head_dropout 사용.
    dropout : float, default 0.3
        Dropout 확률.

        - ``num_layers > 1`` → GRU 내부 층간 dropout 으로 사용
        - ``num_layers == 1`` → 별도 ``nn.Dropout`` 으로 Linear 헤드 앞에 적용
    batch_first : bool, default True
        True 면 입력 shape ``(B, T, F)``, False 면 ``(T, B, F)``.
        DataLoader 기본 출력이 ``(B, T, F)`` 이므로 True 권장.

    Attributes
    ----------
    gru : nn.GRU
        GRU 본체.
    head_dropout : nn.Module
        ``num_layers=1`` 시 ``nn.Dropout(dropout)``, 그 외 ``nn.Identity()``.
        인터페이스 통일을 위해 항상 존재.
    head : nn.Linear
        마지막 시점 hidden → scalar 매핑 (hidden_size → 1).

    Notes
    -----
    GRU 는 LSTM 과 달리 셀 상태(cell state)가 없으며,
    ``forward`` 반환값이 ``(output, h_n)`` 으로 2-튜플이다
    (LSTM 은 ``(output, (h_n, c_n))`` 으로 3-튜플).
    본 모듈은 ``output[:, -1, :]`` 만 사용하므로 차이가 없다.

    Examples
    --------
    설정 A (num_layers=1, dropout=0.3):

    >>> model = GRURegressor(input_size=1, hidden_size=64, num_layers=1, dropout=0.3)
    >>> x = torch.randn(32, 63, 1)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32])

    설정 B (num_layers=2, dropout=0.2) — 2층 dropout 이 GRU 내부에 적용:

    >>> model_b = GRURegressor(1, hidden_size=64, num_layers=2, dropout=0.2)
    >>> isinstance(model_b.head_dropout, nn.Identity)
    True
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.batch_first = batch_first

        # num_layers=1 시 GRU dropout 인자는 무시됨 → 0.0 넘기고 head_dropout 으로 대체
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=batch_first,
        )
        # num_layers=1 에서만 유효한 dropout, 그 외 층은 nn.Identity 로 통일
        self.head_dropout = (
            nn.Dropout(dropout) if num_layers == 1 else nn.Identity()
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GRU 통과 후 마지막 시점 hidden → scalar 회귀.

        Parameters
        ----------
        x : torch.Tensor
            ``batch_first=True``  : shape ``(B, T, F)``
            ``batch_first=False`` : shape ``(T, B, F)``

        Returns
        -------
        torch.Tensor
            shape ``(B,)`` — 각 시퀀스별 예측 scalar.

        Notes
        -----
        GRU forward 반환: ``(output, h_n)``
        LSTM 과 달리 셀 상태(c_n)가 없으므로 ``_`` 언패킹 시 h_n 만 무시됨.
        """
        out, _ = self.gru(x)                        # (B, T, H) 또는 (T, B, H)
        if self.batch_first:
            last = out[:, -1, :]                    # batch_first=True 기준 마지막 시점
        else:
            last = out[-1, :, :]                    # batch_first=False 기준 마지막 시점
        last = self.head_dropout(last)              # num_layers=1 시에만 실제 drop
        return self.head(last).squeeze(-1)          # (B, 1) → (B,)


def count_parameters(model: nn.Module) -> int:
    """학습 가능 파라미터 수 집계.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
        ``sum(p.numel() for p in model.parameters() if p.requires_grad)``.

    Examples
    --------
    >>> model = GRURegressor(1, 64, 1)
    >>> count_parameters(model) > 0
    True
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
