"""Phase 1 — LSTM 회귀 모델 정의 모듈.

공개 인터페이스
--------------
LSTMRegressor                   — PyTorch nn.Module 서브클래스 (시퀀스 → scalar)
count_parameters(model) → int   — 학습 가능 파라미터 수 집계 (보조 유틸)

주요 설계 결정 (학습자료_주의사항.md 참고)
----------------------------------------
1. **batch_first=True 기본값** (§3.3, 02_setting_A_daily21.ipynb §5·§7 명시 요구)
   DataLoader 출력이 ``(B, T, F)`` 이므로, nn.LSTM 을 ``batch_first=True`` 로 맞춘다.
   ``output[:, -1, :]`` (마지막 시점 hidden 추출) 은 ``batch_first=True`` 기준
   축 인덱스이다. False 로 전환 시 ``output[-1, :, :]`` 으로 분기된다.

2. **num_layers=1 dropout 함정** (§4.4)
   PyTorch ``nn.LSTM`` 의 ``dropout`` 인자는 ``num_layers > 1`` 일 때만 효과를
   가지며, 1층 구조에서 사용하면 warning 이 발생한다.
   본 모듈은 1층일 때 LSTM dropout 인자를 0.0 으로 넘기고, 별도
   ``nn.Dropout`` 을 Linear 헤드 앞에 삽입한다.

3. **마지막 시점 hidden 추출** (vs 평균 풀링)
   LSTM 은 순방향으로 과거 정보를 누적하므로 마지막 hidden state 가 전체
   시퀀스를 요약한다. 수익률 예측에서 마지막 시점 정보가 가장 최근·유효한
   신호이므로 ``output[:, -1, :]`` 선택.

4. **Forget gate bias = 1 초기화** (학습자료 02_LSTM_게이트메커니즘.md §6)
   PyTorch 기본은 모든 bias 를 0 으로 초기화 → ``f_t = sigmoid(0) = 0.5``.
   매 시점 정보의 절반이 지워져 학습 초기부터 vanishing 이 발생한다.
   ``b_f = 1`` 로 시작하면 ``f_t = sigmoid(1) ~ 0.73`` 으로 정보 대부분이 유지되며,
   금융 시계열의 장기 의존성 학습에 유리하다 (Jozefowicz et al., 2015).
   PyTorch nn.LSTM bias 순서는 ``[i, f, g, o]`` (각 hidden_size) 이므로
   ``[H : 2H]`` 구간이 forget gate bias 이다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """LSTM 기반 회귀 모델 (시퀀스 → scalar 예측).

    Parameters
    ----------
    input_size : int
        입력 피처 수. 단변량이면 1, 다변량이면 feature 개수.
    hidden_size : int, default 128
        LSTM hidden state 차원.
    num_layers : int, default 2
        LSTM stacking 층수. 2 이상이면 LSTM 자체 dropout 사용, 1 이면 head_dropout 사용.
    dropout : float, default 0.2
        Dropout 확률.

        - ``num_layers > 1`` → LSTM 내부 층간 dropout 으로 사용
        - ``num_layers == 1`` → 별도 ``nn.Dropout`` 으로 Linear 헤드 앞에 적용
    batch_first : bool, default True
        True 면 입력 shape ``(B, T, F)``, False 면 ``(T, B, F)``.
        DataLoader 기본 출력이 ``(B, T, F)`` 이므로 True 권장.

    Attributes
    ----------
    lstm : nn.LSTM
        LSTM 본체.
    head_dropout : nn.Module
        ``num_layers=1`` 시 ``nn.Dropout(dropout)``, 그 외 ``nn.Identity()``.
        인터페이스 통일을 위해 항상 존재.
    head : nn.Linear
        마지막 시점 hidden → scalar 매핑 (hidden_size → 1).

    Examples
    --------
    설정 A (num_layers=2, dropout=0.2):

    >>> model = LSTMRegressor(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
    >>> x = torch.randn(32, 126, 1)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32])

    설정 B (num_layers=1, dropout=0.3) — 1층 dropout 이 head_dropout 으로 적용:

    >>> model_b = LSTMRegressor(1, hidden_size=64, num_layers=1, dropout=0.3)
    >>> isinstance(model_b.head_dropout, nn.Dropout)
    True
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.batch_first = batch_first

        # num_layers=1 시 LSTM dropout 인자는 무시됨 → 0.0 넘기고 head_dropout 으로 대체
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=batch_first,
        )
        # num_layers=1 에서만 유효한 dropout, 그 외 층은 nn.Identity 로 통일
        self.head_dropout = (
            nn.Dropout(dropout) if num_layers == 1 else nn.Identity()
        )
        self.head = nn.Linear(hidden_size, 1)

        # Forget gate bias = 1 초기화 (학습자료 §6, Jozefowicz et al. 2015)
        # PyTorch LSTM bias 레이아웃: [b_ii | b_if | b_ig | b_io], 각 길이 hidden_size.
        # b_ih_l*, b_hh_l* 양쪽 모두 동일 레이아웃이므로 두 bias 모두 forget 구간을 1 로 채움.
        self._init_forget_gate_bias(value=1.0)

    def _init_forget_gate_bias(self, value: float = 1.0) -> None:
        """LSTM forget gate bias 를 ``value`` 로 초기화.

        Notes
        -----
        - ``nn.LSTM`` 은 layer 마다 ``bias_ih_l{k}``, ``bias_hh_l{k}`` 를 가지며
          각각 ``[i, f, g, o]`` 4구간이 hidden_size 길이로 concat 된 형태.
        - forget gate 는 두 번째 구간 → 슬라이스 ``[H : 2H]``.
        - ``num_layers > 1`` 시 모든 layer 에 동일 적용.
        """
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)              # = 4 * hidden_size
                hs = n // 4                    # forget gate 구간 길이
                with torch.no_grad():
                    param[hs : 2 * hs].fill_(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM 통과 후 마지막 시점 hidden → scalar 회귀.

        Parameters
        ----------
        x : torch.Tensor
            ``batch_first=True``  : shape ``(B, T, F)``
            ``batch_first=False`` : shape ``(T, B, F)``

        Returns
        -------
        torch.Tensor
            shape ``(B,)`` — 각 시퀀스별 예측 scalar.
        """
        out, _ = self.lstm(x)                       # (B, T, H) 또는 (T, B, H)
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
    >>> model = LSTMRegressor(1, 128, 2)
    >>> count_parameters(model) > 0
    True
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
