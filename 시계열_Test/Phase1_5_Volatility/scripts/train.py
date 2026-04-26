"""Phase 1 — LSTM 학습 루프 모듈.

공개 인터페이스
--------------
train_one_fold(model, train_loader, val_loader, **hp)  → Dict[str, Any]
get_device(preference='auto')                          → torch.device
save_checkpoint(state_dict, path)                      → None
load_checkpoint(path, device='cpu')                    → Dict

학습 구성 (PLAN.md 확정 사양)
-----------------------------
- Loss          : nn.HuberLoss(delta=0.01)   — 수익률 outlier 강건
- Optimizer     : AdamW(lr=1e-3, weight_decay=1e-4)
- Scheduler     : ReduceLROnPlateau(mode='min', patience=5, factor=0.5)
- Gradient clip : max_norm=1.0
- EarlyStop     : best val loss 기준 patience=10
- Checkpoint    : best state_dict in-memory 보관 + 외부 ``save_checkpoint`` 로 디스크 저장

학습자료_주의사항.md 함정 방어 체크리스트
-----------------------------------------
§3.6 — PyTorch 학습 루프 흔한 실수 10가지
    ✓ model.train() / model.eval() 매 단계 명시 전환
    ✓ optimizer.zero_grad() 매 배치 호출
    ✓ val 단계 torch.no_grad() 컨텍스트
    ✓ loss.item() 으로 그래프 detach (리스트 축적 시 메모리 누수 방지)
    ✓ gradient clipping (LSTM gradient explosion 방지)
§4.4 — LSTM num_layers=1 dropout 함정 (models.py 에서 처리됨)
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Device 유틸
# ---------------------------------------------------------------------------
def get_device(preference: str = 'auto') -> torch.device:
    """학습에 사용할 torch.device 를 결정한다.

    Parameters
    ----------
    preference : str
        ``'auto'``  : cuda > mps > cpu 순 자동 선택
        ``'cuda'`` / ``'mps'`` / ``'cpu'`` : 명시 선택 (가용하지 않으면 실패)

    Returns
    -------
    torch.device
    """
    if preference != 'auto':
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device('cuda')
    # torch.backends.mps 는 구버전 PyTorch 에는 존재하지 않을 수 있음
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    path: Union[str, Path],
) -> None:
    """state_dict 를 디스크에 저장 (부모 디렉토리 자동 생성).

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        ``model.state_dict()`` 또는 ``train_one_fold`` 의 ``best_state_dict``.
    path : str or Path
        저장 경로 (확장자 .pt 권장).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, p)


def load_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
) -> Dict[str, torch.Tensor]:
    """state_dict 를 디스크에서 로드.

    Parameters
    ----------
    path : str or Path
    device : str or torch.device, default 'cpu'
        ``map_location`` 에 전달.

    Returns
    -------
    Dict[str, torch.Tensor]
        ``model.load_state_dict(...)`` 로 바로 사용 가능.
    """
    return torch.load(Path(path), map_location=device)


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------
def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    max_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 0.01,
    grad_clip: float = 1.0,
    early_stop_patience: int = 10,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    device: Union[str, torch.device] = 'auto',
    verbose: bool = True,
    log_every: int = 1,
) -> Dict[str, Any]:
    """단일 fold 에 대한 LSTM 학습 루프.

    Parameters
    ----------
    model : nn.Module
        학습할 모델 (``LSTMRegressor`` 등). 함수 내부에서 ``.to(device)`` 호출.
    train_loader, val_loader : DataLoader
        훈련·검증 DataLoader. 각 배치는 ``(X, y)`` 튜플.
    max_epochs : int, default 100
        최대 학습 epoch.
    lr : float, default 1e-3
        초기 learning rate.
    weight_decay : float, default 1e-4
        AdamW L2 정규화 강도.
    huber_delta : float, default 0.01
        HuberLoss delta (L1/L2 전환점). 일별 수익률 ~1% 스케일 기준.
    grad_clip : float, default 1.0
        Gradient clipping ``max_norm``. 0 또는 None 이면 clip 건너뜀.
    early_stop_patience : int, default 10
        best val loss 갱신 없는 epoch 허용 한계. 초과 시 학습 중단.
    lr_patience : int, default 5
        ReduceLROnPlateau patience.
    lr_factor : float, default 0.5
        ReduceLROnPlateau factor.
    device : str or torch.device, default 'auto'
        학습 장치. ``'auto'`` 면 cuda > mps > cpu.
    verbose : bool, default True
        epoch 단위 진행률 print.
    log_every : int, default 1
        verbose=True 시 몇 epoch 마다 print 할지 (1 = 매 epoch).

    Returns
    -------
    Dict[str, Any]

        - ``best_state_dict`` : 최저 val loss 시점의 ``state_dict`` (deepcopy).
        - ``history`` : dict with keys ``train_loss`` / ``val_loss`` / ``lr``
          (모두 list[float], 길이 = 실제 학습 epoch 수).
        - ``best_epoch`` : int (1-indexed).
        - ``best_val_loss`` : float.
        - ``stopped_early`` : bool — EarlyStopping 으로 중단되었으면 True.

    Notes
    -----
    학습자료_주의사항.md §3.6 함정 방어:
    - model.train()/eval() 명시, optimizer.zero_grad() 매 배치,
      val 단계 torch.no_grad(), loss.item() detach, clip_grad_norm_.
    """
    dev = get_device(device) if isinstance(device, str) else device
    model.to(dev)

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=lr_patience, factor=lr_factor,
    )

    history: Dict[str, list] = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val = float('inf')
    best_state: Dict[str, torch.Tensor] = {}
    best_epoch = 0
    patience_counter = 0
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        # ----- train phase -----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad()                                   # 함정: zero_grad 누락 금지
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_losses.append(loss.item())                        # 함정: item() 으로 그래프 detach
        train_loss = float(np.mean(train_losses))

        # ----- val phase -----
        model.eval()                                                # 함정: eval 모드 전환 필수
        val_losses = []
        with torch.no_grad():                                       # 함정: no_grad 로 메모리 절약
            for xb, yb in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # ----- best checkpoint 갱신 -----
        improved = val_loss < best_val - 1e-8
        if improved:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % log_every == 0 or improved):
            flag = '  * best' if improved else ''
            print(
                f'[ep {epoch:3d}] train={train_loss:.6f}  '
                f'val={val_loss:.6f}  lr={current_lr:.2e}{flag}'
            )

        if patience_counter >= early_stop_patience:
            stopped_early = True
            if verbose:
                print(
                    f'  -> EarlyStopping at epoch {epoch} '
                    f'(patience {early_stop_patience} exceeded, '
                    f'best epoch={best_epoch}, best val={best_val:.6f})'
                )
            break

    return {
        'best_state_dict': best_state,
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'stopped_early': stopped_early,
    }
