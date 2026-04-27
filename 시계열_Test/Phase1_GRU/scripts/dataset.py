"""Phase 1 — GRU 학습용 텐서 데이터셋 유틸리티.

공개 인터페이스
--------------
SequenceDataset                                              — torch.utils.data.Dataset 서브클래스
make_sequences(arr, seq_len, horizon)                        → (X, y)
walk_forward_folds(n, is_len, purge, emb, oos_len, step)    → List[(train_idx, test_idx)]
build_fold_datasets(series, train_idx, test_idx,            → (train_ds, test_ds, scaler)
                    seq_len, extra_features,
                    target_series)

Notes
-----
Phase1_LSTM 의 ``LSTMDataset`` 을 ``SequenceDataset`` 으로 이름 변경.
데이터 파이프라인 자체는 모델에 무관하므로 로직 변경 없음.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """(X, y) 쌍을 보관하는 PyTorch Dataset.

    Parameters
    ----------
    X : np.ndarray, shape (N, seq_len, n_features)
    y : np.ndarray, shape (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_sequences(
    arr: np.ndarray,
    seq_len: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """1D / 2D 배열에서 슬라이딩 윈도우 (X, y) 쌍을 생성한다.

    Parameters
    ----------
    arr : np.ndarray, shape (T,) or (T, n_features)
        입력 시계열. 1D면 단변량.
    seq_len : int
        GRU 입력 시퀀스 길이 (타임스텝 수).
    horizon : int
        예측 대상 시점 오프셋. 기본값 1 = 다음 날.

    Returns
    -------
    X : np.ndarray, shape (N, seq_len, n_features)
    y : np.ndarray, shape (N,)
        첫 번째 피처([:, 0]) 기준 t+horizon 값.
        N = T − seq_len − horizon + 1.

    Raises
    ------
    ValueError
        T < seq_len + horizon 인 경우.
    """
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    T, n_feat = arr.shape
    N = T - seq_len - horizon + 1
    if N <= 0:
        raise ValueError(
            f"시계열 길이 T={T}이 seq_len({seq_len}) + horizon({horizon})보다 짧습니다."
        )
    X = np.stack([arr[i : i + seq_len] for i in range(N)])
    y = arr[seq_len + horizon - 1 : seq_len + horizon - 1 + N, 0]
    return X, y


def walk_forward_folds(
    n: int,
    is_len: int,
    purge: int,
    emb: int,
    oos_len: int,
    step: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Walk-Forward 교차검증 폴드 인덱스를 생성한다.

    각 폴드 구조::

        [  train (is_len)  ][ purge ][ emb ][ test (oos_len) ]
        start              train_end         test_start       test_end

    Parameters
    ----------
    n : int
        전체 시계열 길이.
    is_len : int
        훈련 구간(In-Sample) 길이.
    purge, emb : int
        훈련~테스트 사이 제거·엠바고 기간 (레이블 겹침 누수 방지).
    oos_len : int
        테스트 구간(Out-Of-Sample) 길이.
    step : int
        폴드 간 슬라이딩 스텝.

    Returns
    -------
    List of (train_idx, test_idx)
        각각 numpy.arange 배열. 길이 고정 (is_len, oos_len).
    """
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while True:
        train_end = start + is_len
        test_start = train_end + purge + emb
        test_end = test_start + oos_len
        if test_end > n:
            break
        folds.append((
            np.arange(start, train_end),
            np.arange(test_start, test_end),
        ))
        start += step
    return folds


def build_fold_datasets(
    series: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seq_len: int,
    extra_features: Optional[np.ndarray] = None,
    target_series: Optional[np.ndarray] = None,
) -> Tuple[SequenceDataset, SequenceDataset, StandardScaler]:
    """Walk-Forward 폴드 하나의 SequenceDataset 쌍을 생성한다.

    누수 방지
    ---------
    StandardScaler 는 train_idx 범위 데이터로만 fit() 한다.
    테스트 구간 transform 에는 fit 없이 동일 파라미터를 적용한다.

    테스트 시퀀스 설계
    ------------------
    각 t ∈ test_idx 에 대해::

        X_te = scaled[t − seq_len : t]   # (seq_len, n_features)
        y_te = target_series[t]          # 외부 타깃 (미리 계산된 누적 수익률 등)
               또는 scaled[t, 0]         # target_series=None 시 당일 주 피처 값

    입력 구간이 purge/embargo 기간과 겹쳐도 레이블이 아닌 입력 피처이므로 허용된다.
    이 설계로 test_idx 내 모든 날짜에 대해 예측 샘플이 생성된다.

    target_series 와 타깃 정렬
    --------------------------
    target_series 를 제공할 경우 (Setting A — 21일 누적 수익률 등):
    - 훈련: y_tr[j] = target_series[train_idx[j + seq_len − 1]]
              = j 번째 시퀀스의 마지막 날짜 t 의 타깃 (t+1~t+21 누적 수익률)
    - 테스트: y_te[k] = target_series[test_idx[k]]
    target_series 는 호출 전에 NaN 처리 및 스케일링을 완료해야 한다.

    Parameters
    ----------
    series : np.ndarray, shape (T,)
        주 입력 피처 (log_return).
    train_idx, test_idx : np.ndarray
        walk_forward_folds() 반환값.
    seq_len : int
        GRU 입력 시퀀스 길이.
    extra_features : np.ndarray, shape (T, k), optional
        추가 입력 피처. series 와 열 결합 후 스케일.
        예: np.column_stack([qqq_lr, volume_norm])
    target_series : np.ndarray, shape (T,), optional
        외부 타깃 시계열 (예: build_daily_target_21d() 반환값의 .values).
        None 이면 기존 동작 유지 — scaled 주 피처를 타깃으로 사용.

    Returns
    -------
    train_ds : SequenceDataset
        X.shape = (N_train, seq_len, n_features),  N_train = is_len − seq_len
    test_ds : SequenceDataset
        X.shape = (N_test, seq_len, n_features),   N_test = oos_len
    scaler : StandardScaler
        입력 피처 역변환 및 재사용을 위해 반환.
    """
    data = series[:, np.newaxis]
    if extra_features is not None:
        data = np.column_stack([series, extra_features])

    scaler = StandardScaler()
    scaler.fit(data[train_idx])          # 훈련 구간으로만 fit — 누수 방지
    scaled = scaler.transform(data)      # 전체 시계열 transform (테스트 맥락 포함)

    X_tr, y_tr_default = make_sequences(scaled[train_idx], seq_len)

    if target_series is not None:
        # 외부 타깃 사용: j 번째 시퀀스의 마지막 위치 = train_idx[j + seq_len - 1]
        n_train = len(X_tr)
        y_tr = np.array([
            float(target_series[train_idx[j + seq_len - 1]])
            for j in range(n_train)
        ])
    else:
        y_tr = y_tr_default

    X_te_list: List[np.ndarray] = []
    y_te_list: List[float] = []
    for k, t in enumerate(test_idx):
        if t < seq_len:
            continue  # 충분한 이력 없음 (초반 폴드 방어)
        X_te_list.append(scaled[t - seq_len : t])
        if target_series is not None:
            y_te_list.append(float(target_series[t]))
        else:
            y_te_list.append(float(scaled[t, 0]))

    X_te = np.stack(X_te_list)
    y_te = np.array(y_te_list)

    return SequenceDataset(X_tr, y_tr), SequenceDataset(X_te, y_te), scaler
