# Black-Litterman 실험 프레임워크 — 상세 가이드

---

## 1. 전체 구조

```
final/
├── bl_config.py          ← 실험 정의 (여기만 수정해서 실험 추가)
├── bl_functions.py       ← BL 수식 핵심 함수 (웬만하면 건드리지 않음)
├── 99_run.ipynb          ← 실험 실행 → results/*.pkl 저장
├── 99_analyze.ipynb      ← 결과 비교 분석 + 차트
│
├── results/              ← 실험 결과 (자동 생성)
│   └── {실험명}.pkl
├── data/                 ← 입력 데이터
│   ├── monthly_panel.csv
│   ├── daily_returns.pkl
│   └── ff3_monthly.csv
└── phase3(data_outputs)/ ← Phase3 LSTM 예측 결과 (재천)
    └── data/
        └── ensemble_predictions_stockwise.csv
```

---

## 2. 실험 실행 순서

```
① bl_config.py에 실험 추가 (필요 시)
② 99_run.ipynb 셀 순서대로 실행
   cell-00 : 패키지 임포트 + 경로 설정
   cell-01 : 데이터 로드 (monthly_panel, daily_returns, FF3)
   cell-02 : LSTM 예측 로드 (phase3(data_outputs)/data/ 에서 자동 탐색)
   cell-03 : Dispatcher 함수 정의 (get_vol_series, get_Q, get_omega 등)
   cell-04 : walk_forward() 함수 정의
   cell-05 : 전체 실험 실행 → results/*.pkl 저장
   cell-06 : 빠른 성과 확인
③ 99_analyze.ipynb 실행 → 비교 차트 + 성과 테이블
```

---

## 3. 실험 추가 방법

### Case 1 — 기존 슬롯 파라미터만 바꿀 때 (가장 흔한 경우)

`bl_config.py`의 `EXPERIMENTS` 리스트에 dict 한 줄 추가. **다른 파일은 건드리지 않아도 됨.**

```python
# bl_config.py

EXPERIMENTS = [
    BASELINE,
    ...
    # ── 새로 추가하는 실험 ──────────────────────────────────────
    {**BASELINE, 'name': 'q_0005', 'q_value': 0.005},          # Q값만 변경
    {**BASELINE, 'name': 'no_tc',  'tc': 0.0},                 # 거래비용 제거
    {**BASELINE, 'name': 'my_exp', 'p_mode': 'trailing_vol252', 'p_weight': 'rp'},  # 조합
]
```

`{**BASELINE, ...}` 패턴이 핵심 — BASELINE의 모든 값을 상속받고 바꾸고 싶은 슬롯만 덮어씀.

> **주의**: `name`은 반드시 지정. `results/{name}.pkl`로 저장됨.

---

### Case 2 — 새 계산 방식을 추가할 때

3개 파일을 **이 순서대로** 수정:

#### ① `bl_functions.py` — 새 함수 추가

```python
# 예: 새 Q 계산 방식 추가
def compute_Q_my_method(P, ...) -> float:
    ...
    return float(q)
```

상단 import에서 함수 공개하지 않아도 됨 — 99_run.ipynb에서 직접 호출.

#### ② `bl_config.py` — 슬롯 주석 + 실험 dict 추가

```python
# 슬롯 주석에 새 선택지 추가
#   q_mode : 'fixed' | 'ff3_regression' | ... | 'my_method'  ← 추가

{**BASELINE, 'name': 'q_my', 'q_mode': 'my_method'},
```

#### ③ `99_run.ipynb` — cell-03 dispatcher에 elif 추가

```python
def get_Q(cfg, P, valid_tix, train_dates, pred_date, all_d):
    mode = cfg.get('q_mode', 'fixed')
    ...
    elif mode == 'my_method':          # ← 추가
        return compute_Q_my_method(P, ...)
```

---

## 4. 슬롯 키 레퍼런스

| 슬롯 | 선택지 | 기본값 | 설명 |
|---|---|---|---|
| `p_mode` | `trailing_vol21` / `trailing_vol252` / `lstm_predicted` | `trailing_vol21` | P 행렬 분류 기준 변동성 |
| `p_weight` | `mcap` / `eq` / `rp` / `vol_mcap` | `mcap` | P 행렬 가중 방식 |
| `q_mode` | `fixed` / `ff3_regression` / `realized_spread` / `regime` / `none` / `capm` | `fixed` | Q 추정 방식 |
| `q_value` | float | `0.003` | `q_mode='fixed'`일 때 Q값 (월 기준) |
| `q_regime_table` | dict | — | `q_mode='regime'`일 때 레짐별 Q값 |
| `omega_mode` | `he_litterman` / `scaled` / `rmse` | `he_litterman` | Omega 계산 방식 |
| `omega_scale` | float | `1.0` | `omega_mode='scaled'`일 때 배수 |
| `prior` | `capm_mcap` / `capm_eq` | `capm_mcap` | 시장 Prior 가중 방식 |
| `tc` | float | `0.001` | 편도 거래비용 (10bp = 0.001) |
| `max_weight` | float | `0.10` | 단일 종목 최대 비중 |
| `lstm_pred_path` | str | Phase3 경로 자동 탐색 | LSTM 예측 파일 경로 |

---

### `p_mode` 상세

| 값 | 사용하는 변동성 | 설명 |
|---|---|---|
| `trailing_vol21` | `vol_21d` (과거 21일 실현변동성) | 기본. look-ahead bias 없음 |
| `trailing_vol252` | `vol_252d` (과거 252일 실현변동성) | 장기 안정적 신호 |
| `lstm_predicted` | Phase3 LSTM+HAR 앙상블 예측 변동성 | `phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` 필요 |

---

### `p_weight` 상세

| 값 | Long (저변동 그룹) | Short (고변동 그룹) | 유니버스 |
|---|---|---|---|
| `mcap` | 시총 비례 | 시총 비례 | 하위/상위 30% |
| `eq` | 동일가중 | 동일가중 | 하위/상위 30% |
| `rp` | 1/σ 역변동성 | 1/σ 역변동성 | 하위/상위 30% |
| `vol_mcap` | (1/σ)×mcap | σ×mcap | 전체 유니버스 |

> `rp`는 Pyo & Lee (2018) 방식 — 30% 그룹 선별 후 그 안에서 역변동성 가중.

---

### `q_mode` 상세

| 값 | Q 결정 방식 | 특이사항 |
|---|---|---|
| `fixed` | `q_value` 고정값 | 기본. 가장 단순 |
| `ff3_regression` | FF3 회귀로 종목별 기대수익률 추정 → P@r_hat | 훈련 구간 최소 24개월 필요 |
| `realized_spread` | 훈련 구간 저변동-고변동 수익률 스프레드 평균 | look-ahead bias 없음 |
| `regime` | SPY 변동성 레짐별 Q (low/normal/high_vol) | `q_regime_table` dict 필요 |
| `none` | Q 없음 — BL 스킵, 직접 보유 | naive 비교군용 |
| `capm` | BL 없음 — CAPM prior π로 직접 최적화 | 전체 유니버스 대상 |

---

### `omega_mode` 상세

| 값 | Omega 결정 방식 |
|---|---|
| `he_litterman` | τ·P·Σ·P^T (He-Litterman 1999 표준) |
| `scaled` | he_litterman × `omega_scale` (scale<1: 뷰 더 신뢰, scale>1: 덜 신뢰) |
| `rmse` | LSTM 예측 RMSE 기반 스케일링 (RMSE 높을수록 Omega 증가) |

---

## 5. 현재 실험 목록

### Trailing 변동성 실험

| 실험명 | p_mode | p_weight | prior | q_mode |
|---|---|---|---|---|
| `baseline` | vol21 | mcap | capm_mcap | fixed |
| `prior_eq` | vol21 | mcap | **capm_eq** | fixed |
| `p_vol252` | **vol252** | mcap | capm_mcap | fixed |
| `p_rp` | vol21 | **rp** | capm_mcap | fixed |
| `p_eq` | vol21 | **eq** | capm_mcap | fixed |
| `p_vol_mcap` | vol21 | **vol_mcap** | capm_mcap | fixed |

### LSTM 예측 변동성 실험 (Phase3 앙상블)

| 실험명 | p_mode | p_weight | prior |
|---|---|---|---|
| `p_lstm_mcap` | lstm | mcap | capm_mcap |
| `p_lstm_eq` | lstm | eq | capm_mcap |
| `p_lstm_rp` | lstm | rp | capm_mcap |
| `p_lstm_vol_mcap` | lstm | vol_mcap | capm_mcap |
| `prior_eq_p_lstm_mcap` | lstm | mcap | **capm_eq** |
| `prior_eq_p_lstm_eq` | lstm | eq | **capm_eq** |
| `prior_eq_p_lstm_rp` | lstm | rp | **capm_eq** |
| `prior_eq_p_lstm_vol_mcap` | lstm | vol_mcap | **capm_eq** |

### 비교군 (BL 없음)

| 실험명 | 설명 |
|---|---|
| `capm_no_bl` | BL 없음 — CAPM prior π로 전체 유니버스 최적화 |
| `naive_lowvol` | BL 없음 — 저변동 하위 30% 시총가중 직접 보유 |
| `naive_lowvol_rp` | BL 없음 — 저변동 하위 30% 역변동성 가중 직접 보유 |

---

## 6. LSTM 실험 전제조건

`p_mode='lstm_predicted'` 실험은 `phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` 파일이 있어야 실행됨.

**없으면 자동 스킵** (경고 메시지 출력 후 넘어감 — 에러 아님).

파일 구조:
```
date        ticker  y_pred_lstm  y_pred_har  y_pred_ensemble  y_true  ...
2007-04-23  AAPL    -4.12        -4.36       -4.24            -3.99
2007-04-23  MSFT    ...
```

`y_pred_ensemble` (log-RV 스케일) → `np.exp()` → 실제 변동성 → 월말 기준 종목 랭킹.

---

## 7. 결과 파일 구조

`results/{실험명}.pkl`에 저장되는 dict:

| 키 | 타입 | 내용 |
|---|---|---|
| `config` | dict | 이 실험의 bl_config 설정값 전체 |
| `ret` | pd.Series | 월별 순수익률 (거래비용 차감 후) |
| `gross_ret` | pd.Series | 월별 총수익률 (거래비용 전) |
| `spy_ret` | pd.Series | 월별 SPY 수익률 |
| `weights` | pd.DataFrame | 월별 × 종목 포트폴리오 가중치 |
| `comp` | pd.DataFrame | 월별 구성 지표 (eff_n, top10_share, avg_vol, turnover 등) |
| `meta` | pd.DataFrame | 월별 Q값, lambda 등 실험 메타데이터 |
| `errors` | list | 에러 발생 월 목록 |

---

## 8. 재실행 방법

`99_run.ipynb` cell-05의 `SKIP_IF_EXISTS = True`가 기본 → 이미 저장된 실험은 자동 스킵.

특정 실험 재실행하려면:
```bash
# 해당 pkl 삭제 후 99_run.ipynb 재실행
del final/results/{실험명}.pkl
```

전체 재실행:
```python
SKIP_IF_EXISTS = False  # cell-05에서 변경
```

---

## 9. 주의사항

| 항목 | 내용 |
|---|---|
| **Look-ahead bias** | `fwd_ret_1m`은 성과 평가 전용. BL 입력(P/Q 계산)에 절대 사용 금지 |
| **LSTM 학습** | `ensemble_predictions_stockwise.csv`는 Phase3에서 walk-forward로 학습된 결과. 재학습 시 Phase3/02a_v2.ipynb 실행 (GPU + 수 시간 소요) |
| **rp 가중방식** | Pyo & Lee (2018) 방식 — 30% 선별 후 그룹 내 역변동성 가중. Phase3 구현과 동일 |
| **vol_mcap 가중방식** | 전체 유니버스 대상 (30% 컷 없음). Phase3에 없는 방식 |
| **거래비용 단위** | `tc=0.001` = 편도 10bp. 월 TC 비용 = turnover × tc |
| **데이터 선행** | `01_DataCollection.ipynb` 실행 후 `data/` 폴더가 채워진 상태에서 실행 |
