"""1회용 빌드 스크립트 — 01_volatility_eda.ipynb 생성.

이 스크립트는 노트북을 nbformat 으로 빌드하여 저장한다. 노트북 셀 자체를
직접 편집하기보다 본 스크립트를 갱신 후 재실행하는 흐름이 권장된다.

사용법:
    python _build_01_eda_nb.py

저장 위치: 같은 폴더의 ``01_volatility_eda.ipynb`` (덮어쓰기).
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

NB = nbf.v4.new_notebook()
NB.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
}

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ============================================================================
# 노트북 헤더
# ============================================================================
md("""# Phase 1.5 — 변동성 EDA · 통계 검정 (`01_volatility_eda.ipynb`)

> **목적**: Phase 1.5 본 실험(`02_volatility_lstm.ipynb`) 진입 전, **변동성 타깃의 통계적 성질**을 검증하여 모델 설정의 정합성을 사전에 확보합니다.
>
> **작성자**: 재천 / **데이터**: SPY · QQQ (Phase 1 raw_data 사본 재사용) / **분석 기간**: 2016-01-01 ~ 2025-12-31

## 본 노트북의 핵심 검증 항목

| § | 항목 | 의도 |
|---|---|---|
| §1 | 환경 부트스트랩 | 한글 폰트·시드·경로 (OS 호환) |
| §2 | 데이터 로드 | Phase 1 raw_data CSV 재사용 |
| §3 | 시계열 시각화 (log_ret · log_ret² · \\|log_ret\\|) | 변동성 클러스터링 육안 확인 |
| §4 | **RV 계산 + 분포 진단** | log 변환 채택 정당화 (정규성 개선) |
| §5 | **ACF/PACF on log(RV)** | embargo=21 충분성 검증 |
| §6 | **ADF/KPSS 정상성 검정 on log(RV)** | LSTM 입력 전제 검증 |
| §7 | 체제 진단 (저변동/고변동 분위수) | 시장 체제 변화 시각화 |
| §8 | **타깃 누수 검증** (assert + 5행 표) | `shift(-21)` 부호·`log` 함정 차단 |
| §9 | 종합 결론 | 사용자 체크포인트 |

## 사용자 체크포인트
- §4 → Log-RV 채택 합의 (분포 결과 보고)
- §5 → embargo=21 충분성 합의 (ACF lag 1~30 결과)
- §8 → 누수 검증 PASS 확인
""")

# ============================================================================
# §1 환경 부트스트랩
# ============================================================================
md("""## §1. 환경 부트스트랩

`scripts/setup.py` 의 `bootstrap()` 한 줄로 한글 폰트·시드·경로·표시옵션을 일괄 설정합니다.

**왜 별도 모듈인가?**: Phase 1 과 동일한 환경 설정을 모든 노트북에서 재사용하기 위함입니다 (single source of truth). Phase 1.5 의 `scripts/setup.py` 는 Phase 1 사본이므로 동일 동작입니다.

**OS 호환**: Windows(Malgun Gothic) / macOS(AppleGothic) / Linux(NanumGothic) 자동 분기. Linux 는 `koreanize-matplotlib` 패키지 필요.
""")
code("""# scripts/ 가 import path 에 없으면 추가 (노트북 cwd 가 Phase1_5_Volatility 기준)
import sys
from pathlib import Path
NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

# 환경 부트스트랩 — 한글 폰트, 시드(=42), 경로 상수, pandas/matplotlib 표시 옵션
from scripts.setup import bootstrap, BASE_DIR, RAW_DATA_DIR, RESULTS_DIR
font_used = bootstrap()
print(f"BASE_DIR     = {BASE_DIR}")
print(f"RAW_DATA_DIR = {RAW_DATA_DIR}")
""")

# ============================================================================
# §2 데이터 로드
# ============================================================================
md("""## §2. 데이터 로드

Phase 1 의 `results/raw_data/{SPY,QQQ}.csv` 를 본 분기에 그대로 복사하여 보유 중입니다 (Step 0 작업). 따라서 **재다운로드 불필요**.

**컬럼**: Adj Close, Close, High, Low, Open, Volume / **행수**: 4,336 (2009-01-02 ~ 2026-03-30)

**왜 Adj Close 인가?**: 배당·분할이 반영된 가격으로 장기 시계열 분석에 정합. log_ret 계산은 Adj Close 기준.
""")
code("""import pandas as pd
import numpy as np

# Phase 1 raw_data 사본 로드 (다운로드 X — 격리·자기완결 원칙)
def load_ticker(ticker: str) -> pd.DataFrame:
    \"\"\"raw_data CSV 로드 + 인덱스 정렬 + log_ret 컬럼 추가.\"\"\"
    csv_path = RAW_DATA_DIR / f'{ticker}.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    df['log_ret'] = np.log(df['Adj Close']).diff()  # 일별 로그수익률
    return df

raw_spy = load_ticker('SPY')
raw_qqq = load_ticker('QQQ')

# 분석 기간 절단 (10년) — 워밍업 데이터(2009-2015)는 lookback 채움용으로만
ANALYSIS_START = '2016-01-01'
ANALYSIS_END   = '2025-12-31'
df_spy = raw_spy.loc[ANALYSIS_START:ANALYSIS_END].copy()
df_qqq = raw_qqq.loc[ANALYSIS_START:ANALYSIS_END].copy()

print(f"SPY 분석 구간: {df_spy.index[0].date()} ~ {df_spy.index[-1].date()}, n={len(df_spy)}")
print(f"QQQ 분석 구간: {df_qqq.index[0].date()} ~ {df_qqq.index[-1].date()}, n={len(df_qqq)}")
print(f"\\nSPY 첫 3행:")
print(df_spy.head(3)[['Adj Close', 'log_ret']])
""")

# ============================================================================
# §3 시계열 시각화
# ============================================================================
md("""## §3. 시계열 시각화 — 변동성 클러스터링 육안 확인

3종 시계열을 비교 시각화합니다.

| 시계열 | 직관 |
|---|---|
| `log_ret` | 일별 로그수익률 (양/음 부호 포함) |
| `log_ret²` | 일별 분산 추정값 (instantaneous variance proxy, 음수 X) |
| `\\|log_ret\\|` | 일별 변동 크기 (음수 X, 분포 더 평이) |

**왜 보는가?**: 변동성 클러스터링(고변동성이 군집해서 발생하는 현상)이 실제로 관측되는지 육안 확인. 이는 ARCH 효과의 시각적 증거이자, **변동성 예측이 가능하다**는 핵심 가설의 사전 검증입니다.

**예상 패턴**: COVID 2020-03, 2022 연준 긴축, 2025-04 관세 충격 등에서 변동성 spike 가 며칠~몇 주 지속.
""")
code("""import matplotlib.pyplot as plt

def plot_three_series(df: pd.DataFrame, ticker: str) -> None:
    \"\"\"log_ret, log_ret^2, |log_ret| 3-panel 시계열 그래프.\"\"\"
    log_ret = df['log_ret'].dropna()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(log_ret.index, log_ret.values, lw=0.5, color='steelblue')
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].set_ylabel('log_ret')
    axes[0].set_title(f'{ticker} — 일별 로그수익률 시계열')
    axes[1].plot(log_ret.index, (log_ret ** 2).values, lw=0.5, color='darkorange')
    axes[1].set_ylabel('log_ret²')
    axes[1].set_title(f'{ticker} — 제곱 로그수익률 (variance proxy)')
    axes[2].plot(log_ret.index, log_ret.abs().values, lw=0.5, color='crimson')
    axes[2].set_ylabel('|log_ret|')
    axes[2].set_xlabel('date')
    axes[2].set_title(f'{ticker} — 절댓값 로그수익률')
    plt.tight_layout()
    plt.show()

plot_three_series(df_spy, 'SPY')
plot_three_series(df_qqq, 'QQQ')
""")

# ============================================================================
# §4 RV 계산 + 분포 진단
# ============================================================================
md("""## §4. Realized Volatility 계산 + 분포 진단 ⭐

**핵심 검증 단계**입니다. PLAN 에서 채택한 **Log-RV 타깃**의 정당성을 분포로 확인합니다.

### 4-1. RV 정의 (trailing 21일 표준편차)
```
RV[t] = std( log_ret[t-20], log_ret[t-19], ..., log_ret[t] )
```

### 4-2. 분포 진단 4-panel
| Panel | 시각화 |
|---|---|
| (1) | RV 히스토그램 — right-skewed 확인 |
| (2) | log(RV) 히스토그램 — 정규 분포에 가까워지는지 |
| (3) | RV QQ-plot — 정규 대비 꼬리 |
| (4) | log(RV) QQ-plot — 직선에 얼마나 가까운지 |

### 4-3. 판정 기준
- log(RV) 히스토그램이 종 모양 + QQ-plot 이 직선에 근접 → **Log-RV 타깃 채택 정당화**
- 그렇지 않으면 (a) Rolling Std (b) Sum-of-Squares 등 대안 재검토

### 함정
- `np.log(0)` 또는 `np.log(NaN)` → -inf/NaN. 본 셀에서 `dropna()` 명시.
""")
code("""from scipy import stats

def compute_rv(log_ret: pd.Series, window: int = 21) -> pd.Series:
    \"\"\"Trailing window 일 표준편차 (ddof=1, pandas 기본).\"\"\"
    return log_ret.rolling(window).std()

def plot_distribution_4panel(log_ret: pd.Series, ticker: str) -> dict:
    \"\"\"RV / log(RV) 분포 4-panel 진단 + 분포 통계 반환.\"\"\"
    rv = compute_rv(log_ret).dropna()
    log_rv = np.log(rv).replace([-np.inf, np.inf], np.nan).dropna()  # 누수: log(0) 방어

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(rv, bins=50, color='steelblue', edgecolor='white')
    axes[0, 0].set_title(f'{ticker} — RV 분포 (right-skewed 예상)')
    axes[0, 0].set_xlabel('RV (21d trailing std)')
    axes[0, 1].hist(log_rv, bins=50, color='darkgreen', edgecolor='white')
    axes[0, 1].set_title(f'{ticker} — log(RV) 분포 (정규에 근접 기대)')
    axes[0, 1].set_xlabel('log(RV)')
    stats.probplot(rv, dist='norm', plot=axes[1, 0])
    axes[1, 0].set_title(f'{ticker} — RV QQ-plot (정규 대비)')
    stats.probplot(log_rv, dist='norm', plot=axes[1, 1])
    axes[1, 1].set_title(f'{ticker} — log(RV) QQ-plot')
    plt.tight_layout()
    plt.show()

    return {
        'ticker': ticker,
        'n': len(rv),
        'rv_mean': rv.mean(), 'rv_std': rv.std(),
        'rv_skew': rv.skew(), 'rv_kurt_excess': rv.kurt(),
        'logrv_mean': log_rv.mean(), 'logrv_std': log_rv.std(),
        'logrv_skew': log_rv.skew(), 'logrv_kurt_excess': log_rv.kurt(),
    }

stats_spy = plot_distribution_4panel(df_spy['log_ret'].dropna(), 'SPY')
stats_qqq = plot_distribution_4panel(df_qqq['log_ret'].dropna(), 'QQQ')

# 분포 통계 비교 표
print('\\n분포 비교 (RV vs log(RV)):')
print('-' * 70)
fmt = '{:<12} {:>10} {:>10} {:>10} {:>10}'
print(fmt.format('ticker', 'mean', 'std', 'skew', 'kurt_xs'))
print('-' * 70)
for s in (stats_spy, stats_qqq):
    print(fmt.format(s['ticker']+' RV', f"{s['rv_mean']:.5f}", f"{s['rv_std']:.5f}", f"{s['rv_skew']:.3f}", f"{s['rv_kurt_excess']:.3f}"))
    print(fmt.format(s['ticker']+' logRV', f"{s['logrv_mean']:.4f}", f"{s['logrv_std']:.4f}", f"{s['logrv_skew']:.3f}", f"{s['logrv_kurt_excess']:.3f}"))
""")

# ============================================================================
# §5 ACF/PACF on log(RV)
# ============================================================================
md("""## §5. ACF/PACF on log(RV) — embargo=21 충분성 검증 ⭐

### 5-1. 변동성 자기상관 (Persistence)
변동성은 수익률과 달리 **lag 1 ACF 가 0.9 가깝게 매우 강한 자기상관**을 보이는 것이 일반적입니다 (변동성 클러스터링의 정량 표현).

### 5-2. embargo=21 충분성 판정 기준
Walk-Forward fold 사이의 자기상관 잔존을 차단하기 위해 embargo=21 (영업일) 을 사용합니다 (Phase 1 과 동일 — 비교 공정성).

**충분 조건**: ACF lag 21 ~ 30 이 95% CI 안으로 들어옴 (=자기상관 무시 가능 수준).

**불충분 시 후속 액션**: embargo 확장 (42 등) 또는 `Phase1.5` 별도 walk-forward 파라미터 검토 (재천_WORKLOG 보류 항목으로 등록).
""")
code("""from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def acf_pacf_logrv(log_ret: pd.Series, ticker: str, max_lag: int = 30):
    \"\"\"log(RV) 의 ACF/PACF lag 1~max_lag 정량 + 시각화.\"\"\"
    rv = compute_rv(log_ret).dropna()
    log_rv = np.log(rv).replace([-np.inf, np.inf], np.nan).dropna()

    acf_vals = acf(log_rv, nlags=max_lag, fft=True)
    pacf_vals = pacf(log_rv, nlags=max_lag, method='ywm')

    # 95% CI: ±1.96/sqrt(n) 근사
    n = len(log_rv)
    ci = 1.96 / np.sqrt(n)

    print(f'{ticker} — log(RV) ACF/PACF (n={n}, 95% CI = ±{ci:.4f})')
    print('-' * 60)
    print(f"{'lag':>5} {'ACF':>10} {'PACF':>10} {'ACF outside CI':>16}")
    for lag in [1, 2, 5, 10, 21, 22, 25, 30]:
        outside = '✓' if abs(acf_vals[lag]) > ci else ' '
        print(f"{lag:>5} {acf_vals[lag]:>10.4f} {pacf_vals[lag]:>10.4f} {outside:>16}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(log_rv, lags=max_lag, ax=axes[0])
    axes[0].set_title(f'{ticker} — log(RV) ACF (lag 1~{max_lag})')
    plot_pacf(log_rv, lags=max_lag, ax=axes[1], method='ywm')
    axes[1].set_title(f'{ticker} — log(RV) PACF')
    plt.tight_layout()
    plt.show()
    return acf_vals, pacf_vals

acf_spy, pacf_spy = acf_pacf_logrv(df_spy['log_ret'].dropna(), 'SPY')
print()
acf_qqq, pacf_qqq = acf_pacf_logrv(df_qqq['log_ret'].dropna(), 'QQQ')
""")

# ============================================================================
# §5-확장 ACF 1~252 (long-memory 진단)
# ============================================================================
md("""## §5-확장. ACF lag 1~252 (1년) — Long-memory 진단 + embargo 적정값 탐색

### 5x-1. 확장 목적
- Lag 30 까지의 ACF=0.45 는 단기 의존성을 넘어선다 → **lag 60·100·252 까지 보아 long-memory 여부 확정**
- **Embargo 적정값** 탐색: ACF 가 95% CI 안으로 들어가는 첫 lag 를 권장 embargo 로
- 주기성 진단: weekly(5), monthly(21·22), quarterly(63), semi-annual(126), annual(252) lag 에서 spike 확인

### 5x-2. 판정 기준
| ACF(lag=k) | k=60 | k=126 | k=252 |
|---|---|---|---|
| < 0.1 | 단기 의존 (일반 시계열) | — | — |
| **0.1~0.3** | **장기기억** | **장기기억** | **장기기억** |
| > 0.3 | 매우 강한 장기기억 (near-integrated) | — | — |

### 5x-3. Embargo 권장값 도출
- ACF 가 95% CI ($\\pm 1.96/\\sqrt{n}$) 안으로 들어가는 첫 lag 를 후보로
- 그 lag 보다 큰 값을 embargo 로 채택 → 자기상관 잔존 차단
""")
code("""def acf_extended_logrv(log_ret: pd.Series, ticker: str, max_lag: int = 252) -> dict:
    \"\"\"log(RV) ACF lag 1~max_lag — long-memory 진단 + embargo 권장값 탐색.\"\"\"
    rv = compute_rv(log_ret).dropna()
    log_rv = np.log(rv).replace([-np.inf, np.inf], np.nan).dropna()

    acf_vals = acf(log_rv, nlags=max_lag, fft=True)
    n = len(log_rv)
    ci = 1.96 / np.sqrt(n)

    # 95% CI 안으로 들어가는 첫 lag (embargo 후보)
    inside_ci = np.abs(acf_vals[1:]) < ci
    if inside_ci.any():
        first_inside = int(np.argmax(inside_ci)) + 1
    else:
        first_inside = -1  # 끝까지 CI 밖

    print(f'{ticker} — log(RV) ACF lag 1~{max_lag} (n={n}, 95% CI = ±{ci:.4f})')
    print('-' * 60)
    fmt = '{:>5} {:>10}'
    print(fmt.format('lag', 'ACF'))
    for lag in [1, 5, 10, 21, 22, 30, 42, 60, 63, 100, 126, 200, 252]:
        if lag <= max_lag:
            print(fmt.format(lag, f'{acf_vals[lag]:.4f}'))
    if first_inside > 0:
        print(f'\\n  → ACF 95% CI 안으로 진입한 첫 lag = {first_inside}')
        print(f'    (embargo 후보: 이보다 약간 큰 값. 예: {first_inside + 5})')
    else:
        print(f'\\n  → ACF 가 lag {max_lag} 까지도 95% CI 밖 — extreme long-memory')

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    lags_arr = np.arange(1, max_lag + 1)
    axes[0].plot(lags_arr, acf_vals[1:], lw=1.0, color='steelblue')
    axes[0].axhline(ci, color='red', lw=0.7, ls='--', label=f'95% CI ±{ci:.3f}')
    axes[0].axhline(-ci, color='red', lw=0.7, ls='--')
    axes[0].axhline(0, color='gray', lw=0.5)
    for k, c in [(21, 'orange'), (63, 'purple'), (126, 'green'), (252, 'darkred')]:
        if k <= max_lag:
            axes[0].axvline(k, color=c, lw=0.8, ls=':', alpha=0.6, label=f'lag {k}')
    axes[0].set_title(f'{ticker} — log(RV) ACF lag 1~{max_lag} (long-memory 진단)')
    axes[0].set_xlabel('lag (영업일)')
    axes[0].set_ylabel('ACF')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_ylim(-0.1, 1.05)

    # log-log plot — hyperbolic decay 시그니처 확인 (long-memory 면 직선)
    pos_acf = np.maximum(acf_vals[1:max_lag+1], 1e-6)
    axes[1].loglog(lags_arr, pos_acf, lw=1.0, color='darkgreen')
    axes[1].set_title(f'{ticker} — log-log ACF (long-memory 면 직선 감쇠)')
    axes[1].set_xlabel('log(lag)')
    axes[1].set_ylabel('log(ACF)')
    axes[1].grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        'ticker': ticker,
        'first_inside_ci': first_inside,
        'acf_lag_60': acf_vals[60] if max_lag >= 60 else np.nan,
        'acf_lag_126': acf_vals[126] if max_lag >= 126 else np.nan,
        'acf_lag_252': acf_vals[252] if max_lag >= 252 else np.nan,
    }

ext_spy = acf_extended_logrv(df_spy['log_ret'].dropna(), 'SPY', max_lag=252)
print()
ext_qqq = acf_extended_logrv(df_qqq['log_ret'].dropna(), 'QQQ', max_lag=252)

# 종합 표
print('\\n=== Long-memory 진단 종합 ===')
print(f"{'ticker':<8} {'first_inside_CI':>16} {'ACF@60':>10} {'ACF@126':>10} {'ACF@252':>10}")
for r in (ext_spy, ext_qqq):
    print(f"{r['ticker']:<8} {r['first_inside_ci']:>16} {r['acf_lag_60']:>10.4f} {r['acf_lag_126']:>10.4f} {r['acf_lag_252']:>10.4f}")
""")

# ============================================================================
# §6 ADF/KPSS
# ============================================================================
md("""## §6. ADF + KPSS 정상성 검정 on log(RV)

### 6-1. 두 검정의 귀무가설 차이 (반대 방향)
| 검정 | H₀ (귀무가설) | H₀ 기각 시 |
|---|---|---|
| **ADF** | 단위근 존재 (비정상) | 정상 ✓ |
| **KPSS** | 정상 | 비정상 ✗ |

### 6-2. 종합 판정 표
| ADF | KPSS | 결론 |
|---|---|---|
| 기각 (정상) | 비기각 (정상) | **정상 확정** ✓ |
| 비기각 (비정상) | 기각 (비정상) | **비정상 확정** ✗ |
| 기각·기각 또는 비기각·비기각 | — | 모호 — 차분·구조변화 검토 |

### 6-3. LSTM 입력 전제
LSTM 은 **약정상성**(평균·공분산 시간 불변)을 가정합니다. log(RV) 가 정상이면 추가 차분 없이 직접 입력 가능.
""")
code("""from statsmodels.tsa.stattools import adfuller, kpss
import warnings

def stationarity_test(log_ret: pd.Series, ticker: str) -> dict:
    \"\"\"ADF + KPSS 검정.\"\"\"
    rv = compute_rv(log_ret).dropna()
    log_rv = np.log(rv).replace([-np.inf, np.inf], np.nan).dropna()

    # ADF: H0 = unit root (비정상). p < 0.05 면 정상
    adf_stat, adf_p, *_ = adfuller(log_rv, autolag='AIC')

    # KPSS: H0 = 정상. p > 0.05 면 정상.
    # statsmodels KPSS 는 p-value 가 표 범위(0.01~0.10) 밖이면 InterpolationWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kpss_stat, kpss_p, *_ = kpss(log_rv, regression='c', nlags='auto')

    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p > 0.05
    overall = '정상 ✓' if (adf_stationary and kpss_stationary) else '주의 — 추가 검토'

    print(f'{ticker} log(RV) 정상성 검정')
    print('-' * 50)
    print(f'  ADF  : stat={adf_stat:>8.3f}, p={adf_p:.4f}  →  {"정상" if adf_stationary else "비정상"}')
    print(f'  KPSS : stat={kpss_stat:>8.3f}, p={kpss_p:.4f}  →  {"정상" if kpss_stationary else "비정상"}')
    print(f'  종합 : {overall}')

    return {'ticker': ticker, 'adf_p': adf_p, 'kpss_p': kpss_p, 'overall': overall}

res_spy = stationarity_test(df_spy['log_ret'].dropna(), 'SPY')
print()
res_qqq = stationarity_test(df_qqq['log_ret'].dropna(), 'QQQ')
""")

# ============================================================================
# §7 체제 진단
# ============================================================================
md("""## §7. 체제 진단 — 저변동·고변동 구간 시각화

### 7-1. 분위수 기반 체제 정의
- **저변동 체제**: RV ≤ 30% 분위수 (안정 구간)
- **중립 체제**: 30% 분위수 < RV < 70% 분위수
- **고변동 체제**: RV ≥ 70% 분위수 (위기 구간)

### 7-2. 왜 보는가?
- **체제 변화** (저→고변동 전환)는 변동성 예측 모델의 가장 큰 함정
- COVID 2020-03, 2022 긴축 등 체제 전환 시점에 모델이 어떻게 반응할지 예상
- **본 단계는 단순 시각화** — 체제 분류 모델은 추후 단계 (Regime Detection 활용 시)
""")
code("""def plot_regime(log_ret: pd.Series, ticker: str) -> None:
    \"\"\"RV + 30/70 분위수 영역 표시.\"\"\"
    rv = compute_rv(log_ret).dropna()
    q30 = rv.quantile(0.30)
    q70 = rv.quantile(0.70)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(rv.index, rv.values, lw=0.7, color='steelblue', label='RV (21d)')
    ax.axhline(q30, color='green', lw=1, ls='--', label=f'q30 = {q30:.4f}')
    ax.axhline(q70, color='red', lw=1, ls='--', label=f'q70 = {q70:.4f}')
    ax.fill_between(rv.index, 0, q30, color='green', alpha=0.05)
    ax.fill_between(rv.index, q70, rv.max() * 1.05, color='red', alpha=0.05)
    ax.set_title(f'{ticker} — RV 시계열 + 체제 (저변동: ≤q30 / 고변동: ≥q70)')
    ax.set_ylabel('RV')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # 체제별 일수
    n_low = (rv <= q30).sum()
    n_high = (rv >= q70).sum()
    n_total = len(rv)
    print(f'{ticker}: 저변동 {n_low}/{n_total}일 ({n_low/n_total*100:.1f}%), 고변동 {n_high}/{n_total}일 ({n_high/n_total*100:.1f}%)')

plot_regime(df_spy['log_ret'].dropna(), 'SPY')
plot_regime(df_qqq['log_ret'].dropna(), 'QQQ')
""")

# ============================================================================
# §8 누수 검증 (inline 임시 함수)
# ============================================================================
md("""## §8. 타깃 누수 검증 (inline 임시 함수) ⭐

### 8-1. 타깃 정의 (PLAN §2-2)
```python
target[t] = log( std( log_ret[t+1 : t+22] ) )    # forward 21일
```

### 8-2. 검증 의무 (변동성 특수 함정 5종 — PLAN §9)
| # | 함정 | 검증 |
|---|---|---|
| 1 | `shift(-21)` 부호 누락 | assert 3개 시점 |
| 2 | `np.log(0)` 또는 `np.log(NaN)` | `assert (rv > 0).all()` |
| 3 | ddof 불일치 (pandas 기본 1, numpy 기본 0) | 명시 통일 (`ddof=1`) |
| 4 | rolling vs std 위치 (log 전후) | `log(rolling().std())` 패턴 통일 |
| 5 | 타깃 마지막 21행 NaN | `dropna()` 후 비교 |

### 8-3. 본 셀의 임시 함수
Step 2 에서 `scripts/targets_volatility.py` 로 추출 예정. 본 §8 은 inline 검증으로 빠른 sanity check.
""")
code("""def build_target_logrv_21d_inline(adj_close: pd.Series, window: int = 21) -> pd.Series:
    \"\"\"21일 forward log-realized-volatility 타깃 (Phase 1.5 정의).

    Parameters
    ----------
    adj_close : pd.Series
        Adj Close 가격 시계열 (DatetimeIndex).
    window : int, default=21
        rolling 윈도우 길이 (영업일).

    Returns
    -------
    pd.Series
        target[t] = log( std(log_ret[t+1 : t+window+1], ddof=1) )
        마지막 ``window`` 행 NaN.
    \"\"\"
    log_ret = np.log(adj_close).diff()                      # 누수: t-1 → t 차분, 시점 t 사용 OK
    rv = log_ret.rolling(window).std(ddof=1)                 # 누수: trailing window, t 시점 종료
    target = np.log(rv).shift(-window)                        # 누수: forward shift → t+window 시점 RV 가 t 라벨
    return target

def verify_no_leakage_logrv_inline(
    adj_close: pd.Series, target: pd.Series, n_checks: int = 3, window: int = 21
) -> None:
    \"\"\"누수 검증 — 3개 무작위 시점에서 target[t] == log(std(log_ret[t+1:t+22], ddof=1)) assert.\"\"\"
    log_ret = np.log(adj_close).diff()
    rng = np.random.default_rng(42)
    valid_target = target.dropna()
    valid_idx_int = np.array([target.index.get_loc(d) for d in valid_target.index])
    sample_int = rng.choice(valid_idx_int, size=n_checks, replace=False)

    print('[검증 1] 시점별 target[t] == log(std(log_ret[t+1:t+22])) assert')
    for i_pos in sample_int:
        t_date = target.index[i_pos]
        future_logret = log_ret.iloc[i_pos + 1 : i_pos + 1 + window]
        # ddof=1 명시 — pandas rolling.std 기본과 일치
        expected = np.log(future_logret.std(ddof=1))
        actual = target.iloc[i_pos]
        ok = np.isclose(actual, expected, atol=1e-12, equal_nan=False)
        print(f'  t={t_date.date()}, expected={expected:.6f}, actual={actual:.6f} → {"PASS" if ok else "FAIL ❌"}')
        assert ok, f'타깃 누수 검증 실패 at t={t_date}'

    print('\\n[검증 2] 첫 5행 육안 표 (date, log_ret_t, target_t, 직접계산)')
    # log_ret 은 dropna 하지 않음 — target 과 동일 인덱스 유지 (인덱스 어긋남 방지)
    rows = []
    for i_pos in range(5):
        t_date = target.index[i_pos]
        if i_pos + 1 + window > len(target):
            continue
        future_lr = log_ret.iloc[i_pos + 1 : i_pos + 1 + window]
        # ddof=1 명시 통일. NaN 포함 시 pandas std 는 자동 skip, 그러나 i_pos+1 부터는 NaN 없음
        expected = np.log(future_lr.std(ddof=1)) if len(future_lr) == window else np.nan
        rows.append({
            'date': t_date.date(),
            'log_ret_t': log_ret.iloc[i_pos] if i_pos < len(log_ret) else np.nan,
            'target_t (from func)': target.iloc[i_pos],
            'expected (manual)': expected,
        })
    print(pd.DataFrame(rows).to_string(index=False))

WINDOW = 21  # PLAN §2-2: 21영업일 forward window

# SPY 누수 검증
target_spy_logrv = build_target_logrv_21d_inline(df_spy['Adj Close'], window=WINDOW)
print('=' * 70)
print('SPY 누수 검증')
print('=' * 70)
verify_no_leakage_logrv_inline(df_spy['Adj Close'], target_spy_logrv, window=WINDOW)

# QQQ 누수 검증
target_qqq_logrv = build_target_logrv_21d_inline(df_qqq['Adj Close'], window=WINDOW)
print('\\n' + '=' * 70)
print('QQQ 누수 검증')
print('=' * 70)
verify_no_leakage_logrv_inline(df_qqq['Adj Close'], target_qqq_logrv, window=WINDOW)

# 추가 sanity check: log(0)/log(NaN) 진입 차단
# 정상 NaN 개수 = WINDOW (forward shift 로 인한 마지막 행)
# 메커니즘:
#   - log_ret = adj_close.log().diff() : 첫 1 행 NaN
#   - rolling(WINDOW).std() : 첫 WINDOW 행 NaN (rolling 시작 부족)
#   - shift(-WINDOW) : 인덱스를 WINDOW 만큼 앞으로 당김 → 첫 WINDOW 행 NaN 이 사라지고
#                     마지막 WINDOW 행이 NaN 이 됨
# 결과: 정상 NaN = 마지막 WINDOW 행만 (= 21)
expected_nan = WINDOW
n_nan_spy = target_spy_logrv.isna().sum()
n_nan_qqq = target_qqq_logrv.isna().sum()
print(f'\\nNaN 카운트: SPY={n_nan_spy} / QQQ={n_nan_qqq} (예상={expected_nan}: forward shift(-{WINDOW}) 으로 인한 마지막 {WINDOW}행)')
ok = (n_nan_spy == expected_nan) and (n_nan_qqq == expected_nan)
print(f'  → 예상값과 {"일치 ✓ (log(0)/log(NaN) 진입 없음)" if ok else "불일치 ❌ (rolling NaN 또는 log(0) 진입 발생)"}')
assert ok, f'NaN 카운트 검증 실패: SPY={n_nan_spy}, QQQ={n_nan_qqq}, 예상={expected_nan}'
""")

# ============================================================================
# §9 종합 결론
# ============================================================================
md("""## §9. 종합 결론 — 사용자 체크포인트

### 9-1. 검증 결과 요약 (셀 출력 직접 확인)

| 검증 | 항목 | 기대 결과 |
|---|---|---|
| §3 | 변동성 클러스터링 시각적 존재 | 고변동 spike 가 며칠~수주 지속 |
| §4 | log(RV) 분포 — 정규에 근접 | 히스토그램 종 모양, QQ-plot 직선 근접 |
| §5 | ACF lag 1~30 — 강한 자기상관 (변동성 persistence) | lag 1 > 0.5, lag 21~30 95% CI 근접 |
| §6 | ADF + KPSS — 정상성 확정 | ADF p<0.05, KPSS p>0.05 |
| §7 | 체제 분포 — 저/고 30%·30% 근사 | 분위수 정의상 자동 만족 |
| §8 | 누수 검증 — 3 assert + 5행 표 | PASS, NaN 만 마지막 window 행 |

### 9-2. PLAN 의 핵심 가정 검증

| 가정 | 본 노트북 검증 셀 | 결과 |
|---|---|---|
| Log-RV 분포가 거의 정규 → MSE loss 정합 | §4 | 셀 출력 확인 |
| 변동성 자기상관 강함 → LSTM 학습 가능 | §5 | 셀 출력 확인 |
| log(RV) 정상 → 차분 불필요 | §6 | 셀 출력 확인 |
| embargo=21 이 자기상관 잔존 차단에 충분 | §5 lag 21~30 | 셀 출력 확인 |
| `shift(-21)` 누수 차단 | §8 | PASS |

### 9-3. 다음 단계 (Step 2)

본 노트북에서 PASS 라면:
- **Step 2**: `scripts/targets_volatility.py` · `metrics_volatility.py` · `baselines_volatility.py` 작성 + 단위 테스트
- §8 의 inline 함수를 정식 모듈로 추출

### 9-4. 사용자 검토 포인트

1. **§4 분포**: log 변환 후 정규에 근접하는지 시각적 합의
2. **§5 ACF**: lag 21~30 이 95% CI 근접하는지 (embargo=21 충분성)
3. **§8 누수 검증**: PASS 출력 확인

이상 없으면 Step 2 진행을 승인 부탁드립니다.
""")

# ============================================================================
# 노트북 저장
# ============================================================================
NB.cells = cells

OUT_PATH = Path(__file__).resolve().parent / '01_volatility_eda.ipynb'
nbf.write(NB, str(OUT_PATH))
print(f'노트북 빌드 완료: {OUT_PATH}')
print(f'총 셀 수: {len(cells)} (마크다운 + 코드)')
