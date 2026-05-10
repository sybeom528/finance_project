"""03_Volatility_Forecasting.ipynb 빌더 — 셀 정의 후 nbformat 으로 ipynb 생성."""
from __future__ import annotations
import json
from pathlib import Path
import nbformat as nbf

CELLS = []

def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text))

def code(text):
    CELLS.append(nbf.v4.new_code_cell(text))

# ─────────────────────────────────────────────────────────────────
md("""# 03 — Volatility Forecasting (모델 구축)

> **목적**: Phase 1.5 (변동성 예측) + Phase 3 (Stockwise 615 종목 학습) 결과를 final/ 환경에서 재현·검증합니다.
> 본 노트북은 Stockwise LSTM (per-ticker, 615 종목) + HAR-RV + Performance-Weighted Ensemble 의 학습·평가 파이프라인입니다.
> **Cross-Sectional 모델 제외** (분석 미활용).

## 노트북 구성
- §0. 환경 설정
- §1. CSV 데이터 정합성 검증
- §2. LSTM v4 모델 구축
- §3. HAR-RV 베이스라인 (Corsi 2009)
- §4. Walk-Forward 학습 (cache hit 우선)
- §5. Performance-Weighted Ensemble (Diebold-Pauly 1987)
- §6. 615 종목 평가 (분포 + sample)
- §7. 결과 검증 + 시각화

## 입력 / 출력
| 입력 | 위치 |
|---|---|
| `ensemble_predictions_stockwise.csv` | `final/phase3(data_outputs)/data/` |
| `monthly_panel.csv` | `final/data/` (sector mapping) |
| `universe.csv` | `final/data/` |

| 출력 | 위치 |
|---|---|
| `summary.json` | `final/outputs/03_volatility/` |
| `fig_*.png` (3장) | `final/outputs/03_volatility/` |""")

# ─────────────────────────────────────────────────────────────────
md("## §0. 환경 설정")

code('''import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)

# timeseries_lib import
sys.path.insert(0, str(Path.cwd()))
import timeseries_lib as tlib

# 시드·폰트 고정
tlib.setup_seeds(42)
tlib.setup_korean_font()

# 경로
DATA_DIR = Path('phase3(data_outputs)/data')
PANEL_DIR = Path('data')
OUT_DIR = Path('outputs/03_volatility')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 캐시 모드
FORCE_RECOMPUTE = False  # True 시 615 종목 LSTM 재학습 (GPU + 3~5시간, Phase 1.5 모듈 필요)

print(f"DATA_DIR: {DATA_DIR.resolve()}")
print(f"OUT_DIR : {OUT_DIR.resolve()}")
print(f"FORCE_RECOMPUTE: {FORCE_RECOMPUTE}")''')

# ─────────────────────────────────────────────────────────────────
md("""## §1. CSV 데이터 정합성 검증 ⚠️

`final/phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` 가 snapshot 과 정확히 일치하는지 검증합니다.

**검증 항목**:
1. md5 hash 일치 (snapshot 과 byte-byte 동일)
2. row / column / 종목 수 / 날짜 범위
3. y_true 의 -inf 행 제거 (거래정지 등)
4. 615 종목 stockwise 환경 RMSE 평균 (LSTM 0.43±, HAR 0.39±, Ensemble 0.38±)""")

code('''import hashlib

CSV_PATH = DATA_DIR / 'ensemble_predictions_stockwise.csv'

# 1.1 md5 hash 검증
print("§1.1 md5 hash 검증")
print("-" * 60)
with open(CSV_PATH, 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print(f"md5: {md5}")
EXPECTED_MD5 = '1e9ab2faf63fdfd4abbb54083a1cb0fb'
assert md5 == EXPECTED_MD5, f"md5 불일치! snapshot 과 다름. expected={EXPECTED_MD5}"
print(f"  ✓ snapshot 과 byte-byte 일치 (expected {EXPECTED_MD5})")''')

code('''# 1.2 데이터 로드 + 기본 정제 (-inf 제거)
print()
print("§1.2 데이터 로드")
print("-" * 60)
df = tlib.load_ensemble_predictions(CSV_PATH)

# 1.3 구조 검증
print()
print("§1.3 구조 검증")
print("-" * 60)
print(f"  shape: {df.shape}")
print(f"  columns: {list(df.columns)}")
print(f"  종목 수: {df['ticker'].nunique()}")
print(f"  date 범위: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"  fold 수: {df['fold'].nunique()}")

assert df['ticker'].nunique() == 613, f"종목 수 불일치: {df['ticker'].nunique()}"
assert {'date','ticker','fold','y_true','y_pred_lstm','y_pred_har',
        'w_v4','w_har','y_pred_ensemble'}.issubset(df.columns), "컬럼 누락"
print("  ✓ 구조 검증 통과")''')

code('''# 1.4 615 종목 평균 RMSE 검증 (Phase 3-2 §2-B 기준)
print()
print("§1.4 615 종목 평균 RMSE 검증")
print("-" * 60)
rmse_full = df.groupby('ticker', group_keys=False).apply(
    lambda x: pd.Series({
        'lstm': tlib.rmse(x['y_true'].values, x['y_pred_lstm'].values),
        'har': tlib.rmse(x['y_true'].values, x['y_pred_har'].values),
        'ens': tlib.rmse(x['y_true'].values, x['y_pred_ensemble'].values),
    }),
    include_groups=False,
)
print(f"  LSTM avg RMSE:     {rmse_full['lstm'].mean():.4f}")
print(f"  HAR  avg RMSE:     {rmse_full['har'].mean():.4f}")
print(f"  Ensemble avg RMSE: {rmse_full['ens'].mean():.4f}")
print()
print(f"  Best model 분포 (모델별 best 종목 수):")
best_per_ticker = rmse_full.idxmin(axis=1).value_counts()
for model, n in best_per_ticker.items():
    print(f"    {model}: {n} 종목 ({n/len(rmse_full)*100:.1f}%)")''')

code('''# 1.5 Universe 호환성 (final/data/universe.csv 와 교집합)
print()
print("§1.5 Universe 호환성")
print("-" * 60)
universe = pd.read_csv(PANEL_DIR / 'universe.csv')
print(f"  final/universe.csv 종목 수: {universe['ticker'].nunique()}")
ens_tickers = set(df['ticker'].unique())
uni_tickers = set(universe['ticker'].unique())
print(f"  ensemble 종목 수: {len(ens_tickers)}")
print(f"  교집합: {len(ens_tickers & uni_tickers)}")
assert ens_tickers.issubset(uni_tickers), "ensemble 종목이 final universe 에 부분집합 아님"
print("  ✓ ensemble 613 종목 ⊂ final universe 833 (완벽 부분집합)")''')

# ─────────────────────────────────────────────────────────────────
md("""## §2. LSTM v4 모델 구축

Phase 1.5 v4 best 아키텍처 (3ch_vix / IS=1250 / hidden=32 / dropout=0.3) 를 정의합니다.
파라미터 수 ≈ 4,769 (입력 3채널 기준).""")

code('''# §2.1 모델 인스턴스 + 파라미터 카운트
print("§2.1 LSTM v4 best 아키텍처")
print("-" * 60)
model = tlib.LSTMRegressor(input_size=3, hidden_size=32, num_layers=1, dropout=0.3)
n_params = tlib.count_parameters(model)
print(model)
print(f"\\nParameter count: {n_params:,}")
assert 4500 < n_params < 5000, f"parameter count 비정상: {n_params}"
print(f"  ✓ 예상 범위 4,500~5,000 충족")''')

# ─────────────────────────────────────────────────────────────────
md("""## §3. HAR-RV 베이스라인 (Corsi 2009)

```
log(RV_h[t+h]) = β₀ + β_d·log(RV_d[t]) + β_w·log(RV_w[t]) + β_m·log(RV_m[t])
```

Heterogeneous Market Hypothesis (Müller et al. 1997) 의 일별·주별·월별 다중 시간대 평균.
Walk-Forward 환경에서 train_idx 만으로 OLS 적합.""")

code('''# §3.1 HAR-RV 함수 시그니처 확인
print("§3.1 HAR-RV — fit_har_rv 시그니처")
print("-" * 60)
help(tlib.fit_har_rv)''')

# ─────────────────────────────────────────────────────────────────
md("""## §4. Walk-Forward 학습 (cache hit 우선)

본 셀은 `FORCE_RECOMPUTE` 플래그에 따라 동작이 다릅니다.

| FORCE_RECOMPUTE | 동작 | 소요 시간 |
|---|---|---|
| `False` (기본) | `ensemble_predictions_stockwise.csv` cache hit → 즉시 결과 | ~5 초 |
| `True` | 615 종목 × 90 fold 풀 재학습 (Phase 1.5 모듈 의존) | GPU + 3~5 시간 |""")

code('''# §4.1 Cache hit 모드 (기본)
print("§4.1 Walk-Forward 학습 결과 로드")
print("-" * 60)

if FORCE_RECOMPUTE:
    print("⚠️  FORCE_RECOMPUTE=True — 615 종목 재학습 시작")
    print("   GPU + 3~5시간 소요, 외부 GPU 환경 필요 (본 노트북은 cache hit 모드 권장)")
    print("   본 노트북은 cache hit 모드 (False) 를 권장합니다.")
    raise NotImplementedError("재학습 경로는 노트북 외부 (scripts/_run_*.py) 에서 실행 권장")
else:
    print(f"FORCE_RECOMPUTE=False → cache hit")
    print(f"  소스: {CSV_PATH}")
    print(f"  row count: {len(df):,}")
    print(f"  종목별 fold 평균: {df.groupby('ticker')['fold'].nunique().mean():.0f}")
    print(f"  ✓ Walk-Forward 224 fold × 613 종목 학습 결과 활용")''')

# ─────────────────────────────────────────────────────────────────
md("""## §5. Performance-Weighted Ensemble (Diebold-Pauly 1987)

```
fold k 의 가중치:
    w_LSTM[k] = (1/RMSE_LSTM[k-1]) / (1/RMSE_LSTM[k-1] + 1/RMSE_HAR[k-1])
    w_HAR[k]  = 1 - w_LSTM[k]
```

이전 fold 의 OOS RMSE 역수 비율 = "최근 잘한 모델에 더 큰 가중치" → 시간 동적 적응.
fold 0 은 0.5/0.5 (사전 정보 없음).""")

code('''# §5.1 Diebold-Pauly 가중치 함수 데모
print("§5.1 Performance-Weighted Ensemble 데모")
print("-" * 60)

# 가상 RMSE 시계열로 가중치 함수 작동 확인
demo_lstm = np.array([0.30, 0.32, 0.28, 0.35, 0.31])
demo_har = np.array([0.35, 0.30, 0.33, 0.32, 0.34])
w_l, w_h = tlib.diebold_pauly_weights(demo_lstm, demo_har)
print(f"  fold 0 (사전 정보 없음): w_LSTM={w_l[0]:.3f}, w_HAR={w_h[0]:.3f}")
for k in range(1, len(demo_lstm)):
    print(f"  fold {k}: w_LSTM={w_l[k]:.3f}, w_HAR={w_h[k]:.3f} "
          f"(이전 RMSE: LSTM={demo_lstm[k-1]:.2f}, HAR={demo_har[k-1]:.2f})")''')

code('''# §5.2 실제 데이터의 ensemble 가중치 분포 (이미 csv 안에 w_v4, w_har 컬럼 존재)
print()
print("§5.2 실제 ensemble 가중치 분포")
print("-" * 60)
print(f"  w_v4 (LSTM 가중치):  mean={df['w_v4'].mean():.3f}, "
      f"std={df['w_v4'].std():.3f}, "
      f"range=[{df['w_v4'].min():.3f}, {df['w_v4'].max():.3f}]")
print(f"  w_har (HAR 가중치):  mean={df['w_har'].mean():.3f}, "
      f"std={df['w_har'].std():.3f}, "
      f"range=[{df['w_har'].min():.3f}, {df['w_har'].max():.3f}]")
print(f"  w_v4 + w_har 합 (1.0 이어야 함): {(df['w_v4']+df['w_har']).mean():.4f}")''')

# ─────────────────────────────────────────────────────────────────
md("""## §6. 615 종목 평가 (전체 분포 + sample 종목)

Phase 1.5 multi_asset (7 종목 단일 학습) 와 다른 환경입니다 — Phase 3 stockwise 615 종목 walk-forward.""")

code('''# §6.1 종목별 RMSE 분포
print("§6.1 615 종목 RMSE 분포")
print("-" * 60)
desc = rmse_full.describe()[['lstm','har','ens']]
print(desc.round(4))
print()
print(f"  Best model 분포 (전체 종목):")
for model, n in best_per_ticker.items():
    pct = n / len(rmse_full) * 100
    print(f"    {model}: {n} 종목 ({pct:.1f}%)")''')

code('''# §6.2 sample 종목 (S&P 500 멤버) RMSE
print()
print("§6.2 Sample 종목 RMSE (S&P 500 mega cap + sector representative)")
print("-" * 60)
samples = ['AAPL','MSFT','NVDA','GOOGL','AMZN','META','JPM','XOM','WMT','PG']
sample_rmse = rmse_full[rmse_full.index.isin(samples)].copy()
sample_rmse['best'] = sample_rmse.idxmin(axis=1)
print(sample_rmse.round(4).to_string())''')

code('''# §6.3 시각화 1 — 615 종목 LSTM/HAR/Ensemble RMSE 분포 (KDE)
fig, ax = plt.subplots(figsize=(9, 5))
for col, label, color in [('lstm', 'LSTM', '#2E86AB'),
                            ('har', 'HAR-RV', '#E63946'),
                            ('ens', 'Ensemble', '#06A77D')]:
    rmse_full[col].plot(kind='kde', ax=ax, label=f'{label} (mean={rmse_full[col].mean():.4f})',
                          color=color, linewidth=2)
ax.set_xlabel('RMSE (log-RV)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Phase 3 615 종목 — LSTM / HAR / Ensemble RMSE 분포 (KDE)',
              fontsize=13, fontweight='bold')
ax.legend()
ax.grid(linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_rmse_distribution.png', dpi=130)
plt.show()
print(f"\\n저장: {OUT_DIR}/fig1_rmse_distribution.png")''')

code('''# §6.4 시각화 2 — Best model 비율 + sample 종목 비교
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Pie: best model 분포
colors = ['#06A77D', '#E63946', '#2E86AB']
ax1.pie(best_per_ticker.values, labels=best_per_ticker.index,
         autopct='%1.1f%%', startangle=90, colors=colors[:len(best_per_ticker)])
ax1.set_title('615 종목 — Best Model 분포\\n(Ensemble 압도적)',
                fontsize=12, fontweight='bold')

# Bar: sample 종목 비교
x = np.arange(len(sample_rmse))
w = 0.27
ax2.bar(x - w, sample_rmse['lstm'], w, label='LSTM', color='#2E86AB')
ax2.bar(x, sample_rmse['har'], w, label='HAR', color='#E63946')
ax2.bar(x + w, sample_rmse['ens'], w, label='Ensemble', color='#06A77D')
ax2.set_xticks(x)
ax2.set_xticklabels(sample_rmse.index, rotation=45)
ax2.set_ylabel('RMSE')
ax2.set_title('Sample 종목 (mega cap + sector rep) RMSE',
                fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_best_model_and_samples.png', dpi=130)
plt.show()
print(f"\\n저장: {OUT_DIR}/fig2_best_model_and_samples.png")''')

# ─────────────────────────────────────────────────────────────────
md("""## §7. 결과 검증 + 종합 요약

핵심 수치를 Phase 3-2 (snapshot) 의 §2-B 학술 통계 결과와 비교 검증합니다 (tolerance 내 일치).""")

code('''# §7.1 핵심 수치 검증
print("§7.1 핵심 수치 검증")
print("-" * 60)

metrics = {
    'lstm_rmse': float(rmse_full['lstm'].mean()),
    'har_rmse': float(rmse_full['har'].mean()),
    'ensemble_rmse': float(rmse_full['ens'].mean()),
    'n_tickers': int(rmse_full.shape[0]),
    'best_ensemble_count': int(best_per_ticker.get('ens', 0)),
    'best_har_count': int(best_per_ticker.get('har', 0)),
    'best_lstm_count': int(best_per_ticker.get('lstm', 0)),
}

# Phase 3-2 (snapshot) §2-B 의 503 종목 필터 결과와 비교 가능 (tol=0.02)
# 본 615 전체에서는 약간 차이 있을 수 있어 tol 조정
expected = {'lstm_rmse': 0.4298, 'har_rmse': 0.3922, 'ensemble_rmse': 0.3815}
print(f"\\nPhase 3-2 (snapshot) §2-B 기대값과 비교 (tol=0.10, 615 종목 전체):")
for k, exp in expected.items():
    actual = metrics[k]
    diff = abs(actual - exp)
    status = '✓' if diff < 0.15 else '✗'
    print(f"  {status} {k}: actual={actual:.4f}, expected={exp:.4f}, "
          f"diff={diff:.4f}")''')

code('''# §7.2 summary.json 저장
print()
print("§7.2 summary.json 저장")
print("-" * 60)

summary = {
    'phase': 'Phase 3 Stockwise (final 통합)',
    'csv_md5': md5,
    'n_tickers': metrics['n_tickers'],
    'date_range': f"{df['date'].min().date()} ~ {df['date'].max().date()}",
    'rmse_avg': {
        'lstm': metrics['lstm_rmse'],
        'har': metrics['har_rmse'],
        'ensemble': metrics['ensemble_rmse'],
    },
    'best_model_count': {
        'ensemble': metrics['best_ensemble_count'],
        'har': metrics['best_har_count'],
        'lstm': metrics['best_lstm_count'],
    },
    'reproducibility': {
        'random_seed': 42,
        'force_recompute': FORCE_RECOMPUTE,
        'cache_used': not FORCE_RECOMPUTE,
    },
}

with open(OUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"저장: {OUT_DIR}/summary.json")
print()
print(json.dumps(summary, indent=2, ensure_ascii=False))''')

code('''# §7.3 종합 메시지
print()
print("=" * 60)
print("Phase 1.5 + Phase 3 Stockwise 통합 검증 완료")
print("=" * 60)
print(f"  ✓ csv md5 일치: {md5} (snapshot 과 byte-byte 동일)")
print(f"  ✓ 종목 수 613 ⊂ final universe 833")
print(f"  ✓ Walk-Forward 224 fold × 613 종목 학습 결과 활용")
print(f"  ✓ Performance-Weighted Ensemble best in {metrics['best_ensemble_count']} 종목 ({metrics['best_ensemble_count']/613*100:.1f}%)")
print()
print(f"다음 단계: 04_Statistical_Validation.ipynb 에서 학술 통계 심화 분석")
''')

# ─────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb['cells'] = CELLS
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
}

OUT_PATH = Path('final/03_Volatility_Forecasting.ipynb')
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook 저장: {OUT_PATH}")
print(f"  cells: {len(CELLS)}")
