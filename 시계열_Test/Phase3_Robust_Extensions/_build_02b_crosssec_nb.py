"""1회용 빌드 스크립트 — 02b_phase15_cross_sectional.ipynb 생성.

Phase 3 Step 2b — Cross-Sectional LSTM 학습.

종목별 (Stock-wise) 과 대비하여 모든 종목을 하나의 LSTM 으로 학습.
Ticker Embedding (Gu, Kelly, Xiu 2020) 을 통해 종목 정체성 인식.

⭐ C4+Mj5 수정 적용:
- build_cs_inputs(align_to_common_dates=True) 로 날짜 정렬
- 종목별 date array 동일 보장
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
# 헤더
# ============================================================================
md("""# Phase 3 Step 2b — Cross-Sectional LSTM 학습 (`02b_phase15_cross_sectional.ipynb`)

> **목적**: 모든 종목을 하나의 LSTM 으로 학습 (Cross-Sectional).
>          종목별 (02a) 과의 직접 비교로 "Cross-Sectional 방식의 우위 여부" 검증.

## Cross-Sectional vs 종목별 차이

| 비교 | 종목별 (02a) | **Cross-Sectional (02b)** |
|---|---|---|
| 학습 | 종목마다 별도 LSTM | 모든 종목 공유 LSTM |
| 정보 전달 | 종목 A 의 패턴 → A 만 | 종목 A 의 패턴 → 모든 종목 |
| Ticker 인식 | 없음 | Ticker Embedding (Gu, Kelly, Xiu 2020) |
| 파라미터 수 | 4,513 × n_tickers | **54,913** (embedding 포함) |
| 데이터 효율 | 종목별 독립 | **전 종목 데이터 공유** |

## ⭐ C4+Mj5 수정

`build_cs_inputs(align_to_common_dates=True)` 적용:
- 모든 종목 panel 전체 날짜 축으로 정렬 (IPO 이전 = NaN)
- 동일 position = 동일 market date 보장
- NaN 구간은 `_build_cs_dataset_for_fold` 에서 자동 제외

## 산출물

| 파일 | 내용 |
|---|---|
| `data/ensemble_predictions_crosssec.csv` | CS 예측 (date, ticker, y_pred_lstm_cs, y_pred_har, y_pred_ensemble, y_true) |

## 셀 구성

| § | 내용 |
|---|---|
| §1 | 환경 부트스트랩 |
| §2 | GPU 환경 확인 |
| §3 | C4+Mj5 날짜 정렬 검증 |
| §4 | Cross-Sectional 학습 실행 |
| §5 | 결과 검증 + Ticker Embedding 분석 |
| §6 | 02a (종목별) vs 02b (Cross-Sectional) 비교 |
""")


# ============================================================================
# §1 환경
# ============================================================================
md("""## §1. 환경 부트스트랩""")

code("""%load_ext autoreload
%autoreload 2

import sys, time
from pathlib import Path

NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
from scripts.volatility_ensemble import build_cs_inputs
from scripts.models_cs import CrossSectionalLSTMRegressor, CS_V4_BEST_CONFIG

font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '02b_crosssec'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'OUT_DIR: {OUT_DIR}')""")


# ============================================================================
# §2 GPU
# ============================================================================
md("""## §2. GPU 환경 확인""")

code("""print('=== GPU 환경 ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_mem = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)) / 1e9
    print(f'Total VRAM: {total_mem:.1f} GB')
    print(f'Free  VRAM: {free_mem:.1f} GB')
    device = 'cuda'
else:
    print('⚠️ CPU 모드')
    device = 'cpu'

print(f'\\n=== CS_V4_BEST_CONFIG ===')
for k, v in CS_V4_BEST_CONFIG.items():
    print(f'  {k}: {v}')""")

code("""# CrossSectionalLSTMRegressor 파라미터 수 확인
dummy_model = CrossSectionalLSTMRegressor(
    input_size=CS_V4_BEST_CONFIG['input_size'],
    hidden_size=CS_V4_BEST_CONFIG['hidden_size'],
    num_layers=CS_V4_BEST_CONFIG['num_layers'],
    dropout=CS_V4_BEST_CONFIG['dropout'],
    n_tickers=200,    # 최대 종목 수
    embedding_dim=CS_V4_BEST_CONFIG['embedding_dim'],
)
n_params = sum(p.numel() for p in dummy_model.parameters())
print(f'CS LSTM 파라미터 수: {n_params:,} (기대: 54,913)')
assert n_params > 50000, f'파라미터 수 이상: {n_params}'
print('✅ 모델 구조 검증 완료')""")


# ============================================================================
# §3 C4+Mj5 검증
# ============================================================================
md("""## §3. C4+Mj5 날짜 정렬 검증 (⭐ 핵심)

`build_cs_inputs(align_to_common_dates=True)` 를 적용했을 때
모든 종목의 date array 길이가 동일한지 검증.

- **PASS**: unique 길이가 1 → 모든 종목 동일 날짜 축
- **FAIL**: unique 길이 > 1 → 날짜 불일치 (해결 방안 §3.3 참조)
""")

code("""# 3-1. 데이터 로드
panel_path = DATA_DIR / 'daily_panel.csv'
universe_path = DATA_DIR / 'universe_top50_history_extended.csv'

assert panel_path.exists(), f'panel 없음: {panel_path}'
assert universe_path.exists(), f'universe 없음: {universe_path}'

panel = pd.read_csv(panel_path, parse_dates=['date'])
universe_df = pd.read_csv(universe_path, parse_dates=['cutoff_date'])

print(f'panel: {panel.shape}')
print(f'universe: {universe_df.shape}, unique {universe_df["ticker"].nunique()} 종목')""")

code("""# 3-2. build_cs_inputs (align=True) 실행 + 날짜 정렬 검증
all_tickers = universe_df['ticker'].unique().tolist()

print(f'학습 대상 종목 수: {len(all_tickers)}')
print('build_cs_inputs (align_to_common_dates=True) 실행 중...')

t0 = time.time()
cs_inputs = build_cs_inputs(
    panel=panel,
    tickers=all_tickers,
    align_to_common_dates=True,    # ⭐ C4+Mj5 수정
)
print(f'완료 ({time.time()-t0:.1f}초)')

# 날짜 정렬 검증
date_lengths = {t: len(cs_inputs['date'][t]) for t in cs_inputs['series']}
unique_lengths = set(date_lengths.values())
print(f'\\n=== C4+Mj5 검증 ===')
print(f'종목별 date array 길이 unique 값: {unique_lengths}')

if len(unique_lengths) == 1:
    print(f'✅ 모든 종목 동일 날짜 축 ({list(unique_lengths)[0]} 거래일)')
else:
    print(f'⚠️ 날짜 길이 불일치 ({len(unique_lengths)} 종류) — 해결 방안 §3.3 참조')
    # 주로 panel 에 없는 종목 등의 엣지 케이스
    by_len = {}
    for t, l in date_lengths.items():
        by_len.setdefault(l, []).append(t)
    for l, ts in sorted(by_len.items()):
        print(f'  length={l}: {ts[:5]}...')

if cs_inputs['common_dates'] is not None:
    print(f'common_dates: {cs_inputs["common_dates"][0].date()} ~ {cs_inputs["common_dates"][-1].date()}')""")

code("""# 3-3. IPO 종목 NaN 구조 확인 (예: META, Snowflake 등)
ipo_check_tickers = []
for t in all_tickers:
    if t in cs_inputs['series']:
        series = cs_inputs['series'][t]
        n_nan = np.sum(np.isnan(series))
        n_total = len(series)
        if n_nan > 500:    # 500 일 이상 NaN = 신규 IPO 가능성
            ipo_check_tickers.append((t, n_nan, n_total, round(n_nan/n_total*100, 1)))

ipo_check_tickers.sort(key=lambda x: -x[1])
print(f'NaN 구간 많은 종목 (신규 IPO 추정):')
for t, n_nan, n_total, pct in ipo_check_tickers[:10]:
    print(f'  {t}: {n_nan}/{n_total} NaN ({pct}%)')""")


# ============================================================================
# §4 Cross-Sectional 학습
# ============================================================================
md("""## §4. Cross-Sectional 학습 실행

`run_ensemble_cross_sectional()`:
- `use_har=True` → HAR-RV 결합 (Ensemble)
- 학습 시간: 약 1~2 시간
""")

code("""from scripts.volatility_ensemble import run_ensemble_cross_sectional

print('=== Cross-Sectional 학습 시작 ===')
print(f'입력 채널: {CS_V4_BEST_CONFIG["input_size"]} ch (rv_d, rv_w, rv_m, vix_log)')
print(f'hidden_size: {CS_V4_BEST_CONFIG["hidden_size"]}')
print(f'embedding_dim: {CS_V4_BEST_CONFIG["embedding_dim"]}')
print(f'IS: {CS_V4_BEST_CONFIG["is_len"]}일, embargo: {CS_V4_BEST_CONFIG["embargo"]}일')""")

code("""t0 = time.time()

ensemble_cs = run_ensemble_cross_sectional(
    panel_csv=panel_path,
    universe_csv=universe_path,
    out_dir=DATA_DIR,
    config=CS_V4_BEST_CONFIG,
    device=device,
    use_har=True,
    out_name='ensemble_predictions_crosssec.csv',
    verbose=True,
)

elapsed = time.time() - t0
print(f'\\n⏱️ 총 소요 시간: {elapsed/60:.1f} 분')
print(f'결과: {ensemble_cs.shape}')
ensemble_cs.head()""")


# ============================================================================
# §5 결과 검증 + Embedding 분석
# ============================================================================
md("""## §5. 결과 검증 + Ticker Embedding 분석""")

code("""# 5-1. 기본 통계
print('=== 결과 기본 통계 ===')
print(f'행 수: {len(ensemble_cs):,}')
print(f'unique 종목: {ensemble_cs["ticker"].nunique()}')
print(f'date 범위: {ensemble_cs["date"].min().date()} ~ {ensemble_cs["date"].max().date()}')
print()
pred_cols = [c for c in ensemble_cs.columns if 'y_pred' in c]
print(f'예측 컬럼: {pred_cols}')
print('NaN 수:')
print(ensemble_cs[pred_cols + ['y_true']].isna().sum())""")

code("""# 5-2. 종목별 RMSE
def rmse(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)) if mask.sum() > 0 else np.nan

lstm_col = 'y_pred_lstm_cs' if 'y_pred_lstm_cs' in ensemble_cs.columns else 'y_pred_lstm'
ens_col = 'y_pred_ensemble' if 'y_pred_ensemble' in ensemble_cs.columns else lstm_col

rmse_cs = ensemble_cs.groupby('ticker').apply(
    lambda df: pd.Series({
        'rmse_lstm_cs': rmse(df['y_true'].values, df[lstm_col].values),
        'rmse_har': rmse(df['y_true'].values, df['y_pred_har'].values),
        'rmse_ensemble_cs': rmse(df['y_true'].values, df[ens_col].values),
    })
).reset_index()

print('=== CS 모델 종목별 RMSE 통계 ===')
print(rmse_cs[['rmse_lstm_cs', 'rmse_har', 'rmse_ensemble_cs']].describe())""")

code("""# 5-3. Ticker Embedding PCA 시각화 (학습된 모델에서 추출)
try:
    from sklearn.decomposition import PCA

    # 최신 모델 파일 찾기
    model_files = list((DATA_DIR / 'models_cs').glob('*.pt')) if (DATA_DIR / 'models_cs').exists() else []
    if model_files:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        model = CrossSectionalLSTMRegressor(
            input_size=CS_V4_BEST_CONFIG['input_size'],
            hidden_size=CS_V4_BEST_CONFIG['hidden_size'],
            num_layers=CS_V4_BEST_CONFIG['num_layers'],
            dropout=CS_V4_BEST_CONFIG['dropout'],
            n_tickers=cs_inputs['n_tickers'],
            embedding_dim=CS_V4_BEST_CONFIG['embedding_dim'],
        )
        checkpoint = torch.load(latest_model, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

        # Embedding weight 추출
        emb_weights = model.ticker_embedding.weight.detach().numpy()
        ticker_names = sorted(cs_inputs['ticker_to_id'].keys(),
                              key=lambda t: cs_inputs['ticker_to_id'][t])

        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb_weights)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=30)
        for i, t in enumerate(ticker_names[:30]):    # 처음 30 개만 label
            ax.annotate(t, emb_2d[i], fontsize=7, alpha=0.8)
        ax.set_title('Ticker Embedding PCA (2D)', fontsize=12)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'ticker_embedding_pca.png', dpi=100, bbox_inches='tight')
        plt.show()
    else:
        print('저장된 모델 파일 없음 — embedding PCA 스킵')
except Exception as e:
    print(f'Embedding PCA 실패: {e}')""")


# ============================================================================
# §6 02a vs 02b 비교
# ============================================================================
md("""## §6. 02a (종목별) vs 02b (Cross-Sectional) 직접 비교

동일 종목·동일 기간에서 두 방식의 RMSE 직접 비교.
""")

code("""# 02a 결과 로드
sw_path = DATA_DIR / 'ensemble_predictions_stockwise.csv'
if sw_path.exists():
    ensemble_sw = pd.read_csv(sw_path, parse_dates=['date'])

    # 공통 종목 + 공통 기간
    common_tickers = set(ensemble_sw['ticker'].unique()) & set(ensemble_cs['ticker'].unique())
    print(f'공통 종목 수: {len(common_tickers)}')

    # 종목별 RMSE 비교
    sw_rmse = ensemble_sw[ensemble_sw['ticker'].isin(common_tickers)].groupby('ticker').apply(
        lambda df: rmse(df['y_true'].values, df['y_pred_ensemble'].values)
    ).rename('rmse_sw')

    cs_rmse_ser = ensemble_cs[ensemble_cs['ticker'].isin(common_tickers)].groupby('ticker').apply(
        lambda df: rmse(df['y_true'].values, df[ens_col].values)
    ).rename('rmse_cs')

    compare = pd.concat([sw_rmse, cs_rmse_ser], axis=1).dropna()
    compare['cs_better'] = compare['rmse_cs'] < compare['rmse_sw']

    print(f'\\nCS 가 더 좋은 종목: {compare["cs_better"].sum()}/{len(compare)} ({compare["cs_better"].mean()*100:.1f}%)')
    print(f'Stockwise 평균 RMSE: {compare["rmse_sw"].mean():.4f}')
    print(f'Cross-Sec 평균 RMSE: {compare["rmse_cs"].mean():.4f}')

    # 산점도
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(compare['rmse_sw'], compare['rmse_cs'], alpha=0.6, s=40)
    lim_min = min(compare.min().min(), 0)
    lim_max = max(compare.max().max(), 0)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label='동일 성능 선')
    ax.set_xlabel('RMSE (02a 종목별)', fontsize=11)
    ax.set_ylabel('RMSE (02b Cross-Sectional)', fontsize=11)
    ax.set_title('종목별 vs Cross-Sectional RMSE 비교', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'sw_vs_cs_rmse.png', dpi=100, bbox_inches='tight')
    plt.show()
else:
    print(f'02a 결과 없음: {sw_path}')
    print('02a 학습 완료 후 재실행하십시오.')""")

code("""# 최종 요약
print('=== Phase 3 Step 2b 완료 ===')
print(f'산출물: {DATA_DIR / "ensemble_predictions_crosssec.csv"}')
print(f'종목 수: {ensemble_cs["ticker"].nunique()}')
print(f'기간: {ensemble_cs["date"].min().date()} ~ {ensemble_cs["date"].max().date()}')
print()
print('다음 단계:')
print('  03_BL_backtest_extended.ipynb  (BL 백테스트 — 02a + 02b 완료 후)')""")


# ============================================================================
# 저장
# ============================================================================
NB.cells = cells

OUT_PATH = Path(__file__).parent / '02b_phase15_cross_sectional.ipynb'
nbf.write(NB, str(OUT_PATH))
print(f'✅ 저장 완료: {OUT_PATH}')
