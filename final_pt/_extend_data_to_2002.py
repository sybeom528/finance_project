"""
_extend_data_to_2002.py — daily_returns + vix 를 2002-01 부터 보강.

목적: LSTM 학습의 fold 정의를 이전 시계열_Test 학습 (panel start 2002) 와
일치시키기 위해 panel 시작을 2002 까지 확장. fold 222~226 의 incremental
학습이 정상 동작하도록 함.

처리 흐름:
1. 기존 prices_raw.pkl 로드 (822 ticker, 2004-01-02~)
2. yfinance 로 같은 universe × 2002-01-01 ~ 2003-12-31 추가 다운로드
3. prices_raw.pkl 갱신 (concat) — date 추가
4. daily_returns.pkl 재계산 후 갱신
5. macro_daily.csv 의 vix 2002-2003 보강 (^VIX 단일 yfinance)
6. 백업 자동 생성 (.bak_pre_2002_ext)

영향 범위 (final/data/):
  - prices_raw.pkl     : 2004-01-02 → 2002-01-02 시작
  - daily_returns.pkl  : 동일
  - macro_daily.csv    : vix 컬럼만 2002-2003 보강 (다른 컬럼 무관)

영향 0 (보존):
  - monthly_panel.csv  : panel 빌드 단계와 무관
  - universe.csv, sp500_membership.pkl 등 다른 파일

소요 시간: ~10-15분 (yfinance 다운로드 + 가공)

실행:
    cd final && python _extend_data_to_2002.py
"""
from __future__ import annotations

import io
import shutil
import sys
import time
from pathlib import Path

# Windows cp949 encoding 우회 — 한글 print 정상 동작 (em dash, 화살표 등 unicode)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import pickle

BASE = Path(__file__).parent
DATA = BASE / 'data'

OLD_START = '2004-01-01'   # 기존 prices_raw 시작 (참고용)
NEW_START = '2002-01-01'   # 신규 시작
NEW_END   = '2003-12-31'   # 추가 다운로드 끝점 (2003-12-31 + 다음 영업일 = 2004 시작과 연결)

PRICES_PATH = DATA / 'prices_raw.pkl'
RET_PATH    = DATA / 'daily_returns.pkl'
MACRO_PATH  = DATA / 'macro_daily.csv'

# 백업 경로
BAK_SUFFIX  = '.bak_pre_2002_ext'

print('━' * 70)
print(' _extend_data_to_2002 — daily_returns + vix 를 2002-01 부터 보강')
print('━' * 70)

# ── 1. 기존 데이터 로드 ──────────────────────────────────────────
print('\n[1/6] 기존 데이터 로드')

with open(PRICES_PATH, 'rb') as f:
    prices_old = pickle.load(f)
print(f'  prices_raw       : {prices_old.shape}, {prices_old.index.min().date()}~{prices_old.index.max().date()}')

with open(RET_PATH, 'rb') as f:
    dr_old = pickle.load(f)
print(f'  daily_returns    : {dr_old.shape}, {dr_old.index.min().date()}~{dr_old.index.max().date()}')

macro_old = pd.read_csv(MACRO_PATH, parse_dates=['date'])
print(f'  macro_daily      : {macro_old.shape}, vix finite={macro_old.vix.notna().sum()}')

# 이미 2002 시작이면 skip
if prices_old.index.min() <= pd.Timestamp(NEW_START):
    print(f'\n  ⚡ 이미 prices_raw 가 {prices_old.index.min().date()} 부터 시작 → 작업 불필요')
    sys.exit(0)

# ── 2. universe 추출 (기존 prices_raw 기준) ──────────────────────
print('\n[2/6] universe 추출 (기존 prices_raw 의 822 ticker)')
if isinstance(prices_old.columns, pd.MultiIndex):
    tickers = sorted(prices_old.columns.get_level_values(1).unique().tolist())
else:
    tickers = sorted(prices_old.columns.tolist())
# 매크로 ticker 분리 (^IRX 등)
macro_tickers = [t for t in tickers if t.startswith('^')]
stock_tickers = [t for t in tickers if not t.startswith('^')]
print(f'  stock tickers: {len(stock_tickers)}')
print(f'  macro tickers: {macro_tickers}')

# ── 3. yfinance 로 2002-2003 prices 다운로드 ────────────────────
print(f'\n[3/6] yfinance 로 {NEW_START} ~ {NEW_END} 다운로드')
print(f'  대상: {len(tickers)} ticker (stock + macro)')

import yfinance as yf

# 50 ticker chunk 단위
chunks = [tickers[i:i + 50] for i in range(0, len(tickers), 50)]
chunk_dfs = []
t0 = time.time()
for i, chunk in enumerate(chunks):
    elapsed = (time.time() - t0) / 60
    print(f'  chunk {i+1}/{len(chunks)} ({len(chunk)} ticker, 경과 {elapsed:.1f}분) ...', flush=True)
    try:
        df_chunk = yf.download(
            tickers=chunk,
            start=NEW_START,
            end='2004-01-02',  # exclusive — 2003-12-31 까지 포함
            auto_adjust=False,
            group_by='column',     # 같은 field 끼리 그룹 (Field × Ticker MultiIndex)
            threads=True,
            progress=False,
        )
        if df_chunk is not None and not df_chunk.empty:
            chunk_dfs.append(df_chunk)
    except Exception as e:
        print(f'    chunk {i+1} 실패: {e}')

if not chunk_dfs:
    print('  ❌ 다운로드 실패 — 모든 chunk 실패')
    sys.exit(1)

# concat all chunks (column-wise)
prices_new = pd.concat(chunk_dfs, axis=1)
print(f'\n  다운로드 완료: shape={prices_new.shape}')
print(f'  date 범위: {prices_new.index.min().date()} ~ {prices_new.index.max().date()}')

# ── 4. prices_raw.pkl 갱신 (백업 + concat) ──────────────────────
print('\n[4/6] prices_raw.pkl 갱신')

# 백업
bak = PRICES_PATH.with_suffix(PRICES_PATH.suffix + BAK_SUFFIX)
if not bak.exists():
    print(f'  백업 생성: {bak.name}')
    shutil.copy2(PRICES_PATH, bak)

# 컬럼 정렬: 새로 다운받은 데이터의 컬럼이 기존과 정확히 매칭되도록 reindex
# 기존 prices_old.columns 와 정확히 같은 column 으로 reindex (없는 ticker는 NaN)
prices_new = prices_new.reindex(columns=prices_old.columns)

# 2002-2003 행 + 기존 2004-2026 행 concat
combined = pd.concat([prices_new, prices_old]).sort_index()
# 중복 제거 (2003-12-31 등이 양쪽에 있으면 기존 우선)
combined = combined[~combined.index.duplicated(keep='last')]

print(f'  combined shape: {combined.shape}')
print(f'  date 범위: {combined.index.min().date()} ~ {combined.index.max().date()}')

with open(PRICES_PATH, 'wb') as f:
    pickle.dump(combined, f)
print(f'  ✅ 저장: {PRICES_PATH}')

# ── 5. daily_returns.pkl 재계산 ──────────────────────────────────
print('\n[5/6] daily_returns.pkl 재계산')

# 백업
bak_ret = RET_PATH.with_suffix(RET_PATH.suffix + BAK_SUFFIX)
if not bak_ret.exists():
    shutil.copy2(RET_PATH, bak_ret)
    print(f'  백업 생성: {bak_ret.name}')

# Close 추출 (또는 Adj Close 우선 사용)
if isinstance(combined.columns, pd.MultiIndex):
    close = combined['Close'].copy()
else:
    close = combined.copy()
close = close[~close.index.duplicated(keep='last')]

# log return 계산
log_ret = np.log(close / close.shift(1))
# 첫 행 NaN — 그대로 두면 학습 시 자동 제외

# 기존 daily_returns 의 종목과 동일하게 reindex (^IRX 등 매크로 빼고)
# 단, dr_old 에 있는 종목만 keep
keep_cols = [c for c in dr_old.columns if c in log_ret.columns]
log_ret = log_ret[keep_cols]

print(f'  daily_returns shape: {log_ret.shape}')
print(f'  date 범위: {log_ret.index.min().date()} ~ {log_ret.index.max().date()}')

with open(RET_PATH, 'wb') as f:
    pickle.dump(log_ret, f)
print(f'  ✅ 저장: {RET_PATH}')

# ── 6. macro_daily.csv 의 vix 2002-2003 보강 ────────────────────
print('\n[6/6] macro_daily.csv 의 vix 보강')

# 백업
bak_macro = MACRO_PATH.with_suffix(MACRO_PATH.suffix + BAK_SUFFIX)
if not bak_macro.exists():
    shutil.copy2(MACRO_PATH, bak_macro)
    print(f'  백업 생성: {bak_macro.name}')

# ^VIX 다운로드 (2002-2003)
print(f'  ^VIX 다운로드 ({NEW_START} ~ 2004-01-02) ...')
try:
    vix_new = yf.download(
        '^VIX',
        start=NEW_START,
        end='2004-01-02',
        auto_adjust=False,
        progress=False,
    )
    if vix_new is None or vix_new.empty:
        print('  ⚠ ^VIX 다운로드 결과 비어있음 — vix 보강 skip')
    else:
        # Close 컬럼 사용
        if isinstance(vix_new.columns, pd.MultiIndex):
            vix_close = vix_new[('Close', '^VIX')]
        else:
            vix_close = vix_new['Close']
        # macro 와 동일 형식으로 변환
        vix_df = pd.DataFrame({
            'date': vix_close.index,
            'vix': vix_close.values,
        })
        # 기존 macro 와 concat
        macro_old_keep_cols = [c for c in macro_old.columns]
        # vix 만 추가, 다른 컬럼은 NaN
        vix_df_full = pd.DataFrame(columns=macro_old_keep_cols)
        vix_df_full['date'] = vix_df['date']
        vix_df_full['vix'] = vix_df['vix']
        # concat (2002-2003 + 2004-2026)
        macro_combined = pd.concat([vix_df_full, macro_old]).sort_values('date')
        macro_combined = macro_combined.drop_duplicates(subset='date', keep='last')

        macro_combined.to_csv(MACRO_PATH, index=False)
        print(f'  ✅ macro_daily.csv 갱신: shape={macro_combined.shape}')
        print(f'     date 범위: {macro_combined.date.min().date()} ~ {macro_combined.date.max().date()}')
        print(f'     vix finite: {macro_combined.vix.notna().sum()}')
except Exception as e:
    print(f'  ⚠ ^VIX 다운로드 실패: {e}')
    print(f'     macro_daily.csv 무수정 (vix 2002-2003 NaN 으로 남음)')

print('\n' + '━' * 70)
print(' 작업 완료 — 03 노트북 재실행 준비 완료')
print('━' * 70)
print()
print('다음 단계:')
print('  1. final/03_Volatility_Forecasting.ipynb 의 cell 13 §4.2 에서')
print("     PANEL_MAX_DATE = '2026-04-30' 또는 None 설정")
print('     (현재 2026-02-01 인 경우 변경 필요)')
print('  2. 노트북 종료 → 재시작 → Run All')
print('  3. 자동 동작:')
print('     - daily_panel 재빌드 (2002-01 시작)')
print('     - fold csv 보존 (이미 truncate 됨, fold 0~221)')
print('     - incremental 학습 — fold 222~226 추가 (~30-40분 GPU)')
print('     - BL pkl 재계산 (~25-30분)')
print()
print('롤백 (필요 시):')
print(f'  prices_raw.pkl{BAK_SUFFIX} → prices_raw.pkl')
print(f'  daily_returns.pkl{BAK_SUFFIX} → daily_returns.pkl')
print(f'  macro_daily.csv{BAK_SUFFIX} → macro_daily.csv')
