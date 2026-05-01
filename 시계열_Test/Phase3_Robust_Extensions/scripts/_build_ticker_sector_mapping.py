"""
613 학습 종목의 sector mapping 생성.

Wikipedia S&P 500 (~500 종목) + hardcoded historical (~110 종목) 결합.
출력: data/ticker_sector_mapping.csv (ticker, sector)

활용:
- 05a_v2_lstm.ipynb §2-G (Sector × Best Model)
- 05a_v2_lstm.ipynb §2-C (Forecast Bias by sector)
- 02a_v2.ipynb §7-H (Sector Exposure)

사용:
    python scripts/_build_ticker_sector_mapping.py
"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = NB_DIR / 'data'
OUT_PATH = DATA_DIR / 'ticker_sector_mapping.csv'

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded historical sector mapping (출처: 02a_v2 §7-6)
# ─────────────────────────────────────────────────────────────────────────────

HARDCODED_SECTOR = {
    # ─── 파산 14 종목 ───
    'SIVB': 'Financials',
    'FRC':  'Financials',
    'JCP':  'Consumer Discretionary',
    'MNK':  'Health Care',
    'CHK':  'Energy',
    'FTR':  'Communication Services',
    'DO':   'Energy',
    'DNR':  'Energy',
    'ENDP': 'Health Care',
    'BIG':  'Consumer Discretionary',
    'EK':   'Information Technology',
    'WIN':  'Communication Services',
    'DF':   'Consumer Staples',
    'ANR':  'Energy',
    # ─── Health Care 인수 ───
    'CELG': 'Health Care', 'AGN': 'Health Care', 'STJ': 'Health Care',
    'CERN': 'Health Care', 'ABMD': 'Health Care', 'ALXN': 'Health Care',
    'HSP': 'Health Care', 'LIFE': 'Health Care', 'COV': 'Health Care',
    'BCR': 'Health Care', 'CFN': 'Health Care', 'BXLT': 'Health Care',
    'CEPH': 'Health Care', 'CVH': 'Health Care', 'WCG': 'Health Care',
    'FRX': 'Health Care', 'SGP': 'Health Care', 'MIL': 'Health Care',
    'VAR': 'Health Care', 'AET': 'Health Care', 'ESRX': 'Health Care',
    'MDVN': 'Health Care', 'PRGO': 'Health Care', 'TMH': 'Health Care',
    'ANTM': 'Health Care', 'ZBH': 'Health Care',
    # ─── IT 인수 ───
    'RHT': 'Information Technology', 'XLNX': 'Information Technology',
    'BRCM': 'Information Technology', 'ALTR': 'Information Technology',
    'LLTC': 'Information Technology', 'MXIM': 'Information Technology',
    'SNDK': 'Information Technology', 'LSI': 'Information Technology',
    'NVLS': 'Information Technology', 'CTXS': 'Information Technology',
    'LXK': 'Information Technology', 'NOVL': 'Information Technology',
    'JDSU': 'Information Technology', 'MFE': 'Information Technology',
    'EMC': 'Information Technology', 'CA': 'Information Technology',
    'CPWR': 'Information Technology', 'INFA': 'Information Technology',
    'TLAB': 'Information Technology',
    # ─── Communication Services 인수 ───
    'DTV': 'Communication Services', 'ATVI': 'Communication Services',
    'TWTR': 'Communication Services', 'TWC': 'Communication Services',
    'PCS': 'Communication Services', 'LVLT': 'Communication Services',
    'SNI': 'Communication Services', 'YHOO': 'Communication Services',
    'TWX': 'Communication Services', 'CTL': 'Communication Services',
    'LBTYA': 'Communication Services',
    # ─── Energy 인수 ───
    'APC': 'Energy', 'PXD': 'Energy', 'MRO': 'Energy', 'NBL': 'Energy',
    'BHI': 'Energy', 'CXO': 'Energy', 'CPGX': 'Energy', 'CAM': 'Energy',
    'ESV': 'Energy', 'XEC': 'Energy', 'WPX': 'Energy', 'QEP': 'Energy',
    'SWN': 'Energy', 'BJS': 'Energy', 'RDC': 'Energy', 'XTO': 'Energy',
    'HFC': 'Energy', 'NFX': 'Energy', 'EP': 'Energy', 'COG': 'Energy',
    'NE': 'Energy', 'RIG': 'Energy', 'SE': 'Energy', 'OKE': 'Energy',
    # ─── Financials 인수 ───
    'ETFC': 'Financials', 'PBCT': 'Financials', 'HCBK': 'Financials',
    'LM': 'Financials', 'NYX': 'Financials', 'JNS': 'Financials',
    'XL': 'Financials', 'TSS': 'Financials', 'FII': 'Financials',
    'GR': 'Financials', 'CBE': 'Financials', 'CB': 'Financials',
    'GENZ': 'Financials',
    # ─── Consumer Discretionary 인수 ───
    'TIF': 'Consumer Discretionary', 'IGT': 'Consumer Discretionary',
    'PETM': 'Consumer Discretionary', 'SPLS': 'Consumer Discretionary',
    'CTX': 'Consumer Discretionary', 'FBHS': 'Consumer Discretionary',
    'FDO': 'Consumer Discretionary', 'APOL': 'Consumer Discretionary',
    'JNY': 'Consumer Discretionary', 'JCI': 'Consumer Discretionary',
    'BMC': 'Consumer Discretionary',
    # ─── Consumer Staples 인수 ───
    'WFM': 'Consumer Staples', 'HNZ': 'Consumer Staples',
    'MJN': 'Consumer Staples', 'GMCR': 'Consumer Staples',
    'DPS': 'Consumer Staples', 'KFT': 'Consumer Staples',
    'KRFT': 'Consumer Staples', 'CCE': 'Consumer Staples',
    'LO': 'Consumer Staples', 'RAI': 'Consumer Staples',
    'AVP': 'Consumer Staples', 'SVU': 'Consumer Staples',
    'SWY': 'Consumer Staples',
    # ─── Industrials 인수 ───
    'PCP': 'Industrials', 'JOY': 'Industrials', 'PLL': 'Industrials',
    'KSU': 'Industrials', 'COL': 'Industrials', 'LLL': 'Industrials',
    'FLIR': 'Industrials', 'AYE': 'Industrials', 'SAI': 'Industrials',
    'RTN': 'Industrials', 'TYC': 'Industrials', 'NLSN': 'Industrials',
    'CVG': 'Industrials',
    # ─── Materials 인수 ───
    'MON': 'Materials', 'SIAL': 'Materials', 'ARG': 'Materials',
    'DWDP': 'Materials', 'AKS': 'Materials', 'X': 'Materials',
    'PX': 'Materials', 'POT': 'Materials',
    # ─── Utilities 인수 ───
    'PGN': 'Utilities', 'POM': 'Utilities', 'GAS': 'Utilities',
    'TEG': 'Utilities', 'TE': 'Utilities', 'STR': 'Utilities',
    'PNW': 'Utilities',
    # ─── Real Estate 인수 ───
    'PCL': 'Real Estate', 'GGP': 'Real Estate', 'DRE': 'Real Estate',
    # ─── 추가 누락 핸들 ───
    'TGNA': 'Communication Services',  # TEGNA (private 2026)
    'WLTW': 'Financials',              # Willis Towers Watson -> WTW
}


def main():
    print('=' * 70)
    print('  Sector Mapping 생성 (Wikipedia + Hardcoded historical)')
    print('=' * 70)

    # 1. 학습 종목 (613) 로드
    ens_sw = pd.read_csv(
        DATA_DIR / 'ensemble_predictions_stockwise.csv',
        usecols=['ticker'],
    )
    learned_tickers = sorted(ens_sw['ticker'].unique())
    print(f'  학습 종목: {len(learned_tickers)}')

    # 2. Wikipedia S&P 500 sector (requests + User-Agent 로 403 우회)
    print()
    print('  [1/2] Wikipedia S&P 500 sector 다운로드 (requests + UA)...')
    import requests
    from io import StringIO
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
    }
    try:
        resp = requests.get(sp500_url, headers=headers, timeout=30)
        resp.raise_for_status()
        sp500_df = pd.read_html(StringIO(resp.text))[0]
        # 컬럼명 확인 (변할 수 있음): Symbol, GICS Sector
        cols = list(sp500_df.columns)
        print(f'    Wikipedia table 컬럼: {cols[:6]}')

        # Symbol / Ticker 컬럼 자동 감지
        sym_col = next((c for c in cols if c.lower() in ('symbol', 'ticker')), 'Symbol')
        sect_col = next((c for c in cols if 'sector' in c.lower() and 'sub' not in c.lower()), 'GICS Sector')
        print(f'    사용 컬럼: ticker={sym_col!r}, sector={sect_col!r}')

        wiki_map = sp500_df.set_index(sym_col)[sect_col].to_dict()
        print(f'    ✅ Wikipedia 매핑: {len(wiki_map)} 종목')
    except Exception as e:
        print(f'    ⚠️ Wikipedia 다운로드 실패: {e}')
        print(f'    yfinance fallback 시도')
        wiki_map = {}

    # 2b. yfinance fallback (Wikipedia 누락 종목만)
    learned_tickers_set = set(pd.read_csv(
        DATA_DIR / 'ensemble_predictions_stockwise.csv',
        usecols=['ticker'],
    )['ticker'].unique())
    yf_needed = sorted(learned_tickers_set - set(wiki_map.keys()) - set(HARDCODED_SECTOR.keys()))
    yf_map = {}
    if not wiki_map and yf_needed:
        print(f'  [1b] yfinance 로 {len(yf_needed)} 종목 sector 다운로드 (~5-10분)...')
        try:
            import yfinance as yf
            for i, t in enumerate(yf_needed):
                try:
                    info = yf.Ticker(t).info
                    sector = info.get('sector', None) or info.get('gicsSector', None)
                    if sector:
                        yf_map[t] = sector
                except Exception:
                    pass
                if (i + 1) % 50 == 0:
                    print(f'    {i+1}/{len(yf_needed)} 진행 ({len(yf_map)} sector 받음)')
            print(f'    ✅ yfinance 매핑: {len(yf_map)} 종목')
        except ImportError:
            print(f'    ⚠️ yfinance 미설치 → skip')
    if yf_map:
        wiki_map.update(yf_map)

    # 3. 통합 매핑 (Wikipedia + Hardcoded, hardcoded 우선)
    print()
    print('  [2/2] 통합 매핑 (Wikipedia + Hardcoded historical)...')
    combined = {**wiki_map, **HARDCODED_SECTOR}  # hardcoded 우선
    print(f'    통합 매핑: {len(combined)} 종목')

    # 4. 학습 종목과 매칭
    results = []
    n_wiki, n_hardcoded, n_unknown = 0, 0, 0
    for t in learned_tickers:
        if t in HARDCODED_SECTOR:
            sector = HARDCODED_SECTOR[t]
            source = 'hardcoded'
            n_hardcoded += 1
        elif t in wiki_map:
            sector = wiki_map[t]
            source = 'wikipedia'
            n_wiki += 1
        else:
            sector = 'Unknown'
            source = 'unknown'
            n_unknown += 1
        results.append({'ticker': t, 'sector': sector, 'source': source})

    df = pd.DataFrame(results)

    # 5. 분포 보고
    print()
    print(f'  ─── 매핑 결과 ({len(df)} 종목) ───')
    print(f'    Wikipedia: {n_wiki:3d} ({n_wiki/len(df)*100:.1f}%)')
    print(f'    Hardcoded: {n_hardcoded:3d} ({n_hardcoded/len(df)*100:.1f}%)')
    print(f'    Unknown:   {n_unknown:3d} ({n_unknown/len(df)*100:.1f}%)')
    print()
    print(f'  ─── Sector 분포 ───')
    sector_cnt = df['sector'].value_counts()
    for s, n in sector_cnt.items():
        print(f'    {s:30s}: {n:3d}')

    if n_unknown > 0:
        print()
        print(f'  ─── Unknown 종목 (앞 20개) ───')
        unknown_tickers = df[df['sector'] == 'Unknown']['ticker'].tolist()
        print(f'    {unknown_tickers[:20]}')
        if len(unknown_tickers) > 20:
            print(f'    ... 외 {len(unknown_tickers) - 20} 종목')

    # 6. 저장
    df[['ticker', 'sector']].to_csv(OUT_PATH, index=False)
    df.to_csv(OUT_PATH.with_suffix('.detailed.csv'), index=False)
    print()
    print(f'  💾 저장: {OUT_PATH.name} ({len(df)} 행)')
    print(f'  💾 저장: {OUT_PATH.with_suffix(".detailed.csv").name} (source 포함)')


if __name__ == '__main__':
    main()
