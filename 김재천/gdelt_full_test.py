"""
GDELT 1일치 전체 수집 테스트 (coverage=True)
Windows: if __name__ == '__main__' 필수 (ProcessPoolExecutor 사용)
"""
import gdelt
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

COL_DATE, COL_THEMES, COL_ORGS, COL_TONE = 'DATE', 'Themes', 'Organizations', 'V2Tone'

TICKERS = [
    'SPY','QQQ','IWM','EFA','EEM',
    'TLT','AGG','SHY','TIP',
    'GLD','DBC',
    'XLK','XLF','XLE','XLV','VOX','XLY','XLP','XLI','XLU','XLRE','XLB',
    'AAPL','MSFT','AMZN','GOOGL','JPM','JNJ','PG','XOM'
]

# ── 수정된 AMZN 포함 전체 매핑 테이블 ─────────────────────────
TICKER_TO_GDELT = {
    'AAPL': {'type':'org','names':['APPLE INC','APPLE COMPUTER','APPLE']},
    'MSFT': {'type':'org','names':['MICROSOFT CORP','MICROSOFT CORPORATION','MICROSOFT']},
    # AMZN: org_themed 타입 — 기업명 + 경제/기술 테마 AND 조합으로 지리명 노이즈 차단
    'AMZN': {'type':'org_themed',
             'names':['AMAZON.COM','AMAZON INC','AMAZON WEB SERVICES','AWS'],
             'themes':['ECON_STOCKMARKET','TECHNOLOGY','ECON_RETAIL']},
    'GOOGL':{'type':'org','names':['ALPHABET INC','GOOGLE','ALPHABET']},
    'JPM':  {'type':'org','names':['JPMORGAN CHASE','JP MORGAN','JPMORGAN']},
    'JNJ':  {'type':'org','names':['JOHNSON & JOHNSON','JOHNSON AND JOHNSON']},
    'PG':   {'type':'org','names':['PROCTER & GAMBLE','PROCTER AND GAMBLE']},
    'XOM':  {'type':'org','names':['EXXON MOBIL','EXXONMOBIL','EXXON']},
    'SPY':  {'type':'keyword','themes':['ECON_STOCKMARKET'],
             'keywords':['S&P 500','S&P500','STOCK MARKET']},
    'QQQ':  {'type':'keyword','themes':['ECON_STOCKMARKET','TECHNOLOGY'],
             'keywords':['NASDAQ','TECH STOCKS','TECHNOLOGY SECTOR']},
    'IWM':  {'type':'keyword','themes':['ECON_STOCKMARKET'],
             'keywords':['RUSSELL 2000','SMALL CAP','SMALL-CAP']},
    'EFA':  {'type':'keyword','themes':['ECON_STOCKMARKET'],
             'keywords':['MSCI EAFE','DEVELOPED MARKET','INTERNATIONAL EQUITY']},
    'EEM':  {'type':'keyword','themes':['ECON_STOCKMARKET'],
             'keywords':['MSCI EMERGING','EMERGING MARKET','EM EQUITY']},
    'XLK':  {'type':'keyword','themes':['TECHNOLOGY'],
             'keywords':['TECHNOLOGY SECTOR','TECH SECTOR','INFORMATION TECHNOLOGY']},
    'XLF':  {'type':'keyword','themes':['ECON_STOCKMARKET'],
             'keywords':['FINANCIAL SECTOR','BANKING SECTOR','FINANCIAL STOCKS']},
    'XLE':  {'type':'keyword','themes':['ENV_OIL','ENERGY'],
             'keywords':['ENERGY SECTOR','OIL STOCKS','ENERGY STOCKS']},
    'XLV':  {'type':'keyword','themes':['HEALTH'],
             'keywords':['HEALTH CARE SECTOR','HEALTHCARE STOCKS','PHARMA SECTOR']},
    'VOX':  {'type':'keyword','themes':['TECHNOLOGY'],
             'keywords':['COMMUNICATION SERVICES','TELECOM SECTOR']},
    'XLY':  {'type':'keyword','themes':['ECON_RETAIL'],
             'keywords':['CONSUMER DISCRETIONARY','CONSUMER SPENDING','RETAIL SECTOR']},
    'XLP':  {'type':'keyword','themes':['ECON_RETAIL'],
             'keywords':['CONSUMER STAPLES','DEFENSIVE STOCKS','STAPLES SECTOR']},
    'XLI':  {'type':'keyword','themes':['ECON_TRADE'],
             'keywords':['INDUSTRIAL SECTOR','INDUSTRIALS','MANUFACTURING SECTOR']},
    'XLU':  {'type':'keyword','themes':['ENERGY'],
             'keywords':['UTILITIES SECTOR','ELECTRIC UTILITIES','UTILITY STOCKS']},
    'XLRE': {'type':'keyword','themes':['ECON_REALESTATE'],
             'keywords':['REAL ESTATE SECTOR','REIT SECTOR','PROPERTY SECTOR']},
    'XLB':  {'type':'keyword','themes':['COMMODITY'],
             'keywords':['MATERIALS SECTOR','BASIC MATERIALS','CHEMICALS SECTOR']},
    'TLT':  {'type':'keyword','themes':['ECON_INTEREST_RATES'],
             'keywords':['20-YEAR TREASURY','30-YEAR BOND','LONG-TERM TREASURY']},
    'AGG':  {'type':'keyword','themes':['ECON_INTEREST_RATES'],
             'keywords':['INVESTMENT GRADE','BOND MARKET','AGGREGATE BOND']},
    'SHY':  {'type':'keyword','themes':['ECON_INTEREST_RATES'],
             'keywords':['2-YEAR TREASURY','SHORT-TERM TREASURY','FED FUNDS RATE']},
    'TIP':  {'type':'keyword','themes':['ECON_INTEREST_RATES','ECON_INFLATION'],
             'keywords':['TIPS','INFLATION PROTECTED','REAL YIELD']},
    'GLD':  {'type':'keyword','themes':['ECON_GOLD','COMMODITY'],
             'keywords':['GOLD PRICE','PRECIOUS METALS','GOLD MARKET']},
    'DBC':  {'type':'keyword','themes':['COMMODITY'],
             'keywords':['COMMODITY INDEX','RAW MATERIALS','COMMODITY MARKET']},
}

def extract_gdelt_for_ticker(gkg, ticker):
    if ticker not in TICKER_TO_GDELT:
        return pd.DataFrame()
    config = TICKER_TO_GDELT[ticker]

    if config['type'] == 'org':
        pattern = '|'.join(config['names'])
        mask = gkg[COL_ORGS].str.contains(pattern, na=False, case=False)

    elif config['type'] == 'org_themed':
        # 기업명 매칭 AND 테마 필터 — 지리명 오매핑 방지
        name_pattern = '|'.join(config['names'])
        mask_name = gkg[COL_ORGS].str.contains(name_pattern, na=False, case=False)
        theme_pattern = '|'.join(config['themes'])
        mask_theme = gkg[COL_THEMES].str.contains(theme_pattern, na=False, case=False)
        mask = mask_name & mask_theme

    elif config['type'] == 'keyword':
        theme_pattern = '|'.join(config['themes'])
        mask_theme = gkg[COL_THEMES].str.contains(theme_pattern, na=False, case=False)
        kw_pattern = '|'.join(config['keywords'])
        mask_kw = (
            gkg[COL_ORGS].str.contains(kw_pattern, na=False, case=False) |
            gkg[COL_THEMES].str.contains(kw_pattern, na=False, case=False)
        )
        mask = mask_theme & mask_kw
    else:
        return pd.DataFrame()

    matched = gkg[mask].copy()
    matched['ticker'] = ticker
    return matched


if __name__ == '__main__':
    out = open(
        "C:/Users/gorhk/최종 프로젝트/finance_project/김재천/gdelt_test_result.txt",
        "w", encoding="utf-8"
    )

    def log(msg=""):
        print(msg)
        out.write(msg + "\n")
        out.flush()

    log("=" * 55)
    log("GDELT 1일치 전체 수집 테스트 (coverage=True)")
    log("=" * 55)

    gd = gdelt.gdelt(version=2)
    log("수집 중... (1~3분 소요)")
    gkg = gd.Search(['2025 Apr 14'], table='gkg', coverage=True)
    log(f"수집 완료: {gkg.shape[0]:,}행 x {gkg.shape[1]}열")

    # ── 종목별 필터링 ──────────────────────────────────────────
    log("\n종목별 필터링 실행 중...")
    all_rows = []
    rows_per_ticker = {}
    for ticker in TICKERS:
        m = extract_gdelt_for_ticker(gkg, ticker)
        rows_per_ticker[ticker] = len(m)
        if not m.empty:
            all_rows.append(m)

    # ── 결과 출력 ──────────────────────────────────────────────
    type_map = {t:'대형주' for t in ['AAPL','MSFT','AMZN','GOOGL','JPM','JNJ','PG','XOM']}
    type_map.update({t:'채권ETF' for t in ['TLT','AGG','SHY','TIP']})
    type_map.update({t:'대안ETF' for t in ['GLD','DBC']})
    for t in TICKERS:
        if t not in type_map: type_map[t] = '시장/섹터ETF'

    log("\n=== 종목별 필터링 결과 (coverage=True, 1일치) ===")
    for t in TICKERS:
        cnt = rows_per_ticker[t]
        ok  = "O" if cnt > 0 else "X"
        pct = f"({cnt/gkg.shape[0]*100:.2f}%)" if cnt > 0 else ""
        log(f"  [{ok}] {t:5s} ({type_map[t]:9s}) | {cnt:5d}건 {pct}")

    mapped = sum(1 for v in rows_per_ticker.values() if v > 0)
    log(f"\n커버 종목: {mapped}/{len(TICKERS)}개 ({mapped/len(TICKERS)*100:.0f}%)")

    # ── 집계 ──────────────────────────────────────────────────
    if all_rows:
        gdelt_mapped = pd.concat(all_rows, ignore_index=True)
        tone_split = gdelt_mapped[COL_TONE].str.split(',', expand=True)
        gdelt_mapped['_tone_avg']   = pd.to_numeric(tone_split[0], errors='coerce')
        gdelt_mapped['_tone_neg']   = pd.to_numeric(tone_split[2], errors='coerce')
        gdelt_mapped['_tone_polar'] = pd.to_numeric(tone_split[3], errors='coerce')
        gdelt_mapped['_date'] = pd.to_datetime(
            gdelt_mapped[COL_DATE].astype(str).str[:8], format='%Y%m%d', errors='coerce'
        )
        result = (
            gdelt_mapped.groupby(['_date','ticker'])
            .agg(tone_avg=('_tone_avg','mean'), tone_neg=('_tone_neg','mean'),
                 tone_polar=('_tone_polar','mean'), event_cnt=('_tone_avg','count'),
                 tone_std=('_tone_avg','std'))
            .reset_index().rename(columns={'_date':'date'})
        )
        log(f"\n=== (date, ticker) 집계 결과 ({result.shape}) ===")
        log(result.sort_values('event_cnt', ascending=False).to_string(index=False))

        # 노이즈 점검
        for check_t in ['AAPL','AMZN','GLD','TLT']:
            rows = extract_gdelt_for_ticker(gkg, check_t)
            log(f"\n=== 노이즈 점검: {check_t} ({len(rows)}건) ===")
            if not rows.empty:
                for _, row in rows[[COL_THEMES, COL_ORGS]].head(3).iterrows():
                    log(f"  기관: {str(row[COL_ORGS])[:90]}")
                    log(f"  테마: {str(row[COL_THEMES])[:70]}")
            else:
                # 매핑 실패 시 raw 데이터에서 직접 검색
                kw = check_t.replace('AAPL','APPLE').replace('AMZN','AMAZON').replace('GLD','GOLD').replace('TLT','TREASURY')
                raw = gkg[gkg[COL_ORGS].str.contains(kw, na=False, case=False)]
                log(f"  '{kw}' raw 검색: {len(raw)}건")
                for v in raw[COL_ORGS].dropna().head(3).values:
                    log(f"    {str(v)[:90]}")

        # V2Tone 통계
        log("\n=== 전체 V2Tone 통계 ===")
        tone_all = pd.to_numeric(gkg[COL_TONE].str.split(',').str[0], errors='coerce').dropna()
        log(tone_all.describe().round(3).to_string())

    log("\n완료.")
    out.close()
