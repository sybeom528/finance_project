"""
02a_v2.ipynb 에 §7-K (V자 반등 분석) 셀 추가.
"""
import json
import sys
import io
import uuid
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
NB_PATH = Path(__file__).resolve().parent.parent / '02a_v2.ipynb'

SEC7_K_HEADER = """## §7-K. V자 반등 분석 (Recovery + Top Holdings + 대형주 비중 + Vol Prediction)

**목적:**
- BL_trailing_mcap 의 OOS Sharpe 1.206 우위가 대형주 V자 반등 덕분인지 검증
- BL_ml_sw_mcap 의 OOS MDD -18.13% / Sharpe 1.082 약점이 V자 반등 실패 때문인지 분석

**핵심 발견 (preview):**
- ✅ 부분 맞음: Pre-COVID (2018-19) 시기 trailing_mcap 대형주 비중 +5.9%p 더 컸음
- ❌ 표면적 해석 틀림: **ML 이 2020 COVID 6개월 반등 +21.99% 로 trailing (+21.90%) 보다 우위**
- ⭐ 진짜 원인: drawdown depth (-1.65%p 더 깊음) + 반등 후 빅테크 over-concentration (top10 share 61.7% vs 52.5%)"""


SEC7_K_CODE = '''# §7-K. V자 반등 분석 (사용자 가설 검증)
import pickle
print('=' * 75)
print('  §7-K. V자 반등 분석')
print('=' * 75)

sec7_k_path = DATA_DIR / 'sec7_k_recovery_analysis.pkl'
if not sec7_k_path.exists():
    print(f'⚠️ {sec7_k_path.name} 부재 → standalone 스크립트 (`_run_02a_v2_sec7_recovery.py`) 먼저 실행')
else:
    with open(sec7_k_path, 'rb') as f:
        sec7_k = pickle.load(f)

    # ─── A. Recovery 능력 비교 ───
    print()
    print('  --- A. Recovery 6개월 누적 (drawdown trough 후) ---')
    rec_results = sec7_k['recovery_results']
    rec_df = []
    for name, events in rec_results.items():
        if not events:
            continue
        avg_recov = np.mean([e['recov_6m_%'] for e in events])
        max_recov = max([e['recov_6m_%'] for e in events])
        rec_df.append({
            'scenario': name,
            'n_events': len(events),
            'avg_recov_6m_%': round(avg_recov, 2),
            'max_recov_6m_%': round(max_recov, 2),
        })
    rec_df = pd.DataFrame(rec_df).set_index('scenario')
    print(rec_df.to_string())

    # 2020 COVID event 직접 비교
    print()
    print('  ⭐ 2020 COVID Event (start=2020-01, trough=2020-02): 6개월 반등 비교')
    for name, events in rec_results.items():
        for e in events:
            if e['trough'].startswith('2020-02'):
                print(f'    {name:<25s}: depth {e["depth_%"]:+.2f}% / recov_6m {e["recov_6m_%"]:+.2f}%')
                break

    # ─── B. Top 10 Holdings 비교 ───
    print()
    print('  --- B. Top 10 Holdings 비교 (특정 시기, mcap 시나리오) ---')
    holdings = sec7_k['holdings_comparison']
    for label, data in holdings.items():
        print(f'\\n  [{label} (reb_date: {data["reb_date"]})]')
        for s_name, h in data['top10'].items():
            print(f'    {s_name}: top10 share {h["top10_share_%"]:.1f}% / total long {h["n_total_long"]}')
            for t, w in zip(h['tickers'][:5], h['weights_%'][:5]):
                print(f'      {t:6s} {w:5.2f}%')

    # ─── C. 대형주 (top 50) 비중 ───
    print()
    print('  --- C. 대형주 (Top 50 mcap) 비중 통계 ---')
    lc_ts = sec7_k['largecap_share_timeseries']
    lc_summary = {}
    for name, ts_dict in lc_ts.items():
        ts = pd.Series(ts_dict)
        lc_summary[name] = {
            'mean_%': ts.mean() * 100,
            'std_%': ts.std() * 100,
            'min_%': ts.min() * 100,
            'max_%': ts.max() * 100,
        }
    lc_df = pd.DataFrame(lc_summary).T.round(2)
    print(lc_df.to_string())

    # 시기별 비교 (mcap 시나리오 강조)
    print()
    print('  --- 시기별 평균 대형주 (top 50) 비중 (mcap 환경) ---')
    period_lc = sec7_k['period_largecap']
    period_df = pd.DataFrame(period_lc).T.round(2)
    print(period_df.to_string())

    # 시각화
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # (1) 대형주 비중 시계열
    ax = axes[0]
    for name, ts_dict in lc_ts.items():
        ts = pd.Series(ts_dict).sort_index()
        ts.index = pd.to_datetime(ts.index)
        color = '#1f77b4' if 'ml_sw' in name else '#d62728'
        style = '-' if 'mcap' in name else '--'
        ax.plot(ts.index, ts.values * 100, label=name, color=color, linestyle=style, alpha=0.8)
    ax.set_title('대형주 (Top 50 mcap) 비중 시계열 — 4 시나리오')
    ax.set_ylabel('Top 50 share %')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (2) 시기별 막대 차트
    ax = axes[1]
    period_df.plot(kind='bar', ax=ax, width=0.85)
    ax.set_title('시기별 대형주 비중 (4 시나리오)')
    ax.set_ylabel('Top 50 share %')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha='right')

    plt.tight_layout()
    out_path = OUT_DIR_V2_SW / 'recovery_largecap_share.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f'  💾 {out_path.name}')

    # ─── D. 변동성 예측 비교 ───
    print()
    print('  --- D. ML vs Trailing 변동성 예측 비교 (rank correlation + overlap) ---')
    vol_comp = sec7_k['vol_comparison']
    vol_df = []
    for label, d in vol_comp.items():
        vol_df.append({
            'event': label,
            'reb_date': d['reb_date'],
            'rank_corr': round(d['rank_correlation'], 3),
            'pearson_corr': round(d['pearson_corr'], 3),
            'low_overlap_%': round(d['low_overlap_%'], 1),
            'high_overlap_%': round(d['high_overlap_%'], 1),
        })
    vol_df = pd.DataFrame(vol_df).set_index('event')
    print(vol_df.to_string())

    print()
    print('=' * 75)
    print('  §7-K 종합 결론 (사용자 가설 검증)')
    print('=' * 75)
    print('  ✅ 부분 맞음:')
    print('    - Pre-COVID (2018-19) 시기 trailing_mcap 대형주 비중 +5.9%p 더 큼')
    print('    - 이 차이가 2018-19 강세장 + 2020 회복기 누적 우위 기여')
    print()
    print('  ❌ 표면적 해석 틀림:')
    print('    - ML 이 2020 COVID 6개월 반등 +21.99% 로 trailing (+21.90%) 보다 우위')
    print('    - 즉 "ML 이 V자 반등을 못 따라갔다" 는 잘못된 해석')
    print()
    print('  ⭐ 진짜 원인:')
    print('    - drawdown depth: ML -18.13% vs Trailing -16.48% (-1.65%p 더 깊음)')
    print('    - 반등 후 over-concentration: ML top10 share 61.7% vs Trailing 52.5%')
    print('    - ML 빅테크 4사 집중 (AAPL+MSFT+GOOGL+AMZN = 38.15%)')
    print('    - Trailing 은 META 추가로 분산 (5.82%)')
    print('    - 2020-Q4 ~ 2021 빅테크 약세에 ML 의 over-concentration 부담')
    print()
    print('  📊 ML 의 가치는 충격 시기 차별화:')
    print('    - 정상 시기 (2022-08 긴축 trough): rank corr 0.894 (ML/Trail 거의 동일)')
    print('    - 충격 시기 (2020-02 COVID): rank corr 0.756 (ML 의 차별화 정보 발현)')
    print('    - → ML 의 추가 정보는 충격 시기 entry 결정에서만 의미')
'''


def split_src(s):
    lines = s.rstrip('\n').split('\n')
    return [ln + '\n' for ln in lines[:-1]] + [lines[-1]]


def make_cell(src_str, cell_type='code'):
    cell = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": split_src(src_str),
    }
    if cell_type == 'code':
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


# 노트북에 §7-K 추가
print('=' * 70)
print('  02a_v2.ipynb 에 §7-K 셀 추가')
print('=' * 70)

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)
print(f'  원래 셀 수: {len(nb["cells"])}')

new_cells = [
    make_cell(SEC7_K_HEADER, cell_type='markdown'),
    make_cell(SEC7_K_CODE),
]
nb['cells'] = nb['cells'] + new_cells

print(f'  추가: 1 markdown + 1 code')
print(f'  새 셀 수: {len(nb["cells"])}')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'  💾 저장: {NB_PATH.name}')
