"""모든 results/*.pkl을 로드해서 Sharpe·CAGR·MDD 순위 출력."""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / 'results'
DATA_DIR    = Path(__file__).parent.parent / 'data'

panel_rf = pd.read_csv(DATA_DIR / 'monthly_panel.csv',
                       usecols=['date','ticker','rf_1m'], parse_dates=['date'])
rf = panel_rf.groupby('date')['rf_1m'].first()

results = {}
for pkl in sorted(RESULTS_DIR.glob('*.pkl')):
    with open(pkl, 'rb') as f:
        results[pkl.stem] = pickle.load(f)

def calc(name):
    r = results[name]['ret'].dropna()
    rf_a = rf.reindex(r.index).fillna(0)
    exc  = r - rf_a
    ann  = exc.mean() * 12
    vol  = r.std() * np.sqrt(12)
    sh   = ann / vol if vol > 0 else np.nan
    cum  = (1 + r).cumprod()
    mdd  = ((cum / cum.cummax()) - 1).min()
    cagr = r.mean() * 12
    return dict(sharpe=sh, cagr=cagr, vol=vol, mdd=mdd, total=cum.iloc[-1]-1)

# SPY benchmark
spy_ret = results['baseline']['spy_ret'].dropna()
rf_a_spy = rf.reindex(spy_ret.index).fillna(0)
spy_sh = (spy_ret - rf_a_spy).mean() * 12 / (spy_ret.std() * np.sqrt(12))
spy_cum = (1 + spy_ret).cumprod()
spy_mdd = ((spy_cum / spy_cum.cummax()) - 1).min()

rows = []
for name in results:
    m = calc(name)
    rows.append({
        '실험명': name,
        'Sharpe': round(m['sharpe'], 3),
        'CAGR': round(m['cagr']*100, 2),
        'Vol': round(m['vol']*100, 2),
        'MDD': round(m['mdd']*100, 2),
        '누적수익': round(m['total']*100, 1),
    })

df = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
print(df.to_string(index=False))
print()
print(f"SPY benchmark: Sharpe {spy_sh:.3f} | MDD {spy_mdd*100:.2f}% | 누적 {(spy_cum.iloc[-1]-1)*100:.1f}%")
print(f"기간: {spy_ret.index[0].date()} ~ {spy_ret.index[-1].date()} ({len(spy_ret)} months)")
print(f"실험 수: {len(results)}")
