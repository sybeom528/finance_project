# 성향별 액티브 ETF 펀드 — Low-Risk BL 프로젝트

> **저변동성 anomaly + LSTM σ 예측 + Black-Litterman 단일 view 프레임워크로 위험성향별 펀드 후보를 탐색하는 quant 프로젝트.**
>
> 최종 갱신: 2026-05-07

---

## 파이프라인 (7-Step)

```
1. Data Collection      ✅ 01_DataCollection.ipynb       — DATA_COLLECTION.md
2. Preprocessing/EDA    ✅ 02_LowRiskAnomaly.ipynb       — ANOMALY_ANALYSIS.md
3. LSTM σ 예측           ✅ phase3(data_outputs)/         (다른 팀원 영역)
4. Black-Litterman       ✅ 99_run.ipynb (walk_forward)
5. MVO + 위험성향 매핑    🟡 99_analyze.ipynb (분석 진행 중)
6. 3-레짐 안정성 + 민감도  ✅ 99_analyze.ipynb (I/J/K/L/M/N)
7. Streamlit 대시보드     ❌ 미착수
```

---

## 파일 구조

```
final/
├── 99_run.ipynb              ← walk_forward 실행 → results/*.pkl
├── 99_analyze.ipynb          ← 분석 (K_CUT → I → J → K → L → M → N)
├── 99_slot_effects.ipynb     ← 슬롯 차원 효과 라인플롯 (pivot CSV 자동 생성)
├── 01_DataCollection.ipynb   ← 데이터 수집
├── 02_LowRiskAnomaly.ipynb   ← anomaly EDA
│
├── bl_config.py              ← EXPERIMENTS 정의 (156개)
├── bl_functions.py           ← BL 핵심 함수 (Σ/π/P/Q/Ω/BL/TC/Metrics)
├── master_table.py           ← results/*.pkl → mt/rt 빌더
├── analyze_plots.py          ← 시각화 모듈
│
├── results/                  ← 156 pkl
├── data/                     ← monthly_panel.csv, daily_returns.pkl, ff3_monthly.csv
├── outputs/                  ← 차트 PNG
└── phase3(data_outputs)/     ← LSTM 예측 산출 (다른 팀원)
```

---

## 문서 가이드

| 목적 | 참고 |
|---|---|
| 실험 슬롯 수식 + 추가법 + 실행법 | [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md) |
| 데이터 수집 파이프라인 | [DATA_COLLECTION.md](DATA_COLLECTION.md) |
| 저변동 anomaly 검증 EDA | [ANOMALY_ANALYSIS.md](ANOMALY_ANALYSIS.md) |
| 학술 reference (Pyo & Lee 2018) | [Exploiting_LowRisk_Anomaly_BL_Summary.md](Exploiting_LowRisk_Anomaly_BL_Summary.md) |
| 분석 노트북 셀 해설 | `99_analyze.ipynb` 안 markdown 셀 |

---

## 핵심 결과 (현 시점)

- **Top 후보** (sortino_ir 기준): `mcap_ls_eq_lam_pap` (Sharpe 1.140, CAGR 16.23%, MDD -14.68%)
- **baseline 대비**: Sharpe +0.03, CAGR +2.7%p, MDD 비슷한 수준 유지
- **SPY 대비**: 모든 위험조정 지표에서 우위 (특히 위기 방어)
- **Q 민감도**: q=0.003이 우리 setup의 robust optimum (BAB 학술값 0.0064는 단조 감소)
- **레짐 안정성**: 3-레짐 (R1 회복 / R2 확장 / R3 변동) HMM 기반

---

## 명명 체계 — canonical 5-token

```
{prior}_{p_mode}_{p_weight}_{q}_{omega}
mcap   _ls    _eq      _lam_pap  ← 예
```

토큰 정의 + 슬롯 수식 → [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md) §4 참고
