# Streamlit 대시보드 설계 plan v1 — Low-Risk BL × LSTM σ × Risk-Profile

> **작성일**: 2026-05-08
> **상태**: 방향성 확정, 페이지·KPI·구현 순서 매핑 완료. 다음 단계 = `lib/data_loader.py` 부터 단계적 재구축 착수

---

## 1. 방향성 (Q&A 결과 확정)

| 축 | 결정 |
|---|---|
| **주 청중** | 투자자 / 펀드 운용 의사결정자 |
| **핵심 메시지** | 위험성향별 맞춤 펀드 후보 제공 |
| **Top 모델 처리** | Top 3~5 카드 (단일 모델 강조 X, 비교 도구로 작동) |
| **페이지 깊이** | 단계적 5 페이지 (Home / Overview / Compare / Matrix / RiskProfile) |
| **위험성향 분류** | sortino 우선 + sortino_ir + MDD + vol 종합 점수 |
| **HOLD-OUT 처리** | TEST 168m 우선 노출, HO는 "실전 검증" 탭에 분리 |
| **벤치마크 비교** | 자유 다중 선택 (multi-select dropdown, 최대 7개) |

### 1-1. 디자인 함의 — 위 결정이 만드는 톤

투자자 청중은 *"내 위험 성향에 맞는 후보가 무엇이고, 그게 SPY·baseline 대비 얼마나 매력적이며, 최근 24개월에도 작동하는가"* 가 궁금합니다. 따라서 학술 발표용 대시보드와 다음 3가지가 다릅니다.

1. **숫자 단위가 % 와 $** — log-RMSE 같은 학술 단위는 사이드바 보조표기로만 노출. 메인은 `Sharpe 1.14`, `CAGR 16.2%`, `$10,000 → $X,XXX`.
2. **"왜 이 후보인가" 한 줄 내러티브 필수** — 매 카드마다 한 문장 narrative. (예: *"S&P 500 변동성 하위 30% 시총가중 + LSTM σ 동적 view = 2010년 이후 SPY 대비 ~CAGR +X%p, MDD 작음"*)
3. **HO 24m 부진 정직 공개** — 숨기지 않되 "실전 검증" 탭에 격리해서 메인 첫인상을 망치지 않음. 부진 원인(2024-12 sector rotation, 방어주 77.6% 집중)은 expander 안 narrative.

---

## 2. KPI 체계 — 4 계층

투자자가 화면에서 만나는 숫자는 **계층마다 목적이 다릅니다**. 마구 나열하지 않고 4 계층으로 정리합니다.

### Tier 1 — Hero KPI (Overview 첫화면, 5개 이내)

가장 큰 글씨로 한 화면에 노출. 한눈에 "이 전략이 매력적인가" 결론 가능해야 함.

| KPI | 정의 | 표시 형식 | 왜 이게 중요? |
|---|---|---|---|
| **Top 추천 평균 Sharpe** (TEST 168m) | Top 5 카드의 sharpe_TEST 평균 | `1.14` | 위험조정 수익 한 줄 평가 |
| **Top 추천 평균 CAGR** (TEST 168m) | Top 5 카드의 cagr_TEST 평균 | `+16.2%/yr` | 절대 복리수익 — 투자자 직관 |
| **Top 추천 평균 MDD** (TEST 168m) | Top 5 카드의 mdd_TEST 평균 | `-14.7%` | 최악의 손실 — 투자자가 가장 두려워하는 숫자 |
| **SPY 대비 Alpha** (Jensen, ann.) | Top 5 평균 alpha | `+5.2%` | "SPY 사는 것보다 얼마나 좋은가" |
| **$10,000 시뮬** | TEST 시작 → 종료 누적 (Top 1 기준) | `$10,000 → $108,XXX` | 가장 직관적인 수익 표현 |

### Tier 2 — Deep KPI (Overview 펼침 + Compare 페이지, 메트릭 테이블)

각 모델 카드 / 비교 테이블에 노출.

| KPI | 정의 | 표시 |
|---|---|---|
| Sortino | 하방편차 기준 위험조정 수익 | 소수 3자리 |
| Sortino IR (3-레짐) | 3 regime 평균/표준편차 — 안정성 | 소수 2자리 |
| Volatility | annualized std × √12 | % |
| Calmar | CAGR / |MDD| | 소수 2자리 |
| Beta vs SPY | OLS β | 소수 2자리 |
| Avg eff_n | 평균 effective number of stocks | 정수 |
| Avg turnover | 월평균 turnover | % |
| Win rate | 양수 월 비율 | % |

### Tier 3 — Diagnostic (Compare 6패널, 시계열·분포)

차트로만 표현, 숫자 카드 X.

1. 누적수익률 (log scale, 시작=1)
2. Drawdown 시계열
3. 월별 수익률 분포 (히스토그램, mean·median 표시)
4. 12m rolling Sortino ⭐
5. 12m rolling Sharpe
6. 12m rolling 하방편차 (annualized %) ⭐

### Tier 4 — Investor-context (사이드바·expander, 최소 노출)

투자자 직관과 연결되는 보조 정보.

- 위험성향 슬라이더 (보수/중립/공격)
- 비교 모델 multi-select dropdown
- 기간 토글 (TEST 168m / FULL 192m / 실전 검증 HO 24m)
- "왜 이 후보인가" 한 줄 narrative (모델별)
- 한계·주의사항 expander (HO 부진, 방어주 집중 등)

---

## 3. 위험성향 분류 — 종합 점수 로직

> **사용자 결정**: sortino 우선 + sortino_ir + MDD + vol 종합

156 cfg 각각에 대해 z-score 정규화 후 가중합으로 *risk_profile_score* 산출.

```python
# 1. 정규화 (156 cfg 안에서 z-score; 방향 통일)
z_sortino    = z(mt['sortino_TEST'])         # 클수록 좋음
z_sortino_ir = z(mt['sortino_ir'])           # 클수록 좋음
z_mdd        = z(-mt['mdd_TEST'].abs())      # |MDD| 작을수록 좋음 → 부호 반전
z_vol        = z(-mt['vol_TEST'])            # 작을수록 좋음 → 부호 반전

# 2. 위험성향별 가중치 (총합 = 1)
WEIGHTS = {
    'conservative': dict(sortino=0.30, sortino_ir=0.30, mdd=0.30, vol=0.10),
    'balanced'    : dict(sortino=0.40, sortino_ir=0.20, mdd=0.20, vol=0.20),
    'aggressive'  : dict(sortino=0.50, sortino_ir=0.10, mdd=0.10, vol=0.30),
    # 공격형도 sortino 우선이지만 vol 비중을 낮추진 않음 — 변동성 자체는 정보
}

# 3. 종합 점수
score = (w['sortino']*z_sortino + w['sortino_ir']*z_sortino_ir +
         w['mdd']*z_mdd + w['vol']*z_vol)

# 4. 상위 5개 추출
top5 = mt.nlargest(5, 'score_<tier>')
```

### 3-1. 가중치 설계 근거

- **보수형**: MDD 비중 30% — 손실 회피 우선. sortino_ir 30%으로 regime-stable한 모델 (mat_mcap_rp_lam_pap 류) 자연 부각
- **중립형**: 모든 차원 균형. sortino 40%으로 위험조정 수익이 메인이지만 4 차원 고려
- **공격형**: sortino 50% — 위험조정 수익 최우선. vol 30%은 *변동성을 감수해도 sortino가 높은* 모델 (mat_eq_eq_lam_pap 류) 부각

### 3-2. Risk Profile 페이지의 사용자 인터랙션

- 슬라이더 0~100 → 0~30 보수, 30~70 중립, 70~100 공격으로 매핑
- **고급 옵션 expander**: 4 가중치 슬라이더 직접 조절 가능 (총합=1 자동 정규화)
- 결과 = Top 5 카드 + 각 카드 옆 "왜?" 한 줄 narrative + 비교 그래프 (5 모델 + SPY 누적수익률)

---

## 4. 페이지 매핑 — 5 페이지 상세

### Page 1 — 🏠 Home

**목적**: 30초 안에 "이 대시보드가 무엇이고 어디로 가야 하는지" 안내.

| 섹션 | 내용 |
|---|---|
| Hero 슬로건 | "S&P 500 저변동성 anomaly + LSTM σ 예측 + Black-Litterman 통합으로 **위험성향별 맞춤 펀드 후보**를 탐색합니다." |
| 5 단계 파이프라인 | 1.Data → 2.LowVol Anomaly EDA → 3.LSTM σ 예측 → 4.BL 156 cfg walk-forward (192m) → 5.위험성향별 Top 5 |
| 빠른 시작 가이드 | "투자자라면 → Risk Profile 페이지부터" / "분석가라면 → Compare 페이지부터" |
| 데이터 신선도 표시 | `최종 갱신: 2026-05-08, 156/156 cfg ret 192m 완성, S&P500 universe 600+ ticker` |

### Page 2 — 📊 Overview

**목적**: KPI hero + Top 5 카드 + 누적수익률 한 화면. 발표 시연의 첫 슬라이드.

| 섹션 | 구성 |
|---|---|
| **Tier 1 KPI 5칸** | Top 5 평균 Sharpe / CAGR / MDD / Alpha / $10K 시뮬 (앞장 §2 참고). delta 표시는 vs SPY 또는 vs baseline |
| **위험성향 토글** | 보수 ↔ 중립 ↔ 공격 라디오 (Risk Profile 페이지의 미니 버전) |
| **Top 5 카드 그리드** | 5 카드 가로 배치. 카드당: 모델명·canonical name 5-token + Sharpe/Sortino/CAGR/MDD/sortino_ir + 한 줄 narrative |
| **누적수익률 차트** | Top 5 + SPY (검정 점선) — log scale + 2024-01 HO 시작 점선 |
| **🔍 실전 검증 탭** | HO 24m 한정 KPI + 누적수익률 + "최근 시장 작동 여부" narrative. **여기에만 HO 부진 노출** |
| 한계·주의사항 expander | "TEST 168m 결과는 학습 기간. HO 24m 결과는 sector rotation 환경 영향 ··· 자세히는 실전 검증 탭" |

### Page 3 — ⚖️ Compare

**목적**: 사용자가 직접 모델을 골라서 다각도로 비교. Tier 3 6 패널의 핵심 무대.

| 섹션 | 구성 |
|---|---|
| **multi-select dropdown** | 156 cfg 중 최대 7개 선택. 기본값 = 위험성향별 Top 3 + SPY + baseline (5 모델) |
| **Tier 2 메트릭 테이블** | 선택 모델의 Sortino/Sharpe/Sortino_IR/CAGR/MDD/Calmar/Beta/Vol/eff_n/turnover. 정렬 가능 |
| **Tier 3 6 패널** | 누적수익률 / Drawdown / 월별분포 / 12m rolling Sortino / 12m rolling Sharpe / 12m rolling 하방편차 (이전 백업의 plot_compare_panel 재활용) |
| **🔍 실전 검증 탭** | 동일 6 패널 but HO 24m 한정. 여기서는 ranking이 어떻게 바뀌는지 표시 (TEST rank → HO rank 화살표) |

### Page 4 — 🗺️ Matrix Heatmap

**목적**: 슬롯별 trade-off를 한눈에. 분석가·연구자 친화. 투자자에게는 "왜 이 156개를 만들었는가" 시각적 답변.

| 섹션 | 구성 |
|---|---|
| 메트릭 선택 | sortino / sharpe / cagr / mdd / sortino_ir 라디오 |
| 슬라이스 선택 | omega 고정 (he/pap/rms 라디오) → prior × p_weight × q 매트릭스 → 3 × 3 × 5 = 45 cells 한 화면 |
| 색상 표시 | sortino 기준 viridis colormap, 셀 안에 숫자도 표기 |
| 슬롯 효과 마진 차트 | 우측 panel: 각 슬롯별 평균 sortino bar (예: prior=mcap 평균 1.65 vs eq 1.58 vs rp 1.71) |
| 한 줄 해석 | "rp p_weight + lam q + pap omega 조합이 sortino 1.85 — 슬롯 효과의 가산성 시각적 검증" |

### Page 5 — 🎯 Risk Profile

**목적**: 투자자에게 직접 "당신을 위한 후보" 제공. 사용자 결정의 핵심 페이지.

| 섹션 | 구성 |
|---|---|
| **위험성향 슬라이더** | 0~100, 색상 그라데이션 (파랑→녹색→주황). 현재 위치 = "당신의 위험성향: 보수형(20)" |
| **고급 옵션 expander** | sortino/sortino_ir/mdd/vol 가중치 슬라이더 4개 (총합 자동 정규화) |
| **Top 5 카드 그리드** | 가로 5장. 슬라이더 변경 시 즉시 재계산 (cache_data) |
| **카드 구성** | 모델명 / 5-token canonical / Sharpe·Sortino·CAGR·MDD 메트릭 / "왜 이 후보?" 1~2 문장 narrative / 작은 12m 누적수익률 sparkline |
| **누적수익률 비교 차트** | Top 5 + SPY 함께 / TEST 168m 기본, 실전 검증 토글 시 HO 24m |
| **위험성향별 narrative** | 보수형 = "MDD 작고 regime stable. 시장 변동기에 손실 최소화 우선" / 중립형 / 공격형 각각 |

---

## 5. 정보 아키텍처 — 사이드바 글로벌 옵션

각 페이지 독립이지만 **사이드바에 공통 글로벌 토글**을 둬서 일관성 확보.

| 글로벌 옵션 | 위치 | 영향 페이지 |
|---|---|---|
| 기간 토글 (TEST / FULL / HO) | 사이드바 라디오 | Overview / Compare / Matrix |
| 위험성향 (보수/중립/공격) | 사이드바 슬라이더 | Overview Top 5 카드 / Risk Profile |
| 비교 벤치마크 multi-select | 사이드바 (Compare 페이지에서만 노출) | Compare |
| 차트 언어 (한국어/English) | 사이드바 토글 | 전 페이지 (옵션) |

---

## 6. 데이터 의존성 — 변경 없음 (백업 초안과 동일)

| 경로 | 역할 | 캐시 전략 |
|---|---|---|
| `final/results/*.pkl` (156개) | BL 실험 결과 | `@st.cache_resource` (한 번만 로드, 모든 페이지 공유) |
| `final/data/monthly_panel.csv` | rf, spy_ret, sector | `@st.cache_data` |
| `final/phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` | LSTM 예측 (Volatility Stats 페이지가 사라졌으므로 미사용 가능) | (선택적) |
| `final/master_table.py` | TEST/HO/FULL metric 빌더 | `@st.cache_data` (sort_by 변경 시만 재계산) |

> **이전 6 페이지 → 현재 5 페이지 변경의 의미**: Volatility & Stats 페이지 제거 → LSTM ensemble csv 로드 불필요 (메모리 절약). LSTM σ 예측 정확도는 Compare 페이지의 expander narrative로만 언급.

---

## 7. 파일 트리 — 단계적 재구축

```
final/app/                          (새 디렉터리, _app_draft_backup/ 와 분리)
├── streamlit_app.py                ← Home + 글로벌 사이드바
├── lib/
│   ├── __init__.py
│   ├── data_loader.py              ← 156 pkl + master_table 캐시 (백업 재활용 + 156 갱신)
│   ├── scoring.py                  ← ⭐ 신규: 위험성향 종합 점수 로직
│   ├── plot_helpers.py             ← 6 패널 비교 차트 (백업 재활용)
│   ├── narrative.py                ← 모델별·위험성향별 한 줄 narrative
│   └── matrix_utils.py             ← 매트릭스 히트맵 슬라이스 빌더
├── pages/
│   ├── 1_📊_Overview.py            ← KPI hero + Top 5 카드 + 누적수익률
│   ├── 2_⚖️_Compare.py             ← multi-select 7 모델 + 6 패널
│   ├── 3_🗺️_Matrix_Heatmap.py     ← 3×3×5 셀 + 슬롯 마진
│   └── 4_🎯_Risk_Profile.py        ← 슬라이더 + Top 5 + narrative
├── assets/
│   └── theme.css                   ← 카드 스타일링, 위험성향 색상 그라데이션
└── README.md                       ← 실행법, 5 페이지 안내
```

---

## 8. 구현 단계 — 7 단계 단계적 재구축

각 단계 완료 후 사용자 검증 → 다음 단계 진행.

| Step | 산출 | 검증 포인트 |
|---|---|---|
| 1 | `lib/data_loader.py` (백업 재활용 + TOP_1_NAME 의존성 제거) | 156 pkl 로드, master_table 빌드, TEST/HO/FULL 분리 OK |
| 2 | `lib/scoring.py` (신규, 종합 점수 로직 §3) | 보수/중립/공격 Top 5 각각 산출 → 사용자 검토 후 가중치 미세조정 |
| 3 | `lib/narrative.py` (위험성향·모델별 한 줄 한국어) | 5 카드 narrative 검증 |
| 4 | `streamlit_app.py + Home` (글로벌 사이드바 포함) | 글로벌 토글 동작, Home 안내 OK |
| 5 | `pages/1_Overview.py` (Tier 1 KPI + Top 5 카드 + 실전 검증 탭) | 발표 시연 가능한 첫 화면 완성 |
| 6 | `pages/4_Risk_Profile.py` (슬라이더 + Top 5 + narrative) | 핵심 사용자 가치 페이지 |
| 7 | `pages/2_Compare.py + pages/3_Matrix_Heatmap.py` (보조 페이지) | 분석가·연구자 친화 페이지 마무리 |

> **단계 1~3 완료 후 한 번 사용자 데모 권장**: 종합 점수가 실제로 합리적인 Top 5를 추천하는지, narrative가 투자자에게 자연스러운지 검증 후 페이지 본격 구현.

---

## 9. 성공 기준 — 발표 시연 시나리오 (3분)

대시보드의 가치는 *시연 가능성*으로 검증.

```
[00:00-00:30] Home 페이지 — 5단계 파이프라인 + 슬로건
[00:30-01:30] Risk Profile — 슬라이더 보수형 → Top 5 카드 자동 갱신 시연
              "당신이 보수형이라면 → mat_mcap_rp_lam_pap (regime IR 28.15) 1순위"
[01:30-02:00] 슬라이더 공격형으로 이동 → 카드가 mat_eq_eq_lam_pap 등으로 변경 시연
              "공격형이라면 → CAGR 16.6% 1위, sharpe 1.10 학술적으로 강함"
[02:00-02:40] Overview 실전 검증 탭 — HO 24m도 정직히 공개
              "최근 24m HO에서는 일부 모델 부진. sector rotation 영향. 정직한 한계"
[02:40-03:00] Compare 페이지 — multi-select 자유 비교 시연
              "분석가 청중이라면 156 cfg 자유 탐색 가능"
```

---

## 10. 미해결·이후 결정 사항

| 항목 | 현재 상태 | 결정 시점 |
|---|---|---|
| 위험성향 가중치 미세조정 | §3 초기값 (0.30/0.30/0.30/0.10 등) 사용 | Step 2 완료 후 Top 5 검토 |
| narrative 톤 (한국어/한영 혼용) | 한국어 위주, 5-token canonical은 영문 유지 | Step 3 작성 시 |
| 실전 검증 탭 HO 부진 narrative 수위 | "sector rotation 영향" 정직 서술 | Step 5 작성 시 |
| Volatility & Stats 페이지 부활 | 현재 5 페이지로 제거. 필요 시 6번째 페이지 | 사용자 피드백 후 |
| 차트 언어 토글 (한/영) | 옵션. 발표 청중 다국적이면 추가 | 발표 직전 결정 |

---

## 11. 백업 초안 대비 변경점 요약

| 항목 | _app_draft_backup/ | 새 plan |
|---|---|---|
| 페이지 수 | 6 | **5** (Volatility & Stats 제거) |
| Top 1 강제 노출 | `TOP_1_NAME = 'mat_eq_eq_lam_pap'` 강제 | **없음. Top 5 카드 종합 점수 기반** |
| 위험성향 분류 | risk_score → tier → Top 3 (단순) | **z-score 종합 점수 (sortino/sortino_ir/MDD/vol 4축)** |
| HO 처리 | 페이지마다 토글 (혼란) | **"실전 검증" 탭으로 분리, 메인은 TEST 우선** |
| 비교 대상 | 자유 multi-select | 동일 (최대 7개) |
| narrative.py 모순 | "HO에서도 안정" 잘못된 서술 | **재작성 — sector rotation 한계 정직히** |
| LSTM 예측 csv 로드 | 매번 ~2.4M rows | **제거 (Volatility 페이지 삭제 → 메모리 절약)** |

---

> **다음 단계**: 본 plan에 사용자 추가 피드백 반영 후 Step 1 (`lib/data_loader.py`) 부터 구현 착수.
