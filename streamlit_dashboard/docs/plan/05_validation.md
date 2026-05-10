# 05. Validation — 검증 / 테스트 / 한계 / Future Work

> **관련 decisionlog**: `11_dl_sections.md` (D-5 Startup check + L 섹션 한계) + 모든 페이지 결정 종합
> **상태**: 확정
> **목적**: 대시보드 구현 전후 검증 / 한계 명시 / 향후 개선 방향

---

## 1. 데이터 무결성 검증 (D-5 Startup check)

### 1.1 Startup check 항목 (`lib/validators.py`)

```python
REQUIRED_DATA_FILES = [
    "streamlit_dashboard/data/monthly_panel.csv",
    "streamlit_dashboard/data/daily_returns.pkl",
    "streamlit_dashboard/data/ff5_monthly.csv",
    "streamlit_dashboard/data/universe.csv",
    "streamlit_dashboard/data/ticker_company_map.csv",
    "streamlit_dashboard/data/results/mat_eq_eq_raw_pap.pkl",
]
```

**검증 절차**:
1. 앱 시작 시 1회 실행
2. 모든 필수 파일 존재 여부 검증
3. 누락 시: `st.error` + `st.stop()` (앱 진입 차단)
4. 누락 파일 list + 복구 명령어 안내

### 1.2 데이터 검증 추가 (구현 시 권장)

- [ ] **monthly_panel.csv**:
  - 컬럼: `date`, `rf`, `spy_ret`, `sector`, `log_mcap`
  - 기간: 2010-01 ~ 2025-12 (192개월)
  - NaN 비율 < 5%

- [ ] **daily_returns.pkl**:
  - 형태: pd.DataFrame (date × ticker)
  - 영업일 ~6099 일
  - NaN 비율 < 10% (상장 시기 차이 고려)

- [ ] **universe.csv**:
  - 컬럼: `ticker`, `gics_sector`
  - 833 ticker 모두 GICS 11개 섹터 매핑

- [ ] **ticker_company_map.csv** (yfinance 수집):
  - 컬럼: `ticker`, `company_name`
  - 누락 ticker = ticker 자체로 fallback

- [ ] **results/mat_eq_eq_raw_pap.pkl**:
  - 형태: dict (`weights`, `returns`, `dates`)
  - weights: 월별 종목별 비중 (합 = 1)
  - returns: 월별 펀드 수익률 (192개월)

### 1.3 데이터 검증 코드 예시

```python
def validate_data_integrity():
    """추가 데이터 무결성 검증 (구현 시 권장)."""
    import pandas as pd

    # monthly_panel
    mp = pd.read_csv("streamlit_dashboard/data/monthly_panel.csv")
    assert "date" in mp.columns
    assert "spy_ret" in mp.columns
    assert mp["date"].nunique() == 192

    # universe
    universe = pd.read_csv("streamlit_dashboard/data/universe.csv")
    assert universe["gics_sector"].nunique() == 11  # GICS 11 섹터

    # ticker_company_map
    map_df = pd.read_csv("streamlit_dashboard/data/ticker_company_map.csv")
    assert len(map_df) >= 800  # ~833 ticker

    # 결과 pkl
    import pickle
    with open("streamlit_dashboard/data/results/mat_eq_eq_raw_pap.pkl", "rb") as f:
        result = pickle.load(f)
    assert "weights" in result
    assert "returns" in result
```

---

## 2. Streamlit Cloud 배포 전 체크리스트

### 2.1 사전 점검

- [ ] **`requirements.txt`** 정확 (J-5 Range versions)
- [ ] **`.streamlit/config.toml`** 다크 테마 + Cobalt Blue 적용
- [ ] **`app.py`** Startup check 호출
- [ ] **모든 데이터 파일** `streamlit_dashboard/data/` 에 위치
- [ ] **`scripts/build_ticker_company_map.py`** 1회 실행 완료
- [ ] **`pages/`** 폴더의 9개 페이지 파일 존재
- [ ] **`lib/`** 폴더의 모든 공통 모듈 존재

### 2.2 메모리 효율 점검 (Streamlit Cloud 1GB 제한)

- [ ] **D-3 캐싱** 모든 데이터 로딩 함수에 `@st.cache_data` 적용
- [ ] **큰 pickle** (daily_returns.pkl, 156 config pkl) lazy loading
- [ ] **Sensitivity Test (Backtesting 영역 6)** Top 10 만 메모리 로드 (155 config 전체 X)
- [ ] **Plotly 차트** `@st.cache_data` 활용 (가능 시)

### 2.3 외부 의존 점검

- [ ] **Pretendard CDN fallback chain** (H-1)
- [ ] **streamlit-card 라이브러리** 사용 (Overview 영역 4) 또는 fallback
- [ ] **yfinance** 추가 호출 없음 (1회 수집 후 CSV 캐시)

### 2.4 GitHub Push

- [ ] `streamlit_dashboard/` 폴더 전체 commit
- [ ] `.gitignore`:
  - `__pycache__/`
  - `*.pyc`
  - `.streamlit/secrets.toml` (사용 시)
- [ ] **데이터 파일 크기** > 100MB 시 git LFS 또는 외부 저장소 권장

### 2.5 Streamlit Cloud 설정

- [ ] share.streamlit.io 접속
- [ ] GitHub repo 연동
- [ ] main branch + `streamlit_dashboard/app.py` 지정
- [ ] **Custom subdomain** (J-4): `volcontrol.streamlit.app` 또는 `adaptive-volcontrol.streamlit.app`
- [ ] **Python version**: 3.10+ (requirements.txt 호환)

### 2.6 배포 후 검증

- [ ] URL 접속 정상
- [ ] 사이드바 6 그룹 + 2 토글 동작
- [ ] 모든 9 페이지 정상 렌더
- [ ] 한글 폰트 (Pretendard) 정상 표시
- [ ] 다크 테마 + Cobalt Blue 정상

### 2.7 발표 전 Warmup (J-3 슬립 모드 회피)

- [ ] 발표 30분 전 1회 접속 (슬립 모드 해제)
- [ ] 모든 페이지 1회 클릭 (캐시 warmup)

---

## 3. 한계 (L 섹션)

### 3.1 Methodology 영역 8 의 3가지 한계 (균형 옵션 B 적용)

★ **5가지 → 3가지 축소** (Q-A1):

| 한계 차원 | 위치 | 톤 |
|---|---|---|
| 🟧 **HO 24m 부진 인정** | Methodology 영역 8 카드 1 | 솔직 인정 (E-2) |
| 🟩 **향후 개선 방향** | Methodology 영역 8 카드 2 | 향후 개선 |
| 🟥 **실무 적용 제약** | Methodology 영역 8 카드 3 | 표준 disclosure |

### 3.2 한계 narrative 분포 (L-1 결정)

| 위치 | 한계 내용 | 톤 |
|---|---|---|
| Methodology 영역 8 카드 1 (HO 부진) | -12.9%p / Sector trade-off | 솔직 인정 (E-2) |
| Methodology 영역 8 카드 2 (개선 방향) | Multi-factor / WFV 추가 / Ablation | 향후 개선 |
| Methodology 영역 8 카드 3 (실무 제약) | 가상 펀드 / 운용 규모 / Tax 미반영 | 표준 disclosure |
| About 영역 6 (Selection Bias 부록) | PBO/DSR / Data snooping | 학술 부록 (Expander) |
| About 영역 7 (자세한 Disclosure) | Risk factors 5가지 | 자세한 |
| Sector Watch 영역 8 (HO 정당화) | 분산 trade-off | 자신감 (Markowitz 1952) |
| Footer (모든 페이지) | HO 부진 짧은 언급 | 표준 |

→ **7개 위치 한계 narrative 일관 분포**

### 3.3 동적 한계 추가 (Methodology 영역 7 Case B)

**조건**: Jarque-Bera 결과 정규분포 채택 (p ≥ 0.05)

**동적 카드 추가**:
```
⚠ LSTM 가치 미입증

영역 7 Jarque-Bera 결과 정규분포 채택 → 단순 모델 대비
LSTM 우위 재검토 필요. 향후 ablation study (BL only vs
BL+LSTM) 을 통해 LSTM 가치 정량 검증 권장.
```

### 3.4 Stress Test 제거 (Q-Stress A)

**제거 근거**:
1. **사용자 지적**: 정확한 시뮬레이션 구현 매우 어려움
2. **학술 정직성** — 단순 β 기반 시뮬레이션은 fat tail / 상관관계 변화 / 유동성 미반영
3. **Methodology 영역 7 (Jarque-Bera fat tail) narrative 와 모순**
4. **Sub-events (영역 5) + Sensitivity (영역 6) 로 위기 안정성 narrative 충분**
5. **"감추지 않는 펀드"** 정신 — 못 하는 것을 하는 척 X

→ Backtesting 페이지 9 영역 → 7 영역 (Stress 제거)

### 3.5 Methodology 영역 6 Factor 분석 — 조건부 유지

**리스크**:
- 결과가 부정적일 가능성 (alpha 가 양수가 아닐 수 있음)
- 회귀 결과 해석 어려움 (p-value, R², CI 등)
- HO 부진 강조 가능

**조건**:
- 일단 생성 — 학술 깊이 우선
- 결과 확인 후 재평가:
  - alpha 가 양수 + 통계적 유의 → narrative 강화 + 유지
  - alpha 음수 / 통계적 무의 → 영역 축소 (B) 또는 제거 (C) 검토

---

## 4. 학술 정직성 선언 (L-3 결정 — Methodology 영역 8 만)

```
✅ 학술 정직성 선언

본 펀드는 학술 정직성을 위해 모든 한계를 명시합니다.
한계 인정은 신뢰성 강화의 토대이며, 향후 개선 방향을
통해 지속적 발전을 추구합니다.
```

**위치**: Methodology 영역 8 만 (다른 페이지 X)

**근거** (L-3):
1. Methodology 영역 8 = 한계 영역의 결론 → 자연 위치
2. About 영역 7 = 다른 톤 (자세한 Disclosure / Risk factors)
3. Sector Watch 영역 8 = 자신감 결론 박스 (다른 톤 유지)

---

## 5. 학술 인용 일람 갱신 (L-2 결정)

**페이지별 학술 인용 → 00_README.md 학술 근거 일람 일괄 갱신**

### 5.1 갱신할 학술 인용 (페이지별 → 00_README 일람)

| 학술 인용 | 적용 위치 |
|---|---|
| Markowitz (1952) | Sector Watch 영역 8 |
| Black-Litterman (1990, 1992) | Methodology 영역 4 |
| He-Litterman (1999) | Methodology 영역 4 |
| Idzorek (2005) | Methodology 영역 4 |
| Hochreiter & Schmidhuber (1997) | Methodology 영역 5 LSTM |
| Gers, Schmidhuber, Cummins (2000) | Methodology 영역 5 |
| Kim & Won (2018) | Methodology 영역 5 |
| Jensen (1968) | Methodology 영역 6 |
| Fama-French (1993, 2015) | Methodology 영역 6 |
| Carhart (1997) | Methodology 영역 6 |
| Jarque-Bera (1980) | Methodology 영역 7 |
| Cont (2001) | Methodology 영역 7 fat tail |
| Hill (1975) | Risk Metrics 영역 8 |
| Embrechts, Klüppelberg, Mikosch (1997) | Methodology 영역 7 |
| Lopez de Prado (2018) | Methodology 영역 5/8 walk-forward |
| Engle (2002) | Methodology 영역 8 DCC-GARCH (Expander) |
| AQR Frazzini, Israel, Moskowitz (2018) | Overview 영역 4 + Methodology 영역 8 |
| Bailey & Lopez de Prado (2014) | About 영역 6 PBO/DSR |
| Frazzini-Pedersen (2014) | Overview 영역 3 IVW |
| Modigliani² (1997) | Risk Metrics 영역 7 |
| Sharpe (1966, 1994) | 표준 메트릭 (각 페이지) |

### 5.2 자세한 인용 위치는 `00_README.md` 학술 근거 일람 참조

---

## 6. Future Work (향후 개선 방향)

### 6.1 Methodology 영역 8 카드 2 ("향후 개선 방향" 카드)

**목록**:
1. **Multi-factor 모델 추가**
   - Momentum + Value + Quality factor 추가
   - 학술 근거: Fama-French (2015) FF5

2. **Walk-forward validation 추가**
   - LSTM + BL 자체 walk-forward 외 추가 검증
   - 학술 근거: Lopez de Prado (2018)

3. **실제 매매 시뮬레이션**
   - Slippage 반영
   - Tax (한국 양도세 등) 반영
   - 유동성 제약 반영

4. **Ablation study** (Case B 동적 추가 시)
   - BL only vs BL+LSTM 비교
   - LSTM 의 가치 정량 검증

### 6.2 About 페이지 영역 6 (Selection Bias 부록) — 향후 추가

**현재**: 학술 부록 (Expander)
**향후**:
- PBO (Probability of Backtest Overfitting) 수치 계산
- DSR (Deflated Sharpe Ratio) 수치 계산
- 학술 인용 link (Bailey & Lopez de Prado 2014)

### 6.3 Investment Simulator 확장

**현재**: Lump-sum / DCA / Goal-based 3 시나리오
**향후 가능 확장**:
- **Goal-based 자동 추천** (목표 + 기간 → 필요 초기 + 매월)
- **다중 자산** (Fund + 채권 + 현금 비율 시뮬레이션)
- **Stress 시나리오 입력** (사용자 정의 위기 시기)

### 6.4 인터랙션 확장

**현재**: G-1 같은 페이지 expand + Q-Zoom
**향후 가능**:
- **Modal popup** (선택)
- **Crossfilter** (다중 차트 동기화)
- **모바일 반응형 강화** (현재 Streamlit 기본만)

### 6.5 데이터 갱신

**현재**: D-4 정적 (배포 시점 고정)
**향후**:
- **자동 갱신** (월별 yfinance 호출)
- **Live 데이터** (실시간 SPY 비교)
- **데이터 버전 관리** (Git 또는 별도 시스템)

---

## 7. 테스트 (구현 시 권장)

### 7.1 단위 테스트 (Unit Test)

- [ ] **`lib/data_loader.py`**:
  - `equal_weight_returns()` 정확성
  - `ivw_returns()` 정확성
  - 캐싱 정상 동작

- [ ] **`lib/metric_calculators.py`** (구현 시):
  - `hill_estimator()` (Hill 1975)
  - `compute_var_cvar()`
  - `compute_recovery_time()`

- [ ] **`lib/insight_generator.py`**:
  - 조건부 카드 정상 생성

### 7.2 통합 테스트 (Integration Test)

- [ ] **사이드바 토글 → 페이지 갱신**
- [ ] **Q-Zoom → expand 동작**
- [ ] **카드 클릭 → 페이지 navigation**

### 7.3 사용자 테스트 (User Test)

- [ ] **5분 demo 흐름** (K-2): Overview → Sim → Sector → Methodology
- [ ] **자유 탐색** (가상 투자자 시뮬레이션)
- [ ] **모바일 반응형** (선택)

---

## 8. 모니터링 (배포 후)

### 8.1 Streamlit Cloud 모니터링

- [ ] **메모리 사용량** (1GB 제한 근접 시 캐싱 점검)
- [ ] **응답 시간** (느린 페이지 식별)
- [ ] **에러 로그** (Streamlit Cloud admin)

### 8.2 사용자 피드백

- [ ] **GitHub Issues** (오류 신고)
- [ ] **Streamlit Community** (UX 피드백)

---

## 9. 마무리 체크리스트

### 9.1 Phase 1 완료 (MVP)

- [ ] Setup + lib/* 모두 정상
- [ ] 사이드바 6 그룹 + 2 토글
- [ ] Overview + Performance 정상
- [ ] Pretendard 폰트 + 다크 테마

### 9.2 Phase 2 완료 (확장)

- [ ] Risk + Holdings + Sector + Investment Simulator + Methodology (간략)
- [ ] 5분 demo 흐름 (K-2) 가능
- [ ] HO 정당화 narrative (Sector Watch 영역 8) 정상

### 9.3 Phase 3 완료 (검증)

- [ ] Methodology 완성 (Factor + Jarque-Bera + 한계)
- [ ] Backtesting (Regime + Sub-events + Sensitivity)
- [ ] About 메타 (영역별 자세 = 팀 상의 후 진행)
- [ ] Streamlit Cloud 배포 정상

### 9.4 발표 전 (D-1)

- [ ] Streamlit Cloud Warmup (30분 전)
- [ ] PPT 자료 준비 (학술 분석 10-15분, K-4 결정)
- [ ] 5분 demo 리허설

---

## 10. 학술 인용 (본 plan 의 학술 근거)

자세한 인용 위치는 `00_README.md` 의 학술 인용 일람 (8 절) 참조.

**핵심 학술 근거**:
- Markowitz (1952) — 평균-분산 이론
- Black-Litterman (1990, 1992) — BL 모델
- Hochreiter & Schmidhuber (1997) — LSTM
- Fama-French (1993, 2015) — Factor 모델
- Lopez de Prado (2018) — Walk-forward validation
- Frazzini, Israel, Moskowitz (2018) — 거래비용
- Bailey & Lopez de Prado (2014) — PBO/DSR

---

## 11. 결론

본 plan 은 decisionlog 12개 파일의 모든 결정사항을 구현 가능한 형태로 정리했습니다.

**총 13 파일 plan 구조**:
- 인덱스 (1) + Setup (1) + Common (1) + Pages (9) + Implementation Steps (1) + Validation (1)

**4-7주 구현 계획**:
- Phase 1 (MVP, 1-2주)
- Phase 2 (확장, 2-3주)
- Phase 3 (검증, 1-2주)

**핵심 원칙**:
1. Self-contained — 각 파일만 Read 해도 작업 수행 가능
2. decisionlog 결정 사항 정확 반영
3. 9 페이지 표준 구조 (영역별 와이어프레임)
4. 균형 옵션 (B) 적용 (Backtesting 7 영역, Methodology 3개 한계)

---

[← 04_implementation_steps.md](04_implementation_steps.md) | [00_README.md →](00_README.md)
