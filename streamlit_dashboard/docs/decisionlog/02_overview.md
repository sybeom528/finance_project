# C-1-1. Overview 페이지

> **파일**: `02_overview.md`
> **결정 시점**: 2026-05-10
> **상태**: 확정 (영역 1~6 모두)
> **포함**: Header / Hero KPI / 누적수익 곡선 / 핵심 강점 카드 / Navigation Cards / Footer

---


## C-1-1. Overview 페이지

> **상태**: 진행 중 (영역 6개 중 영역 1 확정)

### 페이지 역할

랜딩 페이지 = URL 진입 직후 노출. 5초 안에 펀드 정체성 + 핵심 성과 전달, 다음 페이지로 navigation 유도.

### 시선 흐름 (위 → 아래 6 영역)

```
1. Header           → 펀드명 + 슬로건 + 메타
2. Hero KPI         → 4-5개 핵심 메트릭 카드
3. 누적수익 곡선     → vs SPY 메인 차트
4. 핵심 강점 카드    → 3개 차별화 포인트
5. Navigation cards → 다음 페이지 연결
6. Footer           → Disclosure 짧은 버전
```

---

### 영역 1: Header — 확정

#### 배경

페이지 진입 직후 가장 먼저 인지되는 부분. 펀드명 + 슬로건 + 메타 정보 노출.

#### 결정 항목 1-1: 펀드명 표시 방식

**검토된 옵션**:
- (a) 영문 강조 + 한글 부제 (H1 영문 / H2 한글)
- (b) 한글 강조 + 영문 부제
- (c) 한/영 동등
- (d) 영문 단독

**결정**: (a) 영문 강조 + 한글 부제

**근거**:
1. A-3 한/영 병기 + 가상 투자자 중점 → 영문이 펀드 brand 역할 (직관적)
2. 한국 청중 친근감 위해 한글 부제 추가
3. 정통 금융 펀드 패턴 (해외 운용사 한국 진출 시 영문명 우선)

#### 결정 항목 1-2: 슬로건 표시

**결정**: (a) 한/영 병기 (대시 — 로 구분)

**예시**: "변동성 예측 기반 적응형 자산배분 — Volatility-Aware Adaptive Allocation"

**근거**: A-3 한/영 병기 + B-3 슬로건 결정 일관 적용

#### 결정 항목 1-3: 메타 정보

**검토된 메타 항목**:
- 펀드 상태 (● Active 배지)
- 벤치마크 표시
- 데이터 기준일
- AUM (가상)

**결정**: 펀드 상태 ✓ / 벤치마크 ✓ / 데이터 기준일 ✓ / AUM ✗

**근거**:
1. 펀드 상태 ✓ — "● Active (Simulated)" 으로 가상 운용임을 명시 (학술 정직성)
2. 벤치마크 ✓ — "Benchmark: S&P 500 (SPY)" 비교 기준 명확
3. 데이터 기준일 ✓ — "Data as of: 2025-12-31" 신뢰성
4. AUM ✗ — Hero KPI 영역에서 다루기로 함 (중복 회피)

#### 결정 항목 1-4: 레이아웃

**결정**: (a) 좌측 정렬 (펀드명 좌측 + 메타 정보 우측)

**근거**: 레퍼런스 PortfolioX360 일관성 + Bloomberg/금융 대시보드 표준

#### 시각화 예시

```
┌─────────────────────────────────────────────────────────────┐
│  Adaptive VolControl Fund                  ● Active         │
│  어댑티브 볼컨트롤 펀드                       Benchmark: SPY  │
│  변동성 예측 기반 적응형 자산배분 —             Data: 2025-12 │
│  Volatility-Aware Adaptive Allocation                       │
└─────────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 모든 페이지 상단에 동일 Header 패턴 적용 (전체 페이지 일관성)
- 사이드바 상단에도 동일 펀드명 텍스트 마크 표시 (B-5 결정 부합)

---

### 영역 2: Hero KPI — 확정

#### 영역 2 통합 배경 (Context)

페이지 진입 직후 가장 먼저 시선이 가는 4-5개 카드. 5초 안에 펀드 정체성 + 핵심 성과 전달. 레퍼런스 PortfolioX360 (Image 3) 의 Hero 4개 (Cumulative Return / Sharpe / MDD / AUM) 패턴 참조.

5개 핵심 결정 항목 (2-1 ~ 2-5) + 3개 통합 결정 (토글 범위 / Sparkline / 레이아웃).

#### 결정 항목 2-1: KPI 개수

**검토된 옵션**:
- (a) 4개: 간결, 강한 인상 (레퍼런스 동일)
- (b) 5개: 정보량 ↑, 펀드 정체성 풍부
- (c) 6개: 너무 산만, 비권장

**결정**: (b) 5개

**근거**:
1. 정보량과 간결성 균형
2. 펀드 정체성 (VolControl) 강조 위해 Volatility 추가 필요
3. 6개는 시각적 산만 → 5개 적정

#### 결정 항목 2-2: 메트릭 선정

**검토된 옵션** (시나리오 5개):
- A: Cumulative Return / Sortino / Volatility / MDD (4개) - VolControl 정체성
- B: Cumulative Return / Net CAGR / Sortino / Volatility / MDD (5개) - 정보량 ↑
- C: Cumulative Return / Sortino / Volatility / Down Capture (4개) - 시장방어
- D: Cumulative Return / Sharpe / MDD / FF5 Alpha (4개) - 학술 표준
- E: 자유 선택

**결정**: 시나리오 B (5개)

**근거**:
1. **Cumulative Return**: 누적수익 (직관적 첫인상, 모든 펀드 표준)
2. **Net CAGR (after TC)**: 연환산 수익률 (실무 친화, 거래비용 차감 후 정직)
3. **Sortino Ratio**: 펀드 핵심 (하방위험 조정, B-1 Adaptive VolControl 정체성과 매칭)
4. **Volatility**: "VolControl" 정체성 직접 표시 (펀드명과 매칭)
5. **Max Drawdown**: 위험 강조 (가상 투자자 우려 직접 응답)

5개 모두 메트릭 풀 (C-2) 에서 picking → 일관성 유지.

#### 결정 항목 2-3: 표시 기간

**검토된 옵션**:
- (a) FULL only: 단일 숫자, HO 부진 묻힘
- (b) TEST only: 단일 숫자, HO 미반영 (학습편향 우려)
- (c) FULL + HO 별도: 두 숫자, FULL 에 HO 일부 반영
- (d) TEST + HO 별도: 두 숫자, in-sample/OOS 명확 분리

**(b) vs (c) 장단점 비교**:

| 차원 | (b) TEST only | (c) FULL + HO 별도 |
|---|---|---|
| HO 반영 | ✗ | △ (FULL 에 24/192=12.5% 비중) |
| 학술 정직성 | "in-sample only" 명시 | FULL 평균값에 묻힘 |
| HO 부진 회피 의심 | △ ("왜 TEST 만?") | ✗ (별도 표시) |

**결정**: (d) TEST + HO 별도

**근거**:
1. **학술 정직성**: in-sample (TEST 168m) / OOS (HO 24m) 분리는 ML 표준
2. **가상 투자자 정직성**: HO 부진 솔직 노출 → 신뢰성 ↑
3. **HO 부진 회피 의심 제거**: 두 숫자 동시 표시로 감추지 않음
4. **메시지 일관성**: TEST (학습) + HO (검증) narrative 직접 연결
5. **(b)/(c) 약점 모두 회피**: TEST only 의 "왜 TEST 만?" 의심 + FULL 의 평균값 묻힘 둘 다 해결

#### 결정 항목 2-4: 카드 디자인

**검토된 옵션**:
- (a) 큰 숫자만: 가장 간결
- (b) 큰 숫자 + delta vs SPY: 레퍼런스 PortfolioX360 스타일
- (c) 큰 숫자 + sparkline: 미니 트렌드 차트
- (d) 큰 숫자 + delta + sparkline: 정보량 max

**결정**: (c) 큰 숫자 + sparkline

**근거**:
1. 시각적 트렌드 즉시 전달 → 청중 이해 ↑
2. 레퍼런스 PortfolioX360 동일 패턴 (정통 금융 톤)
3. delta vs SPY 는 별도 차트 (영역 3 누적수익 곡선) 에서 자세히 다룸
4. (d) 는 카드 복잡도 ↑ → 혼잡 위험

#### 결정 항목 2-5: HO 부진 처리

**검토된 옵션**:
- (a) Hero 에서 회피: HO 별도 페이지에서만
- (b) Hero 에 FULL/HO 동시: 카드 내 두 줄
- (c) 토글로 전환: 사이드바에서 FULL/TEST/HO 선택
- (d) FULL 위주 + HO 작은 글씨 부기

**결정**: (c) 토글로 전환

**근거**:
1. 자유 탐색 모드 (A-2 결정) 와 부합
2. 다양한 청중 (학술/가상 투자자) 의 관심 기간 자유 선택
3. 사용자 적극적 탐색 유도

#### 통합 결정 ① — Hero 토글 영향 범위

**충돌 가능성**: 2-3 (d) TEST+HO 동시 표시 + 2-5 (c) 사이드바 토글 → 토글 시 Hero 어떻게?

**검토된 옵션**:
- (A) Hero 고정 + 토글 다른 페이지 영향: Hero 일관성 ↑
- (B) Hero 도 토글 영향: 모든 곳 일관 적용
- (C) Hero (d) 고정 + 토글 추가 보조 표시: 정보량 max

**결정**: (A) Hero 고정 + 토글이 다른 페이지에만 영향

**근거**:
1. Hero 첫인상 = 펀드 정체성 5초 요약 → 일관성 중요
2. 토글은 자유 탐색 모드 도구 → 다른 페이지에서 더 의미 있는 활용
3. Hero (d) TEST+HO 동시 표시 만으로 학술 정직성 narrative 완결
4. 토글로 사용자가 깊이 탐색 시 → 누적수익 곡선 / Performance 페이지 등 영향

**영향 범위**:
- Hero KPI (영역 2): ✗ 고정
- 누적수익 곡선 (영역 3): ✓ 토글 영향
- Performance 페이지: ✓
- Risk Metrics 페이지: ✓
- 다른 페이지: ✓ (자세한 영향 범위는 페이지별 결정 시 확정)

#### 통합 결정 ② — Sparkline 표시 기간

**검토된 옵션**:
- (a) TEST 구간만 (168m): 깔끔
- (b) FULL 구간 (192m, HO 다른 색): TEST + HO 트렌드 함께 표시
- (c) TEST + HO 분리 표시: 두 줄

**결정**: (b) FULL 구간 표시 (192m, HO 다른 색)

**근거**:
1. 16년 트렌드 한 sparkline 으로 표시 → 정보량 ↑
2. HO 구간 다른 색 (red dashed 등) → 시각적 분리 명확
3. (c) 두 줄은 카드 높이 ↑ → 혼잡

#### 통합 결정 ③ — 카드 레이아웃

**검토된 옵션**:
- (a) 5 가로 배치 (한 줄): 간결
- (b) 3+2 두 줄 배치: 균형
- (c) 반응형 (스크린 너비 자동): 모든 환경 대응

**결정**: (c) 반응형

**근거**:
1. A-2 자유 탐색 모드 → 다양한 device (PC, 태블릿) 대응 필요
2. Streamlit `st.columns` 의 가변 너비 활용
3. 큰 화면: 5개 한 줄 / 작은 화면: 3+2 자동 분할

#### 시각화 예시 (확정 사항 조합)

```
┌──────────────────┬──────────────────┬──────────────────┐
│ Cumulative Return│ Net CAGR (TC 20bp)│ Sortino Ratio    │
│ TEST: +XXX.X%    │ TEST: +X.XX%     │ TEST: X.XX       │
│ HO:   +X.X%      │ HO:   +X.XX%     │ HO:   0.685      │
│ ▁▂▄▆█ sparkline  │ ▁▃▆█ sparkline   │ ▁▂▆█ sparkline   │
│ (FULL, HO red)   │ (FULL, HO red)   │ (FULL, HO red)   │
├──────────────────┼──────────────────┴──────────────────┤
│ Volatility       │ Max Drawdown     │                  │
│ TEST: XX.X%      │ TEST: -XX.X%     │                  │
│ HO:   10.3%      │ HO:   -8.3%      │                  │
│ ▆▂▃▄▆ sparkline  │ ▂▁▆█ sparkline   │                  │
└──────────────────┴──────────────────┘

레이아웃: 반응형 (큰 화면 5개 한 줄 / 작은 화면 3+2 분할)
Sparkline: FULL 192m, HO 24m 다른 색 (red dashed)
```

#### 결과 / 함의

- 영역 3 (누적수익 곡선) 인터랙션 결정 시 사이드바 토글 영향 범위 명확화 필요
- 모든 페이지에 동일한 사이드바 토글 (FULL / TEST / HO) 적용
- C-3 (Hero KPI) 정식 결정 완료 — C-4/C-5 결정 시 토글 통합 고려
- 다른 페이지 (Performance / Risk Metrics) 메트릭 표시 시 토글 반영

### 영역 3: 누적수익 곡선 (Cumulative Return Curve) — 확정

#### 영역 3 통합 배경 (Context)

Hero KPI 바로 아래 메인 차트로, 16년 (2010-2025) 펀드 vs 벤치마크 누적수익을 직관적으로 보여주는 핵심 시각화. 레퍼런스 PortfolioX360 의 Cumulative Return + PortfolioVisualizer 의 누적수익 곡선 패턴 참조.

7개 핵심 결정 + 2개 추가 결정 (Equal-weight 구현 방식 / Naive Low-vol 구현 방식).

#### 결정 항목 3-1: 차트 표시 기간

**검토된 옵션**:
- (a) 고정 FULL only: 항상 192m 표시
- (b) 토글 반응 (FULL/TEST/HO 선택)
- (c) 항상 FULL + 구간 강조: TEST/HO 영역 배경색 분리
- (d) 분할 모드: TEST + HO 두 차트 옆으로 나란히

**결정**: (c) 항상 FULL + 구간 강조

**근거**:
1. 16년 전체 추세 한눈에 보여주는 대시보드 메인 차트 역할
2. 영역 2 토글 영향 범위와 부합 (사이드바 토글로 zoom 가능)
3. (a) 고정 FULL 보다 (c) 가 더 학술 정직 (TEST/HO 시각 분리)
4. (d) 분할 모드는 정보량 ↑ but 시각적 통일감 ↓

#### 결정 항목 3-2: 비교 대상

**검토된 옵션**:
- (a) SPY only
- (b) SPY + Equal-weight
- (c) SPY + Naive Low-vol
- (d) SPY + Equal-weight + Naive Low-vol
- (e) 토글 가능

**결정**: (e) 토글 가능 (기본 SPY only)

**근거**:
1. A-2 자유 탐색 모드 결정 부합
2. 청중이 펀드의 차별화를 직접 비교할 수 있도록
3. EW (액티브 운용 가치) + IVW (LSTM 가치) 별도 검증

##### 3-2 추가 결정 (Q-A): Equal-weight 구현 방식

**배경**: EW baseline 의 universe 일치 여부에 따라 비교 의미가 다름

**검토된 옵션**:
- (A1) 우리 펀드와 동일 universe EW (1/N)
- (A2) 시장 전체 EW (S&P 500 전체)
- (C) 둘 다

**결정**: (A1) 우리 펀드와 동일 universe EW

**근거**:
1. **액티브 운용 가치 측정**: "BL+LSTM 이 weight 결정 vs 단순 1/N" narrative
2. **universe 일치**: 비교 변수 통제 → universe 차이 noise 제거
3. **(A2) 거리**: S&P 500 EW 는 universe 불일치 (우리 펀드 ~822 ticker)

**구현**:
```python
weights_ew = {ticker: 1/N for ticker in our_fund_universe[t]}
return_ew = sum(weights_ew[i] * returns[t][i] for i in our_fund_universe[t])
```

##### 3-2 추가 결정 (Q-B): Naive Low-vol 구현 방식

**배경**: 우리 펀드 (LSTM 변동성 예측 + BL) 의 가치를 측정하기 위한 단순 baseline

**검토된 옵션**:
- (B1) IVW (Inverse Volatility Weighting, 120일): weight = (1/σ_i) / Σ(1/σ_j)
- (B2) MVP (Minimum Variance Portfolio, 공분산 행렬 사용)
- (B3) Low Vol Quintile (변동성 하위 20% 종목 EW)

**결정**: (B1) IVW (120일 변동성 역수 가중)

**근거**:
1. **가장 단순**: 공분산 행렬 불필요 → 수치적 안정
2. **학술 표준 baseline**: Frazzini & Pedersen (2014) "Betting Against Beta"
3. **LSTM 가치 narrative**: "단순 과거 변동성 vs LSTM 미래 예측" 직접 비교
4. **MVP (B2) 거리**: regularization 필요 + naive 라기엔 너무 정교
5. **Low Vol Quintile (B3) 한계**: universe 일부만 사용 → 우리 펀드와 비대칭

**구현 상세**:
- 변동성 윈도우: **60일 (3개월)** ← 2026-05-10 변경 (원안 120일 → 60일)
- Weight: $w_i = (1/\sigma_i) / \sum_j (1/\sigma_j)$
- $\sigma_i$ = 60일 일별 수익률 표준편차
- 매월 rebalancing (우리 펀드와 동일)
- universe = 시점 t-1 sp500_membership (Q-C 결정 참조)

**변경 이력**:
- *2026-05-10 윈도우 120일 → 60일*: `daily_returns` 데이터 결함 발견 (SPY -41% 일별 spike 등) → `monthly_panel` 기반 구현으로 전환 (Q-D 옵션 E). monthly_panel 에는 `vol_60d`, `vol_252d` 만 있어 120d 부재. 60d 가 252d 보다 단기 변동성 반영 (시장 변화 빠른 반응) → 액티브 비교에 적절. Frazzini-Pedersen (2014) BAB 도 다양한 윈도우 사용 (60d 표준).

##### 3-2 추가 결정 (Q-C): EW / IVW universe 정의 — Look-ahead Bias 회피

**배경**: 사용자 지적 (2026-05-10) — "EW/IVW 구현 시 해당 기간에 실제로 S&P500 에 편입되어 있는 종목만을 대상으로 할 것. 현재 의도는 universe.csv 의 모든 종목을 대상으로 계산되어 look-ahead bias 발생 위험."

**검토된 옵션**:
- (C1) `universe.csv` 의 모든 833 ticker 사용 (무시점)
- (C2) 펀드의 `weights[t] > 0` 인 active ticker 사용 (펀드 universe 동기화)
- (C3) **`sp500_membership[t-1]` 의 frozenset 사용** (직전 월말 시점 S&P500 편입 종목)

**결정**: (C3) `sp500_membership[t-1]`

**근거**:
1. **Look-ahead bias 회피**: 시점 t 의 ret 계산 시 universe 정의는 **t-1 시점에 알 수 있는 정보** 만 사용 (사용자 예시: "2월 1일 자산 구성 시 1월 31일 기준")
2. **Survivorship bias 회피**: 시점 t 에서 이미 퇴출된 종목 제외, 이후 편입될 종목 포함 X
3. **Constituent bias 회피**: 시점 t 에 실제로 S&P500 편입된 종목만 universe → 학술 백테스트 표준 (Brown et al. 1992, Banz 1981)
4. **(C1) 거리**: `universe.csv` (833 ticker) 는 전 기간 합집합 — 2010년에 2020년 편입 종목 포함 → 명백한 look-ahead
5. **(C2) 거리**: 펀드 자체가 active universe 결정 = look-ahead 회피되어 있음. 다만 명시적 학술 정의 (S&P500 편입) 가 더 명확

**구현 상세**:
- 데이터: `final/data/sp500_membership.pkl` (dict: Timestamp → frozenset of tickers)
- 시점 매핑: `ret[t]` (예: `2010-02-28`) 의 universe = `sp500_membership[t_prev]` (예: `2010-01-31`)
- universe ∩ `daily_returns.columns` 교집합 사용 (데이터 누락 ticker 제외)
- 첫 항목 (예: `2010-01-31`) 의 경우 직전 월말 = `2009-12-31` 사용

**시점 convention**:
```
fund.ret[t]    = (t-1, t] 기간 수익률 (t-1 결정 weight 로 보유)
ew.ret[t]      = (t-1, t] 기간 EW 수익률
                 universe = sp500_membership[t-1]
                 weight   = 1/N (N = |universe ∩ daily_returns.columns|)
ivw.ret[t]     = (t-1, t] 기간 IVW 수익률
                 universe = sp500_membership[t-1]
                 σ_i      = (t-window, t-1] 일별 수익률 표준편차
                 weight_i = (1/σ_i) / Σ(1/σ_j)
```

**결과 / 함의**:
- `compute_equal_weight_returns(daily_returns, sp500_membership, fund_dates)` 시그니처 채택
- `compute_ivw_returns(daily_returns, sp500_membership, fund_dates, window=120)` 시그니처 채택
- `lib/validators.py` 의 필수 데이터 파일에 `sp500_membership.pkl` 추가
- `scripts/copy_data.py` 복사 대상에 추가

##### 3-2 추가 결정 (Q-D): EW / IVW 데이터 출처 — `monthly_panel` 사용

**배경**: 1차 구현 (Q-C 시점 + `daily_returns` 기반 EW/IVW) 결과 EW 16년 누적 ~3.5배 (CAGR ~8%). RSP (Invesco S&P 500 EW ETF) 의 실제 historical 5-7배 (CAGR ~11%) 대비 명확히 낮음. **2단계 진단** 진행:

**1단계 진단** (outlier 가설):
- `daily_returns` 25개 ticker 에 -400% / +258% 비정상 (인수합병 / 파산 / 감자 처리 오류)
- 분위 winsorize 1%/99% (Tukey 1962) 적용 후에도 EW 3.47배 (CAGR 8.08%) — 미미한 개선

**2단계 진단** (데이터 자체 결함):
- `daily_returns['SPY']` 2010-2025 cumprod = **6.42배** (정확값 8.24배 대비 22% 낮음)
- `daily_returns['SPY']` 일별 min = **-41%** (정상 약 -12%, COVID 2020-03-16)
- `monthly_panel.spy_ret` 2010-2025 cumprod = **8.16배** (fund.spy_ret 8.24배와 0.96% 차이) → **신뢰 가능**
- `monthly_panel.ret_1m` 기반 EW = **7.42배 (CAGR 13.3%)** → RSP 정상 범위 부합

→ **`daily_returns` 자체에 데이터 처리 결함**. winsorize 로 회피 불가.

**검토된 옵션 (2단계)**:
- (E-A) `daily_returns` + 분위 winsorize (1단계 시도) — 결과 부족
- (E-B) `monthly_panel.ret_1m` (EW) + `monthly_panel.vol_60d` (IVW)
- (E-C) `monthly_panel` (EW) + `daily_returns` 보정 (IVW) — 데이터 출처 불일치

**결정**: (E-B) `monthly_panel` 통합 사용

**근거**:
1. **데이터 신뢰성**: `monthly_panel.spy_ret` 가 `fund.spy_ret` 와 일관성 검증 → 펀드 backtest pipeline 과 동일 데이터 출처 유추 가능
2. **EW 결과 정상 회복**: 7.42배 (CAGR 13.3%) — RSP 정상 범위
3. **데이터 출처 일관**: EW + IVW 모두 `monthly_panel` 사용 → 비교 정합성 유지
4. **시점 convention 유지**: sp500_membership[t-1] universe (Q-C 결정 그대로 유지)
5. **윈도우 변경**: `monthly_panel` 에 `vol_120d` 부재 → `vol_60d` 사용 (Q-B 결정 정정)

**구현 상세**:
```python
# EW: 매월 t 시점 sp500[t-1] universe 의 ret_1m 평균
def compute_equal_weight_returns(monthly_panel, sp500_membership, fund_dates):
    for i, t in enumerate(fund_dates):
        t_prev = fund_dates[i-1] if i > 0 else (t - pd.offsets.MonthEnd(1))
        universe = sp500_membership[max(d for d in sp_dates if d <= t_prev)]
        active = panel[(panel.date == t) & panel.ticker.isin(universe)]
        ew_ret[t] = active['ret_1m'].mean()

# IVW: 매월 t-1 시점 vol_60d 역수 가중 → t 시점 ret_1m
def compute_ivw_returns(monthly_panel, sp500_membership, fund_dates):
    for i, t in enumerate(fund_dates):
        t_prev = ...
        universe = sp500_at(t_prev)
        # t-1 시점 vol_60d 로 weight 결정 (look-ahead 회피)
        prev_active = panel[(panel.date == t_prev) & panel.ticker.isin(universe)]
        weights = (1/prev_active.vol_60d) / (1/prev_active.vol_60d).sum()
        # t 시점 ret_1m 의 가중 평균
        curr_active = panel[(panel.date == t) & panel.ticker.isin(prev_active.ticker)]
        ivw_ret[t] = (curr_active.ret_1m * weights).sum() / weights[curr_active.ticker].sum()
```

**결과 / 함의**:
- EW 7.42배 (CAGR 13.3%) — RSP 정상 범위 회복
- 함수 시그니처 변경: `_daily_returns` → `_monthly_panel`
- `app.py` import 조정 (`load_daily_returns` 호출 제거 가능)
- `sp500_membership.pkl` 그대로 필수 데이터로 유지
- `daily_returns.pkl` 은 다른 페이지 (예: Risk Metrics 의 일별 분포) 에서 필요할 수 있어 필수 유지

#### 결정 항목 3-3: Y축 스케일

**검토된 옵션**:
- (a) 선형 (Linear)
- (b) 로그 (Log)
- (c) 토글

**결정**: (c) 토글 (기본 선형)

**근거**:
1. 16년 장기 데이터 → 로그 스케일 의미 있음
2. 청중 친화 (가상 투자자) 위해 기본 선형
3. 학술 검토 시 로그 스케일 직접 전환 가능

#### 결정 항목 3-4: Regime 시각화

**검토된 옵션**:
- (a) 배경색 영역
- (b) 수직선 마커
- (c) annotation 라벨
- (d) 모든 조합
- (e) 표시 안 함

**결정**: (a) 배경색 영역 + (c) annotation 라벨

**근거**:
1. 배경색 (a) — Regime 4개 시각적 즉시 구분 (학술 narrative)
2. 라벨 (c) — "R1 회복기" 등 텍스트로 의미 전달
3. 수직선 (b) 추가 시 차트 혼잡 → 제외
4. (d) 모든 조합은 과한 정보량

#### 결정 항목 3-5: Drawdown 추가

**검토된 옵션**:
- (a) 단일 차트 (누적수익만)
- (b) 이중 차트 (위: 누적수익 / 아래: drawdown 영역)
- (c) 토글로 전환

**결정**: (b) 이중 차트

**근거**:
1. "위험 + 수익" 동시 시각화 → 펀드 정체성 (VolControl) 강조
2. 청중이 한 화면에서 양쪽 정보 동시 비교
3. 레퍼런스 PortfolioVisualizer 표준 패턴

#### 결정 항목 3-6: 인터랙션

**검토된 인터랙션**:
- Hover tooltip (날짜 + 펀드/SPY 수익률) ★★★
- Zoom / Pan ★★★
- 기간 슬라이더 ★★
- 벤치마크 토글 (3-2 와 연결) ★★
- 로그/선형 토글 (3-3 과 연결) ★
- 이벤트 annotation (COVID 등) ★★

**결정**: 모두 적용

**근거**:
1. A-2 자유 탐색 모드 결정 부합 — 인터랙션 풍부
2. Plotly 표준 기능 (구현 비용 ↓)
3. 청중 적극 탐색 유도

#### 결정 항목 3-7: 차트 크기 + 위치

**검토된 옵션**:
- (a) 전체 너비, Hero 아래
- (b) 절반 너비 + 우측 mini chart
- (c) Hero KPI 위에

**결정**: (a) 전체 너비, Hero 아래

**근거**:
1. 메인 차트 → 큰 비중 필수
2. 레퍼런스 PortfolioX360 동일 패턴
3. Hero (5개 KPI) → 누적수익 (대형) 시선 흐름 자연스러움

#### 시각화 예시 (확정 사항 조합)

```
[Hero KPI 5개 카드 — 반응형]

┌──────────────────────────────────────────────────────────┐
│ Cumulative Return — Adaptive VolControl Fund vs SPY      │
│ Y축: [Linear ▼] (토글로 Log 전환)                        │
│ ┌─ R1 회복기 ─┬── R2 확장기 ──┬── R3 변동기 ──┬─ HO ─┐  │
│ │ (배경색 1)  │  (배경색 2)   │  (배경색 3)   │(4)   │   │
│ │             │               │               │      │   │
│ │     ╱─────╱─╲────╱──╲╱─╲──── Adaptive       │      │   │
│ │   ╱╱   ╱─╱╱        ╲╱   ╲   VolControl     │      │   │
│ │ ╱╱ ╱─╱╱             ╲    ╲╱╲                │      │   │
│ │╱─╱─╱                 ╲   ╱──── SPY          │      │   │
│ └─────────────────────────────────────────────┴──────┘   │
│ 2010    2012    2014    2016    2018    2020   2024     │
│ Annotation: ▼COVID-19 (2020-03)  ▼2022 Bear  ▼2024-12  │
│                                                          │
│ ┌─ Drawdown ──────────────────────────────────────────┐  │
│ │  ▔▔▔▔▔▔▔▔▔▔▔▁▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▁▔▔▔▔▔▔▔▔▔▔▔▔▔        │  │
│ │       ╲     ╱  ╲       ╲    ╲   ╲     ╲           │  │
│ │        ╲___╱    ╲___╱   ╲___╱    ╲___╱            │  │
│ └──────────────────────────────────────────────────────┘  │
│                                                          │
│ 비교 라인: [☑ SPY]  [☐ EW (펀드 universe 1/N)]           │
│           [☐ Naive Low-vol (IVW 120d)]                   │
│ 기간 슬라이더: 2010 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2025 │
└──────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 영역 3 메인 차트 로직 확정 → 구현 시 Plotly 사용
- EW + IVW baseline 산출 → 별도 함수 구현 필요:
  - `equal_weight_returns(universe, returns_df)` — universe 1/N
  - `ivw_returns(universe, returns_df, window=120)` — 120일 inverse vol
- 메트릭 풀 (C-2) 의 baseline 활용 가능
- Performance 페이지 / Risk Metrics 페이지 차트 결정 시 동일 baseline 사용 가능
- 학술 근거: Frazzini & Pedersen (2014) — 부록 학술 근거 일람에 등재

### 영역 4: 핵심 강점 카드 (Differentiator Cards) — 확정

#### 영역 4 통합 배경 (Context)

누적수익 곡선 아래에 위치하는 펀드 차별화 포인트 카드 영역. 청중이 "왜 이 펀드인가" 의 답을 5초 안에 이해할 수 있도록 핵심 강점 3-4개를 카드 형태로 노출.

레퍼런스 PortfolioX360 의 "Risk Metrics 카드" + 일반 펀드 fact sheet 의 "Investment Thesis" 패턴 참조.

5개 핵심 결정 항목.

#### 결정 항목 4-1: 카드 개수

**검토된 옵션**:
- (a) 3개: 간결, 강한 인상
- (b) 4개: 정보량 ↑, 약간 분산
- (c) 5개: 너무 많음, 비권장

**결정**: (a) 3개

**근거**:
1. "3가지 차별화 강조" narrative — 청중 기억 용이 ("Three pillars of...")
2. 강한 인상 + 명확 메시지
3. 4개 이상은 시각적 분산 → 핵심 메시지 약화

#### 결정 항목 4-2: 카드 콘텐츠

**후보 풀** (10개):

| # | 강점 | 핵심 메시지 | 근거 메트릭 |
|---|---|---|---|
| A | LSTM 변동성 예측 | "미래 변동성을 학습으로 예측" | Volatility / Sortino |
| B | Black-Litterman 적응형 | "시장 균형 + view 결합" | CAGR / IR |
| C | 3-Regime 검증 | "회복/확장/변동기 모두 검증" | Sortino R1/R2/R3 |
| D | 학술 검증 (FF5 Alpha) | "5-factor 통제 후 양의 alpha" | FF5 Alpha |
| E | 보수적 거래비용 | "20bp 보수 가정에서도 양호" | Net CAGR / Net Sharpe |
| F | Sortino 우수 | "하방위험 대비 우수" | Sortino > SPY |
| G | Down Capture 우수 | "하락장 SPY 대비 방어" | Down Capture |
| H | Sector-balanced | "IT 집중 회피, 분산" | Sector HHI |
| I | OOS 검증 (HO 24m) | "168m 학습 + 24m 검증" | TEST/HO 분리 |
| J | Win Rate 우수 | "양수 월 비율 ↑" | Win Rate |

**검토된 카드 3개 조합**:
- A + C + E (방법론 + 검증 + 실무) ★★★
- A + B + E (방법론 양면 + 실무)
- A + G + I (LSTM + 시장방어 + OOS)

**결정**: A + C + E

| 카드 | 헤드라인 | 본문 | 숫자 | 학술 근거 |
|---|---|---|---|---|
| **카드 1: A** | Volatility-Aware Allocation | LSTM 으로 미래 변동성 예측 → 적응형 비중 결정 | Volatility (HO 10.3%) | LSTM (Hochreiter 1997) |
| **카드 2: C** | Validated Across Market Regimes | 회복/확장/변동기 + 24m OOS 검증 | Sortino IR | Regime stability |
| **카드 3: E** | Net of Conservative Costs | 20bp 거래비용 차감 후에도 양호한 위험조정 수익 | Net Sortino / Net CAGR | AQR Frazzini et al. (2018) |

**근거**:
1. **A (LSTM)**: 펀드명 (Adaptive VolControl) 직접 매칭 — 펀드 정체성 핵심
2. **C (3-Regime)**: 학술 정직성 (in-sample/OOS 분리) — 심사위원 어필
3. **E (Net)**: 실무 친화 (거래비용 차감 후 정직) — 가상 투자자 신뢰

세 카드가 펀드의 **방법론 (A) + 검증 (C) + 실무 (E)** 3축 모두 커버 → 균형 잡힌 narrative.

#### 결정 항목 4-3: 카드 디자인

**검토된 옵션**:
- (a) 헤드라인 + 본문
- (b) 헤드라인 + 본문 + 숫자
- (c) 아이콘 + 헤드라인 + 본문
- (d) 아이콘 + 헤드라인 + 본문 + 숫자
- (e) 위 + sparkline

**결정**: (d) 아이콘 + 헤드라인 + 본문 + 숫자

**근거**:
1. 정보량 max (아이콘 + 텍스트 + 핵심 수치)
2. (e) sparkline 추가는 카드 복잡도 ↑ → 누적수익 곡선 (영역 3) 과 중복
3. 아이콘 = 시각적 인상 강화

#### 결정 항목 4-4: 추가 정보 — Hover + 클릭

**검토된 옵션**:
- (a) Hover tooltip
- (b) 클릭 → 다른 페이지로 이동
- (c) 둘 다
- (d) 정적

**결정**: (c) 둘 다

**근거**:
1. Hover — 자세한 학술 근거 / 추가 메트릭 즉시 노출
2. 클릭 — 카드별 관련 페이지로 navigation
   - 카드 1 → Methodology 페이지 (LSTM 설명)
   - 카드 2 → Backtesting 페이지 (Regime 검증)
   - 카드 3 → Performance 페이지 (Net 메트릭)

##### Streamlit 구현 방식 검토

**검토된 옵션**:
- 옵션 1: `st.button` — 단순, 디자인 제한
- 옵션 2: HTML 카드 + 별도 클릭 버튼 — 외부 의존 X, 카드 자체 클릭 X
- 옵션 3: `streamlit-card` 라이브러리 — 깔끔, 카드 자체 클릭 ✓
- 옵션 4: `streamlit-elements` (Material UI) — 풍부, 학습 곡선 ↑

**결정**: 옵션 3 (`streamlit-card`) 권장 (J 섹션 기술 스택 결정 시 최종 확정)

**근거**:
1. 카드 자체 클릭 가능 (Hover + 클릭 모두 built-in)
2. 학습 곡선 ↓
3. 디자인 표준 (CSS 미세조정 가능)

**대안**: `streamlit-card` 미사용 시 옵션 2 (HTML + 별도 버튼) 으로 fallback

#### 결정 항목 4-5: 레이아웃

**검토된 옵션**:
- (a) 가로 정렬 (한 줄)
- (b) 2x2 그리드 (4개 시)
- (c) 반응형

**결정**: (c) 반응형

**근거**:
1. 영역 2 (Hero KPI) 와 일관 — 모든 카드 영역 반응형 통일
2. 큰 화면: 3개 가로 한 줄 / 작은 화면: 세로 분할 자동
3. Streamlit `st.columns` 의 가변 너비 활용

#### 시각화 예시 (확정 사항 조합)

```
[Cumulative Return + Drawdown 차트]

┌────────────────────┬────────────────────┬────────────────────┐
│  [📊]              │  [✓]               │  [💎]              │
│                    │                    │                    │
│  Volatility-Aware  │  Validated Across  │  Net of            │
│  Allocation        │  Market Regimes    │  Conservative Costs│
│                    │                    │                    │
│  LSTM 으로 미래    │  회복/확장/변동기  │  20bp 거래비용     │
│  변동성 예측 →     │  + 24m OOS 검증   │  차감 후 양호      │
│  적응형 비중       │                    │                    │
│                    │                    │                    │
│  Vol: 10.3%        │  Sortino IR: X.XX  │  Net Sortino: X.XX │
│  (vs SPY 14.2%)    │                    │                    │
│                    │                    │                    │
│  [Methodology→]    │  [Backtesting→]    │  [Performance→]    │
└────────────────────┴────────────────────┴────────────────────┘
        ↑ Hover                ↑ Hover                ↑ Hover
   (학술 근거 tooltip)
```

#### 결과 / 함의

- 영역 4 핵심 강점 narrative 확정 → "방법론 + 검증 + 실무" 3축 균형
- J 섹션 (기술 스택) 결정 시 `streamlit-card` 라이브러리 채택 검토
- 카드 클릭 시 페이지 navigation = `st.session_state.page` 활용
- Methodology / Backtesting / Performance 페이지 결정 시 카드 1/2/3 메시지와 일관 narrative 유지 필요

### 영역 5: Navigation Cards — 확정

#### 영역 5 통합 배경 (Context)

핵심 강점 카드 (영역 4) 아래에 위치. 다른 7개 페이지로의 진입점 역할. 청중이 Overview 에서 깊이 있는 탐색을 시작할 수 있도록 안내.

레퍼런스: 펀드 운용사 fact sheet 의 "Explore More" 또는 일반 대시보드의 "Quick Links" 패턴.

#### 결정 항목 5-1: Navigation 표시 방식

**검토된 옵션**:
- (a) 카드 그리드 (7개 페이지 카드)
- (b) 사이드바 의존 (영역 5 생략)
- (c) 텍스트 링크 리스트
- (d) 강조 카드 3개 + 간단 링크 4개

**결정**: (a) 카드 그리드 (7개)

**근거**:
1. A-2 자유 탐색 모드 부합 — 시각적 navigation 진입점
2. 사이드바 menu 만으로는 청중이 페이지 차별화 인지 ↓
3. (d) 강조 카드는 영역 4 와 중복 (강조는 영역 4 에서 이미 완료)
4. (c) 텍스트 링크는 시각적 약함

#### 결정 항목 5-2: 카드 구성

**검토된 옵션**:
- (a) 모두 동등 (7개 카드 같은 비중)
- (b) 3개 강조 + 4개 간단
- (c) 2개 강조 + 5개 간단

**결정**: (a) 모두 동등 (7개 카드)

**근거**:
1. 영역 4 (강점 카드) 에서 이미 3개 강조 완료 → 영역 5 에서 추가 강조는 중복
2. 청중이 페이지 차별화 자유 판단 — 자유 탐색 정신 부합
3. 시각적 일관성 (7개 동등 그리드)

#### 결정 항목 5-3: 카드 디자인

**검토된 옵션**:
- (a) 영역 4 와 동일 (아이콘 + 헤드라인 + 본문 + 숫자)
- (b) 단순 디자인 (아이콘 + 페이지명 + 1줄 설명)
- (c) 매우 간단 (페이지명 + 화살표)

**결정**: (b) 단순 디자인

**근거**:
1. 영역 4 강점 카드 (정보량 max) 와 차별화 → 정보 위계 명확
2. 영역 5 는 navigation 역할 → 시각적 가벼움 우선
3. (c) 매우 간단은 시각적 약함 → 가상 투자자 친화 ↓

#### 결정 항목 5-4: 영역 5 위치 / 생략 여부

**검토된 옵션**:
- (a) 포함 (영역 4 → 영역 5 → 영역 6)
- (b) 생략 (사이드바만)

**결정**: (a) 포함

**근거**:
1. Overview 페이지의 시선 흐름 자연스러움 (Hero → 차트 → 강점 → 다음 단계 안내 → Footer)
2. 청중이 Overview 마지막에 다음 페이지 명확 안내 받음
3. 사이드바는 보조 navigation, 영역 5 는 메인 navigation

#### 시각화 예시 (영역 5)

```
[영역 5: Navigation Cards — 7개 단순 카드 그리드]
┌─────────┬─────────┬─────────┬─────────┐
│ [📈]    │ [⚠️]    │ [🏢]    │ [🌐]    │
│Performance│Risk     │Holdings │Sector   │
│성과 분석 │위험 지표│보유 종목│Watch    │
│   →     │   →     │   →     │   →     │
└─────────┴─────────┴─────────┴─────────┘
┌─────────┬─────────┬─────────┐
│ [🧪]    │ [✓]     │ [ℹ️]    │
│Method.  │Backtest │About    │
│방법론   │검증     │FAQ      │
│   →     │   →     │   →     │
└─────────┴─────────┴─────────┘
```

#### 결과 / 함의

- 7개 페이지 모두 navigation 진입점 보장
- 카드 클릭 = `st.session_state.page` 변경 → 페이지 전환
- 영역 4 (강한 인상 카드 3개) + 영역 5 (안내 카드 7개) 시각적 위계 분명

---

### 영역 6: Footer — 확정

#### 영역 6 통합 배경 (Context)

페이지 최하단. Disclosure / 데이터 출처 / 펀드 정보 등 신뢰성 보강 정보. 학술 정직성 + 가상 투자자 신뢰 양면 충족 필요.

#### 결정 항목 6-1: Footer 콘텐츠

**검토된 옵션**:
- (a) Disclosure 만
- (b) Disclosure + 데이터 출처
- (c) Disclosure + 데이터 출처 + 메서드 요약
- (d) 전체 fact sheet (학술 근거 링크 / 팀 정보 / 연락처)

**결정**: (c) Disclosure + 데이터 출처 + 메서드 요약

**근거**:
1. 신뢰성 핵심 3요소 (정직 + 출처 + 메서드) 모두 포함
2. (d) 전체 fact sheet 는 About 페이지에서 다루므로 Footer 중복 회피
3. 가상 투자자가 한 화면에서 펀드 신뢰성 즉시 확인

#### 결정 항목 6-2: 표시 형식

**검토된 옵션**:
- (a) 작은 텍스트 한 줄
- (b) 다단 (3-4 column)
- (c) Expander (접기/펼치기)

**결정**: (b) 다단 (3 column: Disclosure / Data Source / Meta)

**근거**:
1. 정보량 (3가지) 균형 분배 → 가독성 ↑
2. (a) 한 줄은 6-1 (c) 콘텐츠 양 대비 압축 부담 ↑
3. (c) Expander 는 기본 접힌 상태 → 신뢰성 정보 노출 부족

#### 결정 항목 6-3: Disclosure 정직성

**필수 항목**:
- "백테스트 시뮬레이션" 명시 ✓
- "실제 운용 보장 X" ✓
- 데이터 기간 명시 ✓
- **HO 24m 부진 언급 ✓** ← 사용자 결정

**결정**: 권장대로 + HO 24m 부진 언급

**근거**:
1. 학술 정직성 핵심 — HO 부진 회피 시 신뢰성 ↓
2. E 섹션 (HO disclosure) 결정과 일관 (전체 페이지에서 솔직 노출)
3. FINRA Rule 2210 (마케팅 규제) 부합 — misleading 회피
4. 가상 투자자 신뢰 ↑ — "감추지 않는 펀드"

**Disclosure 텍스트 안**:
```
※ 본 결과는 백테스트 시뮬레이션이며 실제 운용 성과를 보장하지 않습니다.
   데이터 기간: 2010-01 ~ 2025-12 (학습 168m + 검증 24m)
   ※ HOLD_OUT 24m (2024-2025) 구간에서 SPY 대비 부진 (Net CAGR +X.XX%
     vs SPY +21.2%) — 자세한 분석은 Backtesting 페이지 참조
```

#### 결정 항목 6-4: Copyright + 메타

**검토된 항목**:
- "© 2026 [팀명]"
- "Built with Streamlit + Plotly"
- "Last updated: 2026-05-10"
- GitHub link

**결정**: 모두 포함

**근거**:
1. **Last updated**: 데이터 신선도 표시 (실무 표준)
2. **Built with**: 개발자 정보 — 학술/개발 투명성
3. **Copyright**: 펀드 prototype 의 출처 명시
4. **GitHub link**: 코드 공개 → 학술 검증 가능 (재현성)

#### 시각화 예시 (영역 6)

```
[영역 5: Navigation Cards 7개]

═══════════════════════════════════════════════════════════════════
┌─ Disclosure ───────┬─ Data Sources ─────┬─ Meta ─────────────────┐
│ ※ 본 결과는 백테스트│ Yahoo Finance      │ Last updated:          │
│   시뮬레이션이며   │ (Adj Close 기준)   │ 2026-05-10             │
│   실제 운용 성과를 │                    │                        │
│   보장하지 않습니다│ Fama-French Library│ Built with:            │
│                    │ (FF5 factors)      │ Streamlit + Plotly     │
│ 데이터 기간:       │                    │                        │
│ 2010-01 ~ 2025-12  │ FRED               │ © 2026 [팀명]          │
│ (학습 168m + 검증  │ (Risk-free rate)   │                        │
│  24m)              │                    │ [GitHub →]             │
│                    │ Methodology:       │                        │
│ ※ HO 24m 부진 —   │ Black-Litterman +  │                        │
│   Backtesting 참조 │ LSTM 4-slot        │                        │
└────────────────────┴────────────────────┴────────────────────────┘
═══════════════════════════════════════════════════════════════════
```

#### 결과 / 함의

- Footer 모든 페이지 동일하게 표시 (st.markdown 또는 별도 함수)
- HO 부진 언급 = E 섹션 (HO disclosure) 결정 일관 적용
- GitHub link → 코드 공개 결정 필요 (J 섹션 배포 결정 시 확정)
- About / FAQ 페이지의 자세한 Disclosure 와 차별화 (Footer = 요약 / About = 상세)

---

### Overview 페이지 — 전체 확정 (영역 1~6)

#### 페이지 시각화 통합 (확정 사항 모두 조합)

```
┌───────────────────────────────────────────────────────────────────┐
│ [영역 1: Header — 좌측 정렬]                                       │
│ Adaptive VolControl Fund                              ● Active     │
│ 어댑티브 볼컨트롤 펀드                                Benchmark: SPY│
│ 변동성 예측 기반 적응형 자산배분 —                    Data: 2025-12 │
│ Volatility-Aware Adaptive Allocation                              │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ [영역 2: Hero KPI 5개 — 반응형, sparkline 포함]                    │
│ ┌──────────┬──────────┬──────────┬──────────┬──────────┐          │
│ │Cumulative│Net CAGR  │Sortino   │Volatility│Max DD    │          │
│ │TEST/HO   │TEST/HO   │TEST/HO   │TEST/HO   │TEST/HO   │          │
│ │sparkline │sparkline │sparkline │sparkline │sparkline │          │
│ │(FULL,    │(FULL,    │(FULL,    │(FULL,    │(FULL,    │          │
│ │ HO red)  │ HO red)  │ HO red)  │ HO red)  │ HO red)  │          │
│ └──────────┴──────────┴──────────┴──────────┴──────────┘          │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ [영역 3: 누적수익 곡선 — 전체 너비, 이중 차트]                      │
│ Cumulative Return — Adaptive VolControl Fund vs SPY               │
│ ┌─R1 회복기─┬─R2 확장기─┬─R3 변동기─┬HO─┐ (배경색 + 라벨)         │
│ │  ── Adaptive VolControl Fund                       │           │
│ │  ── SPY                                              │           │
│ │  ── EW (toggle)                                      │           │
│ │  ── Naive Low-vol IVW (toggle)                       │           │
│ └─────────────────────────────────────────────────────┘           │
│ Annotation: ▼COVID-19  ▼2022 Bear  ▼2024-12 Sector                │
│ Drawdown 영역 (이중 차트 하단)                                     │
│ [Linear▼] [기간 슬라이더]  [☑SPY] [☐EW] [☐IVW]                   │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ [영역 4: 핵심 강점 카드 3개 — Hover + 클릭 (streamlit-card)]       │
│ ┌────────────┬────────────┬────────────┐                          │
│ │[📊]Vol-Aware│[✓]Validated│[💎]Net of  │                          │
│ │Allocation  │Across Regimes│Conserv.Costs│                        │
│ │Vol: 10.3%  │Sortino IR:..│Net Sortino  │                        │
│ │[Method.→]  │[Backtest→] │[Performance→]│                        │
│ └────────────┴────────────┴────────────┘                          │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ [영역 5: Navigation Cards 7개 — 단순 디자인]                       │
│ ┌────┬────┬────┬────┐  ┌────┬────┬────┐                          │
│ │📈  │⚠️  │🏢  │🌐  │  │🧪  │✓   │ℹ️  │                          │
│ │Perf│Risk│Hold│Sect│  │Meth│Btst│About│                          │
│ │ →  │ →  │ →  │ →  │  │ →  │ →  │ →  │                          │
│ └────┴────┴────┴────┘  └────┴────┴────┘                          │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ [영역 6: Footer — 다단 3 column]                                   │
│ Disclosure | Data Sources | Meta                                  │
│ HO 부진 언급 + 데이터 출처 + Last updated + Built with + GitHub   │
└───────────────────────────────────────────────────────────────────┘
```

#### Overview 페이지 결정 결과 / 함의

- Overview 페이지 6 영역 모두 확정 → 구현 시 명확한 와이어프레임
- 다른 7개 페이지 (Performance / Risk Metrics / Holdings / Sector Watch / Methodology / Backtesting / About) 의 일관성 기준 확립:
  - Header 영역 (1-1 ~ 1-4) 모든 페이지 동일
  - Footer 영역 모든 페이지 동일
  - 색상 팔레트 (B-4) 일관 적용
  - 카드 디자인 패턴 (영역 4 강점 / 영역 5 navigation) 다른 페이지에서 재사용
- 사이드바 토글 (영역 2 결정) 영향 범위 명확 → 다른 페이지에 적용
- 각 페이지에 Overview 와 동일한 Header + Footer + 사이드바 적용

### Overview 페이지 → Performance 페이지로 진행

---


---

[← 01_meta_A_B_C.md](01_meta_A_B_C.md) | [03_performance.md](03_performance.md) →
