# 세션 정리 — 2026-04-29

> 이 문서는 오늘 대화에서 반영된 사항 + 녹취록 기반 액션 아이템을 정리한 것입니다.

---

## 1. 오늘 반영된 사항

### 1-1. 09_Q_Adaptive_Comparison.ipynb — 신규 생성

6종 BL 전략 + 벤치마크 2종을 단일 walk-forward 루프에서 비교하는 노트북.

| 전략 | 핵심 아이디어 |
|------|---------------|
| Fixed-Q | Q = Q_OPTIMAL 고정 (기준선) |
| Regime3 | 저→Q_OPTIMAL / 중→×0.5 / 고→0 |
| Q_lambda | Q = Q_OPTIMAL × clip(λ/λ_mean, 0.1, 3.0) |
| Regime+λ | 고→0 (하드스탑) + 나머지→Q_lambda |
| GARCH_conf | Q = Q_OPTIMAL × GARCH 예측 신뢰도 |
| π_ratio | Q = Q_OPTIMAL × clip(\|P·π_t\| / expanding_median, 0.1, 3.0) |
| CAPM | BL 없음, π로만 최적화 |
| SPY | S&P 500 지수 |

**의존성**: `05_Q_Sensitivity → Q_OPTIMAL 자동 로드`, `04_VolatilityPrediction → vol_predicted.csv`

**Look-ahead bias 방지 처리**:
- 레짐 분위수: expanding quantile (미래 데이터 미사용)
- GARCH 신뢰도: expanding quantile 정규화 → [0.3, 1.0]
- π_ratio 기준 스프레드: expanding median (`pi_spread_history`)

---

### 1-2. 혼동행렬 및 MSE/QLIKE — 위치 결정

| 분석 | 결정된 위치 | 이유 |
|------|------------|------|
| 레짐 혼동행렬 | 08_Regime_Q_Portfolio | 레짐 설계·검증 노트북 |
| MSE / QLIKE | 04_VolatilityPrediction | GARCH 모델 검증 노트북 |

09는 Q 전략 포트폴리오 성능 비교가 목적이므로, 두 분석 모두 제외.

---

### 1-3. 04_VolatilityPrediction.ipynb — MSE/QLIKE 셀 추가

Rank IC 시각화(cell-04) 이후에 2개 셀 추가:

#### 추가된 마크다운: 손실함수 정의

$$L_{\text{MSE}}(\hat{\sigma}_t,\, \sigma_t) = (\hat{\sigma}_t - \sigma_t)^2$$

$$L_{\text{QLIKE}}(\hat{h}_t,\, \sigma^2_t) = \frac{\hat{h}_t}{\sigma^2_t} - \log\!\left(\frac{\hat{h}_t}{\sigma^2_t}\right) - 1$$

- QLIKE 최솟값 = 0 (완벽 예측), **과소예측에 더 강한 페널티**
- Patton (2011): 노이즈가 있는 변동성 대리지표 하에서도 일관된(consistent) 손실함수

#### Look-ahead bias 구분

| 변수 | 관측 가능 시점 |
|------|--------------|
| `vol_pred` at t | t 이전 데이터로 추정 → t에 관측 가능 |
| `vol_21d` at t | t 시점 기준 trailing 21일 실현 변동성 |

두 변수 모두 시점 t에 관측 가능 → look-ahead bias 없음.

#### 추가된 코드: 월별 RMSE/QLIKE + 시각화 3종
1. RMSE 시계열 (12M MA 포함)
2. QLIKE 시계열 (QLIKE=0 기준선)
3. RMSE vs Rank IC 산점도 (두 지표 상관 확인)

**md-00도 동기화**: 평가 지표 테이블에 RMSE/QLIKE 항목 추가.

---

### 1-4. Prior 1/N vs 시가총액 — 방향 논의

#### 결론: 이 프로젝트에서 1/N이 더 타당

| 근거 | 내용 |
|------|------|
| ETF 시가총액 의미 | AUM은 기초자산의 경제적 비중이 아님 |
| 편향된 유니버스 | 30개 선택 자체가 selection bias → 시가총액 prior가 "균형" 대표 못함 |
| 뷰와 prior 분리 | prior가 편향을 가지면 뷰(Q, P, Ω) 효과가 prior 편향과 혼재 |

#### 변경 범위 (아직 미적용)

**반드시 바뀌는 곳** — `w_mkt` (6개 파일, 한 줄씩):

```python
# 현재
w_mkt = (mcap / mcap.sum()).reindex(valid_tix).fillna(0)

# 변경
w_mkt = pd.Series(1.0 / len(valid_tix), index=valid_tix)
```

| 파일 | 변경 위치 수 |
|------|------------|
| 05_Q_Sensitivity | 2곳 |
| 06_BL_Q_Comparison | 1곳 |
| 07_BL_VolQ_Grid | 1곳 |
| 08_Regime_Q_Portfolio | 2곳 |
| 09_Q_Adaptive_Comparison | 1곳 |
| 99_baseline | 2곳 |

**별도 결정 필요** — `build_P` 내부 가중치:
- 현재: 그룹 내 시가총액 비중
- 대안: 1/N (equal) 또는 변동성 역수 (`1/vol`)
- `build_P`는 prior가 아닌 **뷰 벡터** → 별도 설계 결정

---

### 1-5. 코드 전반 버그 검토 — 영업일/캘린더 불일치 이슈

팀 미팅 녹취록에서 언급된 "영업일 캘린더와 일반 캘린더 날짜 불일치로 21개월 누락" 버그가 우리 코드에도 존재하는지 전수 검토.

**결론: 우리 코드에는 해당 버그 없음**

| 검토 항목 | 결과 |
|----------|------|
| `monthly_panel.csv` 날짜 형식 | 모든 날짜 캘린더 월말(`resample('ME')`) — 264개월 전부 |
| `vol_predicted.csv` 날짜 | 동일한 캘린더 월말 기준 |
| walk-forward 누락 월 | `regime_q_returns.csv` shape=(180, 3), NaN=0 — 누락 없음 |
| 파이프라인 전체 일관성 | 데이터 생성부터 백테스트까지 `resample('ME')` 하나로 통일 |

녹취록 발언 "제꺼는 그랬어요" → 다른 팀원 코드의 문제, 우리 코드와 무관.

---

### 1-6. 코드 전반 추가 검토 — 기타 잠재 버그

| 항목 | 결과 |
|------|------|
| 레짐 look-ahead | expanding quantile 확인 — 클린 |
| GARCH 훈련 윈도우 | `train_dates = all_dates[idx-60:idx]` (pred_date 미포함) — 클린 |
| BL 수식 | P·Σ·P^T + Ω, 조정 방향 모두 정확 |
| 포트폴리오 제약 | long-only bounds=(0,1), w.sum()=1 확인 |
| fwd_ret_1m NaN (2025년) | NaN 비율 0% — PRICE_END='2026-03-31' 버퍼 덕분 |
| `optimize_portfolio` 폴백 | 수렴 실패 시 경고 없이 1/N 대체 → **이슈 발견, 1-7에서 수정** |
| `actual_ret` 평가 기간 | BL(21영업일) vs SPY(캘린더 월) 불일치 → **이슈 발견, 1-8에서 수정** |

---

### 1-7. `optimize_portfolio` 수렴 실패 시 경고 추가

**문제**: optimizer 실패 시 1/N 균등배분으로 폴백하지만 경고 없음 → 어느 달에 수치 문제가 있었는지 파악 불가.

**수정 내용**: 실패 시 `warnings.warn` 출력 추가.

```python
# 수정 전
w = res.x if res.success else np.ones(n)/n

# 수정 후
if not res.success:
    import warnings
    warnings.warn(f'optimize_portfolio 수렴 실패 → 1/N 대체: {res.message}')
w = res.x if res.success else np.ones(n)/n
```

**적용 파일 (7개 노트북)**:

| 파일 | Cell |
|------|------|
| 05_Q_Sensitivity | 3 |
| 06_BL_Q_Comparison | 3 |
| 07_BL_VolQ_Grid | 3 |
| 08_Regime_Q_Portfolio | 3 |
| 09_Q_Adaptive_Comparison | 3 |
| 99_baseline | 3 |
| 98_2006_baseline | 2 |

---

### 1-8. `actual_ret` 평가 기간 통일 — `fwd_ret_1m` → `ret_pivot.loc[next_date]`

**문제**: BL 포트폴리오 수익률이 "pred_date 이후 21 영업일 복리"로 계산되어 SPY 벤치마크(캘린더 월)와 평가 기간 불일치.

```
# pred_date = 2025-01-31 예시
BL 이전: fwd_ret_1m → 2월1일~2월 21영업일 후 (2월 짧으면 3월 초까지 침범)
BL 수정: ret_pivot.loc[2025-02-28] → 1월31일 종가 ~ 2월28일 종가 (캘린더 월과 동일)
SPY:     spy_series.get(2025-02-28) → 동일한 2월28일 기준
```

실측 편향: 평균 0.069%/월, 표준편차 0.82%/월 → 방향이 랜덤이라 180개월에 걸쳐 상쇄되나, 원칙적으로 비교 기간을 통일하는 것이 더 정확.

**수정 내용**:

```python
# 수정 전
actual_ret = month_df['fwd_ret_1m'].reindex(valid_tix).fillna(0)

# 수정 후 (next_date는 루프 내 이미 계산 또는 신규 삽입)
actual_ret = (ret_pivot.loc[next_date].reindex(valid_tix).fillna(0)
              if next_date is not None
              else month_df['fwd_ret_1m'].reindex(valid_tix).fillna(0))
```

마지막 달(`next_date=None`)은 기존 `fwd_ret_1m` 폴백 유지 — PRICE_END='2026-03-31' 버퍼로 유효한 값.

**적용 범위 (10개 셀, 7개 노트북)**:

| 파일 | Cell | 처리 유형 |
|------|------|----------|
| 05_Q_Sensitivity | 5 | next_date 기존 존재 |
| 05_Q_Sensitivity | 7 | next_date 신규 삽입 |
| 06_BL_Q_Comparison | 4 | next_date 순서 재배치 |
| 07_BL_VolQ_Grid | 4 | next_date 신규 삽입 |
| 08_Regime_Q_Portfolio | 7 | next_date 기존 존재 |
| 08_Regime_Q_Portfolio | 15 | next_date 신규 삽입 |
| 99_baseline | 4 | next_date 순서 재배치 |
| 99_baseline | 7 | next_date 순서 재배치 |
| 98_2006_baseline | 3 | next_date 순서 재배치 |
| 09_Q_Adaptive_Comparison | 7 | next_date 기존 존재 |

> ⚠️ **재실행 필요**: 수정된 노트북(05, 06, 07, 08, 09, 99, 98) 전부 walk-forward 재실행 및 기존 출력 CSV 갱신 필요.

---

## 2. 녹취록 기반 액션 아이템 (미완료)

> 출처: 2026-04-29 팀 미팅 녹취록

---

### [즉시] A. 혼동행렬 — 레짐/변동성 그룹 오분류 확인

**위치**: 08_Regime_Q_Portfolio에 셀 추가

**내용**: 예측 시점 t의 레짐 분류 vs 다음 달(t+1) 실현 레짐 비교

```
예측 레짐(t)  ×  실현 레짐(t+1) → 3×3 혼동행렬
저변동성 예측 → 실제 고변동성 (최악의 오분류)
고변동성 예측 → 실제 저변동성 (기회 손실)
```

- 오분류 월의 포트폴리오 수익률도 함께 확인
- 튜터님 직접 요청 사항

---

### [즉시] B. 시가총액 × 변동성 교차 EDA

**위치**: 02_LowRiskAnomaly 또는 별도 EDA 셀

**필요한 근거**:
> "저변동성이면서 시가총액 비중이 높은 애들이 실제로 돈을 버는가"

**방법**:
- 변동성 분위(3분위) × 시가총액 분위(3분위) → 9개 그룹
- 각 그룹의 다음 달 평균 수익률 히트맵
- "저변동·대형주"가 유의미하게 높은 수익률인지 통계 확인

---

### [단기] C. PCT_GROUP(30%) 임계치 동적화

**현재 문제**: 상·하위 30%를 항상 고정 → 논문의 하나의 가설일 뿐, 일반화 안됨

**시도 방향**:

| 방법 | 내용 |
|------|------|
| 레짐 연동 | 저변동성 레짐 → PCT 확대(예: 40%), 고변동성 → 축소(예: 20%) |
| 클러스터링 | K-means 등으로 자연 경계 탐색 |
| 분위 고정 최적화 | 05번 방식처럼 PCT_GROUP 자체를 그리드 서치 |

---

### [단기] D. P 행렬 가중치 — 변동성 역수 실험

**배경**: 시가총액 비중 → prior 변경과 일관성 문제

```python
# 현재 (시가총액 비중)
P[low_risk] = mcap[low_risk] / mcap[low_risk].sum()

# 실험 후보: 변동성의 역수
inv_vol = 1.0 / vol_series[low_risk]
P[low_risk] = inv_vol / inv_vol.sum()
```

**의미**: 변동성이 낮을수록 뷰에서 더 높은 비중 → 저위험 이상현상과 직접 연결

**주의**: 저변동성 종목에 극단적 집중 가능성 → MDD 악화 리스크 확인 필요

---

### [단기] E. Q 전략 — Fixed vs Lambda 레짐별 분리

**배경 (녹취록)**:
> "GARCH(가치)는 Fixed-Q가 더 좋고, 베이스라인은 Lambda가 더 좋다 → 레짐별로 다르게 가져가는 방식"

**실험 방향**:
- 저·중변동성: `Q_lambda` (시장 위험회피계수 반영)
- 고변동성: `Q = 0` (하드스탑)

09_Q_Adaptive_Comparison 결과 확인 후 08에 반영.

---

### [검토] F. HRP 벤치마크 추가

**배경**:
> "Q가 없으면 BL보다 HRP가 방식 관점에서 더 합리적 — 짜치는 정도를 튜터님께 확인 후 결정"

- 튜터님께 현재 구조의 타당성 문의 결과에 따라 결정
- HRP를 추가 벤치마크로만 넣는 방향도 가능

---

### [배경 확인] G. 2025년 후속 논문 검토

**배경**:
> "변동성뿐 아니라 모멘텀 등 4가지 기준으로 P 행렬을 확대 구성한 2025년 논문"

- 키페이퍼: 2018년 논문 유지
- 후속 논문에서 P 행렬 다양화 아이디어 참고 가능
- 서윤범님 DM으로 공유된 논문 확인

---

## 3. 현재 상태 요약

| 노트북 | 상태 |
|--------|------|
| 01_DataCollection | 완료 |
| 02_LowRiskAnomaly | 완료 (EDA B 추가 예정) |
| 03_VolatilityEDA | 완료 |
| 04_VolatilityPrediction | ✅ MSE/QLIKE 추가 완료 (재실행 필요) |
| 04_5_GARCH_Evaluation | 완료 |
| 05_Q_Sensitivity | ⚠️ actual_ret 수정 → 재실행 필요 |
| 06_BL_Q_Comparison | ⚠️ actual_ret 수정 → 재실행 필요 |
| 07_BL_VolQ_Grid | ⚠️ actual_ret 수정 → 재실행 필요 |
| 08_Regime_Q_Portfolio | ⚠️ actual_ret 수정 → 재실행 필요 / 혼동행렬 추가 예정 (액션 A) |
| 09_Q_Adaptive_Comparison | ✅ 신규 생성 / ⚠️ actual_ret 수정 → 재실행 필요 |
| 99_baseline | ⚠️ actual_ret 수정 → 재실행 필요 |
| 98_2006_baseline | ⚠️ actual_ret 수정 → 재실행 필요 |

---

## 4. 보류 중인 결정

| 항목 | 현황 | 결정 필요 시점 |
|------|------|--------------|
| Prior 1/N 전환 여부 | 방향 확정, 코드 미적용 | 5/5 전 |
| `build_P` 가중치 변경 | 3가지 후보 (mcap / 1/N / 1/vol) | 실험 후 결정 |
| PCT_GROUP 동적화 방법 | 레짐연동 vs 클러스터링 | 하윤님과 논의 |
| HRP 추가 여부 | 튜터님 확인 대기 | 확인 후 결정 |

---

## 5. 데드라인

| 날짜 | 목표 |
|------|------|
| 2026-05-01 | 출석 필수 |
| 2026-05-05 | 코드 전체 수합·정리, 결과물 통합 |
| 2026-05-09~12 | 최종 마무리 (서윤범님 일정 제약 있음) |

---

*작성일: 2026-04-29*
