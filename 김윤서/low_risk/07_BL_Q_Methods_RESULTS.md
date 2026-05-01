# 07. GARCH × Q 추정 방식 비교 — 실행 결과 해석

> 실행 조건: 2010-01-31 ~ 2024-12-31 (180개월), TRAIN_WINDOW=60, TAU=0.1, PCT_GROUP=0.30  
> Vol 소스: GARCH(05번 예측) 단일. Q 방식 5종만 비교  
> Q_FIXED는 06번 GARCH 민감도 분석에서 자동 로드 (Sharpe 최대 기준)

---

## ⚠️ 재실행 필요 안내 (2026-05-01)

이 노트북은 옛 `07_BL_Q_Comparison`(Baseline×5Q)과 옛 `08_BL_VolQ_Grid`(2×5 격자)를 통합·단순화한 새 구조다 — 옛 노트북 결과(Baseline 5종 비교, 2×3 부분 격자)는 폐기되었다. **노트북을 다시 실행해 GARCH×5Q 전체 결과를 새로 확보**해야 이 문서의 표를 채울 수 있다.

옛 RESULTS에서 보존 가치가 있는 핵심 발견은 본 문서 하단 "참고: 옛 결과 핵심 발견"에 요약했다.

---

## 비교 대상

| 전략 | Q 산출 방식 | 비고 |
|------|-----------|------|
| `Q_FIXED` | 06 자동 로드 (Sharpe 최대) | 기준선 |
| `Q_hist` | mean(저위험 − 고위험) 60M | 학습기간 평균 실현수익 |
| `Q_momentum` | mean(저위험 − 고위험) 12M | 단기 추세 |
| `Q_lambda` | Q_FIXED × clip(λ_t/λ_mean, 0.1, 3.0) | 위험회피계수 스케일 |
| `Q_ff3` | FF3 회귀 → P·r̂ | 팩터모델 기반 |
| **CAPM** | π = λΣw_mkt (BL 없음) | 기준선 |
| **SPY** | S&P 500 매수보유 | 시장 벤치마크 |

---

## 성과 비교 표 (재실행 후 채울 것)

| 전략 | 연환산수익률 | 연환산변동성 | Sharpe | Calmar | 누적수익률 | MDD |
|------|------------|------------|--------|--------|-----------|-----|
| Q_FIXED | _ | _ | _ | _ | _ | _ |
| Q_hist | _ | _ | _ | _ | _ | _ |
| Q_momentum | _ | _ | _ | _ | _ | _ |
| Q_lambda | _ | _ | _ | _ | _ | _ |
| Q_ff3 | _ | _ | _ | _ | _ | _ |
| CAPM | _ | _ | _ | _ | _ | _ |
| SPY | _ | _ | _ | _ | _ | _ |

저장 위치: `outputs/08_BL_Q_Methods/q_methods_stats.csv`

---

## 참고: 옛 결과 핵심 발견 (Baseline 데이터 기준 — 참고용)

### 발견 1: Q 방식이 vol 소스보다 결정적

같은 Q_FIXED를 쓰면 vol 소스(Baseline vs GARCH)가 달라도 Sharpe가 비슷하게 높았다.  
반면 Q_ff3는 vol 소스와 무관하게 항상 최하위였다.

```
Sharpe 순위 패턴:
Q_FIXED > Q_hist > Q_ff3
```

→ **vol 예측 모델보다 Q 설정이 성과에 더 큰 영향을 미친다.** 이 결론은 LSTM으로 vol 모델을 교체해도 동일하게 검증되어야 한다.

### 발견 2: GARCH가 Baseline보다 Sharpe가 낮다 (Q_FIXED 기준)

```
Baseline+Q_FIXED: Sharpe 1.074, MDD -13.70%
GARCH+Q_FIXED:    Sharpe 1.019, MDD -16.73%
```

변동성은 강한 자기상관(ARCH 효과)을 가지므로 현재 vol_21d 자체가 다음 달 vol의 양호한 예측치다. 05_VolatilityPrediction의 레짐 역설(고변동성 구간 Baseline IC 역전)이 포트폴리오 성과에도 그대로 반영된다.

→ **이것이 LSTM 도입의 동기**: GARCH의 ARCH 효과 활용만으로는 부족하므로 **시계열 패턴 학습 능력이 강한 LSTM으로 교체** 시 Baseline 대비 우위 확보를 목표로 한다.

### 발견 3: Q_ff3는 vol 소스 무관하게 실패

```
Baseline+Q_ff3: Sharpe 0.586, MDD -51.48%
GARCH+Q_ff3:    Sharpe 0.557, MDD -54.37%
```

FF3 회귀 기반 Q는 월별로 크게 흔들리고 음수가 되는 달이 많아 BL이 저변동 종목을 오히려 줄이는 방향으로 왜곡된다.

### 발견 4: BL 프레임워크 자체가 CAPM을 Sharpe에서 이긴다

```
BL (Q_FIXED): Sharpe 1.019~1.074  >  CAPM: 0.882
```

저위험 이상현상이 실증적으로 작동하고 있다는 근거 — vol 모델/Q 방식과 독립적인 BL 자체의 가치.

---

## LSTM 모델로 교체 시

이 노트북은 `data/vol_predicted.csv`만 읽으므로 **LSTM 출력 파일이 같은 스키마**(date, ticker, vol_pred)면 코드 수정 없이 그대로 실행 가능하다. 자세한 절차는 [PIPELINE.md](PIPELINE.md)의 "예측 모델 교체 가이드" 섹션 참조.

LSTM 적용 후 비교 포인트:
1. **Sharpe**: GARCH+Q_FIXED 1.019 대비 LSTM+Q_FIXED 우위 여부
2. **MDD**: GARCH 16.73% 대비 LSTM 개선 여부
3. **레짐별 IC**: 05번 평가에서 GARCH가 고변동성 구간 IC 역전 → LSTM이 이 구간을 메우는지 확인

---

*최종 업데이트: 2026-05-01 (구조 통합. 결과 표는 노트북 재실행 후 채워질 예정)*
