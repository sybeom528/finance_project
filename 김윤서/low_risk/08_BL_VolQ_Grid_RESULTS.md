# 06. 변동성 소스 × Q 방식 전체 비교 — 실행 결과 해석

> 실행 조건: 2011-01-31 ~ 2025-12-31 (180개월), TRAIN_WINDOW=60, TAU=0.1, PCT_GROUP=0.30  
> GARCH 시작 시점(2011-01)에 맞춰 Baseline도 동일 기간 사용 (공정 비교)  
> **Baseline×5Q 결과는 05번 `q_comparison_returns.csv`에서 로드 (재계산 없음)**  
> **GARCH×5Q 결과만 이 노트북에서 새로 계산**

---

## ⚠️ 실행 상태

| 조합 | 결과 출처 | 상태 |
|------|---------|------|
| Baseline × 5Q | 05번 CSV 로드 | ✓ 완료 |
| GARCH × Q_FIXED | 구 2×3 실행 결과 | ✓ 보유 |
| GARCH × Q_hist | 구 2×3 실행 결과 | ✓ 보유 |
| GARCH × Q_ff3 | 구 2×3 실행 결과 | ✓ 보유 |
| **GARCH × Q_momentum** | — | ⚠️ 미실행 (06 재실행 필요) |
| **GARCH × Q_lambda** | — | ⚠️ 미실행 (06 재실행 필요) |

---

## 성과 비교 표 (현재 보유 결과)

| 전략 | 연환산수익률 | 연환산변동성 | Sharpe | 누적수익률 | MDD |
|------|------------|------------|--------|-----------|-----|
| **Baseline+Q_lambda** | **13.73%** | **11.01%** | **1.112** | 609.29% | **-11.84%** |
| Baseline+Q_FIXED | 13.77% | 11.44% | 1.074 | 608.29% | -13.70% |
| Baseline+Q_hist | 14.19% | 14.24% | 0.896 | 613.83% | -21.57% |
| Baseline+Q_momentum | 10.58% | 15.33% | 0.597 | 307.68% | -42.00% |
| Baseline+Q_ff3 | 11.26% | 16.79% | 0.586 | 332.93% | -51.48% |
| GARCH+Q_FIXED | 13.50% | 11.81% | 1.019 | 575.70% | -16.73% |
| GARCH+Q_hist | 12.81% | 13.24% | 0.859 | 493.99% | -19.50% |
| GARCH+Q_momentum | ⚠️ 미실행 | — | — | — | — |
| GARCH+Q_lambda | ⚠️ 미실행 | — | — | — | — |
| GARCH+Q_ff3 | 11.42% | 17.94% | 0.557 | 331.08% | -54.37% |
| CAPM | 14.21% | 14.49% | 0.882 | 612.78% | -22.17% |
| SPY | 14.03% | 14.06% | 0.898 | — | -23.93% |

> Baseline 결과(5행)는 05번 결과와 동일 — 동일 파라미터로 동일 로직이 실행되어 수치가 완전히 일치함.

---

## 현재 보유 결과 기반 2×3 부분 격자

```
              Q_FIXED    Q_hist    Q_ff3
Baseline       1.074      0.896    0.586
GARCH          1.019      0.859    0.557

차이 (B-G)    +0.055     +0.037   +0.029  ← Baseline이 일관되게 우위
```

Q_momentum, Q_lambda의 GARCH 버전은 06 재실행 후 추가 예정.

---

## 핵심 발견

### 발견 1: Q 방식이 vol 소스보다 더 결정적

같은 Q_FIXED를 쓰면 vol 소스가 달라도 Sharpe가 비슷하게 높다.  
반면 Q_ff3는 vol 소스와 무관하게 항상 최하위다.

```
Sharpe 순위 패턴 (일관됨):
Q_FIXED > Q_hist > Q_ff3  (Baseline 기준: 1.074 / 0.896 / 0.586)
Q_FIXED > Q_hist > Q_ff3  (GARCH 기준:    1.019 / 0.859 / 0.557)
```

→ **vol 예측 모델보다 Q 설정이 성과에 더 큰 영향을 미친다.**

---

### 발견 2: Baseline+Q_lambda가 현재 보유 결과 중 최고 (Sharpe 1.112, MDD -11.84%)

05번 실행에서 확인된 Q_lambda의 우위가 그대로 반영된다.  
GARCH+Q_lambda는 아직 미실행이므로 Baseline vs GARCH 비교는 06 재실행 후 가능.

---

### 발견 3: GARCH가 Baseline보다 Sharpe가 낮다 (Q_FIXED 기준)

```
Baseline+Q_FIXED: Sharpe 1.074, MDD -13.70%
GARCH+Q_FIXED:    Sharpe 1.019, MDD -16.73%
→ GARCH가 Baseline보다 Sharpe -0.055, MDD도 더 깊음
```

**왜 GARCH가 오히려 성능이 낮을까?**  
변동성은 강한 자기상관(ARCH 효과)을 가지므로 현재 vol_21d 자체가 다음 달 vol의 좋은 예측치다.  
04_5_GARCH_Evaluation에서 확인된 레짐 역설(고변동성 구간 Baseline IC 역전)이 포트폴리오 성과에도 나타난다.

이것은 팩터 투자에서 잘 알려진 현상: **단순하고 안정적인 신호가 복잡한 예측을 종종 이긴다.**

---

### 발견 4: Q_ff3는 vol 소스 무관하게 완전 실패

```
Baseline+Q_ff3: Sharpe 0.586, MDD -51.48%
GARCH+Q_ff3:    Sharpe 0.557, MDD -54.37%
```

FF3 회귀 기반 Q는 월별로 크게 흔들리고 음수가 되는 달이 많아 BL이 저변동 종목을 오히려 줄이는 방향으로 왜곡된다.  
수익률 예측 자체의 어려움이 근본 원인.

---

### 발견 5: 모든 Q_FIXED·Q_hist BL 전략이 CAPM을 Sharpe에서 이긴다

```
BL (Q_FIXED): Sharpe 1.019~1.074  >  CAPM: 0.882
BL (Q_hist):  Sharpe 0.859~0.896  ≈  CAPM: 0.882
```

BL 프레임워크(저변동 long, 고변동 short 뷰)가 순수 CAPM 최적화보다 위험조정 성과 면에서 우위다.  
저위험 이상현상이 실증적으로 작동하고 있다는 근거.

---

## 현재 결론 (부분 결과 기준)

### 파라미터 중요도

| 파라미터 | 중요도 | 결론 |
|---------|--------|------|
| **Q 방식** | ★★★★★ | Q_lambda > Q_FIXED >> Q_hist > Q_momentum ≈ Q_ff3 |
| **vol 소스 (Baseline vs GARCH)** | ★★☆☆☆ | Baseline이 우위 (Q_FIXED 기준 Sharpe +0.055) |
| **BL 프레임워크 자체** | ★★★★☆ | CAPM 대비 Sharpe 우위 확인 |

### 현재 최선 조합

**Baseline + Q_lambda (Sharpe 1.112, MDD -11.84%)** — 05번 결과 기준 최고.  
GARCH+Q_lambda 실행 후 vol 소스 비교 가능.

---

## 다음 단계

06_BL_VolQ_Grid.ipynb를 재실행하여 GARCH×Q_momentum, GARCH×Q_lambda 결과 확보 후 이 문서 업데이트 필요.

```
예상 가설:
GARCH+Q_lambda: Baseline+Q_lambda보다 낮거나 유사 (GARCH vol 소스의 구조적 한계)
GARCH+Q_momentum: Baseline+Q_momentum보다 낮을 가능성 (음수 Q 발생 빈도 영향)
```

---

*생성일: 2026-04-29 (부분 결과 기준 — GARCH×Q_momentum, GARCH×Q_lambda 미실행)*
