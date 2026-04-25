# 하연_WORKLOG.md

> **작성자**: 김하연 (hayeon)
> **규칙**: 설계 결정·파일 변경 시 날짜·내용·판단 근거를 누적 기록한다.

---

## 2026-04-25

### 담당 파트 확정

**파트**: 통계 검정 + EDA (`01_eda_statistics.ipynb`)

**역할**: `01_data_download_and_eda.ipynb`의 기존 내용에 시계열 모델 입력 준비를 위한 통계 검정 섹션을 추가한 확장판 작성.

---

### 신규: `01_eda_statistics.ipynb`

기존 `01_data_download_and_eda.ipynb` (§1~§6) 에 아래 섹션 추가.

| 섹션 | 내용 | 목적 |
|---|---|---|
| §7 | ADF + KPSS 정상성 검정 | LSTM 입력 전제 확인 |
| §8 | 정규화 판단 + Jarque-Bera | Scaler 불필요 결론 + fat-tail 공식 확인 |
| §9 | Ljung-Box 검정 | 자기상관 종합 p-value (lag-by-lag ACF 보완) |
| §10 | ARCH-LM 검정 | 변동성 클러스터링 공식 검정 |
| §11 | Lookback Window T 결정 | seq_len=126 통계적 검증 |
| §12 | 결론 (업데이트) | 신규 섹션 결과 반영 |

**§11 Lookback T 결정 기준 (세 가지 통계 + 도메인)**:

1. **PACF cutoff**: 첫 비유의 lag → AR 선형 order (lookback 하한)
2. **제곱 수익률 Ljung-Box**: 변동성 자기상관 소멸 lag
3. **AR(p) AIC/BIC**: `AutoReg` p=1~60 탐색, 정보 기준 최적 p
4. **도메인 지식**: 126일 ≈ 반기 (어닝시즌 2회 주기)
5. **학습자료 권장**: T=60 근처 (T<20 의존성 부족, T>200 표본 급감)

**설계 결정**:
- 학습자료_주의사항.md §5.2 경고("p-value만으로 판단 X") → §9 Ljung-Box 마크다운 셀에 명시
- 학습자료_주의사항.md §6.1 경고("ADF/KPSS 해석 주의") → §7 마크다운 셀에 명시
- §8 정규화 판단: 시퀀스 모델 입력 준비 이미지 기준 그대로 반영 (log_return → Scaler 불필요)
- Jarque-Bera 비정규 결론 → Huber loss 사용 근거로 연결

**총 셀 수**: 32개 (마크다운 12 + 코드 20)
