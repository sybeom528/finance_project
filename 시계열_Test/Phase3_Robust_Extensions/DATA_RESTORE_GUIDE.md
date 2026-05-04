# 데이터 복원 가이드 (Data Restoration Guide)

> **작성일**: 2026-05-05
> **대상**: `feature/omega-rmse-rp-prior` 브랜치를 다른 로컬 PC 에 받은 후 데이터 파일 복원하려는 경우.
> **이유**: `data/` + `outputs/` 폴더는 `.gitignore` 처리되어 git push 안 됨 → Google Drive 등 외부 파일 공유 통해 별도 전달 필요.

---

## 📦 압축 파일 정보

| 항목 | 값 |
|---|---|
| **파일명** | `phase3_data_outputs_2026-05-05.tar.gz` |
| **크기** | 776 MB (압축 후) |
| **원본 크기** | 2.2 GB (data/) + 8 MB (outputs/) = 약 2.2 GB |
| **포함 영역** | `시계열_Test/Phase3_Robust_Extensions/data/` + `시계열_Test/Phase3_Robust_Extensions/outputs/` |
| **공유 위치** | Google Drive (사용자 본인 계정) |

---

## 🎯 1. 압축 파일 다운로드

Google Drive 에서 `phase3_data_outputs_2026-05-05.tar.gz` 다운로드.

권장 다운로드 위치:
```
시계열_Test/Phase3_Robust_Extensions/phase3_data_outputs_2026-05-05.tar.gz
```

→ 본 가이드 파일 (`DATA_RESTORE_GUIDE.md`) 가 있는 동일 폴더에 다운로드.

---

## 🚀 2. 압축 해제 — **정확한 위치 + 명령어**

### 압축 해제 결과 (사전 확인)

```
시계열_Test/Phase3_Robust_Extensions/
├── data/                        ← 압축 풀면 여기에 생성 (2.2 GB)
│   ├── ensemble_predictions_stockwise.csv      (316 MB)
│   ├── ensemble_predictions_crosssec.csv       (310 MB)
│   ├── fold_predictions_stockwise.csv          (184 MB)
│   ├── daily_panel.csv                          (941 MB)
│   ├── daily_panel_extended.csv                 (155 MB)
│   ├── universe_full_history.csv                ( 28 KB)
│   ├── market_data.csv, vix_daily.csv, risk_free.csv
│   ├── scenario_weights_03_v2.pkl              (9.9 MB)
│   ├── bl_weights_v2_sanity_check.pkl          (6.0 MB)
│   ├── ticker_sector_mapping.csv               ( 12 KB)
│   ├── prices_close_universe.pkl, shares_outstanding.pkl
│   └── ... (총 약 25 파일)
└── outputs/                     ← 압축 풀면 여기에 생성 (8 MB)
    ├── 02a_v2_stockwise/
    ├── 03_bl_backtest/
    ├── 05a_v2_lstm_diag/        ← B2/B3/B4 학술 심화 결과 포함
    ├── 05a_v2_weighting/
    ├── 05b_v2_eval_crosssec/
    ├── 06_pct_sensitivity/
    └── ... (총 약 13 폴더)
```

### Step 1: 작업 디렉토리 이동

```bash
# Windows (Git Bash) 또는 Mac/Linux (bash)
cd "<repo path>/finance_project/시계열_Test/Phase3_Robust_Extensions"
```

```powershell
# Windows PowerShell
cd "<repo path>\finance_project\시계열_Test\Phase3_Robust_Extensions"
```

→ 이 폴더 안에 `phase3_data_outputs_2026-05-05.tar.gz` 가 있어야 함.

### Step 2: (안전) 기존 폴더 백업

```bash
# data/ 또는 outputs/ 폴더가 이미 있으면 백업
[ -d data ] && mv data data_backup_$(date +%Y%m%d) && echo "data 백업 완료" || echo "data 폴더 없음 (정상)"
[ -d outputs ] && mv outputs outputs_backup_$(date +%Y%m%d) && echo "outputs 백업 완료" || echo "outputs 폴더 없음 (정상)"
```

```powershell
# PowerShell 버전
$today = Get-Date -Format "yyyyMMdd"
if (Test-Path "data") { Rename-Item "data" "data_backup_$today"; Write-Host "data 백업 완료" }
if (Test-Path "outputs") { Rename-Item "outputs" "outputs_backup_$today"; Write-Host "outputs 백업 완료" }
```

### Step 3: 압축 해제 (Git Bash 또는 PowerShell)

```bash
# Git Bash (Windows) 또는 Mac/Linux
tar -xzvf phase3_data_outputs_2026-05-05.tar.gz
```

```powershell
# Windows PowerShell (tar 내장됨, Win10+ 표준)
tar -xzvf phase3_data_outputs_2026-05-05.tar.gz
```

→ 약 2-5분 소요.

---

## ✅ 3. 검증 (압축 해제 정상 여부)

### 검증 1: 폴더 + 핵심 파일 존재 확인

```bash
# Git Bash
echo "=== data/ 폴더 검증 ===" && ls -lh data/ | head -10
echo "=== outputs/ 폴더 검증 ===" && ls -lh outputs/ | head -10
```

```powershell
# PowerShell
Write-Host "=== data/ 폴더 검증 ==="; Get-ChildItem data | Select-Object -First 10 Name, Length
Write-Host "=== outputs/ 폴더 검증 ==="; Get-ChildItem outputs | Select-Object -First 10
```

### 검증 2: 핵심 파일 5개 존재 확인 (필수)

```bash
for f in \
  "data/ensemble_predictions_stockwise.csv" \
  "data/ensemble_predictions_crosssec.csv" \
  "data/daily_panel.csv" \
  "data/scenario_weights_03_v2.pkl" \
  "data/ticker_sector_mapping.csv"
do
  if [ -f "$f" ]; then
    size=$(du -h "$f" | cut -f1)
    echo "  [OK]  $f ($size)"
  else
    echo "  [FAIL] $f (파일 없음)"
  fi
done
```

#### 예상 출력 (정상)
```
[OK]  data/ensemble_predictions_stockwise.csv (316M)
[OK]  data/ensemble_predictions_crosssec.csv (310M)
[OK]  data/daily_panel.csv (941M)
[OK]  data/scenario_weights_03_v2.pkl (9.9M)
[OK]  data/ticker_sector_mapping.csv (12K)
```

### 검증 3: 데이터 파일 수 합계 (선택)

```bash
echo "data/ 의 csv 파일 수: $(ls data/*.csv 2>/dev/null | wc -l)"
echo "data/ 의 pkl 파일 수: $(ls data/*.pkl 2>/dev/null | wc -l)"
echo "outputs/ 의 하위 폴더 수: $(ls -d outputs/*/ 2>/dev/null | wc -l)"
```

#### 예상 출력 (정상)
```
data/ 의 csv 파일 수: 11
data/ 의 pkl 파일 수: 14
outputs/ 의 하위 폴더 수: 13
```

---

## 🔧 4. 가상환경 설정 (필요시)

`.venv` 는 압축에 포함되지 않았으므로 다른 PC 에서 별도 재생성 필요.

### 옵션 A: `uv` (권장, 가장 빠름)

```bash
cd "<repo path>/finance_project"
uv sync
```

→ `pyproject.toml` 의 dependencies 자동 설치 (60+ 패키지).

### 옵션 B: `pip` (uv 미설치 시)

```bash
cd "<repo path>/finance_project"
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -e .
```

→ 약 5-10분 소요.

---

## 📋 5. 첫 작업 시 권장 검증 (선택)

압축 해제 + 가상환경 설정 후 데이터 정상 로드 확인:

```bash
cd "<repo path>/finance_project/시계열_Test/Phase3_Robust_Extensions"
python -c "
import pandas as pd
from pathlib import Path

DATA = Path('data')
ens = pd.read_csv(DATA / 'ensemble_predictions_stockwise.csv', parse_dates=['date'], nrows=10)
print(f'02a 결과 (sample): {ens.shape}')
print(f'  date range: {ens[\"date\"].min()} ~ {ens[\"date\"].max()}')
print(f'  columns: {list(ens.columns)}')
print('OK')
"
```

#### 예상 출력 (정상)
```
02a 결과 (sample): (10, 9)
  date range: 2007-04-23 00:00:00 ~ 2007-04-23 00:00:00
  columns: ['date', 'ticker', 'fold', 'y_true', 'y_pred_lstm', 'y_pred_har', 'w_v4', 'w_har', 'y_pred_ensemble']
OK
```

---

## 🧹 6. 정리 (선택)

압축 파일 + 백업 폴더 삭제 (디스크 공간 회복):

```bash
# 압축 파일 삭제
rm phase3_data_outputs_2026-05-05.tar.gz

# 백업 폴더 삭제 (며칠 사용 후 문제 없으면)
rm -rf data_backup_* outputs_backup_*
```

---

## ⚠️ 트러블슈팅

### 문제 1: `tar: command not found`

```bash
# Windows: Git Bash 사용 (Git for Windows 설치 시 자동 포함)
# 또는 PowerShell 의 내장 tar 사용 (Win10 1803+)
# 또는 7-Zip 사용:
"C:/Program Files/7-Zip/7z.exe" x phase3_data_outputs_2026-05-05.tar.gz
"C:/Program Files/7-Zip/7z.exe" x phase3_data_outputs_2026-05-05.tar
```

### 문제 2: 압축 해제 도중 디스크 공간 부족

- `data/` + `outputs/` = 약 2.2 GB 필요
- 디스크 여유 공간 3 GB 이상 확보 후 재시도

### 문제 3: 파일이 이미 있음 (덮어쓰기 묻기)

`tar -xzvf` 는 기본적으로 덮어쓰지만, 일부 환경에서 prompt 발생:
```bash
tar -xzvf phase3_data_outputs_2026-05-05.tar.gz --overwrite
```

### 문제 4: 한글 파일명 깨짐

```bash
# UTF-8 강제 (Windows)
$env:PYTHONIOENCODING = "utf-8"
tar --options="hdrcharset=UTF-8" -xzvf phase3_data_outputs_2026-05-05.tar.gz
```

---

## 📝 7. 압축 파일 출처 + 재생성 방법 (참고)

원본 PC 에서 다음 명령어로 압축됨:

```bash
cd "<repo path>/finance_project/시계열_Test/Phase3_Robust_Extensions"
tar -czvf phase3_data_outputs_2026-05-05.tar.gz data/ outputs/
```

→ 만약 데이터가 다시 갱신되면 동일 명령으로 재압축 + Google Drive 재업로드 + 다른 PC 에서 본 가이드 따라 재복원.

---

## ✅ 완료 체크리스트

- [ ] Google Drive 에서 `phase3_data_outputs_2026-05-05.tar.gz` 다운로드 완료
- [ ] `시계열_Test/Phase3_Robust_Extensions/` 폴더에 압축 파일 위치
- [ ] 기존 `data/` / `outputs/` 폴더 백업 (있다면)
- [ ] `tar -xzvf phase3_data_outputs_2026-05-05.tar.gz` 실행 완료
- [ ] 검증 2 (핵심 파일 5개) 모두 OK
- [ ] (선택) `uv sync` 또는 `pip install -e .` 가상환경 설정 완료
- [ ] (선택) 검증 5 (Python 데이터 로드) OK
- [ ] (선택) 압축 파일 + 백업 폴더 삭제

---

## 📂 본 가이드 파일 위치

```
시계열_Test/Phase3_Robust_Extensions/DATA_RESTORE_GUIDE.md
```

본 파일은 git 추적되어 다른 PC 에서도 자동으로 받을 수 있습니다.

---

## 📞 문의

- 압축 파일 출처: 사용자 본인 (Google Drive 개인 계정)
- 가이드 작성: Claude 4.7 (1M context)
- 마지막 갱신: 2026-05-05
