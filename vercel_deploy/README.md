# vercel_deploy/ — 정적 랜딩 페이지

Adaptive VolControl Fund의 **단일 페이지 마케팅·소개 사이트**. Vercel에 정적 배포되어 채용 담당자·투자자가 클릭 한 번으로 프로젝트 결과를 확인할 수 있습니다.

> 분석 노트북은 [`../final_pt/`](../final_pt/), 인터랙티브 대시보드는 [`../streamlit_dashboard/`](../streamlit_dashboard/) 참고.

## 빠른 미리보기 (로컬)

```bash
cd vercel_deploy
python -m http.server 8000
# 브라우저에서 http://localhost:8000
```

`index.html`이 자동으로 `adaptive-volcontrol-fund.html`로 리다이렉트됩니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| [`index.html`](index.html) | 진입점 — `adaptive-volcontrol-fund.html`로 즉시 리다이렉트 (12줄, meta refresh + JS fallback) |
| [`adaptive-volcontrol-fund.html`](adaptive-volcontrol-fund.html) | 실제 콘텐츠 — 펀드 소개, 결과 지표, 방법론 시각화 (184줄 슬림 버전, 외부 CDN 폰트·리소스 참조) |
| [`vercel.json`](vercel.json) | Vercel 배포 설정 — `cleanUrls: true` (확장자 없는 URL 활성화) |
| `.gitignore` | 로컬 build 산출물 제외 |

## 디자인 토큰

- **폰트**: Pretendard (CDN), fallback → Noto Sans KR / Malgun Gothic / system-ui
- **컬러·타이포그래피**: `streamlit_dashboard/lib/page_helpers.py`의 디자인 시스템과 일치
- **스타일**: HTML 내 inline `<style>` 블록 (single-file, no build step)

## Vercel 배포

1. https://vercel.com 에서 GitHub repo 연결
2. **Import Project** → `sybeom528/finance_project` 선택
3. **Root Directory** 설정: `vercel_deploy`
4. **Framework Preset**: Other (no build needed — 정적 HTML)
5. **Build Command**: 비워둠
6. **Output Directory**: `.` (현재 디렉터리)
7. Deploy

배포 후 받은 URL을 [상위 README](../README.md)의 "Live Demo" 섹션에 추가하세요.

## 24MB HTML과의 관계

이전에 루트에 `Adaptive VolControl Fund.html` (24MB)가 있었습니다. 이는 폰트·이미지·CSS·JS를 전부 base64로 embedding한 **오프라인 단독 export 버전**으로, vercel_deploy의 슬림 웹 배포 버전과 동일 콘텐츠입니다. 중복이라 제거됐고 archive 브랜치에는 더 이상 없습니다.

## 업데이트 가이드

콘텐츠 수정 시:
1. `adaptive-volcontrol-fund.html` 안의 HTML/CSS 직접 편집
2. 로컬 미리보기로 확인
3. main 브랜치에 push → Vercel 자동 재배포 (보통 1분 이내)
