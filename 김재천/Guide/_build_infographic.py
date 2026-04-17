"""
포스터형 Infographic + 핵심 발견 카드 5종 생성 스크립트.
Final Project 프로젝트 한눈 요약용 이미지.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
import platform
from pathlib import Path

# 한글 폰트
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    import koreanize_matplotlib  # noqa
plt.rcParams['axes.unicode_minus'] = False

IMG_DIR = Path(__file__).parent / 'images'
IMG_DIR.mkdir(exist_ok=True)

# 컬러 팔레트
COLOR_PRIMARY = '#1976D2'      # 파란색 (메인)
COLOR_SUCCESS = '#4CAF50'      # 초록 (성공)
COLOR_WARNING = '#FF9800'      # 주황 (경고)
COLOR_DANGER = '#F44336'       # 빨강 (위기)
COLOR_NEUTRAL = '#607D8B'      # 회색 (중립)
COLOR_GOLD = '#FFB300'         # 금 (강조)
BG_LIGHT = '#F5F5F5'
TEXT_DARK = '#212121'
TEXT_MEDIUM = '#616161'


# ============================================================
# 1. 포스터형 Infographic (A3 스타일, 1920×2400)
# ============================================================

def create_infographic():
    """Final Project 프로젝트 한 장 포스터."""
    fig = plt.figure(figsize=(16, 22), facecolor='white')
    gs = GridSpec(14, 6, figure=fig, hspace=0.5, wspace=0.3,
                  left=0.04, right=0.96, top=0.97, bottom=0.02)

    # --- Header (1행, 전체 너비) ---
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                         boxstyle="round,pad=0.02",
                                         facecolor=COLOR_PRIMARY, edgecolor='none',
                                         transform=ax_header.transAxes))
    ax_header.text(0.5, 0.7, 'Final Project', fontsize=36, fontweight='bold',
                   ha='center', va='center', color='white',
                   transform=ax_header.transAxes)
    ax_header.text(0.5, 0.3, '대안데이터 기반 포트폴리오 시뮬레이터',
                   fontsize=20, ha='center', va='center', color='white',
                   transform=ax_header.transAxes)

    # --- 핵심 KPI 4개 (2행) ---
    kpi_data = [
        ('Sharpe\nRatio', '1.064', COLOR_SUCCESS, '+29% vs EW'),
        ('최대\n낙폭', '-15.53%', COLOR_WARNING, 'SPY의 절반'),
        ('8년\n누적수익', '+151%', COLOR_PRIMARY, '연 +12.2%'),
        ('연율\n변동성', '11.41%', COLOR_NEUTRAL, 'SPY 65%'),
    ]
    for i, (title, value, color, subtitle) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[1, i*1 + (i//2)*1 : i*1 + (i//2)*1 + 2] if False else gs[1, i : i+1] if i < 4 else gs[1, 3:])
        ax.axis('off')
        # border
        ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.9,
                                      boxstyle="round,pad=0.02",
                                      facecolor='white', edgecolor=color, linewidth=3,
                                      transform=ax.transAxes))
        ax.text(0.5, 0.8, title, fontsize=12, ha='center', va='center',
                color=TEXT_MEDIUM, transform=ax.transAxes)
        ax.text(0.5, 0.48, value, fontsize=26, fontweight='bold',
                ha='center', va='center', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.18, subtitle, fontsize=10,
                ha='center', va='center', color=TEXT_MEDIUM, transform=ax.transAxes)

    # --- 프로젝트 개요 (3-4행) ---
    ax_intro = fig.add_subplot(gs[2:4, :])
    ax_intro.axis('off')
    ax_intro.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                        boxstyle="round,pad=0.02",
                                        facecolor=BG_LIGHT, edgecolor='none',
                                        transform=ax_intro.transAxes))
    ax_intro.text(0.03, 0.85, '[ 한 문장 요약 ]', fontsize=16, fontweight='bold',
                  color=COLOR_PRIMARY, transform=ax_intro.transAxes)
    ax_intro.text(0.03, 0.58,
                  '"대안데이터(VIX · HY 스프레드 등)로 매일 경보를 발동해\n'
                  ' 주식 비중을 자동 조절하는 포트폴리오 전략"',
                  fontsize=15, ha='left', va='center', color=TEXT_DARK,
                  transform=ax_intro.transAxes, style='italic')
    ax_intro.text(0.03, 0.25,
                  '- 2016~2025 (10년) · 30자산 · Walk-Forward 검증\n'
                  '- Mean-Variance 최적화 + 일별 경보 대응\n'
                  '- Bootstrap + Bonferroni + FDR 통계 검정',
                  fontsize=11, ha='left', va='center', color=TEXT_MEDIUM,
                  transform=ax_intro.transAxes)

    # --- 경보 시스템 규칙 (5-6행) ---
    ax_alert = fig.add_subplot(gs[4:6, :3])
    ax_alert.axis('off')
    ax_alert.text(0.03, 0.95, '[ 경보 시스템 (Config B) ]', fontsize=15, fontweight='bold',
                  color=COLOR_PRIMARY, transform=ax_alert.transAxes)
    alert_rules = [
        ('L0 정상', 'VIX < 20', '비중 유지', COLOR_SUCCESS),
        ('L1 주의', '20 ≤ VIX < 28', '주식 15% 감축', '#FFC107'),
        ('L2 경계', '28 ≤ VIX < 35', '주식 35% 감축', COLOR_WARNING),
        ('L3 위기', 'VIX ≥ 35', '주식 60% 감축', COLOR_DANGER),
    ]
    for i, (level, cond, action, color) in enumerate(alert_rules):
        y = 0.78 - i * 0.18
        ax_alert.add_patch(Rectangle((0.03, y - 0.03), 0.18, 0.12,
                                       facecolor=color, edgecolor='none',
                                       transform=ax_alert.transAxes))
        ax_alert.text(0.12, y + 0.03, level, fontsize=12, fontweight='bold',
                       ha='center', va='center', color='white',
                       transform=ax_alert.transAxes)
        ax_alert.text(0.24, y + 0.03, cond, fontsize=11,
                       va='center', color=TEXT_DARK, transform=ax_alert.transAxes)
        ax_alert.text(0.58, y + 0.03, '→', fontsize=14, va='center',
                       color=TEXT_MEDIUM, transform=ax_alert.transAxes)
        ax_alert.text(0.65, y + 0.03, action, fontsize=11,
                       va='center', color=TEXT_DARK, fontweight='bold',
                       transform=ax_alert.transAxes)

    # --- 최우수 전략 강조 (5-6행 우측) ---
    ax_best = fig.add_subplot(gs[4:6, 3:])
    ax_best.axis('off')
    ax_best.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                       boxstyle="round,pad=0.02",
                                       facecolor=COLOR_GOLD, edgecolor='none',
                                       alpha=0.2,
                                       transform=ax_best.transAxes))
    ax_best.text(0.5, 0.88, '[ 최우수 전략 ]', fontsize=14, fontweight='bold',
                  ha='center', color=COLOR_PRIMARY, transform=ax_best.transAxes)
    ax_best.text(0.5, 0.70, 'M1_보수형_ALERT_B', fontsize=18, fontweight='bold',
                  ha='center', color=TEXT_DARK, transform=ax_best.transAxes)
    details = [
        ('모드', 'M1 (경로 1만)'),
        ('성향', '보수형 (γ=8)'),
        ('Config', 'B (VIX + Contango)'),
        ('Sharpe', '1.064'),
        ('MDD', '-15.53%'),
    ]
    for i, (k, v) in enumerate(details):
        y = 0.54 - i * 0.09
        ax_best.text(0.15, y, k, fontsize=10, color=TEXT_MEDIUM,
                      transform=ax_best.transAxes)
        ax_best.text(0.65, y, v, fontsize=11, fontweight='bold', color=TEXT_DARK,
                      transform=ax_best.transAxes)

    # --- 11단계 파이프라인 (7-8행) ---
    ax_pipeline = fig.add_subplot(gs[6:8, :])
    ax_pipeline.axis('off')
    ax_pipeline.text(0.5, 0.95, '[ 11단계 파이프라인 ]', fontsize=15, fontweight='bold',
                     ha='center', color=COLOR_PRIMARY, transform=ax_pipeline.transAxes)
    steps = [
        ('1\n데이터', COLOR_PRIMARY),
        ('2\n전처리', COLOR_PRIMARY),
        ('3\n최적화', '#7E57C2'),
        ('4\nWF', '#7E57C2'),
        ('5\n리스크', '#7E57C2'),
        ('6\nHMM', COLOR_WARNING),
        ('7\nAblation', COLOR_WARNING),
        ('8\nΣ 레짐', COLOR_SUCCESS),
        ('9\n64 시뮬', COLOR_SUCCESS),
        ('10\n통계', COLOR_SUCCESS),
        ('11\n시각화', '#EC407A'),
    ]
    n = len(steps)
    box_width = 0.85 / n
    for i, (label, color) in enumerate(steps):
        x = 0.05 + i * (0.9 / n)
        ax_pipeline.add_patch(FancyBboxPatch((x, 0.3), box_width * 0.9, 0.4,
                                               boxstyle="round,pad=0.01",
                                               facecolor=color, edgecolor='none',
                                               transform=ax_pipeline.transAxes))
        ax_pipeline.text(x + box_width * 0.45, 0.5, label, fontsize=9,
                          fontweight='bold', ha='center', va='center',
                          color='white', transform=ax_pipeline.transAxes)
        if i < n - 1:
            ax_pipeline.annotate('', xy=(x + box_width + 0.003, 0.5),
                                  xytext=(x + box_width * 0.9, 0.5),
                                  xycoords='axes fraction',
                                  arrowprops=dict(arrowstyle='->', color=TEXT_MEDIUM))
    ax_pipeline.text(0.5, 0.1,
                      '데이터 → 피처 → 최적화 → 백테스트 → 리스크 → 레짐·경보 → Ablation → 통계 → 시각화',
                      fontsize=9, ha='center', color=TEXT_MEDIUM,
                      transform=ax_pipeline.transAxes, style='italic')

    # --- 5대 핵심 발견 (9-11행) ---
    ax_findings = fig.add_subplot(gs[8:11, :])
    ax_findings.axis('off')
    ax_findings.text(0.02, 0.95, '[ 5대 핵심 발견 ]', fontsize=16, fontweight='bold',
                      color=COLOR_PRIMARY, transform=ax_findings.transAxes)
    findings = [
        ('1', '경보 시스템의 실전 가치 증명',
         'L0 주식 40% → L3 주식 16%의 체계적 감축 패턴 확인', COLOR_SUCCESS),
        ('2', '복잡한 경로 2 (Σ 전환)는 실증 무효',
         '두 차례 재설계 후에도 M3 < M1, 역효과 발견', COLOR_DANGER),
        ('3', '단순성의 가치 (Occam\'s Razor)',
         'VIX + Contango 2변수(Config B)가 7지표 복합보다 우수', COLOR_PRIMARY),
        ('4', '앵커 - 반응 이원 구조',
         '채권·금 = 안정 앵커, 주식 = 경보 반응 버퍼', '#00897B'),
        ('5', '위기 시 약 2~3배 방어력',
         'COVID SPY -34% vs 보수형 -7% (4배 방어)', COLOR_GOLD),
    ]
    for i, (num, title, desc, color) in enumerate(findings):
        y = 0.78 - i * 0.16
        # 동그라미 번호
        ax_findings.add_patch(Circle((0.05, y + 0.04), 0.025,
                                       facecolor=color, edgecolor='none',
                                       transform=ax_findings.transAxes))
        ax_findings.text(0.05, y + 0.04, num, fontsize=14, fontweight='bold',
                          ha='center', va='center', color='white',
                          transform=ax_findings.transAxes)
        ax_findings.text(0.10, y + 0.06, title, fontsize=12, fontweight='bold',
                          color=TEXT_DARK, transform=ax_findings.transAxes)
        ax_findings.text(0.10, y + 0.02, desc, fontsize=10, color=TEXT_MEDIUM,
                          transform=ax_findings.transAxes)

    # --- 벤치마크 비교 (12-13행) ---
    ax_bench = fig.add_subplot(gs[11:13, :3])
    ax_bench.text(0.5, 1.05, '[ 벤치마크 비교 (Sharpe) ]', fontsize=13, fontweight='bold',
                   ha='center', color=COLOR_PRIMARY, transform=ax_bench.transAxes)
    strategies = ['우리\n전략', 'EW\n1/30', 'SPY\n단순', '60/40\n전통']
    sharpes = [1.064, 0.82, 0.76, 0.80]
    colors = [COLOR_SUCCESS, COLOR_NEUTRAL, COLOR_NEUTRAL, COLOR_NEUTRAL]
    bars = ax_bench.bar(strategies, sharpes, color=colors, alpha=0.85,
                         edgecolor='white', linewidth=2)
    ax_bench.axhline(1.0, color=COLOR_WARNING, linestyle='--', alpha=0.5,
                      label='탁월 기준 1.0')
    for bar, s in zip(bars, sharpes):
        ax_bench.text(bar.get_x() + bar.get_width()/2, s + 0.02, f'{s:.3f}',
                       ha='center', fontsize=11, fontweight='bold')
    ax_bench.set_ylim(0, 1.2)
    ax_bench.set_ylabel('Sharpe Ratio', fontsize=11)
    ax_bench.grid(axis='y', alpha=0.3)
    ax_bench.spines['top'].set_visible(False)
    ax_bench.spines['right'].set_visible(False)
    ax_bench.legend(loc='upper right', fontsize=9)

    # --- 위기 대응 성과 (12-13행 우측) ---
    ax_crisis = fig.add_subplot(gs[11:13, 3:])
    ax_crisis.text(0.5, 1.05, '[ 위기 방어력 (MDD) ]', fontsize=13, fontweight='bold',
                    ha='center', color=COLOR_PRIMARY, transform=ax_crisis.transAxes)
    crises = ['2018\nVolmageddon', '2020\nCOVID', '2022\n긴축', '2023\nSVB', '2024\n엔캐리']
    spy_dd = [-10, -34, -25, -8, -6]
    our_dd = [-3, -7, -14, -3, -1]
    x = np.arange(len(crises))
    width = 0.35
    ax_crisis.bar(x - width/2, spy_dd, width, label='SPY', color=COLOR_NEUTRAL, alpha=0.8)
    ax_crisis.bar(x + width/2, our_dd, width, label='우리', color=COLOR_SUCCESS, alpha=0.9)
    ax_crisis.set_xticks(x)
    ax_crisis.set_xticklabels(crises, fontsize=9)
    ax_crisis.set_ylabel('손실 %', fontsize=10)
    ax_crisis.legend(loc='lower right', fontsize=9)
    ax_crisis.grid(axis='y', alpha=0.3)
    ax_crisis.spines['top'].set_visible(False)
    ax_crisis.spines['right'].set_visible(False)

    # --- Footer (14행) ---
    ax_footer = fig.add_subplot(gs[13, :])
    ax_footer.axis('off')
    ax_footer.text(0.5, 0.6, '전체 자료: Guide/ (노트북 11개 + 해설 11개 + 보고서 3종 + Streamlit 앱)',
                    fontsize=11, ha='center', color=TEXT_MEDIUM,
                    transform=ax_footer.transAxes)
    ax_footer.text(0.5, 0.2, '2026-04-17 · Final Project · 김재천',
                    fontsize=9, ha='center', color=TEXT_MEDIUM,
                    transform=ax_footer.transAxes, style='italic')

    fig.savefig(IMG_DIR / 'infographic_poster.png', bbox_inches='tight',
                facecolor='white', dpi=150)
    plt.close(fig)
    print('저장: images/infographic_poster.png')


# ============================================================
# 2. 핵심 발견 카드 5종 (1200×800 each)
# ============================================================

def create_finding_card(n, title, subtitle, main_content, detail_lines, color):
    """개별 발견 카드 1장 (이모지 제거 버전)."""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 왼쪽 색상 바
    ax.add_patch(Rectangle((0, 0), 0.08, 1, facecolor=color, edgecolor='none'))

    # 번호 원형
    ax.add_patch(Circle((0.18, 0.82), 0.07, facecolor=color, edgecolor='none'))
    ax.text(0.18, 0.82, f'#{n}', fontsize=22, fontweight='bold',
            ha='center', va='center', color='white')

    # 제목
    ax.text(0.30, 0.87, title, fontsize=22, fontweight='bold', color=TEXT_DARK)

    # 부제
    ax.text(0.30, 0.77, subtitle, fontsize=13, color=TEXT_MEDIUM, style='italic')

    # 메인 콘텐츠 (큰 수치/문구)
    ax.add_patch(FancyBboxPatch((0.12, 0.44), 0.83, 0.18,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor='none', alpha=0.15))
    ax.text(0.535, 0.53, main_content, fontsize=20, fontweight='bold',
            ha='center', va='center', color=color)

    # 핵심 요지 라벨
    ax.text(0.12, 0.35, 'KEY POINTS', fontsize=11, fontweight='bold',
            color=color, family='sans-serif')

    # 상세 라인
    for i, line in enumerate(detail_lines):
        ax.text(0.14, 0.28 - i * 0.055, '- ' + line, fontsize=12, color=TEXT_DARK)

    # 푸터
    ax.text(0.95, 0.03, 'Final Project · 김재천', fontsize=9,
            ha='right', color=TEXT_MEDIUM, style='italic')

    fig.savefig(IMG_DIR / f'key_finding_{n:02d}.png', bbox_inches='tight',
                facecolor='white', dpi=150)
    plt.close(fig)
    print(f'저장: images/key_finding_{n:02d}.png')


def create_all_finding_cards():
    """5개 카드 생성 (이모지 없음)."""
    cards = [
        (1, '경보 시스템의 실전 가치',
         'VIX 기반 경보가 주식 비중 감축을 체계적으로 유도',
         'L0 40% → L3 16% (-60% 감축)',
         [
             'FDR 유의율 18.8%로 경보 효과 통계적 확인',
             '평균 ΔSharpe +0.167 (경로 1 단독)',
             'M1 평균 Sharpe 0.96 vs M0 0.79 (+21%)',
             'Step 11 시각화 5에서 직접 증명',
         ],
         COLOR_SUCCESS),
        (2, '경로 2의 실증적 무효',
         '복잡한 레짐 기반 공분산 전환이 실제로 효과 없음',
         'M2: -0.045 | M3 vs M1: -0.04',
         [
             '두 차례 재설계(OOS 1회 → 월 단위) 모두 실패',
             'Σ 차이가 MV 비중에 미약하게 반영',
             '재최적화 비용 > 이론적 이득',
             '경로 1과 정보 중복으로 시너지 없음',
         ],
         COLOR_DANGER),
        (3, '단순성의 가치',
         '2변수 Config B가 7지표 복합 Config C 능가',
         'Config B: Sharpe 1.064, MDD -15.5%',
         [
             'VIX + VIX Contango 단 2개로 충분',
             "Occam's Razor: 단순 모델 장기 우위",
             '다수 변수는 노이즈 증가, 과적합 위험',
             '실전 해석·유지보수 용이',
         ],
         COLOR_PRIMARY),
        (4, '앵커-반응 이원 구조',
         '포트폴리오는 두 역할의 자산으로 구성',
         '앵커: 채권·금 | 반응: 주식',
         [
             '채권 (TLT, AGG, SHY, TIP): 꾸준한 높은 비중',
             '금 (GLD): 부앵커 + 위기 방어',
             '주식 24개: 경보에 따라 변동 (반응 버퍼)',
             'Step 11 시각화 8에서 직관 증명',
         ],
         '#00897B'),
        (5, '위기 시 2~3배 방어력',
         '역사적 위기에서 손실을 현저히 축소',
         'COVID: SPY -34% vs 우리 -7% (5배 방어)',
         [
             '2018 Volmageddon: -10% → -3% (3.3배)',
             '2020 COVID: -34% → -7% (4.9배)',
             '2022 긴축: -25% → -14% (1.8배)',
             '2023 SVB: -8% → -3% (2.7배)',
         ],
         COLOR_GOLD),
    ]
    for args in cards:
        create_finding_card(*args)


# ============================================================
# 실행
# ============================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Infographic + Key Finding Cards 생성')
    print('=' * 60)
    create_infographic()
    create_all_finding_cards()
    print('\n모든 이미지 생성 완료!')
