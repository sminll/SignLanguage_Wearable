"""Render exploratory temporal diagnostics from saved predictions only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

try:
    from . import evaluate_six_sign_knn as base
    from .report_six_sign_knn import draw_table, md_table, pct, save_figure
except ImportError:
    import evaluate_six_sign_knn as base
    from report_six_sign_knn import draw_table, md_table, pct, save_figure

NOTICE = '기존 결과를 본 뒤 설계한 탐색적 재분석 · 같은 9명 재사용 · 새 외부 참가자 성능 아님'
REP = {'global': '기존: 5초 전체 요약', 'five_bins': '변경: 5구간 순서 반영'}
SENSOR = {'imu_only': '양손 IMU', 'fsr_only': '양손 FSR', 'all_sensors': 'FSR+IMU',
          'nested_selection': '센서도 안쪽에서 선택'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--refresh-report', action='store_true',
                        help='Regenerate diagnostic figures/text from saved predictions; no fitting')
    args = parser.parse_args()
    root = args.results.resolve()
    share = root / '공유자료'
    if share.exists() and not args.refresh_report:
        raise FileExistsError('Choose fresh share output; do not overwrite')
    meta = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(root / 'all_predictions.csv')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csv')}
    lookup = {(r['protocol'], r['representation'], r['method']): r for r in meta['results']}
    for key, group in predictions.groupby(['protocol', 'representation', 'method']):
        group = group.copy()
        group['correct'] = group.label.eq(group.prediction)
        if group.file.duplicated().any() or len(group) != meta['paired_trials']:
            raise ValueError('Invalid once-per-trial diagnostic predictions')
        r = base.result_summary(group)
        prior = {k: v for k, v in lookup[key].items() if k not in ['protocol', 'representation', 'method']}
        if json.dumps(r, sort_keys=True) != json.dumps(prior, sort_keys=True):
            raise ValueError('Saved metrics do not match predictions')
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((v for v in ['Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic', 'AppleGothic'] if v in available), None)
    if not font:
        raise ValueError('A Korean font is required')
    plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    share.mkdir(exist_ok=args.refresh_report)
    within = {rep: lookup[('same_person_blocked5', rep, 'imu_only')] for rep in REP}
    between = {rep: lookup[('heldout_person_fixed', rep, 'imu_only')] for rep in REP}
    tuned = {rep: lookup[('heldout_person_nested', rep, 'nested_selection')] for rep in REP}
    n = meta['paired_trials']
    controlled = [[REP[rep], f"{within[rep]['correct']}/{n} ({pct(within[rep]['accuracy'])})",
                   f"{between[rep]['correct']}/{n} ({pct(between[rep]['accuracy'])})"] for rep in REP]
    draw_table('원인 확인: 같은 사람과 새로운 사람을 나눠 비교',
               ['동작 표현', '같은 사람의 다른 회차', '학습에서 빠진 사람'], controlled,
               '양손 IMU · K=5 · 맨해튼 거리 · 균등 투표 고정\n' + NOTICE +
               '\n같은 사람 평가는 같은 세션의 반복 회차가 학습에 포함될 수 있음',
               share / '01_같은사람_새로운사람_비교')
    sensor_rows = []
    for rep in REP:
        for sensor in ['imu_only', 'fsr_only', 'all_sensors', 'nested_selection']:
            r = lookup[('heldout_person_nested', rep, sensor)]
            sensor_rows.append([REP[rep], SENSOR[sensor], f"{r['correct']}/{n}", pct(r['accuracy'])])
    draw_table('참가자 제외 평가: 센서·KNN 설정 비교',
               ['동작 표현', '센서 입력', '정답/평가', '정확도'], sensor_rows,
               '평가 대상은 학습·설정 선택에서 제외; 센서별 K/거리/투표는 안쪽 평가에서 선택\n' + NOTICE,
               share / '02_구간분할_설정선택_비교')
    sign_rows = []
    for i, label in enumerate(base.LABELS):
        sign_rows.append([label, within['global']['per_sign'][i]['trials'],
                          pct(within['global']['per_sign'][i]['recall']),
                          pct(within['five_bins']['per_sign'][i]['recall']),
                          pct(between['global']['per_sign'][i]['recall']),
                          pct(between['five_bins']['per_sign'][i]['recall'])])
    draw_table('수어별 정답률 — 설정을 동일하게 고정한 IMU 비교',
               ['수어', '평가 회차', '같은 사람 기존', '같은 사람 5구간',
                '빠진 사람 기존', '빠진 사람 5구간'], sign_rows,
               '양손 IMU · K5/맨해튼/균등 · 동작 회차 전체를 학습/평가로 분리\n' + NOTICE,
               share / '03_수어별_비교')
    person_rows = []
    for i, row in enumerate(within['global']['people']):
        person_rows.append([row['person'], row['trials'], pct(row['accuracy']),
                            pct(within['five_bins']['people'][i]['accuracy']),
                            pct(between['global']['people'][i]['accuracy']),
                            pct(between['five_bins']['people'][i]['accuracy'])])
    draw_table('참가자별 정확도 — 설정을 동일하게 고정한 IMU 비교',
               ['참가자', '평가 회차', '같은 사람 기존', '같은 사람 5구간',
                '빠진 사람 기존', '빠진 사람 5구간'], person_rows,
               '같은 사람 평가는 새 세션·새 날짜 검증이 아님\n' + NOTICE,
               share / '04_참가자별_비교')
    for name, rows, headers in [
        ('수어별_비교', sign_rows, ['label', 'trials', 'within_global', 'within_five_bins', 'between_global', 'between_five_bins']),
        ('참가자별_비교', person_rows, ['person', 'trials', 'within_global', 'within_five_bins', 'between_global', 'between_five_bins']),
        ('설정선택_비교', sensor_rows, ['representation', 'sensors', 'correct_trials', 'accuracy'])]:
        pd.DataFrame(rows, columns=headers).to_csv(share / (name + '.csv'), index=False, encoding='utf-8-sig')
    cm = np.array(tuned['five_bins']['confusion_matrix'])
    rates = 100 * cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11.8, 10.3))
    pic = ax.imshow(rates, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(6), base.LABELS, fontsize=13)
    ax.set_yticks(range(6), base.LABELS, fontsize=13)
    ax.set_xlabel('예측한 수어', fontsize=15)
    ax.set_ylabel('실제 수행한 수어', fontsize=15)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm[i,j]}회\n({rates[i,j]:.1f}%)', ha='center', va='center', fontsize=12,
                    color='white' if rates[i,j] > 55 else '#123b60')
    fig.colorbar(pic, ax=ax, fraction=0.046, pad=0.04, label='실제 수어별 비율 (%)')
    fig.suptitle('5구간 특징 KNN — 탐색적 참가자 제외 평가', fontsize=20, fontweight='bold')
    ax.set_title(f"9명 · {n:,}회 · {tuned['five_bins']['correct']}회 정답 · {pct(tuned['five_bins']['accuracy'])}",
                 fontsize=14, pad=18)
    fig.text(0.5, 0.02, NOTICE + '\n센서 종류·KNN 설정은 각 학습 참가자 안에서 선택', ha='center', fontsize=10)
    fig.subplots_adjust(top=0.88, bottom=0.13)
    save_figure(fig, share / '05_5구간_혼동행렬')
    pd.DataFrame(cm, index=base.LABELS, columns=base.LABELS).to_csv(share / '5구간_혼동행렬.csv', encoding='utf-8-sig')
    paired_changes = []
    for method in base.METHODS:
        old = predictions[(predictions.protocol == 'heldout_person_nested') &
                          (predictions.representation == 'global') & (predictions.method == method)]
        new = predictions[(predictions.protocol == 'heldout_person_nested') &
                          (predictions.representation == 'five_bins') & (predictions.method == method)]
        joined = old[['file', 'label', 'prediction']].merge(new[['file', 'label', 'prediction']], on='file',
                                                         validate='one_to_one', suffixes=('_old', '_new'))
        assert joined.label_old.eq(joined.label_new).all()
        before = joined.label_old.eq(joined.prediction_old)
        after = joined.label_new.eq(joined.prediction_new)
        paired_changes.append(dict(method=method, corrected=int((~before & after).sum()),
                                   newly_wrong=int((before & ~after).sum()),
                                   both_correct=int((before & after).sum()), both_wrong=int((~before & ~after).sum())))
    pd.DataFrame(paired_changes).to_csv(share / '회차별_변화집계.csv', index=False, encoding='utf-8-sig')
    delta_fixed = 100 * (between['five_bins']['accuracy'] - between['global']['accuracy'])
    delta_tuned = 100 * (tuned['five_bins']['accuracy'] - tuned['global']['accuracy'])
    if within['global']['accuracy'] >= .8 and between['global']['accuracy'] < .6:
        interpretation = ('같은 사람의 반복 회차는 잘 구분되지만, 학습하지 않은 사람에서는 크게 낮아졌다. '
                          '참가자·착용·세션 조건에 대한 민감성을 의심할 근거이다. '
                          '다만 같은 사람 평가에는 같은 세션이 포함될 수 있어 착용 방향 등이 원인이라고 확정하지 않는다.')
    else:
        interpretation = ('같은 사람과 다른 사람 평가를 함께 보아야 한다. 같은 사람 평가도 낮다면 사람 차이만으로 '
                          '설명할 수 없으며 동작 구간·라벨·센서 신호와 특징 추출을 함께 검토해야 한다.')
    if delta_fixed > 0 and delta_tuned > 0:
        temporal_interpretation = ('구간별 통계를 넣은 결과 두 비교에서 정답 수가 늘었다. 움직임 흐름 정보가 '
                                   '도움이 될 가능성이 있지만 같은 자료를 재사용한 탐색 결과이므로 새 참가자로 확인해야 한다.')
    elif delta_fixed <= 0 and delta_tuned <= 0:
        temporal_interpretation = ('5구간 통계로 나누는 변경은 이번 참가자 제외 평가에서 개선되지 않았다. '
                                   '단순한 구간 분할만으로 문제를 해결했다고 볼 수 없다.')
    else:
        temporal_interpretation = ('구간 분할 효과가 설정 고정/선택 비교에서 일관되지 않았다. '
                                   '전체적으로 개선됐다고 단정하지 않으며 새 참가자 검증이 필요하다.')
    team = (
        '6종 수어 정확도가 낮은 원인을 확인하기 위해 추가 비교했습니다.\n'
        f"양손 IMU에 같은 KNN 설정을 썼을 때, 기존 방식은 같은 사람의 다른 회차에서 {pct(within['global']['accuracy'])}, "
        f"학습에서 빠진 사람에서는 {pct(between['global']['accuracy'])}였습니다.\n"
        f"5초를 5구간으로 나눠 순서를 반영한 방식은 각각 {pct(within['five_bins']['accuracy'])}, "
        f"{pct(between['five_bins']['accuracy'])}였습니다.\n"
        + interpretation + '\n' + temporal_interpretation + '\n'
        '이번 분석은 기존 결과를 보고 같은 9명 자료로 다시 실험한 탐색 결과입니다. '
        '같은 사람 정확도를 새 사람 정확도처럼 쓰지 않으며, 기존 모델은 자동 교체하지 않았습니다.\n'
    )
    report = f'''# 2026-11-15 6종 수어 — 같은 사람/동작 구간 분석

**{NOTICE}**

## 같은 KNN 설정으로 비교한 결과

{md_table(['동작 표현', '같은 사람의 다른 회차', '학습에서 빠진 사람'], controlled)}

양손 IMU, StandardScaler + KNN(K5, 맨해튼 거리, 균등 투표)를 고정했다.
따라서 위 2×2 비교는 사람 분리 방식과 특징 표현의 영향을 구분한다.
이 설정도 기존 전체 자료에서 선택됐던 것이므로 이번 비교는 탐색적 진단이다.
이 표의 참가자 제외 **고정 설정** 점수와 이전에 안내한 안쪽 **설정 선택** 점수(33.62%, 31.80%)는 평가 절차가 다르다.

{interpretation}

## 동작 흐름 반영 결과

각 손의 자기 기록 시작~끝을 상대 시간 5구간으로 나누고, 구간별 평균/표준편차(ddof=0)/최소/최대를 순서대로 연결했다.
IMU240특징, FSR240특징, 전체480특징이다. 약1초씩의 구간이지만 사람의 속도에 따른 정밀한 동작 정렬이나
양손 순간 동기화를 구현한 것은 아니다. 새로운 구간의 샘플은 모두 3개 이상이어야 하며 미달 시 분석을 중단한다.
원래 1,044회 모두 통과했으므로 구간 방식 때문에 새로 빠진 회차는 없다.

같은 설정의 참가자 제외 IMU 비교 변화: **{delta_fixed:+.2f}%p**.
센서까지 안쪽에서 고르는 참가자 제외 평가 변화: **{delta_tuned:+.2f}%p**.

{temporal_interpretation}

## 센서·설정을 안쪽에서 고른 추가 비교

{md_table(['동작 표현', '센서 입력', '정답/평가', '정확도'], sensor_rows)}

기존 방식의 결과는 이전 예측 CSV를 그대로 읽었다. 5구간은 같은 바깥 참가자 9fold, 나머지8명 안쪽8fold,
동일한36설정(K3/5/7 × 거리2 × 투표2 × 센서3)을 사용했다. 평가 사람은 학습/설정 선택에서 제외했다.
하지만 특징 설계 자체는 이전 결과를 보고 정했으므로 새로운 독립 검증 결과라고 부르지 않는다.
결과를 본 뒤 구간 수나 K 검색 범위를 추가로 바꾸지 않았다.

## 수어별 비교 — 고정 설정 IMU

{md_table(['수어', '평가 횟수', '같은 사람 기존', '같은 사람5구간', '빠진 사람 기존', '빠진 사람5구간'], sign_rows)}

## 참가자별 비교 — 고정 설정 IMU

{md_table(['참가자', '평가 횟수', '같은 사람 기존', '같은 사람5구간', '빠진 사람 기존', '빠진 사람5구간'], person_rows)}

## 데이터와 분리 방법

- 고정 커밋 `{meta['source_commit']}`. 11월13~15일 XIAO 원본1,060회, 양손1,044회,9명.
- 오른손 없는16회는 이전과 동일하게 평가에서 제외하며 원본을 삭제하지 않음. 낮은 FSR/정답률로 제외하지 않음.
- 원본1,060개를 Git archive로 읽어 해시 재검증. 기존 특징96열 재계산 후 이전 결과와 수치 일치 검증.
- 같은 사람: 사람·수어별 시간순 회차를5연속블록으로 나누고 한 블록씩 평가. 각 동작 회차 전체를 분리.
  학습/평가 회차UUID와 파일은 겹치지 않음. 같은 세션, 가까운 반복 회차가 학습에 있을 수 있음.
  **이 점수는 새 세션·새 날짜·새 사람에 대한 정확도가 아님.**
- 다른 사람: 그 사람의 모든 회차·날짜를 학습과 설정 선택에서 제외.
- 표준화는 각각의 학습 자료에서만 계산. 센서 행을 따로 학습/평가로 나누지 않음.
- 일부공유 session_id는 다른 사람의 같은 회차를 뜻하지 않음. Subject와 고유trial/file로 분리.

## 아직 알 수 없는 부분과 다음 단계

CSV만으로 착용 위치·압박·센서 방향·수어 정확성은 확정하지 못한다. 같은 사람/세션 차이도 얽혀 있다.
높은 같은 사람 점수만으로 새 사람이 잘 인식된다고 표시하지 않는다. 반복 회차는 독립적이지 않을 수 있어
통계적 유의성이나 회차 독립 가정의 신뢰구간을 주장하지 않는다.
신규 참가자 또는 같은 사람의 장비 재착용 후 별도 세션을 먼저 평가하면 원인을 더 좁힐 수 있다.
이번에 기존 모델을 덮어쓰거나 자동 적용하지 않았으며 새 최종 후보 파일도 만들지 않았다.

## 재현

```powershell
python -B tools/diagnose_six_sign_knn.py --previous data/knn_results/2026-11-15_six_sign_nested_loso --repo . --out data/knn_results/six_sign_diagnostic_rerun
python -B tools/report_six_sign_diagnostics.py --results data/knn_results/six_sign_diagnostic_rerun
python -B -m unittest discover -s tests -p test_six_sign_diagnostics.py -v
```

출력 폴더는 새 이름을 사용한다. `comparison_plan.json`은 예측 전에 고정한 비교 범위와 이전 자료 해시를 저장한다.
`within_person_splits.json`, `fixed_predictions.csv`, `temporal_nested_predictions.csv`, `temporal_inner_searches.json`,
`all_predictions.csv`, `comparison_summary.csv`, `verification.json`에 분리/예측/검증 근거를 보존했다.
그림과 수치는 실제 측정 예측만 사용한다. 자동 테스트의 합성 fixture는 실험 결과에서 제외한다.
'''
    (root / 'report.md').write_text(report, encoding='utf-8')
    (share / '결과_보고서.md').write_text(report, encoding='utf-8')
    (share / '팀원_공유문구.txt').write_text(team, encoding='utf-8')
    shutil.copy2(Path(__file__), root / Path(__file__).name)
    for p in root.glob('*.csv'):
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.name]:
            raise ValueError('Reporting changed input data')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in share.iterdir()
                if p.name != '파일_SHA256.json'}
    base.dump_json(share / '파일_SHA256.json', manifest)
    print(team)
    print('Verified diagnostics and plots saved: ' + str(share))


if __name__ == '__main__':
    main()
