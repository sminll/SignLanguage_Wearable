"""Make poster tables/plots from saved REAL nested-CV predictions. Never fit."""
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
    from .evaluate_six_sign_knn import LABELS, METHODS, result_summary
except ImportError:
    from evaluate_six_sign_knn import LABELS, METHODS, result_summary

NAMES = {'imu_only': '양손 IMU', 'fsr_only': '양손 FSR',
         'all_sensors': '양손 FSR+IMU', 'nested_selection': '센서·설정 자동 선택'}
PRIMARY = 'nested_selection'
NOTE = '참가자 단위 중첩 교차검증 · 학습/설정 선택에서 평가 참가자 제외 · 실제 측정 데이터'


def pct(v):
    return f'{100 * v:.2f}%'


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def save_figure(fig, base):
    for suffix in ['.png', '.svg']:
        fig.savefig(base.with_suffix(suffix), dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_table(title, headers, rows, footnote, base):
    fig, ax = plt.subplots(figsize=(13.2, max(4.0, 1.9 + len(rows) * 0.48)))
    ax.set_axis_off()
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.97)
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d4dde8')
        cell.set_facecolor('#123b60' if row == 0 else ('#edf4fa' if row % 2 else 'white'))
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
    fig.text(0.5, 0.025, footnote, ha='center', va='bottom', fontsize=10, color='#384b5a')
    fig.subplots_adjust(top=0.84, bottom=0.17)
    save_figure(fig, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    share = root / '공유자료'
    if share.exists():
        raise FileExistsError('Share folder already exists; do not overwrite')
    # Recompute every result from the recorded predictions before drawing.
    source = root / 'outer_predictions.csv'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    predictions = pd.read_csv(source)
    stored = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
    if set(predictions.method) != set(METHODS):
        raise ValueError('Incomplete method comparison')
    cohort = pd.read_csv(root / 'cohort_manifest.csv')
    audit = json.loads((root / 'inventory_summary.json').read_text(encoding='utf-8'))
    candidate = json.loads((root / 'candidate_manifest.json').read_text(encoding='utf-8'))
    results = {}
    for method in METHODS:
        group = predictions[predictions.method.eq(method)].copy()
        if len(group) != len(cohort) or group.file.duplicated().any() or set(group.file) != set(cohort.file):
            raise ValueError('Every cohort trial must appear exactly once')
        if not group.correct.astype(str).str.lower().eq(group.label.eq(group.prediction).astype(str).str.lower()).all():
            raise ValueError('Incorrect saved correctness flag')
        group['correct'] = group.label.eq(group.prediction)
        for row in group.itertuples():
            if row.person in row.train_people.split('|'):
                raise ValueError('Outer participant was in training')
        calculated = result_summary(group)
        if json.dumps(calculated, sort_keys=True) != json.dumps(stored[method], sort_keys=True):
            raise ValueError(f'Saved metrics differ from predictions: {method}')
        results[method] = calculated
    share.mkdir()
    available = {f.name for f in font_manager.fontManager.ttflist}
    korean_font = next((font for font in ['Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic', 'AppleGothic']
                        if font in available), None)
    if korean_font is None:
        raise ValueError('Install a Korean font for readable figures')
    plt.rcParams.update({'font.family': korean_font, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    primary = results[PRIMARY]
    n, correct = primary['trials'], primary['correct']
    people = len(primary['people'])
    overall = []
    comparison_rows = []
    for method in METHODS:
        r = results[method]
        overall.append(dict(method=method, name=NAMES[method], correct=r['correct'], trials=r['trials'],
                            accuracy_percent=100 * r['accuracy'],
                            equal_person_accuracy_percent=100 * r['participant_mean_accuracy'],
                            balanced_accuracy_percent=100 * r['balanced_accuracy'], macro_f1=r['macro_f1']))
        comparison_rows.append([NAMES[method], f"{r['correct']}/{r['trials']}", pct(r['accuracy']),
                                pct(r['participant_mean_accuracy']), f"{r['macro_f1']:.3f}"])
    pd.DataFrame(overall).to_csv(share / '센서별_정확도.csv', index=False, encoding='utf-8-sig')
    sign_rows = [[r['label'], r['trials'], r['correct'], r['trials'] - r['correct'], pct(r['recall'])]
                 for r in primary['per_sign']]
    person_rows = [[r['person'], r['trials'], r['correct'], r['trials'] - r['correct'], pct(r['accuracy'])]
                  for r in primary['people']]
    for filename, data, metric in [('수어별_정확도', primary['per_sign'], 'recall'),
                                   ('참가자별_정확도', primary['people'], 'accuracy')]:
        frame = pd.DataFrame(data)
        frame['accuracy_percent'] = frame[metric] * 100
        frame.to_csv(share / (filename + '.csv'), index=False, encoding='utf-8-sig')
    # Participant x sign counts make partial/missing trials explicit.
    cross = []
    rows = predictions[predictions.method.eq(PRIMARY)].copy()
    rows['correct'] = rows.label.eq(rows.prediction)
    for (who, label), group in rows.groupby(['person', 'label']):
        cross.append(dict(person=who, label=label, trials=len(group), correct=int(group.correct.sum()),
                          accuracy_percent=float(group.correct.mean() * 100)))
    pd.DataFrame(cross).to_csv(share / '참가자_수어별_정확도.csv', index=False, encoding='utf-8-sig')
    draw_table('수어 6종 KNN — 센서 입력 비교',
               ['센서 입력', '정답/평가 회차', '전체 정확도', '참가자 평균', 'Macro F1'],
               comparison_rows, NOTE + '\n각 센서의 K·거리·가중치는 안쪽 평가에서 선택; 마지막 행은 센서 종류도 선택',
               share / '01_센서별_정확도')
    foot = NOTE + '\n주 결과: 각 바깥 평가의 나머지 참가자 안에서 센서 종류와 KNN 설정 선택'
    draw_table(f'수어별 정답률 — 전체 {correct:,}/{n:,} ({pct(primary["accuracy"])})',
               ['실제 수어', '평가 회차', '정답', '오답', '정답률'], sign_rows, foot,
               share / '02_수어별_정확도')
    draw_table(f'참가자별 정확도 — {people}명 각각 본인 제외 후 평가',
               ['평가 참가자', '평가 회차', '정답', '오답', '정확도'], person_rows,
               foot + '\nsubject4 오른손 누락 16회 제외; 다른 참가자는 수집된 유효 회차 모두 평가',
               share / '03_참가자별_정확도')
    cm = np.array(primary['confusion_matrix'])
    rates = cm / cm.sum(axis=1, keepdims=True) * 100
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(share / '혼동행렬_횟수.csv', encoding='utf-8-sig')
    pd.DataFrame(rates, index=LABELS, columns=LABELS).to_csv(share / '혼동행렬_행백분율.csv', encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(11.8, 10.2))
    plot = ax.imshow(rates, vmin=0, vmax=100, cmap='Blues')
    ax.set_xticks(range(6), LABELS, fontsize=14)
    ax.set_yticks(range(6), LABELS, fontsize=14)
    ax.set_xlabel('예측한 수어', fontsize=16, labelpad=12)
    ax.set_ylabel('실제 수행한 수어', fontsize=16, labelpad=12)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm[i, j]}회\n({rates[i, j]:.1f}%)', ha='center', va='center', fontsize=13,
                    color='white' if rates[i, j] > 55 else '#102c45')
    fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, label='실제 수어별 비율 (%)')
    fig.suptitle('XIAO 수어 6종 — 참가자 제외 평가 혼동행렬', fontsize=20, fontweight='bold', y=0.98)
    ax.set_title(f'{people}명 · {n:,}회 평가 · {correct:,}회 정답 · 전체 정확도 {pct(primary["accuracy"])}',
                 fontsize=14, pad=18)
    fig.text(0.5, 0.02, NOTE + '\n센서 종류·KNN 설정은 각 바깥 평가의 학습 참가자 안에서 선택',
             ha='center', fontsize=10)
    fig.subplots_adjust(top=0.88, bottom=0.13)
    save_figure(fig, share / '04_혼동행렬')
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    values = [r['recall'] * 100 for r in primary['per_sign']]
    bars = ax.bar(LABELS, values, color='#2176ae', width=0.62)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('실제 수어별 정답률 (%)')
    for bar, row in zip(bars, primary['per_sign']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{pct(row['recall'])}\n{row['correct']}/{row['trials']}회", ha='center', fontsize=12)
    ax.set_title('XIAO 수어 6종 — 수어별 정답률', fontsize=20, pad=18, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.5, 0.02, NOTE, ha='center', fontsize=10)
    fig.subplots_adjust(bottom=0.15)
    save_figure(fig, share / '05_수어별_정확도_그래프')
    raw_n = audit['files']
    exclusions = pd.read_csv(root / 'excluded_trials.csv')
    cfg = candidate['config']
    sensor_usage = rows.groupby('sensors').person.nunique().to_dict()
    caption = (
        'XIAO 기반 양손 센서를 이용한 6종 수어 KNN 분류 평가\n\n'
        f'2026년 11월 13~15일 수집된 참가자 {people}명의 실제 측정 데이터를 사용하여 '
        '괜찮다·아니다·무엇·마시다·화장실·싫다의 6개 수어를 분류하였다. '
        '약 5초 동안 측정한 하나의 동작 회차를 하나의 평가 단위로 정의하고, 양손의 센서값을 '
        '각각 평균·표준편차·최솟값·최댓값으로 요약하였다. '
        f'총 {raw_n:,}회 중 오른손 기록이 없는 {len(exclusions)}회를 양손 평가에서 제외하여 '
        f'{n:,}회를 분석하였다. 제외한 원본 데이터는 삭제하지 않았다.\n\n'
        f'참가자 한 명의 모든 데이터를 평가용으로 분리하고 나머지 {people-1}명만으로 학습하는 '
        f'과정을 {people}명에게 반복하였다. 센서 입력(IMU·FSR·전체)과 KNN 설정은 각 평가의 '
        '학습 참가자 안에서만 추가로 사람을 나누어 선택하였고, 표준화도 해당 학습 데이터에서만 '
        '계산하였다. 따라서 평가 참가자는 학습과 설정 선택 모두에서 제외되었다.\n\n'
        f'실제 평가 결과 {n:,}회 중 {correct:,}회를 올바르게 분류하여 전체 정확도 '
        f'{pct(primary["accuracy"])}를 얻었다. 참가자별 정확도의 동일 가중 평균은 '
        f'{pct(primary["participant_mean_accuracy"])}이며 Macro F1은 {primary["macro_f1"]:.3f}이다. '
        '수어별 정답률은 ' + ', '.join(f"{r['label']} {pct(r['recall'])}"
                                          for r in primary['per_sign']) + '이다.\n\n'
        '혼동행렬은 세로축이 실제 수어, 가로축이 예측 수어이다. 각 칸은 예측 횟수와 해당 실제 '
        '수어 안에서의 비율을 표시하며, 대각선은 정답이다. 전체 정확도는 참가자별 백분율의 단순 '
        '평균이 아니라 전체 평가 회차의 정답 횟수를 합산하여 계산하였다.\n\n'
        '이 결과는 수집된 9명의 데이터를 이용한 참가자 단위 중첩 교차검증 결과이다. '
        '앞으로 새로 참가할 사람에 대한 전향적 외부 평가나 실시간 연속 수어·문장 인식 성능을 '
        '의미하지 않는다. 기존 세 수어 실험과는 수어 종류·참가자·평가 절차가 달라 정확도를 '
        '직접적인 향상 수치로 비교하지 않는다.\n'
    )
    team = (
        f'6개 수어(괜찮다/아니다/무엇/마시다/화장실/싫다) KNN 평가 결과 공유드립니다!\n'
        f'9명의 데이터에서 본인 데이터를 학습과 설정 선택에서 모두 빼고 평가하는 방식을 반복했습니다. '
        f'양손 기록이 있는 {n:,}회 중 {correct:,}회 정답으로 전체 정확도는 {pct(primary["accuracy"])}입니다.\n'
        '수어별 정답률: ' + ' / '.join(f"{r['label']} {pct(r['recall'])}" for r in primary['per_sign']) + '\n'
        '정확도 표와 혼동행렬을 같이 보내드립니다. 포스터에는 “참가자 단위 중첩 교차검증” 결과로 '
        '표기해 주세요. 오른손이 누락된 16회는 이번 양손 비교에서만 제외하고 원본은 보관했습니다. '
        '새 참가자의 데이터로 검증한 최종 모델 정확도는 아직 미측정입니다.\n'
    )
    (share / '포스터_설명.txt').write_text(caption, encoding='utf-8')
    (share / '팀원_공유문구.txt').write_text(team, encoding='utf-8')
    report = f'''# 2026-11-15 XIAO 수어 6종 KNN 평가

주 결과: **{correct:,}/{n:,}회 정답, 전체 정확도 {pct(primary['accuracy'])}**.
참가자 평균 {pct(primary['participant_mean_accuracy'])}, balanced accuracy {pct(primary['balanced_accuracy'])}, Macro F1 {primary['macro_f1']:.3f}.

## 데이터와 분리

- 고정 Git 커밋: `{audit['commit']}` (`training-data`, 결과 도중 새 업로드를 섞지 않음).
- 11월 13~15일, 9명(subject1·2·3·4·5·6·8·9·10), 원본 {raw_n:,}회; 약 5초 동작 하나가 한 샘플.
- subject4 오른손 누락 16회(마시다9·무엇6·화장실1) 제외, 양손 공통 비교 {n:,}회. 원본 보존.
- 낮은 FSR/정답률을 이유로 사람·회차를 제외하지 않음. subject8 마시다24회 모두 포함.
- 표기가 다른 동일 수어는 사용자 확인에 따라 통일. 원본 라벨은 수정하지 않음.
- 같은 사람의 모든 날짜를 함께 분리. Subject 번호를 정규화하고 train/test 사람·회차 UUID·파일 분리 검증.
- 일부 session_id가 서로 다른 Subject에 공유되므로 session_id만으로 참가자를 묶지 않음.

## 평가 방법

바깥쪽에서 한 명을 빼고 8명으로 학습하여 9번 평가했다. 안쪽에서는 그 8명만으로 다시 사람을 나누어
센서 입력 3종 × K(3/5/7) × 거리(맨해튼/유클리드) × 투표(균등/거리 가중), 총36설정을 비교했다.
안쪽 참가자별 balanced accuracy의 동일 가중 평균으로 선택하며 동률은 사전 고정된 config_id 순서로 결정했다.
StandardScaler는 매번 학습 집합에서만 fit했다. 주 결과는 센서 종류까지 안쪽에서 고르는 절차(`nested_selection`)의 바깥 예측이다.
센서별 행은 센서 종류를 고정하고 K·거리·투표만 안쪽에서 선택한 별도 비교이다.
바깥 결과를 보고 센서 한 가지를 고른 뒤 그 점수를 독립 검증 점수로 바꾸어 부르지 않는다.

각 손의 FSR6/IMU6축에서 평균·표준편차(ddof=0)·최솟값·최댓값을 추출했다.
IMU48특징 / FSR48특징 / 전체96특징. 공통 시계 미검증 상태이므로 손별 통계를 합쳤으며 부팅 시각으로 행을 강제 정렬하지 않았다.

## 센서 비교

{md_table(['입력', '정답/평가', '전체 정확도', '참가자 평균', 'Macro F1'], comparison_rows)}

주 결과에서 선택된 센서 종류의 바깥 fold 수: `{json.dumps(sensor_usage, ensure_ascii=False)}`.
전체 회차의 정확도와 참가자별 동일 가중 평균은 평가 수가 달라 서로 다를 수 있다.

## 수어별 결과 — 주 결과

{md_table(['실제 수어', '평가 회차', '정답', '오답', '정답률'], sign_rows)}

## 참가자별 결과 — 주 결과

{md_table(['참가자', '평가 회차', '정답', '오답', '정확도'], person_rows)}

## 해석과 한계

실제 CSV의 라벨을 정답으로 삼은 개별 수어 분류 결과이다. 동작 영상·착용 위치·압박·센서 방향의 적절성은
CSV만으로 확인하지 못했다. calibration_verified는 모두 false이며, 보정 상태가 검증되지 않았다는 의미이다.
FSR가 0인 기록은 일괄 제거하지 않았다. 참가자·장치·날짜 조건이 완전히 독립인 외부 실험은 아니다.
반복 동작은 서로 상관될 수 있어 회차를 독립으로 가정한 신뢰구간이나 통계적 우월성은 주장하지 않는다.
순서 조합·문장 성공률·자동 구간 분할 성능을 측정한 결과가 아니다.
이전 3종 수어의 73.9%와는 데이터/종류/절차가 달라 직접 성능 향상 비교를 하지 않는다.

## 학습한 최종 후보

평가가 끝난 뒤 전체 9명 안에서 같은 선택 절차로 설정을 정하고 {n:,}회 모두로 별도 후보를 학습했다.
설정: `{json.dumps(cfg, ensure_ascii=False)}`.
`six_sign_candidate_model.joblib`은 기존 모델을 덮어쓰지 않았으며 실제 프로그램에 자동 적용하지 않았다.
**이 최종 후보 파일의 새로운 외부 참가자 정확도는 미측정**이다. 위 점수는 학습/선택 절차의 교차검증 점수이다.
이번 여섯 수어가 없는 subject7 등의 새 데이터를 받으면 이 후보를 고정한 채 먼저 평가하고 추가 학습한다.

## 검증 및 재현 자료

- 고정 커밋에서 원본 {raw_n:,}파일 SHA256 확인; 평가값을 CSV 예측으로 다시 계산하여 JSON과 일치 확인.
- 각 방법에서 모든 유효 회차가 정확히 한 번 평가됐는지 확인; 실제 정답 합계와 혼동행렬 대각 합 일치.
- 10개 합성 fixture 자동 테스트: 라벨/ID, 손 누락, 0FSR, 비유한값, 부팅 시계, scaler 분리, 안쪽/바깥 참가자 분리, 동률, 실제 정답 집계 검증.
- 합성 fixture의 정확도는 실험 결과에 포함하지 않음. 실제 센서의 재측정이나 새 외부 참가자 검증은 이번에 수행하지 않음.
- `outer_predictions.csv`: 방법별·회차별 실제/예측/정답, 학습 참가자, 선택 설정.
- `outer_folds.csv`, `inner_searches.json`: 바깥 평가 설정과 안쪽 선택 근거.
- `trial_features.csv`, `cohort_manifest.csv`, `excluded_trials.csv`: 학습 샘플 및 제외 근거.
- `preregistered_settings.json`, `reproducibility.json`: 검색 범위, 버전, 코드·파일 해시.
- 공유자료: 01센서표 / 02수어표 / 03참가자표 / 04혼동행렬 / 05수어별 그래프, PNG·SVG 및 CSV.
'''
    (share / '결과_보고서.md').write_text(report, encoding='utf-8')
    (root / 'report.md').write_text(report, encoding='utf-8')
    shutil.copy2(Path(__file__), root / Path(__file__).name)
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise ValueError('Prediction source changed during reporting')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in share.iterdir() if p.is_file()}
    (share / '파일_SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(team)
    print('Verified every plotted count against predictions; saved: ' + str(share))


if __name__ == '__main__':
    main()
