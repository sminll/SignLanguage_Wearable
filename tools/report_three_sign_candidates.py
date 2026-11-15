"""Poster-ready tables from the actual combined three-class nested predictions."""
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
    from . import evaluate_three_sign_candidates as tri
    from .report_six_sign_knn import draw_table, md_table, pct, save_figure
except ImportError:
    import evaluate_three_sign_candidates as tri
    from report_six_sign_knn import draw_table, md_table, pct, save_figure

NAMES = {'imu_only':'양손 IMU', 'fsr_only':'양손 FSR', 'all_sensors':'양손 FSR+IMU',
         'nested_selection':'학습 참가자 안에서 센서 선택'}
NOTE = '결과를 보고 고른 3종의 탐색적 내부 평가 · 수어별 측정 날짜가 다름 · 새 외부 검증 아님'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--refresh-report', action='store_true', help='Refresh report only; never fit')
    args = parser.parse_args()
    root = args.results.resolve()
    out = root/'공유자료'
    if out.exists() and not args.refresh_report:
        raise FileExistsError('Share folder already exists')
    predictions = pd.read_csv(root/'outer_predictions.csv')
    features = pd.read_csv(root/'trial_features.csv')
    results = json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    candidate = json.loads((root/'candidate_manifest.json').read_text(encoding='utf-8'))
    source_sha = hashlib.sha256((root/'outer_predictions.csv').read_bytes()).hexdigest()
    for method, rows in predictions.groupby('method'):
        rows = rows.copy()
        rows['correct'] = rows.label.eq(rows.prediction)
        if rows.file.duplicated().any() or set(rows.file) != set(features.file):
            raise ValueError('Incomplete/duplicate test predictions')
        for row in rows.itertuples():
            if row.person in row.train_people.split('|'):
                raise ValueError('Outer participant leakage')
        if json.dumps(tri.summary(rows), sort_keys=True) != json.dumps(results[method], sort_keys=True):
            raise ValueError('Stored metrics differ from real predictions')
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ['Malgun Gothic','Noto Sans CJK KR','NanumGothic','AppleGothic'] if f in available), None)
    if not font:
        raise ValueError('Korean font required')
    plt.rcParams.update({'font.family':font, 'axes.unicode_minus':False, 'svg.fonttype':'none'})
    out.mkdir(exist_ok=args.refresh_report)
    r = results['nested_selection']
    n, correct = r['trials'], r['correct']
    sensor_rows = [[NAMES[m], f"{results[m]['correct']}/{n}", pct(results[m]['accuracy']),
                    f"{results[m]['macro_f1']:.3f}"] for m in tri.base.METHODS]
    sign_rows = [[s['label'], s['trials'], s['correct'], s['trials']-s['correct'], pct(s['recall'])]
                 for s in r['per_sign']]
    people_rows = [[p['person'], '/'.join(p['labels']), p['trials'], p['correct'], pct(p['accuracy'])]
                   for p in r['people']]
    draw_table('삼키다·아니다·상처 — 한 모델에서 분류한 결과',
               ['센서 입력', '정답/평가', '전체 정확도', 'Macro F1'], sensor_rows,
               '한 사람의 모든 데이터를 학습·설정 선택에서 제외해 10명 반복\n' + NOTE,
               out/'01_센서별_비교')
    draw_table(f"3종 수어별 정답률 — {correct}/{n} ({pct(r['accuracy'])})",
               ['실제 수어','평가 회차','정답','오답','정답률'], sign_rows,
               '각 바깥 평가의 학습 참가자 안에서 센서·KNN 설정 선택\n' + NOTE,
               out/'02_수어별_정답률')
    draw_table('참가자별 결과 — 해당 사람을 학습에서 제외',
               ['참가자','실제로 측정한 수어','평가 회차','정답','정확도'], people_rows,
               'subject5는 아니다만, subject7은 삼키다·상처만 측정; 해당 회차 모두 포함\n' + NOTE,
               out/'03_참가자별_정확도')
    cm = np.array(r['confusion_matrix'])
    rates = cm/cm.sum(axis=1, keepdims=True)*100
    fig, ax = plt.subplots(figsize=(9.8, 8.8))
    plot = ax.imshow(rates, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(3), tri.LABELS, fontsize=15)
    ax.set_yticks(range(3), tri.LABELS, fontsize=15)
    ax.set_xlabel('예측한 수어', fontsize=16, labelpad=12)
    ax.set_ylabel('실제 수행한 수어', fontsize=16, labelpad=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm[i,j]}회\n({rates[i,j]:.1f}%)', ha='center', va='center', fontsize=19,
                    color='white' if rates[i,j]>55 else '#123b60')
    fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, label='실제 수어별 비율 (%)')
    fig.suptitle('XIAO 삼키다·아니다·상처 — 참가자 제외 평가', fontsize=20, fontweight='bold')
    ax.set_title(f"10명 · {n}회 중 {correct}회 정답 · {pct(r['accuracy'])}", fontsize=15, pad=18)
    fig.text(.5, .02, NOTE+'\n3종이 모두 있는 참가자8명 + 일부 수어만 있는 참가자2명', ha='center', fontsize=10)
    fig.subplots_adjust(top=.87, bottom=.14)
    save_figure(fig, out/'04_혼동행렬')
    pd.DataFrame(cm, index=tri.LABELS, columns=tri.LABELS).to_csv(out/'혼동행렬.csv', encoding='utf-8-sig')
    pd.DataFrame(r['per_sign']).to_csv(out/'수어별_정답률.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(r['people']).to_csv(out/'참가자별_정확도.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sensor_rows, columns=['센서 입력','정답/평가','전체 정확도','Macro F1']).to_csv(
        out/'센서별_비교.csv', index=False, encoding='utf-8-sig')
    primary_rows = predictions[predictions.method.eq('nested_selection')]
    person_sign, weak = [], []
    for who in tri.base.people_order(primary_rows.person):
        for label in tri.LABELS:
            group = primary_rows[primary_rows.person.eq(who) & primary_rows.label.eq(label)]
            if group.empty:
                continue
            k = int(group.label.eq(group.prediction).sum())
            row = dict(person=who, label=label, trials=len(group), correct=k, recall=k/len(group))
            person_sign.append(row)
            if row['recall'] < .5:
                weak.append(f"{who} {label} {k}/{len(group)} ({pct(row['recall'])})")
    pd.DataFrame(person_sign).to_csv(out/'참가자_수어별_정답률.csv', index=False, encoding='utf-8-sig')
    weak_text = ' / '.join(weak) if weak else '50% 미만인 참가자·수어 조건 없음'
    used = primary_rows.groupby('sensors').person.nunique().to_dict()
    dates = features.groupby('label').date_kst.unique().to_dict()
    date_rows = [[s, ', '.join(sorted(dates[s]))] for s in tri.LABELS]
    sub = r['complete_people_subset']
    team = (
        '삼키다·아니다·상처를 따로 나온 점수로 비교하는 대신, 세 수어를 한 모델에 함께 넣어 평가했습니다.\n'
        f"한 사람의 데이터를 학습과 설정 선택에서 모두 빼는 과정을 10명에게 반복했고, {n}회 중 "
        f"{correct}회 정답으로 전체 정확도는 {pct(r['accuracy'])}였습니다.\n"
        '수어별: '+' / '.join(f"{s['label']} {pct(s['recall'])} ({s['correct']}/{s['trials']})" for s in r['per_sign'])+'\n'
        '낮게 나온 조건: '+weak_text+'. 이 회차들도 전체 결과에 포함했습니다.\n'
        '일부 수어만 있는 두 분의 데이터도 제외하지 않았습니다. 다만 기존 결과를 보고 고른 수어이고 '
        '측정 날짜도 서로 달라서, 현재는 시연 후보를 검토하기 위한 내부 실험 결과입니다. '
        '같은 착용 조건에서 세 동작을 함께 새로 측정해 실제 시연 전에 확인하면 좋겠습니다.\n'
    )
    report = f'''# 2026-11-15 삼키다·아니다·상처 통합 KNN

**{correct}/{n}회 정답, 전체 정확도 {pct(r['accuracy'])}.**
이전 실험의 수어별 백분율을 평균/곱한 값이 아니라, 세 수어를 같은 분류기에 넣고 실제 회차별로 예측한 결과이다.

**{NOTE}**

## 데이터와 참가자

- XIAO 양손 센서, 동작약5초가 한 학습 샘플. 삼키다180회, 아니다180회, 상처179회, 총539회.
- 10명 중8명은 세 종류를 모두 보유. subject5는 아니다20회, subject7은 삼키다20·상처20회만 있음.
- 일부 수어만 있는 사람도 최종 평가/학습에서 보존. 없는 수어의 회차를 만들거나0점으로 채우지 않음.
- 초기 제외 자료는 다시 섞지 않았고, 현재 선택된 자료에서 낮은 성능/0FSR을 이유로 새로 제외하지 않음.
- 소스커밋: `{', '.join(candidate['source_commits'])}`.

{md_table(['수어','측정일(KST)'], date_rows)}

중요한 한계: 수어와 측정 날짜·세션 조건이 얽혀 있다. 예를 들어 아니다는 다른 두 수어보다 나중에 수집했다.
따라서 점수가 높아도 수어 자체의 차이만 배웠다고 단정할 수 없다. 같은 날·같은 착용 조건에서
세 수어를 함께 새로 측정해 평가하는 것이 필요하다.

## 평가 방법

바깥쪽: 한 사람의 모든 날짜·회차를 제외하고 나머지9명으로 학습하는 과정을10번 반복했다.
안쪽: 그 학습 참가자 안에서 세 수어가 모두 있는 사람을 한 명씩 빼서 설정을 고른다.
일부 수어만 있는 참가자는 안쪽에서도 학습에 포함하지만, 3종 균형 점수를 고르는 평가자로는 쓰지 않는다.
안쪽 참가자별 macro recall(세 수어 정답률 평균)을 동일 가중 평균하여 설정을 선택한다.
바깥 평가에는 일부 수어만 있는 두 사람의 실제 회차도 포함한다.

센서3종 × K3/5/7 × 맨해튼/유클리드 × 균등/거리 가중, 총36설정을 사전에 고정했다.
StandardScaler는 각 학습 split에서만 계산한다. 동률은 config_id 오름차순으로 선택한다.
양손별 평균·표준편차(ddof0)·최소·최대를 합쳐 IMU48, FSR48, 전체96특징을 쓴다.
부팅 시계를 손끼리 직접 맞추지 않으며 동작의 순간 동기화나 연속 문장 인식 실험이 아니다.

주 결과는 `nested_selection`: 센서 종류까지 안쪽 학습 참가자만으로 선택한 절차이다.
선택된 입력 종류의 바깥fold 수: `{json.dumps(used, ensure_ascii=False)}`.

## 센서별 비교

{md_table(['센서 입력','정답/평가','전체 정확도','Macro F1'], sensor_rows)}

각 센서별 행은 센서 종류를 고정하고 K/거리/투표만 안쪽에서 선택한다.
이 결과를 보고 가장 높은 센서 행을 선택한 점수를 새 외부 평가라고 표시하지 않는다.

## 수어별 결과 — 주 결과

{md_table(['실제 수어','평가 회차','정답','오답','정답률'], sign_rows)}

## 참가자별 결과 — 주 결과

{md_table(['참가자','측정한 수어','평가 회차','정답','정확도'], people_rows)}

50% 미만인 참가자·수어 조건: **{weak_text}**.
특히 아니다는 일부 참가자에게 실패가 집중되어 있다. 전체90.35%만으로 모든 참가자에게 안정적이라고 표현하지 않는다.

주 결과는 모든 실제회차 합계 {correct}/{n}이다. 10명 정확도의 단순평균은 {pct(r['participant_mean_accuracy'])}이며 분모가 다르다.
참고로 세 수어가 모두 있는8명의 같은 예측만 모으면 {sub['correct']}/{sub['trials']}({pct(sub['accuracy'])})다.
이는 낮은 사람을 제외해 다시 학습한 결과가 아니라, 동일한 바깥 예측의 명시적 부분집합 집계이다.

## 최종 후보와 아직 검증하지 않은 것

평가 후 전체539회로 별도 `three_sign_candidate_model.joblib` 후보를 학습했다.
설정은 `{json.dumps(candidate['config'], ensure_ascii=False)}`.
기존 모델을 덮어쓰거나 서버·앱에 자동 적용하지 않았다.
**이 최종 후보 파일의 새 외부 참가자 정확도는 미측정**이다.
현재 점수는 수어를 이전 결과로 고른 뒤 기존 참가자로 수행한 탐색적 내부 교차검증이다.
참가자 제외는 지켰지만 수어 선정 자체를 독립 평가한 것은 아니다.
무동작/미등록 수어 거부, 실제 시연, 재착용 후 다른 날짜의 성능도 아직 검증하지 않았다.

## 재현과 근거

```powershell
python -B tools/evaluate_three_sign_candidates.py --results-root data/knn_results --repo . --out data/knn_results/three_sign_combined_rerun
python -B tools/report_three_sign_candidates.py --results data/knn_results/three_sign_combined_rerun
python -B -m unittest discover -s tests -p test_three_sign_candidates.py -v
```

원본539파일을 각 고정 커밋에서 읽어 SHA256을 확인하고,96특징을 재계산해 이전 특징표와 일치 확인했다.
`comparison_plan.json`, `cohort_manifest.csv`, `outer_predictions.csv`, `outer_folds.csv`, `inner_searches.json`,
`evaluation.json`, `candidate_manifest.json`, `verification.json`에 분리·설정·예측·해시를 보존한다.
보고서는 저장된 예측으로 수치를 다시 계산해 JSON과 일치하는지 확인했다.
합성 fixture 자동 테스트의 숫자는 실제 실험 결과에 포함하지 않는다.
'''
    (root/'report.md').write_text(report, encoding='utf-8')
    (out/'결과_보고서.md').write_text(report, encoding='utf-8')
    (out/'팀원_공유문구.txt').write_text(team, encoding='utf-8')
    shutil.copy2(Path(__file__), root/Path(__file__).name)
    assert hashlib.sha256((root/'outer_predictions.csv').read_bytes()).hexdigest() == source_sha
    tri.base.dump_json(out/'파일_SHA256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in out.iterdir() if p.name != '파일_SHA256.json'})
    print(team)
    print('Validated plot/report output:', out)


if __name__ == '__main__':
    main()
