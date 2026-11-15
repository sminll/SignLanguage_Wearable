"""Read-only preliminary sign screening from saved participant-held-out results.

No new fitting, row removal, model replacement or combined-model performance claim.
The old three-class and new six-class experiments stay separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def screen(predictions, features, cohort):
    if predictions.file.duplicated().any() or features.file.duplicated().any():
        raise ValueError('Expected one prediction and feature row per trial')
    if set(predictions.file) != set(features.file):
        raise ValueError('Predictions/features use different cohorts')
    joined = predictions.merge(features, on='file', validate='one_to_one', suffixes=('', '_feature')).copy()
    for field in ['person', 'session_id', 'trial_id', 'label']:
        if not joined[field].eq(joined[field + '_feature']).all():
            raise ValueError('Identity mismatch: ' + field)
    joined['correct'] = joined.label.eq(joined.prediction)
    for hand in ['left', 'right']:
        low = [f'{hand}_fsr{i}_min' for i in range(1, 7)]
        high = [f'{hand}_fsr{i}_max' for i in range(1, 7)]
        std = [f'{hand}_fsr{i}_std' for i in range(1, 7)]
        if not np.isfinite(joined[low + high + std].to_numpy(float)).all():
            raise ValueError('Nonfinite FSR features')
        joined[hand + '_FSR_all_zero'] = joined[low + high].eq(0).all(axis=1)
        joined[hand + '_FSR_all_constant'] = joined[std].eq(0).all(axis=1)
    rows, people = [], []
    for label, group in joined.groupby('label'):
        by_person = group.groupby('person').agg(trials=('correct', 'size'),
                                                correct=('correct', 'sum'), recall=('correct', 'mean'))
        selected = joined[joined.prediction.eq(label)]
        precision = float(selected.label.eq(label).mean()) if len(selected) else 0.0
        recall = float(group.correct.mean())
        strong_fraction = float(by_person.recall.ge(.8).mean())
        lowest = float(by_person.recall.min())
        # Heuristic review buckets only, not an established reliability standard.
        if recall >= .8 and precision >= .8 and strong_fraction >= .8 and lowest >= .5:
            status = '우선 검토'
        elif recall >= .6 and precision >= .6:
            status = '조건부 후보'
        else:
            status = '현재 보류'
        row = dict(cohort=cohort, label=label, trials=len(group), correct=int(group.correct.sum()),
                   recall=recall, predicted_trials=len(selected), precision=precision,
                   people=len(by_person), people_recall_at_least_80=int(by_person.recall.ge(.8).sum()),
                   people_at_least_80_fraction=strong_fraction,
                   equal_person_mean_recall=float(by_person.recall.mean()),
                   minimum_person_recall=lowest, maximum_person_recall=float(by_person.recall.max()),
                   low_people='|'.join(by_person[by_person.recall.lt(.5)].index), status=status,
                   left_FSR_all_zero_trials=int(group.left_FSR_all_zero.sum()),
                   right_FSR_all_zero_trials=int(group.right_FSR_all_zero.sum()),
                   both_hands_FSR_all_zero_trials=int((group.left_FSR_all_zero & group.right_FSR_all_zero).sum()),
                   left_FSR_all_constant_trials=int(group.left_FSR_all_constant.sum()),
                   right_FSR_all_constant_trials=int(group.right_FSR_all_constant.sum()))
        rows.append(row)
        for person, p in by_person.iterrows():
            people.append(dict(cohort=cohort, label=label, person=person,
                               trials=int(p.trials), correct=int(p.correct), recall=float(p.recall)))
    return rows, people, joined[['file', 'person', 'label', 'left_FSR_all_zero', 'right_FSR_all_zero',
                                'left_FSR_all_constant', 'right_FSR_all_constant']]


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]])


def pct(v):
    return f'{v*100:.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory')
    root = args.results_root
    sources = dict(
        old_three_predictions=root/'2026-11-08_combined_candidate_loso/loso_predictions.csv',
        old_three_features=root/'2026-11-08_knn_development_holdout/trial_features.csv',
        six_predictions=root/'2026-11-15_six_sign_nested_loso/outer_predictions.csv',
        six_features=root/'2026-11-15_six_sign_nested_loso/trial_features.csv')
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()}
    old_pred = pd.read_csv(sources['old_three_predictions'])
    old_feat = pd.read_csv(sources['old_three_features'])
    six_pred = pd.read_csv(sources['six_predictions'])
    six_pred = six_pred[six_pred.method.eq('imu_only')].copy()
    six_feat = pd.read_csv(sources['six_features'])
    all_rows, all_people, fsr = [], [], []
    for predictions, features, cohort in [(old_pred, old_feat, '기존3종'), (six_pred, six_feat, '새6종')]:
        rows, people, quality = screen(predictions, features, cohort)
        all_rows.extend(rows)
        all_people.extend(people)
        quality['cohort'] = cohort
        fsr.append(quality)
    if len(all_rows) != 9 or sum(r['trials'] for r in all_rows) != len(old_pred) + len(six_pred):
        raise ValueError('Trial count mismatch')
    args.out.mkdir(parents=True)
    pd.DataFrame(all_rows).to_csv(args.out/'수어별_후보검토.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(all_people).to_csv(args.out/'사람별_수어정답률.csv', index=False, encoding='utf-8-sig')
    pd.concat(fsr, ignore_index=True).to_csv(args.out/'FSR_상태.csv', index=False, encoding='utf-8-sig')
    report = '''# 2026-11-15 시연 수어 후보 예비 선별

학습하지 않은 사람의 기존 예측을 읽어 후보를 검토했다. 새 학습이나 운영 모델 변경은 하지 않았다.
기존3종과 새6종은 수어 수·참가자·설정 선택 절차가 다른 실험이다. 각각의 실험 안에서 해석하며,
아래 수어를 합쳤을 때의 정확도나 하나의9종 모델 정확도를 의미하지 않는다.

## 선별 기준

- 수행했을 때 맞히는 비율(recall)과 해당 수어라고 판정했을 때 실제 맞는 비율(precision)을 함께 확인.
- 사람별 정답률, 80%이상인 사람 수, 최저 사람의 정답률 확인. 평균만으로 안정적이라고 판단하지 않음.
- 우선 검토: recall≥80%, precision≥80%, 80%이상인 참가자≥80%, 최저 참가자≥50%.
- 조건부 후보: recall≥60%와 precision≥60%이나 위 안정성 기준은 미충족.
- 나머지는 현재 보류. 이 숫자는 이번 예비 검토의 내부 기준이며 공인 기준이나 성능 보장이 아님.
- FSR 전체0/일정값 여부는 손별 참고 기록. 어느 손을 사용하는지와 착용/보정이 검증되지 않아 자동 제외 기준으로 쓰지 않음.

## 분류 결과

'''
    for cohort in ['기존3종', '새6종']:
        selected = [r for r in all_rows if r['cohort'] == cohort]
        report += f'### {cohort}\n\n' + table(
            ['수어', '정답/평가', '수행 정답률', '그 수어 판정의 정답 비율', '80%이상 참가자', '사람별 범위', '예비 판단'],
            [[r['label'], f"{r['correct']}/{r['trials']}", pct(r['recall']), pct(r['precision']),
              f"{r['people_recall_at_least_80']}/{r['people']}",
              f"{pct(r['minimum_person_recall'])}~{pct(r['maximum_person_recall'])}", r['status']]
             for r in selected]) + '\n\n'
    report += '''## 먼저 확인할 후보와 예외

- 아니다: 수행 정답률90%, 다른 수어를 아니다로 잘못 판정한 경우는 이6종 평가에서0회.
  하지만 subject1은2/20(10%)이고 나머지8명은20/20이므로 착용·신호·동작 구간을 먼저 확인해야 한다.
- 삼키다: 수행 정답률92.78%,9명 중8명은95%이상, subject7은8/20(40%).
  다른 수어도 삼키다로 잘못 판정해서, 삼키다라고 예측한230회 중 실제삼키다는167회(72.61%)다.
- 상처: 수행 정답률77.09%이나 subject1은2/19(10.53%), subject7은6/20(30%)이다. 차선후보로 검토한다.
- 교통사고·마시다 및 나머지4종은 현재 저장된 평가에서 낮거나 사람별 편차가 커 우선 시연 후보로 추천하기 어렵다.

이번의 엄격한 예비 기준에서 '우선 검토'를 모두 충족하는 수어는 없다. 아니다·삼키다는 먼저 확인할 조건부 후보이며
어떤 참가자에게나 안정적으로 작동한다고 확정한 것은 아니다. 해당 참가자의 낮은 결과를 이유로 데이터를 지우지 않는다.

## FSR 참고 정보

'''
    report += table(['실험', '수어', '평가회차', '왼손 FSR6개 전체0', '오른손 FSR6개 전체0', '양손 모두 전체0'],
                    [[r['cohort'], r['label'], r['trials'], r['left_FSR_all_zero_trials'],
                      r['right_FSR_all_zero_trials'], r['both_hands_FSR_all_zero_trials']] for r in all_rows])
    report += '''

0은 저장된 측정값이0이라는 뜻이며, 전송 누락과는 다르다. 사용하지 않는 손에서0이 나올 수도 있다.
FSR가0이 아닌 것만으로 사람 간 센서 패턴이 일관되거나 수어가 잘 구분된다고 판단하지 않는다.
위 FSR 숫자는 해당 실험에 포함된 회차만 센다. 손이 누락된16회는 새6종에 이전부터 미포함이며 원본을 보존한다.
이번 예비 선별에서 새로운 회차를 제외하지 않았다. FSR+IMU 원본 대신 검증된 min/max/std 특징으로
전체0/전체일정값 여부를 확인했으며 raw 데이터를 변경하지 않았다.

## 다음 확인

후보 수어를 함께 사용하려면 같은 하나의 분류기에 넣고 참가자별로 다시 평가해야 한다.
각기 다른 실험에서 높은 수어별 숫자를 모아 합쳐진 모델의 정확도라고 표시하면 안 된다.
먼저 낮은 결과가 나온 사람의 착용 방향/압박/기록 구간을 확인하고, 새 참가자 또는 재착용한 별도세션에서 확인한다.
무동작이나 미등록 수어를 거부하는 성능, 실시간 연속 인식 성능은 이번 자료로 검증되지 않았다.
기존3종은 설정 선택 후 내부 교차검증이며 새6종은 참가자 단위 중첩 교차검증의 IMU 비교이다.
후보 선정에 이미 결과를 사용했으므로 최종 선택의 외부 검증이 필요하다.
'''
    (args.out/'후보선별_보고서.md').write_text(report, encoding='utf-8')
    manifest = dict(source_paths={k: str(v) for k,v in sources.items()}, input_sha256=hashes,
                    evaluated_trials=len(old_pred)+len(six_pred), signs=9, new_training=False,
                    input_data_or_models_changed=False, combined_model_accuracy=None,
                    criteria_are_review_heuristics=True, decisions=all_rows)
    for key, p in sources.items():
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[key]:
            raise ValueError('Source changed during screening')
    (args.out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(Path(__file__), args.out/Path(__file__).name)
    print(pd.DataFrame(all_rows)[['cohort','label','correct','trials','recall','precision',
                                'people_recall_at_least_80','people','minimum_person_recall','status']].to_string(index=False))
    print('FSR_ZERO_COUNTS')
    print(pd.DataFrame(all_rows)[['label','left_FSR_all_zero_trials','right_FSR_all_zero_trials']].to_string(index=False))


if __name__ == '__main__':
    main()
