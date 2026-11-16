"""Summarize saved compensation experiments; never train or edit sensor data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .report_six_sign_knn import md_table, pct
except ImportError:
    from report_six_sign_knn import md_table, pct

NAMES = {
    'baseline_all': '기존 양손 FSR+IMU',
    'fsr_scale': 'FSR 크기 정규화 + 기존 IMU',
    'imu_magnitude_all': '기존 FSR + IMU 크기',
    'fsr_scale_imu_magnitude': 'FSR 크기 정규화 + IMU 크기',
    'robust_all': '중앙값·사분위 범위·5/95% 분위수',
    'baseline_imu': '기존 양손 IMU만',
    'imu_magnitude_only': '양손 IMU 크기만',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    out = root/'report.md'
    if out.exists():
        raise FileExistsError('Report exists; never replace a prior result')
    pred = pd.read_csv(root/'predictions.csv')
    summary = pd.read_csv(root/'comparison.csv')
    signs = pd.read_csv(root/'per_sign.csv')
    people = pd.read_csv(root/'per_person.csv')
    audit = json.loads((root/'calibration_audit.json').read_text(encoding='utf-8'))
    for row in summary.itertuples():
        part = pred[pred.cohort.eq(row.cohort) & pred.representation.eq(row.representation)]
        assert len(part) == row.trials and not part.file.duplicated().any()
        assert int(part.label.eq(part.prediction).sum()) == row.correct
        assert np.isclose(part.label.eq(part.prediction).mean(), row.accuracy)
        assert all(r.person not in r.training_people.split('|') for r in part.itertuples())
    three = summary[summary.cohort.eq('three_sign')].set_index('representation')
    baseline, candidate = three.loc['baseline_all'], three.loc['fsr_scale']
    sub = pred[pred.cohort.eq('three_sign')]
    a = sub[sub.representation.eq('baseline_all')].set_index('file')
    b = sub[sub.representation.eq('fsr_scale')].set_index('file').loc[a.index]
    gained = int((~a.correct & b.correct).sum())
    lost = int((a.correct & ~b.correct).sum())
    tables = {}
    for cohort in ['three_sign', 'six_sign']:
        rows = [[NAMES[r.representation], f'{r.correct}/{r.trials}', pct(r.accuracy)]
                for r in summary[summary.cohort.eq(cohort)].itertuples()]
        tables[cohort] = md_table(['처리 방식', '정답/평가', '정확도'], rows)
    sign_table = []
    for label in ['삼키다', '아니다', '상처']:
        part = signs[signs.cohort.eq('three_sign') & signs.label.eq(label)].set_index('representation')
        sign_table.append([label, pct(part.loc['baseline_all', 'recall']), pct(part.loc['fsr_scale', 'recall'])])
    person_table = []
    selected = people[people.cohort.eq('three_sign')]
    for who in sorted(selected.person.unique(), key=lambda p: int(p.replace('subject', ''))):
        part = selected[selected.person.eq(who)].set_index('representation')
        person_table.append([who, pct(part.loc['baseline_all', 'accuracy']), pct(part.loc['fsr_scale', 'accuracy'])])
    report = f'''# 2026-11-16 센서 코드 보정 후보 검토

현재 3종에서 손별 FSR의 크기 차이를 줄이는 정규화가 가장 유망한 후보였다.
기존 {int(baseline.correct)}/{int(baseline.trials)}({pct(baseline.accuracy)})에서
{int(candidate.correct)}/{int(candidate.trials)}({pct(candidate.accuracy)})로 순정답13회,
정확도 {100*(candidate.accuracy-baseline.accuracy):.2f}%p 증가했다.
그러나 일부 참가자와 상처의 정답률은 낮아졌으며, 기존 모델을 교체하지 않았다.

## 무엇을 바꾸었나

기존 코드는 센서별 평균/표준편차/최소/최대를 만들고 학습 참가자에서만 StandardScaler를 계산한다.
휴식 기준 압박값 제거와 센서의 물리적 힘·장착 방향 보정은 구현되어 있지 않다.

- FSR 크기 정규화: 한 손의 한 완료 회차에서 6개 FSR 전체의 95% 분위수를 구하고,
  같은 손의 6채널을 모두 이 수치로 나눈다. 상대적인 채널 분포와 시간 변화는 보존하되
  그 회차의 전체 압박 크기 정보는 약화한다. 분모는 최소50 ADC값으로 제한하여 작은 잡음을 확대하지 않는다.
  50은 사전 고정한 탐색용 값이며 센서 정상/불량 판정 기준이 아니다.
- IMU 크기: 가속도3축과 자이로3축을 각각 벡터 크기로 요약한다. 회전된 축의 영향이 줄지만
  수어를 구분하는 방향 정보도 잃는다. 자세 추정/좌표계 정렬이나 중력 제거를 한 것은 아니다.
- 극단값 영향 완화: 기존4개 통계 대신 중앙값, 사분위 범위,5%/95% 분위수4개를 채널별로 사용했다.
  빠른 동작의 실제 신호도 약화할 수 있으므로 무조건 좋은 필터라고 볼 수 없다.

모든 처리는 해당 손의 **완료된 한 회차**만 사용하며 정답라벨, 다른 회차, 평가 참가자의 전체 자료로
변환값을 추정하지 않는다. 시작 부분을 임의의 휴식으로 간주하지 않는다. 실시간 프레임 단위 인식 결과는 아니다.
원본 값에서 휴식 압박을 빼는 보정이나 힘 단위 변환을 수행한 것은 아니다.

## 3종: 삼키다·아니다·상처,10명539회

{tables['three_sign']}

{md_table(['수어', '기존 FSR+IMU', 'FSR 크기 정규화'], sign_table)}

{md_table(['참가자', '기존 FSR+IMU', 'FSR 크기 정규화'], person_table)}

FSR 정규화로 기존 오답 {gained}회를 새로 맞혔지만, 기존 정답 {lost}회가 새 오답이 됐다.
아니다의 오답은49회에서22회로 줄었고 상처는2회에서14회로 늘었다.
subject5는0/20에서2/20으로 여전히 낮다. 전체 상승만으로 모든 사람의 문제가 해결됐다고 해석하지 않는다.

## 6종: 괜찮다·아니다·무엇·마시다·화장실·싫다,9명1,044회

{tables['six_sign']}

6종에서는 FSR 정규화가 도움이 되지 않았고 IMU 크기만 쓸 때351/1,044(33.62%)에서
381/1,044(36.49%)로 증가했다. 높은 실용 정확도에 도달한 것은 아니며 같은 보정이 모든 수어에
통하는 상황은 아니다. 기존 양손 FSR+IMU 기준25.10%와 IMU만의33.62%를 구분한다.

## 기존 휴식 보정 CSV 연결 검사

- 고정 커밋의 보정 CSV {audit['calibration_files']}개 중 명시적 휴식은 {audit['explicit_rest_files']}개.
- 평가 두 묶음의 중복을 합친 원본 {audit['unique_training_trials']}회, 양손 {audit['training_hand_records']}개 기록 검사.
- 참가자·앱 계정·세션·보드·손·보정 버전이 같고 측정 전30분 이내인 휴식만 연결 후보로 허용.
- 양손을 모두 연결할 수 있는 회차는150회이며, 연결 후보는 모두 calibration_verified=false다.
- 현재3종에서는 subject10의삼키다·상처40회와 subject9의아니다20회, 총60/539회만 연결된다.
- 6종에서는 subject9의110/1,044회만 연결된다. 나머지 사람에게 이 휴식값을 복사하지 않았다.

따라서 기존539회 전체에 검증된 휴식 보정이 이미 준비돼 있다고 볼 수 없다. 참가자별/수어별
구성이 다른 이 부분집합만 뽑아 전체 성능처럼 비교하지 않았다. 실제로 휴식 기준 보정을 평가하려면
같은 착용 세션의 모든 참가자·수어에 휴식 기록을 연결하고 유효성을 확인해야 한다.

## 평가 범위와 재현

이번 분석은 기존 사람 제외 결과를 본 뒤 설계한 **탐색적 재분석**이다. 새로운 독립 외부 검증이 아니다.
이전 바깥fold마다 학습 참가자 안에서 고른 K/거리/투표 설정을 그대로 재사용하여 입력 표현의 영향을
비교했다. 새 표현마다 하이퍼파라미터를 다시 최적화한 중첩평가는 아니다. 따라서 후보 순위도 확정하지 않는다.
전체539회와1,044회를 모든 표현에서 동일하게 유지했고 저성능/0FSR을 이유로 회차를 삭제하지 않았다.
기존 IMU/전체센서 예측이 각각 한 회차도 다르지 않게 재현되는지 먼저 검사했다.
원본1,403파일의SHA256과기존96특징도 다시 확인했다.

```powershell
python -B tools/study_sensor_compensation.py --results-root data/knn_results --repo . --calibration-commit 88fed688ddc5cf3842ed2cfe8e257f0773be66e6 --out data/knn_results/sensor_compensation_rerun
python -B tools/report_sensor_compensation.py --results data/knn_results/sensor_compensation_rerun
python -B -m unittest discover -s tests -v
```

입력은 11/15의3종/6종 원본 실험 폴더이며 새 출력 폴더를 사용한다. 코드 브랜치는
`feature/sensor-calibration-study`. `comparison_plan.json`에 원본 커밋,입력해시,7가지표현,
분모 하한과 평가 방식을 기록했다. `predictions.csv`, `comparison.csv`, `per_sign.csv`,
`per_person.csv`, `calibration_inventory.csv`, `calibration_links.csv`, `verification.json`이 근거다.
기존 CSV/모델,수집·BLE·펌웨어는 수정하지 않았으며 새 모델 파일도 만들거나 배포하지 않았다.

다음 단계는 FSR 정규화 후보를 같은 조건의 새 측정에서 확인하고, 재착용 후 휴식 기록으로
기본 압박·자이로 바이어스 보정을 별도 비교하는 것이다. IMU 방향 정보를 보존하는 특징과
크기 특징을 함께 쓰는 방식도 후속 후보이며 이번에 검증한 결과와 섞지 않는다.
'''
    out.write_text(report, encoding='utf-8')
    print(out)
    print('Verified report against saved actual predictions; no fitting performed')


if __name__ == '__main__':
    main()
