# 앱 수집 XIAO 6종 수어 평가

대상: 괜찮다 / 아니다 / 무엇 / 마시다 / 화장실 / 싫다.
기존 3종 모델과 분리된 실험이며 기존 데이터·모델·BLE·서버를 변경하지 않는다.

## 실행

Python 의존성: numpy, pandas, scikit-learn, matplotlib, joblib, threadpoolctl.
이번 실험 환경은 Python 3.15.2, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4이며
정확한 실행 버전과 파일 SHA256은 결과의 `reproducibility.json`을 따른다.
그래프에는 Windows의 Malgun Gothic 등 한글 폰트가 필요하다.

아래 명령은 해당 기능 브랜치의 저장소 루트에서 실행한다. `git fetch`는 원격 브랜치를 갱신하지만
분석 커밋은 아래처럼 명시적으로 고정하므로 다음 업로드를 자동으로 섞지 않는다.

```powershell
git fetch origin training-data
python -B tools/audit_six_sign_data.py --commit 88fed688ddc5cf3842ed2cfe8e257f0773be66e6 --out data/knn_results/six_sign_inventory_reproduction
python -B tools/evaluate_six_sign_knn.py --inventory-dir data/knn_results/six_sign_inventory_reproduction --repo . --out data/knn_results/six_sign_nested_reproduction
python -B tools/report_six_sign_knn.py --results data/knn_results/six_sign_nested_reproduction
python -B -m unittest discover -s tests -p test_six_sign_knn.py -v
```

이미 존재하는 결과·공유자료 폴더는 덮어쓰지 않는다. 재실행할 때 새 출력 폴더를 지정한다.
원본 CSV는 Git archive를 메모리로 읽고 파일별 해시를 검증한다. 작업 폴더로 raw CSV를 복사하거나
서버로 재전송하지 않는다. 위 실험에 합성 fixture 데이터는 포함하지 않는다.

## 학습과 평가

- 각 CSV의 단일 `(person, session_id, trial_id)`가 한 동작 회차이다.
- 사용자 확인 별칭만 통일한다. subject-8과 subject8은 같은 사람이다.
- 두 손의 시간축을 강제 정렬하지 않고 손별 통계 특징을 결합한다.
- 36가지 사전 설정: 양손 IMU / 양손 FSR / 전체 × K3/5/7 × 맨해튼/유클리드 × 균등/거리 가중.
- 한 사람의 모든 날짜를 제외하는 바깥 9fold; 나머지 8명 안에서만 선택하는 안쪽 8fold.
- 표준화는 각 학습 split에서만 fit. 안쪽 선택은 참가자별 balanced accuracy의 동일 가중 평균.
- 사전 주 결과 `nested_selection`은 센서 종류도 안쪽에서 선택한다.
- 센서별 비교는 입력 종류를 고정하고 K/거리/가중치를 안쪽에서 선택한다.
- 평가 결과가 낮다는 이유로 참가자·회차를 제거하거나 설정 검색 범위를 늘리지 않는다.
- 공유 session_id가 있어도 Subject와 고유 trial UUID·파일을 기준으로 분리를 확인한다.

이전 3종 KNN의 설정 선택 후 교차검증 결과와 평가 절차가 다르다.
6종의 바깥 평가를 본 다음 선택한 가장 높은 행을 새 외부 참가자 평가라고 표시하지 않는다.

## 저장 결과

`outer_predictions.csv`에 각 회차의 실제·예측·정답, 학습 참가자, 선택 설정을 기록한다.
`inner_searches.json`에 바깥 평가별 안쪽 참가자와 모든 설정의 점수를 보존한다.
`evaluation.json`, `outer_folds.csv`와 공유자료 표/혼동행렬은 이 예측 결과를 따른다.
보고서 생성기는 예측에서 결과를 다시 계산하여 저장된 JSON과 같은지 확인한다.

`six_sign_candidate_model.joblib`은 평가 후 전체 9명의 1,044회로 학습한 별도 후보이다.
`pipeline`, `feature_columns`, `metadata`를 저장한다. raw 센서 프레임을 직접 predict하지 않고
`extract_features()`와 같은 손별 특징을 만들어 지정된 열 순서로 입력해야 한다.
이 파일을 실제 앱·서버에 자동 적용하지 않는다. 신뢰할 수 있는 로컬 joblib 파일만 로드한다.

최종 후보의 향후 외부 참가자 정확도는 미측정이다. 교차검증 점수는 학습/선택 절차의 성능이며,
전체 데이터로 만든 이 단일 후보를 새 사람에게 실제로 평가한 점수가 아니다.

## 포스터 자료

`공유자료/` 안에 PNG(240dpi)·SVG 표 3개, 혼동행렬, 수어별 그래프와 CSV,
`포스터_설명.txt`, `팀원_공유문구.txt`, `결과_보고서.md`를 만든다.
혼동행렬의 행은 실제 수어, 열은 예측 수어이며 횟수와 행별 비율을 함께 표시한다.
전체 정확도, 사람별 동일 가중 평균, 수어별 recall을 서로 구분한다.

동작 영상·착용 위치·압박·보정·공통 시계는 CSV 구조 검사만으로 확인되지 않는다.
본 실험은 고정 구간의 개별 6종 분류이며 자동 구간 분할·조합 문장 성공률을 측정하지 않는다.

## 추가 원인 분석 — 같은 사람과 동작 흐름

기존 결과를 본 뒤 다음 진단을 설계했으므로 **같은 9명의 자료를 재사용한 탐색적 분석**이다.
새로운 외부 검증으로 표시하지 않는다. 원본/기존 결과/기존 모델은 보존하며 자동 모델 교체를 하지 않는다.

```powershell
python -B tools/diagnose_six_sign_knn.py --previous data/knn_results/2026-11-15_six_sign_nested_loso --repo . --out data/knn_results/six_sign_diagnostic_rerun
python -B tools/report_six_sign_diagnostics.py --results data/knn_results/six_sign_diagnostic_rerun
python -B -m unittest discover -s tests -p test_six_sign_diagnostics.py -v
```

`--previous`는 앞서 완료한 결과 폴더이다. 다른 이름으로 재현했다면 그 폴더로 바꾼다.
계산을 다시 하지 않고 그림 배치/설명을 갱신할 때만 보고서 명령에 `--refresh-report`를 추가할 수 있다.

같은 사람 진단은 각 사람·수어의 시간순 회차를 5개 블록으로 나누어 한 블록씩 평가한다.
학습한 회차를 다시 맞히는 방식이 아니며 회차UUID와 파일은 분리한다. 다만 같은 세션의 가까운
반복 회차가 학습에 있을 수 있으므로 이 점수를 새 세션·새 날짜·새 사람의 정확도로 사용하지 않는다.

기존 5초 전체 통계와 상대 시간5구간 통계의 비교는 K5/맨해튼/균등을 고정한 양손 IMU로 먼저 수행한다.
이 고정 설정 비교는 이전의 안쪽 설정 선택 평가와 다르므로 점수가 같을 필요는 없다.
동작 흐름의 5구간 버전은 추가로 같은 36설정의 참가자 단위 중첩 평가도 수행한다.
공통 시계는 검증되지 않았으므로 손별 상대 진행률을 사용하고 두 손의 프레임을 강제 정렬하지 않는다.

확인된 고정 설정 IMU 결과(각1,044회):

| 표현 | 같은 사람의 다른 회차 | 학습에서 빠진 사람 |
|---|---:|---:|
| 기존 전체 통계 | 1,025/1,044 (98.18%) | 368/1,044 (35.25%) |
| 상대 시간5구간 | 1,008/1,044 (96.55%) | 282/1,044 (27.01%) |

같은 사람/세션 안의 반복은 잘 구분되지만 참가자가 바뀌면 크게 낮아졌다. 사람·착용·세션 조건에
민감하다는 근거이며, 착용 방향이나 압박 중 어느 것이 원인인지는 CSV만으로 확정하지 못한다.
단순한 5구간 통계는 이번 실험에서 개선되지 않았다. 센서까지 안쪽에서 선택한 결과도 기존332/1,044
(31.80%)에서 5구간229/1,044(21.93%)로 낮아졌다. 이 변경을 운영 모델에 자동 적용하지 않는다.

새 참가자 또는 같은 참가자의 장비 재착용 후 별도 세션을 이용한 평가가 다음 확인 과제이다.
같은 사람의 높은 점수를 일반적인 수어 인식 정확도로 대체해 표시하지 않는다.

## 삼키다·아니다·상처를 같은 분류기로 평가

수어별로 따로 나온 점수를 합치는 대신, 선택한 세 수어를 한 분류기에 실제로 함께 넣어 평가했다.
이 세 수어는 이전 결과를 보고 골랐으므로 탐색적 재분석이며, 새로운 외부 검증이 아니다.

```powershell
python -B tools/evaluate_three_sign_candidates.py --results-root data/knn_results --repo . --out data/knn_results/three_sign_combined_rerun
python -B tools/report_three_sign_candidates.py --results data/knn_results/three_sign_combined_rerun
python -B -m unittest discover -s tests -p test_three_sign_candidates.py -v
```

원래 두 실험 폴더의 특징표와 해시 manifest가 필요하다. 원본은 각 고정Git커밋에서 다시 읽어 해시와96특징을 검증한다.
삼키다180 + 아니다180 + 상처179 = 539회,10명. subject5는 아니다20회, subject7은 삼키다·상처40회만 있다.
두 사람도 바깥 평가와 학습에 포함하며, 없는 수어를 만들어 넣지 않는다. 안쪽 설정 선택 점수는
세 수어가 모두 있는 학습 참가자에서 계산하되 일부 수어만 있는 사람도 학습에는 포함한다.
바깥에서는 한 사람의 모든 날짜와 회차를 제외하여10번 반복한다.36설정, StandardScaler 학습분리,
동률 고정 순서를 유지한다. 저성능이나0FSR 때문에 새로 제외한 회차는 없다.

센서 종류까지 학습 참가자 안에서 선택한 주 결과는 **487/539(90.35%)**다.
모든 바깥fold에서 FSR+IMU 입력이 선택됐다. 수어별로 삼키다179/180(99.44%),
아니다131/180(72.78%), 상처177/179(98.88%)다. 아니다는 subject1 6/20, subject5 0/20,
subject9 5/20로 일부 참가자에게 실패가 집중되므로 평균만으로 안정적인 모델이라고 표시하지 않는다.

큰 한계는 수어와 측정 날짜가 얽혀 있다는 점이다. 삼키다·상처는11월4일/7일,
아니다는11월13~15일 자료다. 같은 날과 같은 착용조건에서 세 동작을 함께 새로 평가해야 한다.
전체539회로 별도 후보 `three_sign_candidate_model.joblib`을 학습했지만 기존 모델에 자동 적용하지 않았다.
후보의 새로운 외부 참가자 정확도는 미측정이다. 결과 폴더에 예측·분리·설정·해시와 공유 그림/설명을 보존한다.

## 기존3종에 아니다를 더한4종 및5종 비교

```powershell
python -B tools/compare_sign_vocabularies.py --results-root data/knn_results --out data/knn_results/vocabulary_comparison_rerun
python -B tools/report_sign_vocabularies.py --results data/knn_results/vocabulary_comparison_rerun
```

기존3종을 재평가하고, 교통사고/삼키다/상처/아니다4종, 그4종에 괜찮다/무엇/마시다/화장실/싫다를 각각
하나씩 더한5종을 비교했다. 모든9종에서 가능한5종 조합을 전부 비교한 것은 아니다.
같은36설정과 참가자단위 중첩평가를 사용한다. 각구성의 바깥10fold, 안쪽은 모든현재라벨을 가진 학습참가자로
설정선택점수를 계산한다. 부분라벨참가자는 학습과 바깥평가에서 보존한다.

- 기존3종의 이번선택절차: 390/559(69.77%). 과거73.9%의 고정설정 사후내부평가와 절차가 다름.
- 아니다를 붙인4종: 499/739(67.52%).
- 그4종을 유지한5종 중 전체정확도최고: 괜찮다추가,601/913(65.83%), 최저수어45%.
- 미리정한 최저수어정답률 우선검토 기준의5종: 화장실추가,590/911(64.76%), 최저수어52%.

각수어구성을 고른 것은 같은내부예측을 본 뒤의 선택이다. 최종외부검증 성능으로 표시하지 않는다.
수어별날짜/착용조건차이가 남아있고 수어수가 다른문제끼리의 비교이다.
기존4종739개의평가파일은 모든5종에 동일하게 들어갔는지 확인하고 별도로정답수를 저장했다.
기존3종후보(삼키다/아니다/상처)의90.35% 기록과 모델은 덮어쓰지 않았다.
새각구성 후보는 별도하위폴더에 저장했고 운영에 자동적용하지 않았다.
82개자동테스트 통과,70개바깥fold와 안쪽선택설정 기록을 별도검증했다.
