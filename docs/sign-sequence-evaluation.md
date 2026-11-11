# 두 수어 고정 구간 조합 평가

현재 상태: 기능 구현 및 가상 데이터 테스트 단계. 실제 조합 인식 정확도는 **미측정**입니다.
이 기능은 동작 시작·종료를 자동 검출하는 연속 인식이 아닙니다.

구현 후 확인 결과와 파일 목록: [검증 기록](sign-sequence-validation.md).

## 확인한 기존 기능과 모델 (2026-11-12)

작업 기준은 `origin/main`의 `e0b65dc`입니다. 원래 VS Code 작업 폴더에 미커밋 변경이 있어,
`feature/fixed-window-sign-combinations` 브랜치를 `data/backups/sequence-feature` 별도 worktree로 만들었습니다.
기존 작업 파일, 원시 CSV, 모델은 수정하지 않습니다.

| 모델 | 실제 지원 수어 | 특징·학습 이력 | 이번 기능 사용 |
|---|---|---|---|
| `data/knn_results/2026-11-01_XIAO/knn_model.joblib` | 갑자기, 머리, 아프다 | 왼손 FSR 6 + IMU 6, 각 mean/std/min/max 48개. subject1, 회차 분할 학습 45개 | 가능. 이전 개별 수어 14/15 결과는 조합 정확도가 아님 |
| `data/knn_results/2026-11-02_subject1_to_subject4/combined_model/knn_model.joblib` | 갑자기, 머리, 아프다 | 같은 48개 특징. subject1·4의 6개 세션, 120회로 결합 학습 | 이번 실험의 고정 기준 모델로 권장. 최고 성능을 보장한다는 의미는 아님 |
| `data/knn_results/2026-11-02_subject1_subject4_to_subject2_retake/combined_model/knn_model.joblib` | 갑자기, 머리, 아프다 | subject2 재측정 포함 과거 결합 후보. 착용·FSR 조건 차이 기록 있음 | 보존, 자동 선택하지 않음 |
| `data/knn_results/2026-11-08_knn_development_holdout/all_current_candidate_model.joblib` | 교통사고, 삼키다, 상처 | 양손 IMU 48개 특징, 초기 데이터 제외 10명·559회 | 머리·아프다 미지원. 이번 평가에서 대체 사용 불가 |

로컬 joblib의 실제 `pipeline.classes_`와 열 목록을 확인했습니다. 기존 모델은 Git에 없는 로컬 산출물입니다.
모델 파일이 없는 PC에서는 소유자가 제공한 동일 모델을 별도로 받아야 합니다. 신뢰하는 모델만 읽으세요.
동일한 48개라는 숫자라도 양손 IMU 48개와 기존 왼손 FSR·IMU 48개는 다른 입력입니다.

Nov1 머리·아프다·갑자기는 각각 독립 세션에 20회씩 저장돼 있습니다. Nov2에도 subject4와
subject2 재측정 자료가 남아 있습니다. 이 자료를 임의로 짝지어 새 조합 실험의 정답률로 사용하지 않습니다.
기존 정확도·삭제/보존 이력은 `실험기록/모델_현황.md`, 날짜별 실험 기록을 참고하세요.

기존 연결은 XIAO BLE → MATLAB 패킷 해석 → 로컬 CSV → 명시적 HTTP 전송 → Python 수신 CSV →
회차별 특징 추출 → KNN입니다. `dual_hand_features.py`는 **같은 동작의 두 손을 결합**합니다.
이번 기능은 **서로 다른 두 동작의 예측 결과를 순서대로 결합**합니다. 기존 자동 조합 평가 기능은 발견되지 않았습니다.

## 수집: 실제 센서 연결

MATLAB R2025b에서 기존 수집 코드와 같은 BLE 환경을 사용합니다. Arduino·UUID·패킷 해석은 바꾸지 않습니다.
새 브랜치의 `MATLAB_노트북용_수정본` 폴더를 Current Folder로 열고 실행합니다.

```matlab
blelist
xiao_ble_sequence_experiment("왼쪽_XIAO_BLE주소", "오른쪽_XIAO_BLE주소")
```

1. 두 장치를 다른 주소로 연결합니다. 두 손 그래프와 FSR 반응을 확인합니다.
2. 별도 ‘두 수어 조합 실험’ 창에서 Subject, 첫 수어 `머리`, 두 수어 `아프다`, 반복 수 `20`을 확인합니다.
   이번 수집기는 기존 왼손 모델에 맞춘 실험입니다. 두 수어 모두 기존 학습과 같은 왼손 동작으로 수행하고,
   오른손은 자연스럽게 내립니다. 양손 수어 모델의 성능을 측정하는 창이 아닙니다.
3. ‘새 실험 시작’을 누르면 준비 3초 후 시작합니다. 시작 시 양쪽에 최근 3초 내 수신이 있어야 합니다.
4. `머리 5초 → 휴식 3초 → 아프다 5초 → 다음 시도 전 휴식 3초`를 반복합니다.
   20조합은 수어 구간 40개이며, 준비 후 320초입니다. 각 구간은 별도로 지정되므로 자동 동작 분할이 아닙니다.
5. ‘중지’는 현재 파일을 보존합니다. 완료 전 중지한 구간과 미시작 구간은 계획표에 남으며 평가에서 제외됩니다.
   중단 후 재개로 구간을 이어 붙이지 않습니다. 새 실험은 새로운 폴더와 ID를 생성합니다.

기존 두 손 모니터는 그래프 표시·수신에만 사용하며, 이 실행 모드에서는 기존 ‘양손 20회 수집 시작’이 비활성화됩니다.
일반 `xiao_ble_dual_monitor(left,right)` 호출의 수집·CSV·서버 전송 기능은 그대로입니다.
하드웨어 실연결, 실제 수행 동작과 안내 표시의 일치, 두 보드 장시간 안정성은 아직 검증하지 않았습니다.

### 저장 구조와 서버 호환

새 출력은 `MATLAB_노트북용_수정본/sensor_data/sequences/seq_<UUID>/`에 생성합니다.

- `attempt_manifest.csv`: 참가자, 조합 시도 ID, 동작 순서 1/2, 실제 수어, CSV 위치, 세션/회차/손,
  동일 실험 시계 ID, 구간 시작·종료(ms), 수집 상태, 실제/가상 구분, 수집 시작 UTC 시각.
  측정 시작 전에 전체 시도 계획을 기록합니다. 테스트 시계를 주입한 수집 결과는 자동으로 synthetic을 기록하며,
  이를 실제 실험으로 표시하려 하면 평가기가 거부합니다.
- `window_001_L.csv`, `window_001_R.csv`, …: 각 구간의 양손 원본(v3, 기존 CSV 저장 함수 사용).
  두 손은 같은 구간 세션을 공유하고, 다음 수어는 별도 세션을 사용합니다.
- 각 CSV의 `host_elapsed_ms`와 `trial_elapsed_ms`는 **그 구간 내부 시간**입니다.
  조합 전체 순서는 manifest의 공통 `recording_id`와 `window_start_ms`로 연결합니다.
  장치 부팅 시간 `time_ms`로 서로 다른 보드/세션을 정렬하지 않습니다.

두 손 원본은 모두 보존하지만 이번에 확인한 머리·아프다 모델은 왼손 학습 모델이라 **왼손만 예측**합니다.
오른손을 0으로 채우거나 다른 모델을 대신 사용하지 않습니다. 양손 통합 머리·아프다 모델은 별도 학습/검증이 필요합니다.
수집 UI의 `complete`는 예정 구간의 시간이 지났다는 의미이며 품질 통과를 뜻하지 않습니다.

로컬 CSV로 평가하므로 서버 전송은 필수가 아닙니다. 기존 재전송 도구로 각 구간의 CSV를 서버에 보낼 수 있습니다.

```powershell
python resend_sensor_v2.py --endpoint http://100.92.15.45:5000/sensor
```

선택 창에서 보낼 CSV 하나를 선택합니다. 양손 파일 각각을 전송할 수 있습니다. **manifest도 별도 보관/공유해야 합니다.**
기존 `/sensor`는 센서만 저장하므로 서버 CSV만으로 조합 시도·순서를 복원하지 않습니다.
이번 수집기는 기존 서버의 ‘세션당 수어 하나’ 제약을 유지합니다. 자동 일괄 업로드나 새 서버 스키마는 추가하지 않았습니다.

## 평가 실행

학습과 평가를 분리하기 위해 평가기는 **predict만 호출**합니다. fit·재학습·기존 CSV 수정은 하지 않습니다.
수집 종료 후 실행하며, 출력 폴더는 매번 존재하지 않는 경로를 지정합니다.

이번 PC에서는 새 worktree에 모델 원본을 복사하지 않았으므로 다음처럼 원래 프로젝트 경로를 지정합니다.

```powershell
$projectRoot = 'C:\Users\sminl\OneDrive\바탕 화면\SignLanguage_Wearable_MERGED'
Set-Location -LiteralPath (Join-Path $projectRoot 'data\backups\sequence-feature')
python -m pip install -r requirements-sequences.txt
$model = Join-Path $projectRoot 'data\knn_results\2026-11-02_subject1_to_subject4\combined_model\knn_model.joblib'
$manifest = '.\MATLAB_노트북용_수정본\sensor_data\sequences\seq_실제UUID\attempt_manifest.csv'
python evaluate_sign_sequences.py --model $model --manifest $manifest --output '.\data\knn_results\sequence_new_person_001' --mode new_participant
```

모델을 저장했던 scikit-learn 환경에서 실행하세요. 산출물에는 실제 실행한 패키지 버전을 기록합니다.
별도 clone이라면 `$model`을 전달받은 실제 모델 경로로 바꾸고, 그 옆 `trial_features.csv`도 함께 받으세요.
현재 모델 클래스/특징이 맞지 않으면 결과를 만들지 않고 오류로 중단합니다.

### 참가자 관계 구분

| 옵션 | 의미 | 검사 |
|---|---|---|
| `--mode new_participant` | 모델 학습에 없던 참가자의 새 조합 | 학습 참가자와 불일치, 학습 세션과 겹치지 않아야 함 |
| `--mode same_participant_new_session` | 학습에 있던 참가자가 새 세션에서 반복 | 학습 참가자와 일치, 학습 세션과 겹치지 않아야 함 |
| `--mode exploratory` | 학습 이력 불명/겹침 등을 확인하는 탐색 | 독립 정확도라고 주장하지 않으며 겹친 세션을 기록 |

기본값은 `exploratory`입니다. 권장 120회 모델은 subject1·4를 학습했으므로 이 두 사람은
`same_participant_new_session`, 그 외 사람은 `new_participant`로 **실험별로 나눠 실행**하세요.
`subject-8`과 `subject8`은 같은 참가자로 취급합니다.

모델의 `training_sessions`와 옆의 `trial_features.csv`를 학습 이력으로 사용합니다.
최초 모델의 특징 파일에 학습 후보 60개가 있고 실제 fit은 45개인 경우, 세션 전체를 보수적으로 겹침 금지 대상으로 잡습니다.
다른 위치에 있으면 `--training-features 경로`로 지정할 수 있습니다. 세션·참가자 이력이 없으면
독립 평가 모드는 허용하지 않습니다. 이 검사는 사용자가 제공한 이력이 해당 모델의 실제 학습 이력이라는 전제입니다.

## 품질 기준과 정확도의 분모

시도별로 정확히 순서 1/2, 같은 참가자·시도 ID·실험 시계가 있어야 합니다. 실제 수어 순서와 계획된
5초·3초·5초 일정이 일치해야 하며 하나의 측정 구간을 여러 시도에 재사용할 수 없습니다.

한 구간이라도 중단/누락되거나 아래 기준을 어기면 **조합 전체**를 제외하고 사유를 저장합니다.

- 선택한 왼손 구간에 8행 이상, 모든 FSR·IMU 값과 시각이 유한해야 함. IMU 누락을 0/중앙값으로 숨기지 않음.
- 첫 샘플은 시작 500ms 이내, 마지막은 4500ms 이후·5000ms 이전, 내부 최대 누락 간격 500ms.
- 샘플 번호 중복·역행, 시간 역행·장치 재시작, 장치와 호스트 경과 시간 차이 500ms 초과를 제외.
- RAW에 실제 수어 열이 있다면 manifest의 정답과 일치해야 함. 정답 라벨은 모델 입력으로 쓰지 않음.

이 기준은 예측 결과를 보고 완화하지 않습니다. 실패한 예측은 제외 사유가 아닙니다.
오른손 파일은 보존하지만 현재 왼손 모델의 유효성 판단에는 사용하지 않습니다.

`첫 수어 정확도 = 첫 수어 정답 시도 수 / 유효 조합 시도 수`

`두 번째 수어 정확도 = 두 번째 수어 정답 시도 수 / 유효 조합 시도 수`

`조합 정확도 = 두 수어를 순서대로 모두 맞힌 시도 수 / 유효 조합 시도 수`

개별 정확도의 곱을 사용하지 않습니다. 유효 0회면 정확도는 JSON에서 null, 보고서/그림에서는 **미측정**입니다.

## 산출물과 포스터 사용

| 파일 | 내용 |
|---|---|
| `attempt_results.csv` | 참가자·시도·실제/예측1·2·개별 정답·조합 성공·변환 문장·유효성·제외 사유·모델 hash·평가 구분 |
| `summary.csv`, `summary.json` | 조합/개별 정확도, 전체·유효·제외·성공·실패 수, 제외 사유별 건수, 모델과 원본 hash, 버전·품질 기준 |
| `participant_summary.csv` | 참가자별 유효 수, 성공 수, 조합 정확도 |
| `manifest_snapshot.csv` | 사용한 조합 시도 연결표 원본 |
| `report.md` | 한국어 결과 요약 |
| `combination_accuracy.png` | 같은 유효 시도에 대한 첫 수어·두 번째·둘 다 정답 비율. 유효 시도/제외 수와 평가 구분 표시 |

규칙 변환은 예측 순서가 `머리 → 아프다`일 때 `머리가 아픕니다`를 출력합니다.
평가 성공 여부는 두 예측을 실제 수행 순서와 비교해 별도로 계산합니다. 생성형 AI는 사용하지 않습니다.
다른 문장 규칙은 `RULES`에 명시적으로 추가해야 합니다.

실제 실험에서는 참가자, 착용 위치(손목→팔꿈치 2/5), 센서 방향, 밴드 압박, 앉음/섬, 동작 영상 기준,
수집 일시를 함께 기록하세요. 기존 모델과 새 측정의 착용 조건 차이가 정확도에 영향을 줄 수 있습니다.
같은 사람 반복, 새 사람 결과를 합쳐 하나의 독립 성능으로 표시하지 마세요. 비교 후 모델/기준을 변경하면 새 평가 데이터를 확보하세요.

비어 있는 계획표로 실행 점검은 가능합니다. 이는 실제 실험이 아니며 정확도 **미측정**과 빈 결과만 생성합니다.

```powershell
python evaluate_sign_sequences.py --model $model --manifest docs/examples/sign_sequence_manifest_empty.csv --output data/knn_results/sequence_unmeasured_check
```

## 자동 테스트와 남은 검증

```powershell
python -B -m unittest discover -s tests -v
```

```matlab
results=runtests({'test_sign_sequence_collection','test_sensor_dual_metadata'});
assertSuccess(results);
```

Python의 가상 센서 값·가상 모델로 조합 성공/오인식, 누락·중단·중복·시간 오류,
학습 세션 겹침 차단, 새 참가자/동일 참가자 구분, 모델 호환성과 원본 보존을 검증합니다.
MATLAB은 가상 프레임과 테스트 시계로 일정, 양손 CSV, 연결표, 중단 보존을 검증합니다.
테스트 가상 결과는 `--data-kind synthetic`으로 표시하며 임시 폴더에서만 생성합니다.

실제 센서로 새 조합을 수행한 결과는 아직 없습니다. 다음 작업은 실제 양손 BLE 연결로 1조합을 먼저 점검한 뒤,
동일 조건으로 여러 참가자에게 20조합씩 수집하여 평가하는 것입니다. 자동 분할, 휴식/미등록 수어 거부,
양손 머리·아프다 모델, 서버에 조합 metadata 저장, 실시간 예측 표시는 별도 후속 작업입니다.
