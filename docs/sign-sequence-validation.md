# 조합 평가 기능 구현·검증 기록 (2026-11-12)

실제 조합 정확도: **미측정**. 아래 테스트 통과 수는 인식 정답률이 아닙니다.

기준: `origin/main e0b65dc` → `feature/fixed-window-sign-combinations`.
원래 작업 폴더의 미커밋 변경을 보존하기 위해 `data/backups/sequence-feature` worktree에서 구현했습니다.

## 기존 자료 확인

`data/sensor_dataset_v2.csv`를 읽어 확인한 REAL·왼손 동작 데이터입니다. 휴식은 아래 행 수에 포함하지 않습니다.

| 데이터 | 갑자기 행/회차 | 머리 행/회차 | 아프다 행/회차 |
|---|---:|---:|---:|
| 11/1 subject1 | 991 / 20 | 992 / 20 | 995 / 20 |
| 11/2 subject4 | 994 / 20 | 992 / 20 | 995 / 20 |
| 11/2 subject2 재측정 | 991 / 20 | 995 / 20 | 993 / 20 |

이는 수어별 독립 측정 자료이며 조합 시도로 사후 연결하지 않았습니다.
최근 교통사고·삼키다·상처 후보는 양손 IMU 입력 모델이며 위 세 수어를 지원하지 않습니다.

실제 모델을 읽어 지원 클래스와 특징을 확인했습니다. 이번 기준 모델:

- 경로: `data/knn_results/2026-11-02_subject1_to_subject4/combined_model/knn_model.joblib`
- 클래스: 갑자기, 머리, 아프다. 왼손 FSR·IMU 48특징, fit 120회.
- 학습 세션: subject1·4의 6개 세션. 별도 새 조합 실험을 학습하지 않음.
- 모델 SHA256: `8dd6f1d5b6b89b4fb06fd2b02961a0c9cec78f9686a3f1c49dc71dbcec9b8806`
- 학습 특징 CSV SHA256: `d7d87c6b7fb763fdb4076aca019fd2da64f08ceb85f1bce69335e0d65f6e15b5`

기존 Nov1 모델도 갑자기·머리·아프다를 지원하며 SHA256은
`f0321e20d3cc8afe16073ba5c17ebb4383c2280f94f9219a04ba229796cbaf9a`입니다.

## 새 기능과 파일

| 파일 | 역할 |
|---|---|
| `evaluate_sign_sequences.py` | 고정 모델 예측, 명시적 조합 연결, 품질/겹침 검사, CSV·JSON·한국어 보고서·그래프 |
| `MATLAB_노트북용_수정본/sign_sequence_phase.m` | 두 수어 5초+휴식3초 일정 계산 |
| `MATLAB_노트북용_수정본/sign_sequence_collection.m` | Subject 선택, 조합 계획표·양손 원본 저장, 중지 보존 |
| `MATLAB_노트북용_수정본/xiao_ble_sequence_experiment.m` | 기존 두 장치 BLE 모니터와 새 수집기 연결 |
| `MATLAB_노트북용_수정본/xiao_ble_dual_monitor.m` | 선택적 프레임 전달 callback 추가. 연결·패킷 해석은 기존 함수 그대로 |
| `tests/test_sign_sequences.py` | 가상 데이터 기반 Python 테스트 23개 |
| `MATLAB_노트북용_수정본/test_sign_sequence_collection.m` | 새 수집기 테스트 3개 |
| `MATLAB_노트북용_수정본/test_sensor_dual_metadata.m` | 기존 수집 회귀 검사 + 새 observer 경로 검사 |
| `requirements-sequences.txt` | 평가 도구의 Python 의존성 |
| `docs/examples/sign_sequence_manifest_empty.csv` | 정확도 미측정 동작 확인용 빈 양식 |
| `docs/sign-sequence-evaluation.md`, `README.md`, 이 문서 | 실행 방법, 모델 구분, 검증 기록 |

## 실행한 검증

- Python `python -B -m unittest discover -s tests -v`: **74개 통과** (기존 51 + 신규 23).
- MATLAB R2025b `runtests({'test_sign_sequence_collection','test_sensor_dual_metadata'})`: **6개 통과**.
  가상 프레임 사용. 왼손/오른손 원본, 1·2 순서, 5초·3초 일정, 중지 보존,
  타이머 저장 중복 방지, 기존 세 유형 수집, 새 observer 전달을 확인했습니다.
- 실제 기존 120회 모델 + 빈 manifest: 유효 0회, 성공 0회, 정확도 null/미측정.
  로컬 `data/knn_results/sequence_unmeasured_check/`에 빈 결과·보고서·미측정 그림을 생성했습니다.
- 기존 모델·CSV를 수정하거나 삭제하지 않았으며 새 학습/배포도 하지 않았습니다.
- `git diff --check` 통과.

Python 실행 환경: Python 3.15.2, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, joblib 1.6.0.
결과 JSON에는 실행 시점의 실제 라이브러리 버전·코드·모델·입력 해시가 추가로 저장됩니다.

## 아직 검증하지 않은 부분과 다음 단계

실제 두 XIAO의 BLE 안정성, 안내 구간과 실제 동작의 일치, 새로운 착용 조건에서의 머리·아프다 인식 성능은 미검증입니다.
먼저 실제 1조합을 수집·평가하여 양쪽 CSV와 안내 시간을 확인하고, 이후 여러 참가자에게 20조합씩 수집하세요.
신규 참가자와 기존 참가자의 새 세션은 평가 모드를 나눕니다. 기존 모델에서 subject1·4는 학습 참가자입니다.

현재는 수집 후 평가입니다. 실시간 예측 화면, 자동 구간 분할, 미등록 수어·휴식 거부,
양손 머리·아프다 모델과 서버의 조합 메타데이터 저장은 후속 구현 대상입니다.
