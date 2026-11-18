# IMU 크기·동작 구간 정렬 재현

결과는 [2026-11-18 기록](../실험기록/2026-11-18/IMU크기_동작구간_비교/README.md)에 있다. 기존 6종·10명·1,170회와 11/16의 바깥 5-fold·내부 3-fold 분할을 그대로 사용한다. 원본, 수집 UI, BLE, 펌웨어, 서버와 기존 모델은 변경하지 않는다.

먼저 [KNN·SVM 재현 안내](group-model-study.md)의 명령으로 `data/knn_results/group_models_new`를 생성한다. 이 폴더에 기본 특징·분할·모델 설정·예측이 함께 만들어지므로 개인 PC의 백업 폴더가 필요 없다. 이어서 저장소 루트에서 실행한다.

```powershell
python -B tools/run_imu_motion_study.py --repo . --previous data/knn_results/group_models_new --out data/knn_results/imu_motion_new
python -B tools/report_imu_motion_study.py --results data/knn_results/imu_motion_new --out data/knn_results/imu_motion_report_new
```

첫 도구는 이전 원본 특징과 예측이 재현되는지 확인하고, 크기 추가/동작 구간 정렬/둘 다 적용/원래 특징을 비교한다. 센서·전처리·SVM 설정을 유지한 통제 비교와 각 특징 안에서 설정을 다시 고른 비교를 구분한다. 원래 특징 473/1,170=40.43%와 이전 설정을 유지한 크기+정렬 후보 489/1,170=41.79%가 재현 기준이다. 특징 종류까지 자동 선택한 전체 절차는 473/1,170=40.43%였다. 41.79%는 여러 비교 중 발견한 후보로, 새 데이터의 확정 성능이 아니다.

IMU 크기는 가속도·자이로 벡터의 크기를 추가하며 축별 특징도 보존한다. 구간 정렬은 한 동작 안에서 지속적인 자이로 활동의 처음과 끝에 여유 구간을 붙이고, 품질 조건을 만족하는 창 안에서만 32점으로 재표본화한다. 정지 손·순간 튐·너무 짧은 창·큰 간격은 원래 샘플로 되돌린다. 임계값은 시험 결과를 보고 조정하지 않았다.

검증된 휴식값이 없는 기록에는 휴식 보정을 적용하지 않는다. 이 처리는 측정이 끝난 고정 구간의 특징 추출이며 실시간 연속 인식, 양손 절대 시각 동기화, 영상으로 검증한 시작/끝 검출을 의미하지 않는다.

전체 테스트는 이 저장소 루트에서 실행한다. PowerShell에서 UTF-8 테스트 로그를 만드는 예:

```powershell
python -B -c "import subprocess; from pathlib import Path; r=subprocess.run(['python','-B','-m','unittest','discover','-s','tests','-v'],capture_output=True,text=True,encoding='utf-8',errors='replace'); Path('test_run.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8'); print(r.stderr); raise SystemExit(r.returncode)"
python -B tools/verify_imu_motion_study.py --repo . --results data/knn_results/imu_motion_new --previous data/knn_results/group_models_new --test-log test_run.log --out data/knn_results/imu_motion_new/independent_verification.json
```

검사 도구는 81개 자동 테스트 통과, 같은 분할과 이전 예측, 선택 기준, 11,700개 예측·정답 수·혼동행렬, 입력/코드 해시를 다시 확인한다. 검사는 새 출력 파일을 요구한다. 테스트를 추가하거나 라이브러리를 바꾸면 원래 재현 검증 조건과의 차이를 별도로 기록한다.

게시된 JSON의 실험 당시 해시/절대 경로는 실행 당시의 근거이며, 새 PC에서는 새 manifest의 실제 경로가 달라진다. 발표용 해석에는 기록의 같은 조건 비교와 미측정 범위를 함께 사용한다. 기존 운영 모델은 교체하지 않았다.
