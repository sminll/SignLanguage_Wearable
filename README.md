# 3종 데이터 전처리·방향 보정 결과

브랜치: `training-data-preprocessed`. 2026-11-23 기준 현재 앱의 삼키다·아니다·상처 725회 전용입니다. 원본 전체는 training-data-raw, 3종 원본은 training-data-3signs에 있습니다. 세 브랜치는 고정 복사본이며 training-data 업로드가 자동 반영되지 않습니다. 이 데이터 전용 브랜치는 main이나 원본 수집 브랜치로 통째로 병합하지 않습니다.

| 경로 | 내용 |
|---|---|
| sensor-data/processed/ | 725개 양손 CSV. 가속도·자이로를 같은 중력 기준 회전으로 변환. FSR은 원본 값 유지 |
| features/gravity_features.csv | 725행, 메타데이터 + f000~f531 특징 532개. 아직 StandardScaler를 적용하지 않은 특징 |
| features/app_standardized_features.csv | 같은 특징에 앱의 고정 StandardScaler를 적용한 값 |
| preprocessing/app_scaler.json | 앱 모델에 저장된 평균·scale, fit 범위 |
| preprocessing/sign_gravity_model.py | main `e03386b8ad24185c7679ed3d58d06db2caebf4c2`의 실제 특징 계산 코드 |
| quality_checks.csv | 회차별·손별 샘플 수, 시간 길이·최대 간격·IMU 오류 비율, 추정 중력 기준 |
| data_manifest.csv | 정규화한 참가자·수어명과 원본 파일 경로·SHA256 |
| snapshot_info.json | 원본/코드/앱 커밋, 적용·비적용 처리 목록 |

10명·725회: 삼키다 260, 아니다 226, 상처 239. 처리 순서는 품질 검사 → 손별 중력 기준 추정 → 가속도·자이로 방향 보정 → 구간 통계·동작 구간·12점 재표본화 → 왼손 266개 + 오른손 266개입니다. 선택적으로 별도 파일에서 앱 StandardScaler를 적용했습니다. `f000~f265`는 왼손, `f266~f531`은 오른손입니다. 각 손의 266개 구성은 전체/3분할 통계 96개 + 움직임 구간 전체/3분할 통계 96개 + 비율 2개 + 12시점×6채널 72개입니다. 채널 순서는 A_z, 수평 가속도 크기, W_z, 수평 자이로 크기, 기준 가속도 대비 변화 크기, 자이로 크기입니다. 통계는 평균→표준편차(ddof=0)→최솟값→최댓값입니다.

방향 보정 CSV의 센서 숫자는 원본과 다릅니다. 추가 열 processing_version, raw_source_sha256로 구분합니다. time·보드 time_ms·sample_seq·FSR·참가자·세션·정답 라벨은 원본 의미를 유지했습니다. ‘아니다’ 별칭은 특징/목록에서 통합하되 방향 보정 CSV의 원래 label은 보존했습니다. 원본의 조임·착용 위치 차이를 완전히 제거하거나 FSR을 실제 힘 단위로 바꾼 데이터가 아닙니다.

휴식값 빼기·약한 필터·FSR 정규화는 적용하지 않았습니다. 기준 휴식값이 없는 회차에 임의 값을 빼지 않았습니다. 여러 후보 보정을 한꺼번에 적용한 결과가 아니라 실제 앱과 같은 처리입니다.

새 K-fold/LOSO 평가에는 gravity_features.csv를 사용하고 매 fold의 학습 자료로만 StandardScaler를 fit하세요. app_standardized_features.csv는 725회 전체로 fit한 배포용 앱 기준을 재현하기 위한 파일이며 이를 그대로 교차검증하면 평가 자료 정보가 섞입니다. 이 브랜치는 정확도 결과를 새로 만들거나 보장하지 않습니다. 참가자 번호와 회차 단위가 학습/평가에 섞이지 않도록 분리하세요.

재현 입력은 원본 커밋 `0f4b1eef59723ea42ebebe320ed6094db289bb4d`, 앱 모델은 `cebb6461102b75563c4eadb31f960c599c0631a6`의 android-app/models/sign_model.json입니다. 삼키다·상처와 아니다의 측정 날짜가 달라 같은 날 세 수어를 측정하는 후속 검증이 필요합니다. 미래의 새 사람·재착용·실제 앱 정확도는 이 파일 생성으로 검증되지 않습니다.
