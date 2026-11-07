# 분석·관리 도구

프로젝트 루트에서 `python tools/파일명.py ...` 형태로 실행한다. 기본 서버와 학습 진입점은 루트에 유지했다. 원본 데이터·모델은 정리 목적으로 옮기거나 삭제하지 않았다.

| 도구 | 용도 | 주의 |
| --- | --- | --- |
| `evaluate_combined_candidate.py` | 559회 통합 후보의 설정 고정 LOSO·훈련 재예측 분리 | 모델 원본 불변, 사후 내부 검증 |
| `tune_knn_with_new_people_holdout.py` | 기존 319회만 설정 선택 → 11/7 신규 4명 평가 → 559회 후보 학습 | 새 평가자료로 튜닝하지 않음, 후보 독립 성능 미정 |
| `compare_loso_sensor_inputs.py` | 동일 초기 제외 LOSO로 전체·왼손 FSR 제외·IMU 전용 비교 | 사후 진단, 행 삭제·모델 교체 없음 |
| `evaluate_app_loso.py` | 숫자 동일인 LOSO, 초기 포함·제외 비교와 후보 2개 저장 | 같은 사람의 모든 날짜를 함께 분리, 고정 k=3 |
| `evaluate_frozen_app_model.py` | 기존 8+10 모델로 새 6·9·1 재측정 평가 | fit 없이 실행, 새 파일만 선택, 모델 해시 검증 |
| `evaluate_standardized_cohort.py` | 오늘 subject8·10 교차 평가, 초기 1·2 추가 효과 비교, 8+10 후보 학습 | 착용 조건 확인 전 탐색 평가; 사람 분리·고정 설정 |
| `inspect_app_quality.py` | Git 커밋의 앱 CSV 회차·센서 품질검사 | 원본 읽기 전용, 학습 아님 |
| `compare_app_subjects.py` | 검사 결과 기준 사람별 센서 비교 | 품질 결과 폴더 필요 |
| `analyze_health_sessions.py` | 건강/건강(앉아서) 두 세션 분석 | 정밀 양손 시간 정렬 미검증 |
| `evaluate_app_then_fit.py` | 앱 subject-1 학습 → subject-2 평가 → 통합 후보 생성 | 현재 제외 목록 적용; 기존 출력 덮어쓰기 거부 |
| `evaluate_app_new_subject.py` | 앱 subject-1·2 학습 → 새 참가자 평가 | 새 참가자는 학습에 포함하지 않음 |
| `evaluate_then_update_knn.py` | 과거 MATLAB 특정 세션 평가 후 별도 학습 | 명시된 세션만 사용, 삭제 실험은 차단 |
| `analyze_knn_transfer.py` | 과거 subject2 재측정 전이 진단 | 사후 분석을 독립 평가로 해석하지 않음 |
| `package_dataset.py` | 기존 날짜의 공유용 고정 ZIP 생성 | 실시간 전체 데이터 패키저가 아님 |

결과는 `data/knn_results/`에 생성되며 Git에 자동 업로드하지 않는다. 학습 실행 전 [모델 현황](../실험기록/모델_현황.md)과 [데이터 선별 기준](../실험기록/데이터_선별_기준.md)을 확인한다.
