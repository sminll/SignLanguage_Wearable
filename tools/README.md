# 분석·관리 도구

프로젝트 루트에서 `python tools/파일명.py ...` 형태로 실행한다. 기본 서버와 학습 진입점은 루트에 유지했다. 원본 데이터·모델은 정리 목적으로 옮기거나 삭제하지 않았다.

| 도구 | 용도 | 주의 |
| --- | --- | --- |
| `inspect_app_quality.py` | Git 커밋의 앱 CSV 회차·센서 품질검사 | 원본 읽기 전용, 학습 아님 |
| `compare_app_subjects.py` | 검사 결과 기준 사람별 센서 비교 | 품질 결과 폴더 필요 |
| `analyze_health_sessions.py` | 건강/건강(앉아서) 두 세션 분석 | 정밀 양손 시간 정렬 미검증 |
| `evaluate_app_then_fit.py` | 앱 subject-1 학습 → subject-2 평가 → 통합 후보 생성 | **오늘 120회차는 학습 제외로 현재 실행 차단** |
| `evaluate_then_update_knn.py` | 과거 MATLAB 특정 세션 평가 후 별도 학습 | 명시된 세션만 사용, 삭제 실험은 차단 |
| `analyze_knn_transfer.py` | 과거 subject2 재측정 전이 진단 | 사후 분석을 독립 평가로 해석하지 않음 |
| `package_dataset.py` | 기존 날짜의 공유용 고정 ZIP 생성 | 실시간 전체 데이터 패키저가 아님 |

결과는 `data/knn_results/`에 생성되며 Git에 자동 업로드하지 않는다. 학습 실행 전 [모델 현황](../실험기록/모델_현황.md)과 [데이터 선별 기준](../실험기록/데이터_선별_기준.md)을 확인한다.
