# 6종 수어 KNN·SVM 비교 재현

실험 기록은 [2026-11-16 결과](../실험기록/2026-11-16/KNN_SVM_시간특징_5fold/README.md)에 있다. 10명·1,170회에서 참가자를 두 명씩 시험하는 바깥 5-fold와 설정을 선택하는 내부 사람별 3-fold를 사용한다. LOSO 또는 같은 사람의 회차를 섞은 일반 K-fold와 구분한다.

원본 CSV나 기존 모델을 변경하지 않는 독립 분석 도구다. 게시된 회차 명단과 Git 원본 커밋에서 중간 특징표를 복원하므로 개인 PC의 `data/backups` 경로가 필요 없다. 실험 때만 존재했던 이전 결과 폴더를 별도로 전달받을 필요도 없다.

저장소 루트의 PowerShell에서 실행한다. Python과 numpy, pandas, scikit-learn, threadpoolctl, matplotlib가 필요하다. 실행 버전은 결과의 manifest에 기록하며, 실험은 scikit-learn 1.9.1로 수행했다.

```powershell
git fetch origin training-data
python -B tools/rebuild_group_study_inputs.py --repo . --cohort 실험기록/2026-11-16/KNN_SVM_시간특징_5fold/cohort_manifest.csv --out data/knn_results/group_inputs_new
python -B tools/run_group_model_study.py --repo . --cohort data/knn_results/group_inputs_new/cohort_manifest.csv --reference-raw data/knn_results/group_inputs_new/raw_features.csv --reference-denoise data/knn_results/group_inputs_new/denoise_features.csv --out data/knn_results/group_models_new
python -B tools/report_group_model_study.py --results data/knn_results/group_models_new --out data/knn_results/group_models_report_new
```

출력 폴더는 항상 새 이름을 사용한다. 첫 도구는 커밋 `4d4d24a4d672c78d934bf120a59a193c2340637e`의 명단별 파일 해시·양손·XIAO/REAL·시간·센서를 검사하고 원시/필터 전체 통계를 만든다. 정확도를 계산하거나 모델을 저장하지 않는다.

두 번째 도구는 같은 Git 원본에서 전체·앞/중간/뒤 특징을 추출하고 KNN/SVM을 평가한다. StandardScaler와 설정 선택은 각 학습 부분에만 적용한다. 원시/약한 필터/손별 FSR 크기 정규화와 IMU/FSR/전체 센서 등을 포함한 768개 후보를 내부 검증에서 비교한다. 모든 참가자는 학습에서 제외된 상태로 한 번씩 자기 기록을 시험받는다.

재현 기준은 고정 IMU KNN 416/1,170=35.56%, SVM+시간 특징 473/1,170=40.43%다. 알고리즘·특징을 자동 선택한 전체 절차는 396/1,170=33.85%였다. 자료를 여러 번 분석한 탐색 결과이며 새 사람·재착용 세션의 확정 성능이 아니다. 결과가 다르면 데이터 커밋/해시, 라이브러리 버전, 설정부터 비교한다.

전체 테스트: `python -B -m unittest discover -s tests -v`. 가상 테스트 점수는 실험 결과에 포함하지 않는다. 원본·중간 특징표·모델은 이번 게시물에 중복 업로드하지 않는다. 결과 CSV는 실제 예측과 평가 근거다.

PR #8 FSR 일반 K-fold를 재현하려면 `docs/kfold-validation.md`를 참고한다. PR의 85.02%는 같은 사람/세션의 회차 시험이며 위 사람별 점수와 직접 비교하지 않는다.
