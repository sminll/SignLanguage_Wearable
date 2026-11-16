# 2026-11-16 센서 코드 보정 후보

[상세 검토 기록](센서_코드보정_검토.md)에 보정 방법, 전체·수어별·참가자별 결과와 한계를 정리했다.

삼키다·아니다·상처의 동일539회에서 FSR 크기 정규화 후보는 기존487회(90.35%)에서500회(92.76%)로 정답이 늘었다. 아니다는 개선됐지만 상처와 일부 참가자는 낮아졌다. 기존 결과를 본 뒤 같은 자료로 수행한 탐색적 비교이며, 새 데이터의 독립 검증이나 운영 모델 교체는 아니다.

## 근거 파일

- `comparison.csv`: 3종·6종에 적용한7가지 표현의 전체 결과.
- `per_sign.csv`, `per_person.csv`: 수어별·참가자별 결과.
- `predictions.csv`: 실제11,081개 예측(두 평가 묶음의 회차 합계1,583 × 표현7가지), 정답, 학습 참가자와 설정. 원본 동작 회차 수가11,081개라는 뜻은 아니다.
- `evaluation.json`: 표현별 혼동행렬과 요약.
- `calibration_inventory.csv`, `calibration_links.csv`, `calibration_audit.json`: 보정 파일56개와 엄격한 이전 휴식 연결 후보 검사. 검증되지 않은 휴식값은 적용하지 않았다.
- `comparison_plan.json`, `verification.json`, `test_validation.json`: 고정 입력, 처리·평가 방식과검증 결과.
- `재현입력/three_sign_trial_features.csv`: 이전3종의 기존 특징표를 바이트 그대로 보관한 입력. 센서 원시 CSV의 중복본이나 새로 보정한 측정값이 아니다.
- `파일_SHA256.json`: 이 폴더에 포함한 근거 파일의 바이트 해시.

## 실행

저장소 루트에서 실행한다. Python 의존성은 numpy, pandas, scikit-learn, matplotlib, threadpoolctl, joblib이며,원본은 고정Git커밋에서 읽는다. 아래 입력 복원은 `data/knn_results`에 해당 폴더가 없는 새 체크아웃에서만 수행한다. 기존 결과가 있으면 덮어쓰지 않는다.

```powershell
git fetch origin training-data
New-Item -ItemType Directory -Force data/knn_results | Out-Null
Copy-Item -LiteralPath 실험기록/2026-11-15/재현자료/2026-11-15_three_sign_combined_candidates -Destination data/knn_results -Recurse
Copy-Item -LiteralPath 실험기록/2026-11-15/재현자료/2026-11-15_six_sign_nested_loso -Destination data/knn_results -Recurse
Copy-Item -LiteralPath 실험기록/2026-11-16/재현입력/three_sign_trial_features.csv -Destination data/knn_results/2026-11-15_three_sign_combined_candidates/trial_features.csv
python -B tools/study_sensor_compensation.py --results-root data/knn_results --repo . --calibration-commit 88fed688ddc5cf3842ed2cfe8e257f0773be66e6 --out data/knn_results/sensor_compensation_reproduction
python -B tools/report_sensor_compensation.py --results data/knn_results/sensor_compensation_reproduction
python -B -m unittest discover -s tests -v
```

기존 출력은 덮어쓰지 않는다. 이번에92개 자동 테스트가 통과했으며 기존 센서 데이터·모델·BLE·펌웨어는 변경하지 않았다.
