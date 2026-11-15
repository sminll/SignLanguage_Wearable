# 2026-11-15 XIAO 수어 KNN 실험 자료

## 포스터에 사용할 자료

- [포스터용 혼동행렬 PNG](삼키다_아니다_상처_공유자료/04_혼동행렬_포스터용.png)
- [인쇄용 SVG](삼키다_아니다_상처_공유자료/04_혼동행렬_포스터용.svg)
- [포스터 오른쪽 설명 및 팀원 전달 문구](포스터_설명과_공유문구.md)
- [3종 통합 평가 보고서](삼키다_아니다_상처_통합_KNN.md)

선정한 삼키다·아니다·상처를 같은 분류기에 넣고, 한 참가자의 모든 회차를 학습 및 설정 선택에서 제외하는 평가를 10명에게 반복했다. 양손 FSR+IMU를 사용한 결과는 **487/539회, 90.35%**이다. 수어별 정답률은 삼키다 179/180회(99.44%), 아니다 131/180회(72.78%), 상처 177/179회(98.88%)다.

기존 결과를 참고해 수어를 선정한 탐색적 내부 교차검증이다. 삼키다·상처는 11월 4·7일, 아니다는 13~15일에 측정했다. 수어와 날짜·착용 조건의 영향을 분리한 새 외부 검증은 아니다. 기존 교통사고·삼키다·상처의 73.88%와는 수어 구성·센서 입력·설정 선택 절차가 달라 알고리즘 개선률로 비교하지 않는다. 부분 수어만 측정한 subject5·subject7도 평가에 포함했다.

## 함께 보관한 분석

| 자료 | 내용 |
|---|---|
| [6종 중첩 평가](6종_KNN_중첩평가.md) | 괜찮다·아니다·무엇·마시다·화장실·싫다, 9명 1,044회 |
| [원인 분석](6종_KNN_원인분석.md) | 같은 사람의 다른 회차와 학습에서 제외한 사람 비교, 상대 시간 구간 특징 비교 |
| [수어별 센서값 비교](센서값_일관성_직접비교.md) | 원본 1,603회에서 사람 간 FSR·IMU 파형 차이를 직접 비교한 기술 통계 |
| [3·4·5종 비교](3종_4종_5종_수어구성_비교.md) | 기존 3종에 아니다 및 추가 수어를 넣은 7개 구성 비교. 가능한 모든 조합의 전수 탐색은 아님 |

낮은 정답률이나 한 손 FSR 0을 이유로 이번 자료에서 참가자를 임의 제외하지 않았다. 0 기록 검사는 '0에 가까운 값'의 센서 정상 여부를 판정한 검사가 아니다. 합성 자동 테스트는 실제 정확도에 포함하지 않는다.

## 계산 근거와 재현

`재현자료/`에는 기존 실험 입력 특징표·원본 해시 목록, 실제 예측, 설정 선택 기록을 저장했다. 센서 원본 CSV의 중복본이나 모델 바이너리는 포함하지 않는다. 모델은 기존 파일을 유지하며 필요하면 평가 코드로 별도 생성한다.

예측의 `nested_selection`이 주 결과다. 3·4·5종 구성별 `primary_outer_predictions.csv`는 전체 저장 예측 중 이 주 결과만 추린 파일이다. 다른 센서 비교 수치는 각 `evaluation.json`에 있다.

저장소 루트에서 PowerShell로 실행한다. 입력 폴더가 이미 있으면 복사하지 않고 새 체크아웃에서 진행한다.

```powershell
git fetch origin training-data
python -B -m unittest discover -s tests -v
# 저장된 3종 예측으로 포스터 그림만 다시 생성 (재학습하지 않음)
python -B tools/plot_three_sign_poster.py --results 실험기록/2026-11-15/재현자료/2026-11-15_three_sign_combined_candidates

# 새 체크아웃에서 3종 학습·평가를 재현할 입력 복원
New-Item -ItemType Directory -Force data/knn_results | Out-Null
Copy-Item -LiteralPath 실험기록/2026-11-15/재현자료/2026-11-08_knn_development_holdout -Destination data/knn_results -Recurse
Copy-Item -LiteralPath 실험기록/2026-11-15/재현자료/2026-11-15_six_sign_nested_loso -Destination data/knn_results -Recurse
python -B tools/evaluate_three_sign_candidates.py --results-root data/knn_results --repo . --out data/knn_results/three_sign_reproduction
python -B tools/report_three_sign_candidates.py --results data/knn_results/three_sign_reproduction
```

Python 의존성과 그 밖의 분석 실행법은 [평가 안내](../../docs/six-sign-knn-evaluation.md)를 참고한다. 원본은 고정 Git 커밋에서 읽어 해시를 검사한다. `재현자료/파일_SHA256.json`은 이번에 묶은 근거 파일의 해시다. 일부 원래 기록의 경로 표기는 Windows 기준이다.

현재 82개 자동 테스트가 통과했다. 실제 새 참가자·재착용·다른 날짜의 성능과 자동 동작 구간 분할 및 문장 조합 성능은 이번 점수에 포함하지 않는다.
