# 삼키다·아니다·상처 전용 원본 데이터

브랜치: `training-data-3signs`. 기준 날짜 2026-11-23, 원본 training-data 커밋 `0f4b1eef59723ea42ebebe320ed6094db289bb4d`의 고정 복사본입니다. 원본 브랜치를 수정하지 않았으며 자동 갱신하지 않습니다.

| 구분 | 브랜치 | 내용 |
|---|---|---|
| 수집 원본 | training-data | 팀 앱이 계속 올리는 원본 |
| 원본 전체 복사본 | training-data-raw | 기준 커밋 전체와 동일한 스냅샷 |
| 3종 원본 | training-data-3signs | 현재 폴더. 3종 725회 원본 CSV |
| 3종 전처리 결과 | training-data-preprocessed | IMU 방향 보정 CSV·특징·정규화 특징 |

10명·725회: 삼키다 260, 아니다 226, 상처 239. `sensor-data/training/`에 원래 경로·파일명·전체 바이트 그대로 넣었습니다. 별칭 ‘아니다 (아니, 부인(하다))’는 목록을 집계할 때만 아니다로 통합했고, 원본 CSV의 label이나 센서값은 바꾸지 않았습니다. 세 수어에 해당하는 원본 725개가 모두 양손 품질 검사를 통과했습니다. 수집 전체에는 다른 수어·과거 한 손 기록도 있으며 해당 자료는 원본/전체 복사본에 남아 있습니다.

`data_manifest.csv`는 참가자·세션·회차·측정일·통합 수어명·원본 경로·SHA256입니다. 회차 하나는 양손 행이 들어 있는 CSV 하나입니다. CSV를 센서 행 단위로 임의 분할하면 같은 회차가 학습과 평가에 섞일 수 있습니다.

이 브랜치는 데이터를 나눠 보관하기 위한 것이며 모델이나 새로운 정확도 결과를 포함하지 않습니다. 삼키다·상처와 아니다의 측정 날짜가 달라 같은 날 세 수어를 측정한 후속 검증이 필요합니다. 원본을 보려면 [training-data](https://github.com/sminll/SignLanguage_Wearable/tree/training-data), 고정 원본 전체는 [training-data-raw](https://github.com/sminll/SignLanguage_Wearable/tree/training-data-raw)를 확인하세요.
