# 데이터 업로드 → 디스코드 알림

GitHub에 공유 데이터를 push하면 **Data upload check**가 기본 검사를 실행하고 디스코드 채널에 결과를 보냅니다. 별도 봇이나 상시 실행 서버는 필요 없습니다.

앱에서 업로드한 CSV는 **`training-data` 브랜치의 `sensor-data/training/`**에 저장됩니다. 이 경로도 자동 검사·알림 대상입니다. 앱에서 GitHub로 직접 업로드하는 경우 PC의 `receive_sensor.py` 실행 여부와 관계없습니다. 해당 브랜치에도 workflow와 검사 스크립트를 설치해야 하며, 현재 `main`과 `training-data`에 반영합니다.

## 처음 한 번 설정하기

1. 팀 디스코드 서버에 `데이터-알림` 텍스트 채널을 만듭니다.
2. 채널 설정 → **연동(Integrations) → 웹훅(Webhooks) → 새 웹훅**에서 웹훅을 만들고 URL을 복사합니다. 웹훅 관리 권한이 필요합니다.
3. [저장소의 Actions Secrets 설정](https://github.com/sminll/SignLanguage_Wearable/settings/secrets/actions) → **New repository secret**을 누릅니다.
4. Name은 `DISCORD_WEBHOOK_URL`, Secret은 복사한 URL로 저장합니다. URL은 채널에 글을 쓸 수 있는 비밀 주소이므로 코드·커밋·스크린샷에 넣지 않습니다.
5. 이 문서와 함께 추가된 `.github/workflows/data-discord.yml`, `tools/data_upload_report.py`를 GitHub에 올립니다. 테스트 파일도 함께 올려두면 이후 수정 시 확인할 수 있습니다.
6. 기본 브랜치에 workflow를 반영한 뒤 [Actions](https://github.com/sminll/SignLanguage_Wearable/actions) → **Data upload check → Run workflow**로 첫 알림을 확인합니다. 수동 실행은 현재 브랜치의 Git 추적 데이터 전체를 검사합니다.

수동 실행 시 브랜치를 `training-data`로 선택하고 `dataset_path`에 `sensor-data/training/사용자/파일.csv`를 입력하면 그 파일만 다시 검사·알림합니다. 경로를 비우면 선택한 브랜치 전체를 검사합니다. 수동 재검사도 새 알림을 보내므로 과거 업로드가 새로 발생했다는 의미는 아닙니다.

연결 전에는 로컬 검사만 완료된 상태입니다. Secret이 없으면 검사와 보고서 저장은 실행되지만 디스코드 전송은 생략되며 Actions 로그에 안내가 남습니다.

## 알림 대상

- 앱 업로드: `training-data` 브랜치의 `sensor-data/training/` 아래 CSV
- `datasets/` 아래 새로 추가되거나 수정·삭제된 ZIP 및 센서 CSV
- `data/sensor_dataset_v숫자.csv` (v2, v3 등) 중 **이미 Git에 올려진 파일**의 변경
- `session_devices.csv`는 센서 CSV 검사가 아니므로 직접 알림 대상에서 제외합니다.
- push에 여러 커밋이 포함되면 마지막 커밋 하나가 아닌 push 전체의 변경을 감지합니다. 삭제도 알리고, 이름 변경은 기존 파일 삭제와 새 파일 추가로 표시합니다.

이 저장소는 PC에서 수집하는 실시간 `data/*.csv`와 생성 모델을 대부분 `.gitignore`로 제외합니다. **PC에서 수신 서버에 데이터가 들어오기만 한 경우에는 알림이 없습니다.** 이 방식의 새 측정은 기존 방침대로 `datasets/`에 공유용 스냅샷을 만들어 GitHub에 올리세요. 앱에서 `sensor-data/training/`으로 직접 업로드하는 방식은 그 업로드 자체로 알림이 실행됩니다. 알림을 위해 로컬 원본 측정 파일을 강제 추가할 필요는 없습니다.

## 검사 내용과 의미

- CSV: 필수 컬럼·식별값, 행의 열 수, left/right, 센서의 유한한 수 여부, FSR 0~4095, 시간의 음수 여부, 샘플 번호 중복, 세션 메타데이터 충돌을 검사합니다.
- v3 CSV: 동작 유형, 회차와 공통 시간의 관계, 5초 측정 구간을 추가로 검사합니다.
- ZIP: 읽기/CRC, 파일 목록과 `manifest.json`의 SHA-256 일치 여부, 통합 CSV의 구조·manifest 집계를 확인합니다. 압축을 해제하거나 포함된 코드·모델을 실행하지 않습니다.
- ZIP은 기존 `tools/package_dataset.py` 형식처럼 `manifest.json`과 `data/sensor_dataset_v*.csv` 통합본을 정확히 하나씩 포함해야 합니다. 총 압축 해제 크기는 512 MiB, 항목은 10,000개까지입니다.
- ZIP 안 날짜별 CSV는 통합본의 부분집합이므로 센서 수량은 **통합본만 집계**합니다. 날짜별 CSV 등 다른 항목은 파일 무결성을 확인합니다.
- 과거 데이터의 빈 `sample_seq`, 누락 IMU, REAL이 아닌 행, v3 한쪽 손 누락은 경고로 알립니다. 과거 한 손 v2 데이터에 두 손을 강제하지 않습니다.
- 수량은 유효한 REAL 행을 기준으로 집계합니다. TEST 등 REAL이 아닌 행은 경고를 남기고 측정 집계에서 제외합니다. v2는 같은 세션에 여러 라벨을 허용하고, v3는 기존 수신 규약대로 라벨·동작 유형을 고정합니다.

**✅ 기본 검사 통과**: 검사 범위의 오류·경고 없음. **⚠️ 확인 필요**: 경고가 있으니 확인. **❌ 오류**: 구조나 값에 오류가 있고 Actions 검사도 실패합니다. 실패한 데이터도 디스코드로 결과를 보냅니다.

알림의 행 수는 **해당 파일 전체 행 수**입니다. 이번에 새로 추가한 행 수가 아니며, 서로 중복되는 여러 ZIP/CSV의 수량을 합산하지 않습니다. 업로드 계정은 GitHub에서 push한 계정이며 실제 측정자와 다를 수 있습니다. 알림에는 subject_id나 센서 원본값을 싣지 않습니다.

이 검사는 학습 가능 여부·기대 회차 충족·양손 시간 정렬·동작 정확도를 보증하지 않습니다. 기존의 분석 및 학습 품질 검사는 별도로 진행하세요.

## 팀원이 보는 내용

```text
⚠️ 데이터 확인 필요
업로드 계정 / 브랜치: teammate / main
검사 대상 1개 · 오류 0건 · 경고 1건

datasets/새_스냅샷.zip
총 1,000행 / 유효 1,000행 · 왼손 500 / 오른손 500
세션 2개 · 회차 20개
라벨: 안녕하세요 1,000행
10행: 일부 IMU 값이 없음
```

알림 제목을 누르면 Actions 결과로 이동합니다. **Summary**에 파일별 집계가 있고 **Artifacts → data-inspection**에서 전체 오류·경고 보고서(JSON, 각 파일 최대 12개 예시)를 내려받을 수 있습니다. 보고서는 14일 보관합니다. 비공개 저장소 링크는 GitHub 접근 권한이 있는 팀원만 열 수 있습니다. 휴대폰 푸시 알림은 각 팀원의 디스코드 채널·앱 알림 설정을 따릅니다.

## 로컬에서 데이터만 검사하기

Python 표준 라이브러리만 사용하며 데이터를 수정하거나 메시지를 보내지 않습니다.

```powershell
python tools/data_upload_report.py datasets/2026-11-01_v1.zip --report "$env:TEMP/data-report.json"
```

오류면 종료 코드 1, 경고만 있거나 통과하면 0입니다. 인자 없이 실행하면 Git 추적 대상 전체를 검사합니다. GitHub에 올라가지 않은 로컬 파일은 경로를 직접 지정해야 검사합니다.

디스코드 URL을 GitHub Secret에 등록한 뒤에도 알림이 오지 않으면 Actions의 **Notify Discord** 로그를 확인하세요. 삭제·재생성한 웹훅은 Secret의 URL도 갱신해야 합니다. HTTP 429는 최대 두 번 재시도하며, 다른 전송 실패는 해당 step을 실패 처리합니다. 수동 재실행하면 같은 데이터의 알림을 다시 보냅니다.

공식 안내: [Discord 웹훅](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks), [GitHub Secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).
