# 수어 업로드 현황과 디스코드 요약 알림

앱의 `training-data` 브랜치 `sensor-data/training/` 업로드를 기준으로, **측정자·세션·한국 날짜별 메시지 한 개**에 여러 수어의 업로드 횟수를 표시합니다. PC 수신 서버를 켜지 않아도 앱에서 GitHub로 직접 업로드할 수 있습니다.

## 실제 동작

1. 여러 CSV가 연속 업로드되면 마지막 변경 이후 **60초의 조용한 구간**을 기다립니다. GitHub 실행 대기 시간은 별도로 걸릴 수 있습니다.
2. 해당 브랜치의 최신 Git 스냅샷을 확인해 누적 수량과 오류를 계산합니다.
3. 처음 확인된 수집 건에는 요약 메시지를 한 번 보내고, 같은 수집 건의 추가 업로드·수정·삭제는 기존 메시지를 갱신합니다.
4. 동일 결과 재실행은 전송을 생략합니다. 삭제된 디스코드 메시지를 복구할 때는 무음 메시지로 재생성합니다. 메시지 수정으로 반복 새 글을 만들지 않으며 @everyone 등 멘션을 넣지 않습니다.

평소처럼 한 사람이 같은 세션에 세 수어를 20회씩 업로드하면 **60개 파일 → 요약 메시지 1개**입니다. 중간에 60초 이상 쉬더라도 같은 세션·날짜라면 이후 수량을 기존 메시지에 반영합니다. 사람이나 세션·측정 날짜가 다르면 별도 수집 건입니다.

진행 중 작업을 새 업로드가 강제로 취소하지 않습니다. 대기 실행은 하나만 유지하며, 이후 실행은 저장된 기준점과 최신 스냅샷을 비교하므로 생략된 중간 실행의 파일도 집계합니다. 10분 넘게 업로드가 계속되면 대기 중인 다음 실행으로 넘깁니다.

## 표시 예시

```text
📥 수어 데이터 업로드 현황
측정자 subject-2 · 측정일 2026-11-07
저장 위치: GitHub / training-data

✅ 교통사고 · 완료
Git 저장 20/20회 · 검사 통과 20회 · 오류 0회 · 경고 0회

✅ 삼키다 · 완료
Git 저장 20/20회 · 검사 통과 20회 · 오류 0회 · 경고 0회

🟡 상처 · 1회 부족
Git 저장 19/20회 · 검사 통과 19회 · 오류 0회 · 경고 0회

합계 59회 저장 · 오류 0회 · 목표까지 1회 부족
GitHub 저장 파일 · 상세 검사 결과
```

기본 목표는 수어당 20회입니다. `docs/data-notification-settings.json`에서 수어별 목표를 바꿀 수 있습니다. 현재 이름에 `테스트` 또는 `디스코드`가 포함된 수어는 1회 시험 목표로 구분합니다. 등록되지 않은 수어는 20회이며, 한 번도 도착하지 않은 예정 수어는 자동으로 추측하지 않습니다.

횟수는 GitHub에서 확인한 `(측정자, 세션, 수어, trial_id)` 기준입니다. 양손 샘플을 합쳐 한 회로 집계하고, 같은 회차의 동일 파일 재전송은 횟수를 늘리지 않습니다. 동일 회차의 내용이 다른 파일은 충돌 오류로 표시합니다. 같은 경로의 파일 수정은 최신 버전으로 검사합니다. 샘플 행 수·손별 수량·파일별 오류 예시는 상세 JSON에 보존합니다.

검사 통과는 파일 구조와 센서값의 기본 검사입니다. 측정 동작의 정확성, 착용/보정, 학습 적합성, 모델 인식 정확도를 보증하지 않습니다. GitHub 업로드 전에 잃은 데이터는 확인할 수 없으므로 표시 횟수는 앱에서 실제 수행한 횟수와 다를 수 있습니다.

## 정확도 표시

지원 수어인 삼키다·상처·교통사고의 수집 현황에는 검토된 **이전 내부 교차검증 73.88% (413/559)**와 원본 기록 링크를 참고로 표시합니다. **이번 업로드 정확도는 미평가**로 구분합니다. 모델 추론이나 재학습은 수신 알림에서 실행하지 않습니다. [구현 가능성 판단과 필요한 조건](model-accuracy-notifications.md)을 확인하세요.

## 설정과 상태 보관

- Discord 텍스트 채널의 웹훅 URL을 GitHub Actions Secret **`DISCORD_WEBHOOK_URL`**로 저장합니다. URL은 코드나 Git 파일에 넣지 않습니다.
- `.github/workflows/data-discord.yml`, `tools/data_upload_report.py`, `tools/data_notification_digest.py`, `tools/notification_store.py`, `docs/data-notification-settings.json`을 업로드가 발생하는 `main`과 `training-data` 브랜치에 둡니다.
- Actions의 자동 `GITHUB_TOKEN`은 별도 **`discord-notification-state` 브랜치**에 검사 캐시, 처리 기준 커밋, 메시지 ID를 기록하기 위해 `contents: write`를 사용합니다. 상태 파일에는 웹훅/인증키나 센서 원본값을 저장하지 않습니다. 원본 CSV는 읽기만 합니다.
- 처음 설치할 때는 과거 자료를 기준점으로 등록하고 일괄 알림을 보내지 않습니다. 수동 실행의 **bootstrap**을 선택해도 알림 없이 기준점을 갱신합니다.
- 운영 중 상태 브랜치를 지우거나 bootstrap을 반복하면 처리 기준점이 바뀌므로 주의합니다. 처리 도중 전송에 실패하면 미전송 수집 건을 남겨 다음 실행에서 다시 처리합니다.

`datasets/`의 공유 ZIP/센서 CSV와 `data/sensor_dataset_v숫자.csv` 변경도 기본 검사합니다. 이들은 파일별 결과를 별도 공유 데이터 메시지 하나에 모아 갱신합니다. ZIP의 날짜별 부분집합은 통합본과 이중 합산하지 않습니다. 기존 manifest/CRC/SHA-256 검사와 크기 제한은 유지합니다.

## 직접 확인

[Actions](https://github.com/sminll/SignLanguage_Wearable/actions) → **Data upload check → Run workflow**에서 브랜치를 선택합니다. `dataset_path`에 CSV 경로를 지정하면 그 파일이 속한 측정자·세션의 전체 현황을 다시 확인합니다. 수동 결과에는 재검사임을 표시합니다. 이미 같은 내용이면 메시지를 다시 보내지 않습니다.

**Summary**에는 수어별 회차가 표시되고 **Artifacts → data-inspection**에는 카드별 집계와 파일 검사 JSON이 14일간 보관됩니다. 비공개 저장소 링크는 접근 권한이 필요합니다. 휴대폰 소리·푸시는 각자의 디스코드 설정을 따릅니다.

로컬 미리보기는 데이터를 수정하거나 메시지를 보내지 않습니다.

```powershell
python tools/data_notification_digest.py --preview --report "$env:TEMP/digest-preview.json"
python -m unittest discover -s tests -p 'test_data_*.py'
```

공식 API: [Discord 메시지 수정](https://docs.discord.com/developers/resources/webhook#edit-webhook-message), [GitHub 실행 동시성](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
