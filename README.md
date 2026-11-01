# 수어 웨어러블 통합 프로젝트

**XIAO 수집: [서버전송 사용법](MATLAB_노트북용_수정본/서버전송_사용법.md)**

**원격 수신 서버 실행: [Tailscale 수신 안내](TAILSCALE_수신안내.md)**

기존 코드와 CSV는 보존했으며 새 데이터 저장은 `data/sensor_dataset_v2.csv`를 사용합니다.

## 폴더 안내

- `MATLAB_노트북용_수정본/`: 현재 센서 수집 화면과 테스트
- `firmware/`: XIAO 펌웨어
- `data/`: 측정 원본, 장치 기록, 학습 결과 및 백업
- [실험기록/](실험기록/README.md): 날짜별 분석 결과
- [legacy/](legacy/README.md): 구버전 Python 코드, 가상 데이터 예제, 과거 안내

현재 Python 실행 파일은 최상위에 유지합니다. 기존 원본 폴더 백업은 `archive_original_folder/`에 그대로 보관합니다.

## 날짜·보드별 KNN 실행

```powershell
python train_sign_knn.py --date 2026-11-01 --board XIAO
```

`--date`는 서버 수신 날짜, `--board`는 세션 장치 기록을 기준으로 선택합니다.
옵션이 없으면 전체 데이터를 사용합니다. 결과는 `data/knn_results/2026-11-01_XIAO/`처럼 분리됩니다.
다른 조건의 결과를 보존하려면 `--output 결과폴더`를 지정하세요.

보드 선택에는 로컬 `data/session_devices.csv`가 필요합니다. 첫 줄은
`session_id,board_type`이며, 이후 각 행에 실제 세션 ID와 `XIAO` 또는 `ESP32`를 기록합니다.
세션 ID는 중복될 수 없습니다. 이 파일과 실측 CSV, 생성 모델, 백업은 GitHub에 올리지 않습니다.
다른 PC에서는 별도로 전달받은 측정 데이터와 장치 기록을 준비해야 같은 분석을 재현할 수 있습니다.

---

# SignLanguage_Wearable
XIAO nRF52840 Sense 기반 수어 인식 웨어러블 프로젝트
