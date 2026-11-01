# 성윤님 PC → 내 PC 센서 데이터 전송

## 내 PC에서 실행

두 PC가 Tailscale에서 서로 접근 가능한 상태인지 확인합니다.
이 PC에서 확인한 Tailscale IPv4 주소는 `100.92.15.45`입니다.

프로젝트 폴더의 터미널에서 실행합니다.

```powershell
python prepare_sensor_v2.py
python receive_sensor.py --host 100.92.15.45
```

수신 중에는 터미널을 열어 둡니다. 기존 서버가 5000번 포트를 사용 중이면
그 서버를 먼저 종료하고 위 명령으로 다시 실행합니다.
주소가 바뀌면 `tailscale ip -4`로 확인한 실제 주소를 사용합니다.

## 성윤님께 전달

Tailscale 연결 후 브라우저에서 아래 주소를 열어 주세요.

`http://100.92.15.45:5000/health`

`status: ok`가 나오면 연결 성공입니다.
접속되지 않으면 두 PC의 Tailscale 상태, 접근 정책, 수신 PC의 TCP 5000 방화벽 허용을 확인합니다.

MATLAB에서 실제로 실행하는 수집 프로그램의 endpoint를 다음과 같이 바꿔 주세요.

```matlab
endpoint = 'http://100.92.15.45:5000/sensor';
```

- XIAO 양손 BLE 수집: `xiao_ble_dual_monitor.m`의 endpoint
- 기존 시리얼 수집: `fsr_http_collection.m`의 endpoint
- 단일 XIAO 수집: `xiao_ble_single_ui_test.m`의 endpoint. `xiao_ble_reconnect`로도 같은 수집 창을 열 수 있습니다.
- 단일 보드의 상세 사용법은 `MATLAB_노트북용_수정본/서버전송_사용법.md`를 참고합니다.

최신 Subject/Session 형식의 수집 코드를 사용하고, 실제 동작 수행자의 Subject와
합의한 동작명, 손을 확인합니다. 우선 1~2회 시험 수집 후 전송하여 수신 PC에서
`data/sensor_dataset_v2.csv`에 저장되는지 확인합니다. 확인 후 동작별 본 수집을 진행합니다.
MATLAB 원본 `sensor_data` CSV도 전달 완료 확인 전까지 보관합니다.

2026-11-01 두 PC 간 실제 센서 데이터 수신과 통합 CSV 추가 저장을 확인했습니다.
측정 CSV와 개인별 세션·장치 매핑은 로컬에 보관하며 코드 업로드에서 제외합니다.
