# SIGN:AI 안드로이드 · 1단계

삼키다·아니다·상처를 휴대폰 안에서 판정하는 Kotlin / Jetpack Compose 앱입니다. Python 서버 없이 사용합니다. 모델을 PC에서 한 번 학습해 숫자 JSON으로 APK에 포함합니다. 1단계 앱에는 인터넷 권한·네트워크 요청·GitHub 로그인·측정 데이터 업로드가 없습니다. 이전 Flutter 서버 연결형 앱과 별개의 프로젝트이며 패키지 이름은 `kr.ac.signai.offline`, 표시 이름은 `SIGN:AI`입니다.

## 기준과 범위

- 작성일: 2026-11-23 (한국 시간).
- 시작 전 `git fetch origin main training-data feature/gravity-frame-model` 실행.
- main: `e03386b8ad24185c7679ed3d58d06db2caebf4c2`.
- training-data: `0f4b1eef59723ea42ebebe320ed6094db289bb4d`.
- feature/gravity-frame-model: `c33576af2167841831288164a90835593ba56960`.
- 작업 브랜치: `feature/android-app`, main 기반. 중력 기준 PR #11은 시작 전에 병합되어 별도 PR 의존성이 없습니다.
- 기존 Python 변경은 `sign_gravity_model.py`의 `export-android` 추가뿐입니다. 기존 데이터·모델·펌웨어·MATLAB·tools/·docs/는 보존합니다. 테스트용 CSV 5개는 고정된 Git 원본을 android-app/ 아래로 복사한 자료입니다.

## 모델

양손이 모두 유효한 실제 CSV 725회(삼키다 260, 아니다 226, 상처 239), 참가자 10명으로 StandardScaler + LogisticRegression(C=0.05, max_iter=5000)을 학습했습니다. `아니다 (아니, 부인(하다))`는 `아니다`로 통합합니다. JSON의 클래스 순서는 sklearn의 `classes_` 순서이며 계수·확률도 같은 순서입니다. FSR은 유효성 검사·그래프에 사용하고 판정 특징에는 사용하지 않습니다. 손별 266개, 왼손→오른손 532개 double 특징을 씁니다.

삼키다·상처는 11/4~7, 아니다는 11/13~15 측정이므로, 같은 날 세 동작을 함께 측정한 검증이 필요합니다. 같은 사람·착용 세션의 특성이 결과에 영향을 줄 수 있습니다. 날짜가 다른 자료의 후속 내부 평가는 새 날짜·재착용·실제 앱 인식 성능을 보증하지 않습니다. softmax 출력은 확률 합이 1인 모델 출력이며 보정된 정답 확률이 아닙니다. 앱의 실제 시연 정확도는 미측정입니다. JSON note와 설정 화면에도 이 한계를 표시합니다.

```powershell
# 저장소 루트, 새 출력 경로만 허용
python sign_gravity_model.py export-android --repo . --ref origin/training-data --signs "삼키다,아니다,상처" --out 새출력폴더
```

출력: `sign_model.json`, `parity_vectors.json`, 원본 해시가 포함된 실제 CSV 5개. 지정 3종이 725회 구성과 다르면 출력 디렉터리를 만들기 전에 중단합니다. 모델 JSON은 `models/sign_model.json`과 `app/src/main/assets/sign_model.json`에 같은 내용으로 포함했습니다. 새 모델을 만들었을 때는 두 파일을 함께 교체하고 parity 자료도 함께 갱신해야 합니다. 앱은 자동으로 모델을 받지 않습니다.

## 화면별 상태

| 탭 | 1단계 구현 | 이후 확인/미구현 |
|---|---|---|
| 동작 인식 | 양손 밴드 검색·주소별 손 지정, 3·2·1 + 5초 측정, 품질 검사, 판정·문장 추가, 길게 눌러 삭제·전체 지우기, 수동 한국어 말하기, CSV 재생 | 영상 출처 미확인: 영상 버튼·체크박스 비활성 |
| 센서 | 좌우 FSR 6개·가속도·각속도 그래프, 최대 ADC·샘플·누락 수/비율 | 실제 기기 수신·그래프 검증 필요 |
| 설정 | 수어·데이터 커밋·회차 수·모델 방식·한계, 판정 임계값 저장 | 업데이트·로그인·업로드는 비활성, 네트워크 코드 없음 |
| 보정 | 준비 중 안내·좌우 항목 표시 | 측정·JSON 가져오기 비활성, 구현하지 않음 |
| 학습 | 준비 중 안내, 5초/3초·20회 방식 설명 | 참가자 선택·측정 저장·삭제·공유·업로드 구현하지 않음 |

1단계 실기기 검증 전에는 2단계를 시작하지 않습니다. 단계 3 연속 인식도 구현하지 않았습니다. 단계 2/3을 구현한 것처럼 보이지 않도록 버튼을 비활성 상태로 유지합니다. iOS 소스·실제 영상은 제공되지 않아 지시서의 화면 설명을 기준으로 구성했으며 픽셀 단위 동일성은 확인하지 못했습니다.

## 설치·실행

1. Android 8.0 이상 휴대폰으로 `app/build/outputs/apk/debug/app-debug.apk`를 전송하고 내 파일에서 눌러 설치합니다. 허용 안내가 나오면 해당 파일 앱의 설치를 허용합니다. 개발용 디버그 서명이며 Play 스토어 배포본은 아닙니다.
2. 앱 이름 `SIGN:AI`를 엽니다. 서버 주소를 입력하지 않습니다.
3. 한국어 음성을 오프라인으로 사용하려면 휴대폰의 글자 읽어주기/TTS 설정에서 한국어 음성 데이터를 미리 설치합니다. 앱은 네트워크 필요 음성을 선택하지 않으며 없으면 안내를 표시합니다.
4. MATLAB·nRF Connect·다른 수집 앱과 밴드 연결을 종료합니다. 앱의 검색을 누르고 권한을 허용합니다. Android 11 이하는 위치 기능도 켭니다.
5. 실제 어느 손에 착용했는지 확인하여 기기별 왼손/오른손을 지정하고 각각 연결합니다. 기기 이름이나 발견 순서로 손을 추측하지 않습니다. 같은 손에 두 밴드를 연결할 수 없습니다.
6. 양손 모두 데이터 수신 상태가 된 후 측정을 누릅니다. 안내 자세를 지키고 카운트다운 뒤 5초 동작합니다. 양손 각각 샘플 30개·3초 이상·시간 엄격 증가·최대 간격 500ms·IMU 오류 비율 5% 이하·센서 유한값을 모두 통과해야 판정합니다.
7. 출력이 임계값보다 낮으면 ‘다시 해 주세요’로 보류하며 문장에 넣지 않습니다. 이 기능은 휴식/기타 수어 검출기로 검증된 기능이 아닙니다. 등록되지 않은 동작도 높은 출력으로 3종 중 하나가 될 수 있습니다.
8. 말하기로 현재 단어 문장을 읽습니다. 별도 문법 생성은 하지 않습니다.

CSV 재생은 예시 1~5 또는 파일 선택으로 실행합니다. 이 예시들은 학습에 사용된 자료이며 파이프라인 확인용입니다. 예시 판정 성공을 독립 정확도나 실제 센서 시연 성능으로 보고하면 안 됩니다. 화면·단어에는 ‘재생’ 표시가 붙습니다. CSV 파일은 최대 2MB이며 파일 선택으로 허용한 자료만 읽습니다.

백그라운드로 가면 진행 중 카운트다운·측정·CSV 판정만 취소합니다. 완료된 문장과 판정은 ViewModel에 유지하며, 앱 프로세스 종료 후 저장하는 기능은 1단계에 없습니다. 백그라운드 검색·새 재연결은 중단하고 다시 화면에 왔을 때 재개합니다. 손 지정과 임계값은 로컬 설정에 저장합니다. iOS가 전체 초기화하는 동작과 이 점이 다릅니다. 다른 사용자 센서가 섞이지 않도록 사용자를 바꾸면 연결 해제 후 손 지정을 확인하세요.

## 빌드와 검사

JDK 17, Gradle 9.5.0, AGP 9.3.1, Kotlin/Compose compiler 2.4.20, Compose BOM 2026.09.00. minSdk 26, compile/targetSdk 37입니다. [Android 17 SDK 설정](https://developer.android.com/about/versions/17/setup-sdk), [AGP 9.3 호환성](https://developer.android.com/build/releases/agp-9-3-0-release-notes), [Compose compiler 설정](https://developer.android.com/develop/ui/compose/setup-compose-dependencies-and-compiler)을 확인했습니다. 새 DSL 전환 대신 명시적 Kotlin 플러그인을 사용해 Gradle 설정에 호환성 플래그가 있습니다. 버전이 고정되어 있으며 AGP 10으로 변경할 때는 해당 플래그를 마이그레이션해야 합니다.

```powershell
# 저장소 루트
python -B -m unittest discover -s android-app/python_tests -v
cd android-app
# local.properties에 sdk.dir=<Android SDK 경로>, JAVA_HOME=JDK17 설정
.\gradlew.bat assembleDebug testDebugUnitTest
```

macOS/Linux에서는 `bash gradlew assembleDebug testDebugUnitTest`를 사용합니다. 자동 테스트는 펌웨어의 명시적 byte offset과 같은 방식으로 패킷을 구성하여 조각 순서 뒤섞임·NaN·누락·타임아웃·uint32 순환·중복·좌우 분리를 검사합니다. 실제 CSV 5개의 532 특징·클래스별 출력이 Python과 1e-6 이내이고 최고 클래스가 같은지 검사합니다. 이 자료와 임의 3D 회전 검사는 실제 실험 정확도 분모에 넣지 않습니다. 검사 결과와 실제 기기 미검증 항목은 `verification.json`에 기록합니다.

현재 Windows의 한글·공백 경로에서는 APK 컴파일은 되지만 Gradle JUnit worker가 테스트 클래스를 찾지 못했습니다. 같은 소스를 ASCII 경로로 복사해 빌드하면 모두 통과합니다. Windows에서는 다음 보조 스크립트를 권장합니다. 기존 소스를 이동·삭제하지 않고, 내용 해시별 새 작업 폴더에 앱 소스·테스트만 복사합니다. 원본 데이터 전체·인증 파일은 복사하지 않습니다.

```powershell
# android-app 폴더. 시스템 전체 실행 정책은 변경하지 않음
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build-windows.ps1 -WorkRoot C:\SignAI-build -SdkRoot C:\Android\Sdk -JavaHome C:\jdk-17
```

실행 결과: `assembleDebug testDebugUnitTest` 통과, Kotlin 검사 19개 통과. 실제 CSV 5개에서 특징 최대 절대 차이 `2.416e-13`, 출력 확률 최대 차이 `5.899e-17`, 최고 클래스 모두 일치. Python 내보내기 검사와 기존 CLI 분기 검사도 통과했습니다. 기존 evaluate·train·predict 함수는 실제 725회로 실행했으며, 별도의 임시 경로에만 결과·후보를 저장했습니다. evaluate는 기존 683/725(94.21%)를 재현했습니다. 이 결과는 내부 중첩 사람별 평가이며 APK의 고정 LR 실시간 정확도가 아닙니다. 기존 함수들의 AST가 main 기준과 동일한지도 검사했습니다.

## BLE와 데이터 처리

펌웨어 `XIAO_NRF52840_FSR6_IMU_BLE.ino`, `sensor_packet.h`를 원본으로 확인했습니다. 서비스 `19B10010-E8F2-537E-4F6C-D104768A1214`, Notify `19B10011-E8F2-537E-4F6C-D104768A1214`, CCCD `0x2902`입니다. 100ms마다 A0/A1/A2 각 20바이트, 공통 uint32 LE sequence, 본문은 timeMs uint32 + FSR 6×uint16 + IMU/온도 7×float32 + imuOk 1바이트입니다. 두 GATT와 디코더를 완전히 분리하고 fragment를 sequence로 모읍니다. 500ms 이상 불완전 조각·지난 번호를 버리고 누락 수를 중복 계산하지 않습니다. 연결을 다시 열 때 디코더와 현재 연결 카운터를 초기화합니다. 새 GATT마다 HIGH 우선순위 요청·서비스 검색·CCCD 활성화를 다시 수행하며 실패 시 연결 카드에 원인을 표시하고 최대 30초 간격으로 재연결합니다.

보드 time_ms는 각 손 내부 품질 검사에 사용하고 수신 시각(UTC 및 단조 시계)은 측정 창·그래프에 사용합니다. 두 보드 시계가 같은 시작값이라고 가정하지 않습니다. 같은 휴대폰 5초 창으로 양손을 수집하되 물리적인 동시 샘플링이나 패킷 전달 성공을 보증하지 않습니다. 측정 실패 시 판정하지 않습니다. 센서 샘플은 메모리에서만 사용하며 자동 학습·디스크 저장·네트워크 송신은 없습니다.

## 실기기 검증 체크리스트 · 현재 미검증

- 양손 밴드 동시 연결, 주소와 실제 손 대응 확인.
- 10초 이상 수신한 각 손의 샘플 수/누락 수/누락률 기록.
- 각 수어 5초 측정→판정, 취소·백그라운드 전환 시 미완료 판정 방지.
- 한 밴드 끊기기→자동 재연결→새 Notify 활성화, 측정 재개 확인.
- CSV 예시와 파일 선택으로 결과·재생 표시 확인.
- 비행기 모드에서 Bluetooth만 다시 켜고, 이미 설치된 한국어 오프라인 음성으로 판정·말하기 확인.
- 모델 정확도 실험은 학습에 쓰지 않은 별도 측정으로 실시. 실제 앱 정확도는 아직 ‘미측정’.

## iOS 담당자에게 확인할 항목 · 추측하여 구현하지 않음

1. 개별·문장 영상의 파일 출처·이용 조건·단어 매핑과 ‘맞아요’ 동작.
2. 표준 보정 JSON 구조와 calibration CSV 경로·label 규칙.
3. 기존 CSV `time`이 센서 측정 시각인지 휴대폰 수신 시각인지.
4. `sensor-data/training/<숫자 폴더>/<소문자 UUID>.csv`의 숫자 폴더 배정 규칙, 참가자 번호·이름 원본.
5. 팀 OAuth App Client ID와 디바이스 흐름 활성화 여부.

확인 이후에도 2단계 업로드는 우선 `android-upload-test`에서 검사하고, 사용자 승인 후에만 training-data로 전환해야 합니다. 현재는 업로드 코드·OAuth·토큰 저장을 추가하지 않았습니다. iOS는 CSV로 기기에서 KNN(k=5)을 학습하지만 이 앱은 PC 학습 LR JSON을 사용합니다. 이 차이를 설정 화면과 PR에 명시합니다.
