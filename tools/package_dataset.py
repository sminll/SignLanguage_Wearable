"""Build the reviewed 2026-11-01 team snapshot without changing live data."""
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'datasets' / '2026-11-01_v1.zip'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT.exists():
        raise RuntimeError('Snapshot already exists. Use a new version for new data.')
    fields, original = read_csv(ROOT / 'data/sensor_dataset_v2.csv')
    _, mapping = read_csv(ROOT / 'data/session_devices.csv')
    assert len({r['session_id'] for r in mapping}) == len(mapping)
    devices = {r['session_id']: r['board_type'] for r in mapping}
    rows = [r for r in original if r['source'] == 'REAL'
            and r['label'].strip().lower() not in {'test', '시험', '연결시험'}
            and r['time'][:10] <= '2026-11-01']
    assert len(rows) == 9754
    expected = [('ESP32', '2026-10-29', 968), ('ESP32', '2026-10-30', 5808),
                ('XIAO', '2026-11-01', 2978)]
    groups = {}
    for board, day, count in expected:
        subset = [r for r in rows if devices[r['session_id']] == board and r['time'].startswith(day)]
        assert len(subset) == count, (board, day, len(subset))
        groups[(board, day)] = subset
    assert sum(map(len, groups.values())) == len(rows)
    with tempfile.TemporaryDirectory(prefix='wearable_share_') as temp:
        package = Path(temp) / 'SignLanguage_dataset_2026-11-01_v1'
        package.mkdir()
        write_csv(package / 'data/sensor_dataset_v2.csv', fields, rows)
        sessions = {r['session_id'] for r in rows}
        write_csv(package / 'data/session_devices.csv', ['session_id', 'board_type'],
                  [{'session_id': r['session_id'], 'board_type': r['board_type']}
                   for r in mapping if r['session_id'] in sessions])
        for (board, day), subset in groups.items():
            write_csv(package / board / (day + '.csv'), fields, subset)
        source_results = ROOT / 'data/knn_results/2026-11-01_XIAO'
        _, features = read_csv(source_results / 'trial_features.csv')
        xiao_sessions = {r['session_id'] for r in groups[('XIAO', '2026-11-01')]}
        assert len(features) == 60 and {r['session_id'] for r in features} == xiao_sessions
        for label in ('갑자기', '머리', '아프다'):
            assert sum(r['label'] == label for r in features) == 20
        result_dir = package / 'KNN_2026-11-01_XIAO'
        result_dir.mkdir()
        for name in ('trial_features.csv', 'knn_model.joblib', 'evaluation.txt',
                     'confusion_matrix.csv', 'confusion_matrix.png',
                     'subject_label_counts.csv', 'class_index.txt'):
            shutil.copy2(source_results / name, result_dir / name)
        shutil.copy2(ROOT / 'train_sign_knn.py', package / 'train_sign_knn.py')
        dependencies = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'joblib']
        (package / 'requirements.txt').write_text(''.join(
            f'{name}=={importlib.metadata.version(name)}\n' for name in dependencies), encoding='utf-8')
        (package / 'README.md').write_text('''# 팀 공유 측정 데이터 v1

대상: 2026-11-01까지 측정한 ESP32·XIAO 실측. 패키지 작성일: 2026-11-02.
실측 9,754행이며 과거 TEST 1행과 삭제된 시험 세션은 제외했습니다.
참가자는 이름 대신 subject 번호로 표시했습니다. 팀 내부에서 사용하세요.

## 파일 구분

| 파일 | 보드 | 날짜 | 실측 행 수 |
|---|---|---|---:|
| ESP32/2026-10-29.csv | ESP32 | 10월 29일 | 968 |
| ESP32/2026-10-30.csv | ESP32 | 10월 30일 | 5,808 |
| XIAO/2026-11-01.csv | XIAO | 11월 1일 | 2,978 |
| data/sensor_dataset_v2.csv | 전체 | 위 세 날짜 | 9,754 |

**날짜별 파일은 통합 파일과 같은 데이터입니다. 함께 합치면 중복됩니다.**
data/session_devices.csv는 session_id별 보드를 구분하는 기록입니다.
ESP32와 XIAO는 장치·장착 조건이 달라 우선 별도로 평가하세요.
time은 서버 수신 시간, time_ms는 장치 시간입니다. 날짜별 파일은 수신 날짜 기준입니다.
회차는 subject_id/session_id/label/hand/trial_id를 함께 사용해 구분합니다.

## 11월 1일 XIAO KNN 실행

ZIP을 풀고 이 README와 train_sign_knn.py가 있는 폴더에서 실행합니다.

```powershell
python -m pip install -r requirements.txt
python train_sign_knn.py --date 2026-11-01 --board XIAO
```

결과는 data/knn_results/2026-11-01_XIAO/에 생성됩니다.
원본은 읽기만 하며, 수신 서버나 Tailscale 연결 없이 실행할 수 있습니다.
이 패키지를 현재 수신 서버의 data 폴더에 덮어쓰지 마세요. 별도 폴더에서 분석하세요.

## 포함된 KNN 결과

KNN_2026-11-01_XIAO/는 XIAO ‘갑자기·머리·아프다’ 각각 20회, 총 60회차의 결과입니다.
FSR+IMU 특징을 사용하고 k=3, random_state=42, 45회 학습/15회 평가입니다.
14/15 정답(93.3%)이며 ‘갑자기’ 1회를 ‘머리’로 예측했습니다.
같은 참가자·같은 세션의 회차를 나눈 초기 평가로, 다른 사람/날짜나 ‘편두통’ 전체 인식률이 아닙니다.
포함 모델은 평가에 사용한 학습 45회차로 학습된 모델입니다.
joblib 모델은 신뢰할 수 있는 이 팀 패키지에서만 로드하고, requirements 버전을 맞추세요.

최상위 manifest.json의 SHA-256으로 각 파일의 무결성을 확인할 수 있습니다.
''', encoding='utf-8')
        manifest = {'snapshot': '2026-11-01_v1', 'real_rows': len(rows),
                    'session_count': len(sessions), 'files': {}}
        for item in sorted(package.rglob('*')):
            if item.is_file():
                manifest['files'][item.relative_to(package).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
        (package / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        temp_zip = Path(temp) / OUTPUT.name
        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(package.rglob('*')):
                if item.is_file():
                    archive.write(item, item.relative_to(package.parent))
        with zipfile.ZipFile(temp_zip) as archive:
            assert archive.testzip() is None
            prefix = package.name + '/'
            for name, digest in manifest['files'].items():
                assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest
            bundled = list(csv.DictReader(io.StringIO(archive.read(prefix + 'data/sensor_dataset_v2.csv').decode('utf-8-sig'))))
            assert bundled == rows
        OUTPUT.parent.mkdir(exist_ok=True)
        with temp_zip.open('rb') as source, OUTPUT.open('xb') as target:
            shutil.copyfileobj(source, target)
    print(f'Package: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)')
    print('Real samples: 9754; ESP32: 6776; XIAO: 2978; sessions: 10')
    print('SHA256:', hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
