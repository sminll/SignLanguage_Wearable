"""Separate, backward-compatible CSV storage for timed two-board capture."""
import csv
import math
from pathlib import Path

from prepare_sensor_v2 import FIELDS

EXTRA_FIELDS = ['gesture_type', 'host_elapsed_ms', 'trial_elapsed_ms']
V3_FIELDS = FIELDS + EXTRA_FIELDS
GESTURE_TYPES = {'left_only', 'right_only', 'both'}


def validate_metadata(data):
    mode = data.get('gesture_type')
    if not isinstance(mode, str) or mode not in GESTURE_TYPES:
        raise ValueError('gesture_type은 left_only/right_only/both여야 합니다.')
    values = {}
    for name in EXTRA_FIELDS[1:]:
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name}은 유한한 비음수 밀리초여야 합니다.')
        values[name] = float(value)
    # This protocol records action windows, not the 3 s rest interval.
    trial = int(data['trial_id'][len(data['label']):])
    within = values['host_elapsed_ms'] - (trial - 1) * 8000
    if not 0 <= within < 5000 or abs(within - values['trial_elapsed_ms']) > 0.01:
        raise ValueError('공통 시간과 회차가 일치하지 않거나 측정 구간 밖입니다.')
    return {'gesture_type': mode, **values}


def sample_key(row):
    return row['session_id'], row['hand'], str(row['sample_seq'])


def fingerprint(row):
    numeric = {'time_ms', 'sample_seq', 'host_elapsed_ms', 'trial_elapsed_ms',
               *(f'fsr{i}' for i in range(1, 7)), 'ax', 'ay', 'az', 'gx', 'gy', 'gz'}
    return tuple('' if row[k] in ('', None) else float(row[k]) if k in numeric else str(row[k])
                 for k in V3_FIELDS if k != 'time')


class V3Store:
    """Caller holds the receiver's lock. Never migrates or edits v2 files."""

    def __init__(self, path):
        self.path = Path(path)
        self.seen = {}
        self.sessions = {}
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        seen, sessions = {}, {}
        if self.path.exists():
            with self.path.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != V3_FIELDS:
                    raise ValueError('v3 CSV 헤더 불일치: 기존 파일을 변경하지 않습니다.')
                for row in reader:
                    key, value = sample_key(row), fingerprint(row)
                    meta = (row['subject_id'], row['label'], row['gesture_type'])
                    if row['session_id'] in sessions and sessions[row['session_id']] != meta:
                        raise ValueError('v3 세션 메타데이터 충돌')
                    if key in seen and seen[key] != value:
                        raise ValueError('v3 기존 샘플 키 충돌')
                    seen[key], sessions[row['session_id']] = value, meta
        self.seen, self.sessions, self.loaded = seen, sessions, True

    def append(self, row):
        self.load()
        key, value = sample_key(row), fingerprint(row)
        meta = (row['subject_id'], row['label'], row['gesture_type'])
        if row['session_id'] in self.sessions and self.sessions[row['session_id']] != meta:
            raise ValueError('동일 세션의 측정자·수어·유형이 다릅니다.')
        if key in self.seen:
            if self.seen[key] != value:
                raise ValueError('동일 세션·손·번호의 데이터가 다릅니다.')
            return 'duplicate'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        with self.path.open('a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=V3_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        self.seen[key], self.sessions[row['session_id']] = value, meta
        return 'saved'
