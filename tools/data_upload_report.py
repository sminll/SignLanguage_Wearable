"""Inspect Git-tracked sensor uploads and send an optional Discord summary.

Uses only Python's standard library. Never modifies data or executes ZIP contents.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SENSORS = [*(f'fsr{i}' for i in range(1, 7)), 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
BASE_FIELDS = ['time', 'source', 'hand', 'trial_id', 'label', 'time_ms', *SENSORS]
FIELDS = BASE_FIELDS + ['subject_id', 'session_id', 'sample_seq']
TIMED_FIELDS = ['gesture_type', 'host_elapsed_ms', 'trial_elapsed_ms']
MAX_ZIP_BYTES = 512 * 1024 * 1024


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8')


def is_dataset(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        return False
    return (
        path.parts[:2] == ('sensor-data', 'training')
        and path.suffix.lower() == '.csv'
    ) or (
        path.parent == PurePosixPath('data')
        and re.fullmatch(r'sensor_dataset_v\d+\.csv', path.name) is not None
    ) or (
        'datasets' in path.parts[:1]
        and path.suffix.lower() in {'.csv', '.zip'}
        and path.name != 'session_devices.csv'
    )


def changed_files():
    """Inspect final versions of files changed across the whole pushed range."""
    event_path = os.environ.get('GITHUB_EVENT_PATH')
    event = json.loads(Path(event_path).read_text(encoding='utf-8')) if event_path else {}
    before = event.get('before', '')
    after = event.get('after') or os.environ.get('GITHUB_SHA') or 'HEAD'
    if os.environ.get('GITHUB_EVENT_NAME') == 'push' and before.strip('0'):
        entries = git('diff', '--name-status', '-z', '--no-renames', before, after).split('\0')
        files = [(entries[i + 1], entries[i]) for i in range(0, len(entries) - 1, 2)]
    else:
        files = [(name, 'current') for name in git('ls-files', '-z').split('\0') if name]
    return [(name, status) for name, status in files if is_dataset(name)]


def issue(result, kind, message):
    result[f'{kind}_count'] += 1
    if len(result[kind]) < 12:
        result[kind].append(message)


def check_csv(stream, result):
    reader = csv.DictReader(stream, strict=True)
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not set(FIELDS).issubset(fields):
        missing = ', '.join(sorted(set(FIELDS) - set(fields)))
        issue(result, 'errors', f'CSV 필수 컬럼 누락 또는 중복: {missing}')
        return
    timed = any(name in fields for name in TIMED_FIELDS)
    if timed and not set(TIMED_FIELDS).issubset(fields):
        issue(result, 'errors', 'v3 시간 메타데이터 컬럼이 일부만 있습니다.')
        return
    hands, labels = Counter(), Counter()
    sessions, trials, seen, trial_hands = {}, set(), set(), defaultdict(set)
    blank_seq = missing_imu = non_real = 0
    for number, row in enumerate(reader, 2):
        result['rows'] += 1
        if None in row or any(value is None for value in row.values()):
            issue(result, 'errors', f'{number}행: 헤더와 데이터의 열 수가 다릅니다.')
            continue
        if row['source'] != 'REAL':
            non_real += 1
            continue
        required = ['time', 'source', 'hand', 'trial_id', 'label', 'subject_id', 'session_id']
        if any(not row[name].strip() for name in required):
            issue(result, 'errors', f'{number}행: 필수 식별 정보가 비어 있습니다.')
            continue
        try:
            if row['hand'] not in {'left', 'right'}:
                raise ValueError('hand는 left/right여야 합니다.')
            for name in ['time_ms', *SENSORS, *(TIMED_FIELDS[1:] if timed else [])]:
                value = row[name]
                if not value and name in SENSORS[6:]:
                    continue  # The receiver explicitly supports missing IMU readings.
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f'{name}: 유한한 수가 아닙니다.')
                if name.startswith('fsr') and not 0 <= value <= 4095:
                    raise ValueError(f'{name}: 0~4095 범위를 벗어납니다.')
                if name in {'time_ms', *TIMED_FIELDS[1:]} and value < 0:
                    raise ValueError(f'{name}: 음수입니다.')
            if row['sample_seq']:
                seq = float(row['sample_seq'])
                if not math.isfinite(seq) or seq < 1 or not seq.is_integer():
                    raise ValueError('sample_seq: 양의 정수가 아닙니다.')
                key = row['session_id'], row['hand'], int(seq)
                if key in seen:
                    raise ValueError('동일 세션·손·sample_seq가 중복됩니다.')
                seen.add(key)
            else:
                blank_seq += 1
            if timed:
                if row['gesture_type'] not in {'left_only', 'right_only', 'both'}:
                    raise ValueError('gesture_type 값이 잘못됐습니다.')
                suffix = row['trial_id'].removeprefix(row['label'])
                if not suffix.isdigit() or not 1 <= int(suffix) <= 20:
                    raise ValueError('trial_id와 label/회차가 일치하지 않습니다.')
                elapsed = float(row['trial_elapsed_ms'])
                expected = float(row['host_elapsed_ms']) - (int(suffix) - 1) * 8000
                if not 0 <= elapsed < 5000 or abs(expected - elapsed) > 0.01:
                    raise ValueError('v3 측정 시간과 회차가 일치하지 않습니다.')
            # v2 permits multiple labels in one session; v3 fixes the label/mode.
            meta = ((row['subject_id'], row['label'], row['gesture_type'])
                    if timed else (row['subject_id'],))
            if row['session_id'] in sessions and sessions[row['session_id']] != meta:
                raise ValueError('동일 세션의 측정자·라벨·동작 유형이 다릅니다.')
            sessions[row['session_id']] = meta
        except (ValueError, OverflowError) as exc:
            issue(result, 'errors', f'{number}행: {exc}')
            continue
        missing_imu += any(not row[name] for name in SENSORS[6:])
        hands[row['hand']] += 1
        labels[row['label']] += 1
        trial = (row['subject_id'], row['session_id'], row['label'], row['trial_id'])
        trials.add(trial)
        trial_hands[trial].add(row['hand'])
    result.update(valid_rows=sum(hands.values()), real_rows=sum(hands.values()),
                  hands=dict(hands), labels=dict(labels),
                  sessions=len(sessions), trials=len(trials))
    if not result['rows']:
        issue(result, 'errors', 'CSV에 측정 행이 없습니다.')
    for count, text in [(blank_seq, 'sample_seq가 없어 중복 확인 불가'),
                        (missing_imu, '일부 IMU 값이 없음'),
                        (non_real, 'source가 REAL이 아니므로 측정 집계에서 제외')]:
        if count:
            issue(result, 'warnings', f'{count:,}행: {text}')
    if timed:
        unpaired = sum(h != {'left', 'right'} for h in trial_hands.values())
        if unpaired:
            issue(result, 'warnings', f'{unpaired:,}회차: 한쪽 손 데이터가 없어 양손 분석 확인 필요')


def check_zip(path, result):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (len(names) != len(set(names)) or len(names) > 10000
                or sum(info.file_size for info in infos) > MAX_ZIP_BYTES):
            raise ValueError('ZIP 중복 파일명 또는 검사 크기 한도(512 MiB/10,000개) 초과')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('ZIP 내부 파일 경로가 잘못됐습니다.')
        # Read every entry for CRC verification, without extracting or running code.
        for info in infos:
            with archive.open(info) as member:
                while member.read(1024 * 1024):
                    pass
        manifests = [n for n in names if PurePosixPath(n).name == 'manifest.json']
        if len(manifests) != 1:
            raise ValueError('공유 ZIP에는 manifest.json이 정확히 하나 필요합니다.')
        manifest_name = manifests[0]
        manifest = json.loads(archive.read(manifest_name))
        prefix = PurePosixPath(manifest_name).parent
        files = manifest.get('files')
        if not isinstance(files, dict) or not files:
            raise ValueError('manifest.json의 files 목록이 비어 있거나 잘못됐습니다.')
        expected = {str(prefix / name): digest for name, digest in files.items()}
        actual = {info.filename for info in infos if not info.is_dir()} - {manifest_name}
        if set(expected) != actual:
            raise ValueError('ZIP 파일 목록과 manifest 목록이 다릅니다.')
        for name, digest in expected.items():
            with archive.open(name) as member:
                if hashlib.file_digest(member, 'sha256').hexdigest() != digest:
                    raise ValueError(f'manifest SHA-256 불일치: {name}')
        masters = [n for n in names if re.search(r'(^|/)data/sensor_dataset_v\d+\.csv$', n)]
        if len(masters) != 1:
            raise ValueError('ZIP에는 data/sensor_dataset_v*.csv 통합본이 정확히 하나 필요합니다.')
        result['dataset_member'] = masters[0]
        with archive.open(masters[0]) as member:
            check_csv(io.TextIOWrapper(member, encoding='utf-8-sig', newline=''), result)
        for field, measured in [('real_rows', result['real_rows']), ('session_count', result['sessions'])]:
            if field in manifest and manifest[field] != measured:
                issue(result, 'errors', f'manifest의 {field}와 통합 CSV 집계가 다릅니다.')


def inspect_file(name, change='current'):
    result = dict(path=name, change=change, rows=0, valid_rows=0, real_rows=0, hands={}, labels={},
                  sessions=0, trials=0, errors=[], warnings=[], errors_count=0, warnings_count=0)
    if change == 'D':
        issue(result, 'warnings', 'GitHub에서 삭제된 파일입니다. 데이터 검사는 생략했습니다.')
        return result
    path = ROOT / name
    try:
        if path.suffix.lower() == '.zip':
            check_zip(path, result)
        else:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                check_csv(stream, result)
    except (OSError, ValueError, KeyError, TypeError, csv.Error, RuntimeError,
            zipfile.BadZipFile, NotImplementedError) as exc:
        issue(result, 'errors', f'파일 검사 실패: {str(exc)[:300]}')
    return result


def clean(value, limit=200):
    """Make user-supplied labels/paths readable without Markdown or mentions."""
    value = re.sub(r'[`*~|<>\[\]\\]', '', str(value)).replace('@', '＠')
    return ' '.join(value.split())[:limit]


def describe(result):
    if result['errors_count']:
        status = '❌ 오류'
    elif result['warnings_count']:
        status = '⚠️ 확인 필요'
    else:
        status = '✅ 기본 검사 통과'
    lines = [f"{status} · {clean(result['path'])}"]
    if result['change'] != 'D':
        hands = result['hands']
        lines.append(f"총 {result['rows']:,}행 / 유효 {result['valid_rows']:,}행 · "
                     f"왼손 {hands.get('left', 0):,} / 오른손 {hands.get('right', 0):,}")
        lines.append(f"세션 {result['sessions']:,}개 · 회차 {result['trials']:,}개")
        labels = ', '.join(f'{clean(k, 40)} {v:,}행' for k, v in sorted(result['labels'].items())[:8])
        if labels:
            if len(result['labels']) > 8:
                labels += f" 외 {len(result['labels']) - 8}개 라벨"
            lines.append(f'라벨: {labels}')
    lines.extend(clean(message, 250) for message in (result['errors'] + result['warnings'])[:3])
    return '\n'.join(lines)


def payload(report):
    files = report['files']
    errors = sum(f['errors_count'] for f in files)
    warnings = sum(f['warnings_count'] for f in files)
    title = '❌ 데이터 검사 오류' if errors else '⚠️ 데이터 확인 필요' if warnings else '✅ 데이터 기본 검사 완료'
    details = '\n\n'.join(describe(f) for f in files)
    details = details[:3500] + ('\n… 전체 결과는 검사 결과 링크에서 확인하세요.' if len(details) > 3500 else '')
    embed = dict(title=title, description=details,
                 color=0xED4245 if errors else 0xFEE75C if warnings else 0x57F287,
                 fields=[dict(name='업로드 계정 / 브랜치',
                              value=clean(f"{report['actor']} / {report['branch']}", 200)),
                         dict(name='파일 기준 집계', value=f"검사 대상 {len(files)}개 · 오류 {errors}건 · 경고 {warnings}건")],
                 footer=dict(text='파일별 전체 행 수입니다. 신규 추가 행 수나 학습 적합성 판정은 아닙니다.'))
    if report.get('run_url'):
        embed['url'] = report['run_url']
    return dict(username='수어 데이터 알림', allowed_mentions={'parse': []}, embeds=[embed])


def send_discord(report):
    if not report['files']:
        print('검사 대상 변경이 없어 Discord 전송을 생략합니다.')
        return
    url = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()
    if not url:
        print('::warning::DISCORD_WEBHOOK_URL Secret이 없어 알림을 생략합니다. 검사 보고서는 생성됐습니다.')
        return
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.netloc != 'discord.com'
            or not re.fullmatch(r'/api(?:/v\d+)?/webhooks/\d+/[A-Za-z0-9._-]+', parsed.path)):
        raise ValueError('DISCORD_WEBHOOK_URL은 discord.com의 공식 웹훅 주소여야 합니다.')
    # Always request a response so a successful HTTP request confirms delivery.
    target = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, 'wait=true', ''))
    body = json.dumps(payload(report), ensure_ascii=False).encode('utf-8')
    for attempt in range(3):
        request = urllib.request.Request(target, data=body,
                                         headers={'Content-Type': 'application/json',
                                                  'User-Agent': 'SignLanguage-Data-Notifier/1.0'})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                response.read()
            print('Discord 알림 전송 완료')
            return
        except urllib.error.HTTPError as exc:
            try:
                if exc.code == 429 and attempt < 2:
                    try:
                        delay = float(json.loads(exc.read()).get('retry_after', 1))
                    except (ValueError, TypeError):
                        delay = 1
                    time.sleep(min(max(delay, 1), 30))
                    continue
                # Do not print the exception URL: it contains the webhook secret.
                raise RuntimeError(f'Discord 전송 실패: HTTP {exc.code}') from None
            finally:
                exc.close()
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('Discord 연결 실패 또는 응답 시간 초과. 웹훅과 네트워크를 확인하세요.') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='로컬에서 검사할 CSV/ZIP (생략하면 Git 변경 감지)')
    parser.add_argument('--report', type=Path, default=Path('data-report.json'))
    parser.add_argument('--notify', type=Path, help='기존 JSON 보고서를 Discord에 전송')
    args = parser.parse_args()
    if args.notify:
        send_discord(json.loads(args.notify.read_text(encoding='utf-8')))
        return 0
    selected = args.paths
    if not selected and os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch':
        if requested := os.environ.get('DATASET_PATH', '').strip():
            if not is_dataset(requested):
                parser.error('수동 검사 경로는 저장소 내부의 알림 대상 CSV/ZIP이어야 합니다.')
            selected = [requested]
    files = [(name, 'current') for name in selected] if selected else changed_files()
    server = os.environ.get('GITHUB_SERVER_URL', 'https://github.com')
    repo, run = os.environ.get('GITHUB_REPOSITORY', ''), os.environ.get('GITHUB_RUN_ID', '')
    report = dict(actor=os.environ.get('GITHUB_ACTOR', 'local'),
                  branch=os.environ.get('GITHUB_REF_NAME', 'local'),
                  run_url=f'{server}/{repo}/actions/runs/{run}' if repo and run else '',
                  files=[inspect_file(name, status) for name, status in files])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = ('## 데이터 업로드 검사\n\n' + '\n\n'.join(describe(f) for f in report['files'])
               if files else '## 데이터 업로드 검사\n\n검사 대상 데이터 변경이 없습니다.')
    summary += '\n\n파일별 전체 수량이며 ZIP의 날짜별 부분집합은 합산하지 않습니다.\n'
    summary += '기본 구조 검사입니다. 학습 적합성·수집 완료 여부·양손 시간 동기화를 보증하지 않습니다.\n'
    print(summary)
    if summary_path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(summary_path, 'a', encoding='utf-8') as stream:
            stream.write(summary)
    return 1 if any(f['errors_count'] for f in report['files']) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f'::error::{clean(exc, 300)}', file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, subprocess.CalledProcessError):
        # Avoid accidentally leaking authentication URLs in Actions logs.
        print('::error::처리 실패: 보고서 경로·이벤트 파일·Git 이력 또는 Discord 설정을 확인하세요.', file=sys.stderr)
        sys.exit(1)
