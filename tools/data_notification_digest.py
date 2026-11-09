"""Debounce upload bursts and update one Discord card per participant/session/day."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

try:
    from tools import data_upload_report as checks
    from tools.notification_store import Discord, GitHubStore
except ModuleNotFoundError:  # Direct execution from tools/ on a GitHub runner.
    import data_upload_report as checks
    from notification_store import Discord, GitHubStore

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
CONFIG = ROOT / 'docs/data-notification-settings.json'


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env)


def card_key(subject, session, day):
    return json.dumps([subject, session, day], ensure_ascii=False, separators=(',', ':'))


def capture_day(value):
    # App times carry Z; historical receiver times without an offset are Korean local time.
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return (stamp.replace(tzinfo=KST) if stamp.tzinfo is None else stamp.astimezone(KST)).date().isoformat()


def inspect_blob(name, blob):
    result = dict(path=name, change='current', rows=0, valid_rows=0, real_rows=0,
                  hands={}, labels={}, sessions=0, trials=0, errors=[], warnings=[],
                  errors_count=0, warnings_count=0)
    records = []
    try:
        if name.lower().endswith('.zip'):
            with tempfile.TemporaryDirectory(prefix='wearable-inspection-') as directory:
                path = Path(directory) / 'snapshot.zip'
                path.write_bytes(blob)
                checks.check_zip(path, result)
            return {'inspection': result, 'records': records}
        text = blob.decode('utf-8-sig')
        checks.check_csv(io.StringIO(text), result)
        if name.startswith('sensor-data/training/'):
            groups = defaultdict(list)
            for row in csv.DictReader(io.StringIO(text)):
                if row.get('source') != 'REAL':
                    continue
                subject, session, label, trial = (row.get(k, '') or '' for k in
                                                   ('subject_id', 'session_id', 'label', 'trial_id'))
                groups[subject, session, label, trial].append(row)
            for (subject, session, label, trial), rows in groups.items():
                days = []
                for row in rows:
                    try:
                        days.append(capture_day(row.get('time', '')))
                    except ValueError:
                        checks.issue(result, 'errors', '측정 시간(time)을 날짜로 해석할 수 없습니다.')
                day = min(days) if days else '날짜 미확인'
                # Stable across a byte-for-byte resend and row ordering; never save raw sensor data in state.
                normalized = sorted(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
                fingerprint = hashlib.sha256('\n'.join(normalized).encode()).hexdigest()
                records.append(dict(subject=subject or '측정자 미확인', session=session or 'unidentified-upload',
                                    label=label or '수어 미확인', trial=trial or name, day=day,
                                    fingerprint=fingerprint, rows=len(rows),
                                    hands=dict(Counter(r.get('hand', '') for r in rows)),
                                    errors=result['errors_count'], warnings=result['warnings_count']))
    except (OSError, ValueError, TypeError, KeyError, csv.Error, zipfile.BadZipFile, RuntimeError) as exc:
        checks.issue(result, 'errors', f'파일 검사 실패: {str(exc)[:200]}')
    if name.startswith('sensor-data/training/') and not records:
        checks.issue(result, 'errors', '측정자·세션·수어를 식별할 수 있는 REAL 측정 기록이 없습니다.')
        records.append(dict(subject='측정자 미확인', session='unidentified-upload',
                            label='식별 불가 파일', trial=name, day='날짜 미확인',
                            fingerprint=hashlib.sha256(blob).hexdigest(), rows=0, hands={},
                            errors=result['errors_count'], warnings=result['warnings_count']))
    return {'inspection': result, 'records': records}


def inventory(head, previous):
    """Read the latest complete Git snapshot; do not depend on canceled push ranges."""
    tree = git('ls-tree', '-r', '-z', head).decode('utf-8')
    blobs = {}
    for entry in tree.split('\0'):
        if entry:
            meta, name = entry.split('\t', 1)
            if meta.split()[1] == 'blob' and checks.is_dataset(name):
                blobs[name] = meta.split()[2]
    changed = {name for name in set(previous) | set(blobs)
               if previous.get(name, {}).get('blob') != blobs.get(name)}
    current = {name: item for name, item in previous.items() if name in blobs and name not in changed}
    if any(name in blobs for name in changed):
        # Archive reads include ZIP contents as bytes only; no extraction or execution.
        roots = sorted({name.split('/')[0] for name in blobs})
        raw = git('archive', head, *roots)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for member in archive:
                if member.isfile() and member.name in changed and member.name in blobs:
                    item = inspect_blob(member.name, archive.extractfile(member).read())
                    current[member.name] = dict(blob=blobs[member.name], **item)
    return current, changed


def expected(label, config):
    if label in config.get('targets', {}):
        return config['targets'][label]
    if any(word in label for word in config.get('test_label_contains', [])):
        return 1
    return config.get('default_target', 20)


def group_cards(files, config):
    cards = {}
    for name, item in sorted(files.items()):
        for record in item['records']:
            key = card_key(record['subject'], record['session'], record['day'])
            card = cards.setdefault(key, dict(subject=record['subject'], session=record['session'],
                                              day=record['day'], labels={}, files=[]))
            card['files'].append(name)
            label = card['labels'].setdefault(record['label'], {})
            label.setdefault(record['trial'], []).append((name, record))
    for card in cards.values():
        rows = []
        for label, trials in sorted(card['labels'].items()):
            passed = errors = warnings = duplicates = samples = left = right = 0
            problems = []
            for copies in trials.values():
                versions = {record['fingerprint'] for _, record in copies}
                duplicates += max(0, len(copies) - 1)
                bad = len(versions) > 1 or any(record['errors'] for _, record in copies)
                warned = any(record['warnings'] for _, record in copies)
                errors += bad
                warnings += warned and not bad
                passed += not bad
                first = copies[0][1]
                samples += first['rows']
                left += first['hands'].get('left', 0)
                right += first['hands'].get('right', 0)
                if bad or warned or len(copies) > 1:
                    reason = ('동일 회차의 파일 내용 충돌' if len(versions) > 1 else
                              '중복 업로드 · 1회로 집계' if len(copies) > 1 and not bad else
                              '; '.join(files[copies[0][0]]['inspection']['errors' if bad else 'warnings'][:2]))
                    problems.append(dict(path=copies[0][0], reason=reason))
            target = expected(label, config)
            status = ('오류 확인' if errors else '경고 확인' if warnings else
                      f'{target - len(trials)}회 부족' if len(trials) < target else
                      f'목표보다 {len(trials) - target}회 많음' if len(trials) > target else '완료')
            rows.append(dict(label=label, uploaded=len(trials), passed=passed, errors=errors,
                             warnings=warnings, duplicates=duplicates, target=target,
                             missing=max(0, target-len(trials)), status=status,
                             samples=samples, left_samples=left, right_samples=right, problems=problems))
        card['labels'] = rows
        card['files'] = sorted(set(card['files']))
    return cards


def affected_keys(changed, before, after):
    return {card_key(r['subject'], r['session'], r['day'])
            for name in changed for source in (before, after)
            for r in source.get(name, {}).get('records', [])}


def model_reference(config, labels):
    report = config.get('model_reference')
    if not report or not set(labels).intersection(report['labels']):
        return '이번 업로드 정확도: 미평가 · 수신 검사만 실행'
    scores = []
    for label in sorted(set(labels)):
        if score := report.get('per_label', {}).get(label):
            scores.append(f"{checks.clean(label, 40)} {score['correct']/score['trials']:.1%} ({score['correct']}/{score['trials']})")
    detail = ('이전 검증 수어별: ' + ' · '.join(scores) + '\n') if scores else ''
    return (f"이전 내부 교차검증 **{report['correct']/report['trials']:.2%}** "
            f"({report['correct']}/{report['trials']}) · {report['evaluated_on']}\n"
            f"대상: {', '.join(report['labels'])}\n"
            + detail +
            '이번 업로드 정확도: **미평가** · 저장 후보의 새 사람 성능 미확인\n'
            f"[평가 조건과 모델 상태]({report['source_url']})")


def render_card(card, config, repo, branch, run_url, *, removed=False, manual=False):
    labels = card['labels']
    errors = sum(r['errors'] for r in labels)
    attention = errors or any(r['warnings'] or r['missing'] or r['uploaded'] > r['target'] for r in labels)
    title = '📥 수어 데이터 업로드 현황'
    if removed:
        title = '⚠️ 수집 데이터 삭제 확인'
    lines = [f"측정자 **{checks.clean(card['subject'], 60)}** · 측정일 **{card['day']}**",
             f"저장 위치: GitHub / {checks.clean(branch, 60)}", '']
    if removed:
        lines.append('이 수집 건의 데이터 파일이 GitHub에서 삭제됐습니다.')
    for row in labels:
        icon = '🔴' if row['errors'] else '🟡' if row['status'] != '완료' else '✅'
        lines.extend([f"{icon} **{checks.clean(row['label'], 60)} · {row['status']}**",
                      f"Git 저장 **{row['uploaded']}/{row['target']}회** · 검사 통과 {row['passed']}회"
                      f" · 오류 {row['errors']}회 · 경고 {row['warnings']}회"])
        if row['duplicates']:
            lines.append(f"중복 파일 {row['duplicates']}개는 횟수에서 제외")
        lines.append('')
    lines.append(f"합계 **{sum(r['uploaded'] for r in labels)}회 저장** · "
                 f"오류 {errors}회 · 목표까지 {sum(r['missing'] for r in labels)}회 부족")
    if manual:
        lines.append('수동 재검사 결과 · 새 측정 알림이 아닙니다.')
    description = '\n'.join(lines)
    if len(description) > 3800:
        description = description[:3750] + '\n… 나머지는 상세 결과에서 확인'
    base = f'https://github.com/{repo}/'
    fields = [dict(name='모델 정확도 · 수신 결과와 별도',
                   value=model_reference(config, [r['label'] for r in labels])[:1024])]
    problems = [p for row in labels for p in row['problems']]
    if problems:
        fields.append(dict(name='확인할 내용', value='\n'.join(checks.clean(p['reason'], 200) for p in problems[:3])[:900]))
    from urllib.parse import quote
    link = base + 'tree/' + quote(branch, safe='') + '/sensor-data/training'
    fields.append(dict(name='바로 확인', value=f'[GitHub 저장 파일]({link}) · [상세 검사 결과]({run_url})'))
    return dict(username='수어 데이터 알림', embeds=[dict(title=title, description=description,
                color=0xED4245 if errors else 0xFEE75C if attention or removed else 0x57F287,
                fields=fields, footer=dict(text='Git에 확인된 고유 회차 기준 · 목표 기본 20회 · 동작 정확성/학습 적합성은 별도 평가'))])


def wait_for_quiet(store, seconds=60, *, sleep=time.sleep, clock=time.monotonic):
    head = store.latest()
    since = clock()
    deadline = since + 600
    while clock() - since < seconds:
        if clock() >= deadline:
            raise RuntimeError('업로드가 10분 이상 계속됩니다. 대기 중인 다음 실행에서 다시 집계합니다.')
        sleep(min(10, seconds - (clock() - since)))
        latest = store.latest()
        if latest != head:
            head, since = latest, clock()
    return head


def deliver_cards(cards, keys, state, store, discord, config, repo, branch, run_url, manual=False):
    delivered = 0
    for key in sorted(keys):
        old = state['messages'].get(key)
        card = cards.get(key)
        if card is None:
            subject, session, day = json.loads(key)
            card = dict(subject=subject, session=session, day=day, labels=[], files=[])
        digest = hashlib.sha256(json.dumps({'card': card, 'model_reference': config.get('model_reference')},
                                          sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if old and old['digest'] == digest:
            state['pending_keys'] = [k for k in state.get('pending_keys', []) if k != key]
            continue
        body = render_card(card, config, repo, branch, run_url, removed=key not in cards, manual=manual)
        message_id = discord.deliver(body, old['id'] if old else None)
        state['messages'][key] = dict(id=message_id, digest=digest)
        state['pending_keys'] = [k for k in state.get('pending_keys', []) if k != key]
        store.save(state)  # Persist each confirmed delivery before processing another participant.
        delivered += 1
    return delivered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--preview', action='store_true', help='로컬 미리보기만 생성. 상태 저장·전송 없음')
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    branch = os.environ.get('GITHUB_REF_NAME', 'training-data')
    repo = os.environ.get('GITHUB_REPOSITORY', 'sminll/SignLanguage_Wearable')
    run_url = (f'https://github.com/{repo}/actions/runs/{os.environ["GITHUB_RUN_ID"]}'
               if os.environ.get('GITHUB_RUN_ID') else f'https://github.com/{repo}/actions')
    store = None if args.preview else GitHubStore(repo, branch, os.environ['GITHUB_TOKEN'])
    state = store.load() if store else dict(version=2, head='', files={}, messages={})
    if store:
        latest = store.latest()
        event = os.environ.get('GITHUB_EVENT_NAME')
        if event == 'push' and latest == state['head'] and not state.get('pending_keys') and not state.get('pending_legacy'):
            print('이미 처리한 Git 상태입니다. 새 알림 없음.')
            args.report.write_text(json.dumps({'unchanged': True}), encoding='utf-8')
            return 0
        if event == 'push':
            wait_for_quiet(store, config.get('quiet_seconds', 60))
        git('fetch', 'origin', branch, env=store.git_environment())
        head = git('rev-parse', 'FETCH_HEAD').decode().strip()
    else:
        head = git('rev-parse', 'origin/' + branch).decode().strip()
    before = state['files']
    files, changed = inventory(head, before)
    cards = group_cards(files, config)
    requested = os.environ.get('DATASET_PATH', '').strip()
    bootstrap = args.bootstrap or os.environ.get('BOOTSTRAP') == 'true'
    if requested:
        if not checks.is_dataset(requested) or requested not in files:
            raise ValueError('수동 검사 파일이 해당 브랜치에 없습니다.')
        keys = affected_keys({requested}, before, files)
    else:
        keys = affected_keys(changed, before, files)
    if not state['head'] and not requested:
        bootstrap = True  # First deployment never announces all historical uploads.
    keys |= set(state.get('pending_keys', []))
    legacy_results = dict(state.get('pending_legacy', {}))
    for name in changed:
        if not name.startswith('sensor-data/training/'):
            legacy_results[name] = files[name]['inspection'] if name in files else checks.inspect_file(name, 'D')
    state.update(files=files, pending_keys=sorted(keys), pending_legacy=legacy_results)
    report = dict(source_commit=head, branch=branch, bootstrap=bootstrap,
                  changed_files=sorted(changed), cards=[cards[k] for k in sorted(keys) if k in cards],
                  file_checks={n: files[n]['inspection'] for n in sorted(changed) if n in files})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = ['## 수어 업로드 요약', '']
    for card in report['cards']:
        summary += [f"### {checks.clean(card['subject'])} · {card['day']}", '',
                    '| 수어 | Git 저장 | 검사 통과 | 오류 | 상태 |', '|---|---:|---:|---:|---|']
        summary += [f"| {checks.clean(r['label'])} | {r['uploaded']}/{r['target']}회 | {r['passed']}회 | {r['errors']}회 | {r['status']} |" for r in card['labels']]
        summary.append('')
    if bootstrap:
        summary.append('기존 기록을 기준점으로 등록했습니다. 과거 업로드의 알림은 보내지 않았습니다.')
    print('\n'.join(summary))
    if path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(path, 'a', encoding='utf-8') as stream:
            stream.write('\n'.join(summary))
    if args.preview:
        return 0
    if bootstrap:
        state.update(head=head, pending_keys=[], pending_legacy={})
        store.save(state)
        print('알림 기준점 등록 완료 · Discord 전송 없음')
        return 0
    url = os.environ.get('DISCORD_WEBHOOK_URL', '')
    if not url:
        raise ValueError('DISCORD_WEBHOOK_URL Secret이 필요합니다. 전송 상태를 갱신하지 않습니다.')
    store.save(state)  # Save the pending work before sending; a failed retry cannot lose a participant.
    count = deliver_cards(cards, keys, state, store, Discord(url), config, repo, branch, run_url,
                          manual=os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch')
    # Non-app snapshots retain their file-level inspection, grouped in a single message per push burst.
    legacy = [legacy_results[n] for n in sorted(legacy_results)]
    if legacy:
        body = checks.payload(dict(files=legacy, actor=os.environ.get('GITHUB_ACTOR', ''), branch=branch, run_url=run_url))
        key = 'shared-datasets'
        digest = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
        old = state['messages'].get(key)
        if not old or old['digest'] != digest:
            message_id = Discord(url).deliver(body, old['id'] if old else None)
            state['messages'][key] = dict(id=message_id, digest=digest)
            count += 1
    state.update(head=head, pending_keys=[], pending_legacy={})
    store.save(state)
    print(f'Discord 요약 반영 완료: {count}개 · 기존 메시지는 수정, 동일 결과 재전송 없음')
    failed = any(files[n]['inspection']['errors_count'] for n in changed if n in files)
    failed |= any(row['errors'] for card in report['cards'] for row in card['labels'])
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        # Network clients sanitize errors; never log request URLs, headers or tokens.
        message = str(exc) if isinstance(exc, RuntimeError) else '설정·상태·입력 파일을 확인하세요.'
        print('::error::' + message, file=sys.stderr)
        sys.exit(1)
