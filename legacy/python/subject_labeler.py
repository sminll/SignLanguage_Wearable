"""Tag previously received sensor rows by subject without changing the original CSV.

First run: python subject_labeler.py init
Later, AFTER a session's HTTP transfer: python subject_labeler.py subject3

Requires data/received_sensor.csv with 'time' and 'source' columns.
"""
import csv
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / 'data' / 'received_sensor.csv'
OUTPUT = BASE / 'data' / 'received_sensor_with_subject.csv'
FIELDS_TO_SKIP = {'subject_id'}
INITIAL_DATES = {'2026-10-29': 'subject1', '2026-10-30': 'subject2'}


def load_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'파일에 헤더가 없습니다: {path}')
        fields = reader.fieldnames
        rows = list(reader)
    return fields, rows


def date_from_time(timestamp):
    # The Python receiver stores local reception time, not the ESP32 millis().
    match = re.search(r'(20\d{2})[-/.]?(\d{2})[-/.]?(\d{2})', timestamp or '')
    if not match:
        return None
    return f'{match.group(1)}-{match.group(2)}-{match.group(3)}'


def is_test(row):
    return (row.get('source') or '').strip().upper() == 'TEST'


def write_atomic(fields, rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig', newline='', dir=OUTPUT.parent,
                                         prefix='.subject_', suffix='.csv', delete=False) as f:
            tmp_name = f.name
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, OUTPUT)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.remove(tmp_name)


def main():
    if len(sys.argv) != 2 or (sys.argv[1] != 'init' and not re.fullmatch(r'subject[1-9]\d*', sys.argv[1])):
        print('사용법: python subject_labeler.py init   또는   python subject_labeler.py subject3')
        return 2
    if not SOURCE.exists():
        print(f'원본 CSV가 없습니다: {SOURCE}')
        return 1
    fields, incoming = load_csv(SOURCE)
    if not {'time', 'source'}.issubset(fields):
        print(f'원본 CSV에 time/source 열이 없습니다. 현재 열: {fields}')
        return 1
    result_fields = [f for f in fields if f != 'subject_id'] + ['subject_id']

    if sys.argv[1] == 'init':
        if OUTPUT.exists():
            print('이미 결과 파일이 있습니다. 기존 사람 구분 정보 보호를 위해 init을 중단합니다.')
            print(f'결과 파일: {OUTPUT}')
            return 1
        dated = Counter()
        unresolved = []
        for number, row in enumerate(incoming, 1):
            if is_test(row):
                row['subject_id'] = ''
                continue
            day = date_from_time(row.get('time', ''))
            dated[day] += 1
            row['subject_id'] = INITIAL_DATES.get(day, '')
            if not row['subject_id']:
                unresolved.append((number, row.get('time', ''), row.get('label', '')))
        if unresolved:
            print('다음 날짜의 센서 데이터는 사람을 확정할 수 없어 저장하지 않았습니다:')
            for item in unresolved[:10]:
                print(f'  행 {item[0]}: time={item[1]} label={item[2]}')
            print('날짜별 실제 측정자를 먼저 확인하세요.')
            return 1
        write_atomic(result_fields, incoming)
        print('어제/오늘 수신 데이터 구분을 완료했습니다.')
        for key, value in sorted(dated.items()):
            print(f'{key}: {INITIAL_DATES[key]} / {value}행')
        print(f'저장: {OUTPUT}')
        return 0

    subject = sys.argv[1]
    if not OUTPUT.exists():
        print('먼저 python subject_labeler.py init 을 실행하세요.')
        return 1
    previous_fields, previous = load_csv(OUTPUT)
    if previous_fields != result_fields:
        print('기존 결과 파일 열 구조가 달라 중단합니다. 데이터를 수정하지 않았습니다.')
        return 1
    if len(incoming) < len(previous):
        print('원본 데이터가 이전보다 줄었습니다. 이전 매핑을 보호하기 위해 중단합니다.')
        return 1
    raw_fields = [f for f in fields if f != 'subject_id']
    for i, old in enumerate(previous):
        if any(incoming[i].get(k, '') != old.get(k, '') for k in raw_fields):
            print(f'원본 CSV의 {i+1}번째 데이터가 바뀌었습니다. 오분류 방지를 위해 중단합니다.')
            return 1
    new_rows = incoming[len(previous):]
    for row in new_rows:
        row['subject_id'] = '' if is_test(row) else subject
    combined = previous + new_rows
    write_atomic(result_fields, combined)
    counted = sum(not is_test(row) for row in new_rows)
    print(f'{subject}: 새로 추가된 실제 센서 데이터 {counted}행에 사람 번호를 부여했습니다.')
    print(f'새 행 전체 {len(new_rows)}개 (TEST 데이터는 사람 번호 공란)')
    print(f'총 {len(combined)}행, 저장: {OUTPUT}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, csv.Error) as exc:
        print(f'오류: {exc}')
        sys.exit(1)
