
import csv
import math
from pathlib import Path

# CSV 파일 위치
file_path = Path(__file__).parent / "data" / "sensor_test.csv"

# 양손 센서 24개 이름
sensor_names = [
    f"{side}_{sensor}"
    for side in ["left", "right"]
    for sensor in [
        "fsr1", "fsr2", "fsr3",
        "fsr4", "fsr5", "fsr6",
        "ax", "ay", "az",
        "gx", "gy", "gz"
    ]
]

# CSV 파일 읽기
with open(file_path, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    columns = reader.fieldnames or []
    data = list(reader)

# 필요한 항목이 모두 있는지 확인
required = ["time", "label", "source"] + sensor_names
missing = [name for name in required if name not in columns]

if missing:
    print("누락된 항목:", missing)
    raise SystemExit("CSV 형식 오류!")

print("총 데이터 개수:", len(data))
print("센서 항목 개수:", len(sensor_names))

error_count = 0

# 각 데이터의 오류 확인
for row_number, row in enumerate(data, start=2):
    for sensor in sensor_names:
        value = row.get(sensor)

        if value is None or value.strip() == "":
            print(f"{row_number}행 {sensor}: 값 누락")
            error_count += 1
            continue

        try:
            number = float(value)

            if not math.isfinite(number):
                print(f"{row_number}행 {sensor}: 비정상 값")
                error_count += 1

        except ValueError:
            print(f"{row_number}행 {sensor}: 숫자 아님")
            error_count += 1

print("--------------------")

if error_count == 0:
    print("검사 완료! 센서값 형식 오류 없음")
else:
    print(f"총 {error_count}개의 오류 발견")

print("현재 데이터는 가상 센서 데이터입니다.")

