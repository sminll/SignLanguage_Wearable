import csv
from pathlib import Path

# CSV 파일 위치
file_path = Path(__file__).parent / "data" / "sensor_test.csv"

# 데이터 불러오기
with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    data = list(reader)

print("데이터 불러오기 성공!")
print("총 데이터 개수:", len(data))
print("센서 항목 개수:", len(data[0]) - 3)

print("\n첫 번째 데이터:")
for name, value in data[0].items():
    print(f"{name}: {value}")

print("\n주의: 현재 데이터는 가상 센서 데이터입니다.")
