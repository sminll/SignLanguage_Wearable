import csv
import random
from datetime import datetime
from pathlib import Path

# 저장할 폴더 생성
# 컴퓨터마다 터미널 위치가 달라도 이 파일 옆에 data 폴더를 생성합니다.
data_folder = Path(__file__).parent / "data"
data_folder.mkdir(exist_ok=True)

file_path = data_folder / "sensor_test.csv"

# 양쪽 손목의 센서 항목
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

with open(file_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["time", "label", "source"] + sensor_names)

    for i in range(10):
        values = [round(random.uniform(0, 1), 3)
                  for _ in sensor_names]

        writer.writerow([
            datetime.now().isoformat(),
            "test",
            "SIMULATED"
        ] + values)

print("CSV 파일 저장 완료!")
print(file_path.resolve())
