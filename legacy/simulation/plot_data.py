
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# 기존 CSV 파일 불러오기
file_path = Path(__file__).parent / "data" / "sensor_test.csv"

with open(file_path, "r", encoding="utf-8-sig") as file:
    data = list(csv.DictReader(file))

# 측정 순서
x = list(range(1, len(data) + 1))

# 왼손과 오른손의 첫 번째 FSR 값
left_fsr1 = [float(row["left_fsr1"]) for row in data]
right_fsr1 = [float(row["right_fsr1"]) for row in data]

# 그래프 그리기
plt.plot(x, left_fsr1, marker="o", label="Left FSR 1")
plt.plot(x, right_fsr1, marker="o", label="Right FSR 1")

plt.title("FSR Sensor Data - Simulation")
plt.xlabel("Sample Number")
plt.ylabel("Simulated Value")
plt.legend()
plt.grid(True)

plt.show()
