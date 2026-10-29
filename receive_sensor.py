
from flask import Flask, request
from datetime import datetime
from pathlib import Path
from threading import Lock
import csv

app = Flask(__name__)

# CSV 파일 저장 위치
data_folder = Path(__file__).parent / "data"
data_folder.mkdir(exist_ok=True)

file_path = data_folder / "received_sensor.csv"

# 저장할 데이터 항목
sensor_names = [
    "fsr1", "fsr2", "fsr3",
    "fsr4", "fsr5", "fsr6",
    "ax", "ay", "az",
    "gx", "gy", "gz"
]

columns = [
    "time", "source", "hand",
    "trial_id", "label", "time_ms"
] + sensor_names

save_lock = Lock()

@app.route("/sensor", methods=["POST"])
def receive_sensor():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return {"error": "JSON 데이터 오류"}, 400

    # 저장할 한 줄의 데이터 준비
    row = {name: data.get(name, "") for name in columns}
    row["time"] = datetime.now().isoformat()
    row["source"] = data.get("source", "UNKNOWN")

    # CSV 파일에 한 줄씩 추가
    with save_lock:
        new_file = not file_path.exists() or file_path.stat().st_size == 0

        with open(file_path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=columns)

            if new_file:
                writer.writeheader()

            writer.writerow(row)

    print("센서값 수신 및 저장 완료!", flush=True)
    print("받은 데이터:", data, flush=True)

    return {"status": "success"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

