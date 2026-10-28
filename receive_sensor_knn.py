"""수어 웨어러블: MATLAB 센서값 수신 + KNN용 동작 라벨 기록.

실행: python receive_sensor.py
POST /start  {"label":"A"}      동작 측정 시작
POST /sensor {"hand":"left", ...} 센서값 수신 및 CSV 저장
POST /stop                       동작 측정 종료
"""

import csv
import math
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import time_ns

from flask import Flask, request, render_template_string

app = Flask(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_FILE = DATA_DIR / "knn_sensor_log.csv"
SENSORS = [f"fsr{i}" for i in range(1, 7)] + ["ax", "ay", "az", "gx", "gy", "gz"]
COLUMNS = [
    "received_at", "received_at_ms", "trial_id", "label", "hand",
    "time_ms", "source", *SENSORS, "valid", "error_reason"
]
LOCK = Lock()
CURRENT = {"trial_id": "", "label": "", "count": 0}


@app.get("/")
def dashboard():
    """측정 시작/종료를 브라우저에서 쉽게 조작하는 임시 화면."""
    return render_template_string("""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>수어 웨어러블 - KNN 데이터 수집</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#1f2937;margin:0;padding:32px}
main{max-width:650px;margin:auto;background:white;border:1px solid #e5e7eb;border-radius:16px;padding:28px}
h1{font-size:24px;margin:0 0 4px}p{color:#64748b}select,button{font:inherit;border-radius:10px;padding:10px 14px}
select{border:1px solid #cbd5e1;min-width:185px}button{border:0;cursor:pointer;color:white;background:#1764c0}
button.stop{background:#475569}button:disabled{opacity:.45;cursor:not-allowed}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.info{background:#eef2f7;padding:16px;border-radius:10px}
small{color:#64748b}#message{min-height:24px;color:#1764c0;font-weight:600;word-break:break-word}
</style></head><body><main>
<h1>손동작 데이터 수집</h1>
<p>센서가 전송 중일 때 동작을 선택하고 측정을 시작하세요. 기록은 CSV에 자동 저장됩니다.</p>
<label for="label">동작 선택</label>
<div class="controls">
<select id="label"><option value="A">A — 손 펴기 (시험용)</option><option value="B">B — 주먹 쥐기 (시험용)</option><option value="C">C — 손목 들어 올리기 (시험용)</option></select>
<button id="start" onclick="send('/start',{label:document.querySelector('#label').value})">측정 시작</button>
<button id="stop" class="stop" onclick="send('/stop',{})">측정 종료</button>
</div>
<div class="info" id="status">상태 확인 중…</div><p id="message"></p>
<small>동작별로 시작 → 동작 수행 → 종료를 반복하세요. 센서값은 data/knn_sensor_log.csv에 계속 추가됩니다. 실제 한국수어 인식 전의 시험 화면입니다.</small>
</main><script>
async function refresh(){
 try{const r=await fetch('/status');const d=await r.json();
 document.querySelector('#status').textContent=d.recording
 ? `측정 중: ${d.label} / 회차 ${d.trial_id} / 저장 ${d.samples}행`
 : '측정 대기 중 — 센서값은 라벨 없이도 저장됩니다.';
 document.querySelector('#start').disabled=d.recording;
 document.querySelector('#stop').disabled=!d.recording;
 }catch(e){document.querySelector('#status').textContent='서버 연결 오류';}
}
async function send(endpoint,payload){
 try{const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 const d=await r.json();document.querySelector('#message').textContent=r.ok?(endpoint==='/start'?'측정을 시작했습니다.':'측정을 종료했습니다.'):(d.error||'요청 오류');
 await refresh();
 }catch(e){document.querySelector('#message').textContent='요청 실패: '+e.message;}
}
refresh();setInterval(refresh,1500);
</script></body></html>""")


@app.get("/status")
def status():
    with LOCK:
        return {
            "recording": bool(CURRENT["trial_id"]),
            "trial_id": CURRENT["trial_id"],
            "label": CURRENT["label"],
            "samples": CURRENT["count"],
        }


@app.post("/start")
def start_trial():
    data = request.get_json(silent=True)
    label = data.get("label") if isinstance(data, dict) else None
    if not isinstance(label, str) or not (1 <= len(label.strip()) <= 40):
        return {"error": "label을 입력해 주세요. 예: A"}, 400

    with LOCK:
        if CURRENT["trial_id"]:
            return {"error": "이미 측정 중입니다. /stop을 먼저 실행하세요."}, 409
        CURRENT["label"] = label.strip()
        CURRENT["trial_id"] = (
            f"{CURRENT['label']}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        CURRENT["count"] = 0
        result = {"status": "recording", "trial_id": CURRENT["trial_id"], "label": CURRENT["label"]}
    print("동작 측정 시작:", result, flush=True)
    return result


@app.post("/sensor")
def receive_sensor():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return {"error": "JSON 객체를 보내주세요."}, 400

    received_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
    received_at_ms = time_ns() // 1_000_000
    errors = []

    hand = data.get("hand", "")
    if hand not in ("left", "right"):
        errors.append("hand 누락/오류")

    sample_ms = data.get("time_ms", "")
    if sample_ms == "" or sample_ms is None:
        errors.append("time_ms 누락")
    else:
        try:
            if not math.isfinite(float(sample_ms)):
                errors.append("time_ms 비정상")
        except (TypeError, ValueError):
            errors.append("time_ms 숫자 아님")

    for sensor in SENSORS:
        val = data.get(sensor, "")
        if val == "" or val is None:
            errors.append(f"{sensor} 누락")
            continue
        try:
            if not math.isfinite(float(val)):
                errors.append(f"{sensor} nan/inf")
        except (TypeError, ValueError):
            errors.append(f"{sensor} 숫자 아님")

    with LOCK:
        row = {name: data.get(name, "") for name in SENSORS}
        row.update({
            "received_at": received_at,
            "received_at_ms": received_at_ms,
            "trial_id": CURRENT["trial_id"],
            "label": CURRENT["label"],
            "hand": hand,
            "time_ms": sample_ms,
            "source": data.get("source", "UNKNOWN"),
            "valid": "YES" if not errors else "NO",
            "error_reason": "; ".join(errors),
        })
        new_file = not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0
        with CSV_FILE.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        if CURRENT["trial_id"]:
            CURRENT["count"] += 1
            count = CURRENT["count"]
        else:
            count = 0

    if count <= 1 or count % 20 == 0 or errors:
        print(f"센서값 저장: label={row['label'] or '(미지정)'}, "
              f"hand={hand}, valid={row['valid']}, count={count}", flush=True)
    return {"status": "saved", "valid": row["valid"], "trial_id": row["trial_id"]}


@app.post("/stop")
def stop_trial():
    with LOCK:
        result = {
            "status": "stopped",
            "trial_id": CURRENT["trial_id"],
            "label": CURRENT["label"],
            "samples": CURRENT["count"],
        }
        CURRENT.update(trial_id="", label="", count=0)
    print("동작 측정 종료:", result, flush=True)
    return result


if __name__ == "__main__":
    # 오늘은 이 PC에서만 테스트. 내일 Tailscale 연결 후 접근 주소를 변경합니다.
    app.run(host="127.0.0.1", port=5000, debug=False)
