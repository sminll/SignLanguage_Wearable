
import random
import matplotlib.pyplot as plt
from collections import deque

# 최근 50개의 데이터만 저장
times = deque(maxlen=50)
left_values = deque(maxlen=50)
right_values = deque(maxlen=50)

# 그래프 준비
plt.ion()
fig, ax = plt.subplots()

left_line, = ax.plot([], [], label="Left FSR1")
right_line, = ax.plot([], [], label="Right FSR1")

ax.set_title("Real-Time FSR Simulation")
ax.set_xlabel("Sample Number")
ax.set_ylabel("Simulated Value")
ax.set_ylim(0, 1)
ax.set_xlim(0, 50)
ax.legend(loc="upper left")
ax.grid(True)
plt.show(block=False)
plt.pause(0.1)

# 가상 센서 데이터 생성 및 그래프 갱신
i = 0

while plt.fignum_exists(fig.number):
    i += 1

    times.append(i)
    left_values.append(random.uniform(0, 1))
    right_values.append(random.uniform(0, 1))

    left_line.set_data(list(times), list(left_values))
    right_line.set_data(list(times), list(right_values))

    ax.set_xlim(max(0, i - 49), max(50, i))

    fig.canvas.draw_idle()
    plt.pause(0.3)

print("실시간 그래프 종료")
