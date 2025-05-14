import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
algorithms = [
    "A*", "Backtracking", "Beam Search", "BFS", "DFS",
    "Hill Climbing", "Q-Learning", "No Observation", "Simulated Annealing"
]

visited_nodes = [4, 6, 8, 8, 4, 6, 3, 9, 26]
avg_times = [
    1.0040, 0.9620, 1.3020, 0.9940, 1.0330,
    0.7140, 1.3680, 0, 1.0523  # 0 để chỉ No Observation bị giới hạn
]

# Đổi 0 thành NaN cho biểu đồ thời gian
avg_times = [np.nan if t == 0 else t for t in avg_times]

# Tạo biểu đồ song song
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.canvas.manager.set_window_title('So sánh thuật toán Tooth')

# --- Biểu đồ số node ---
bars1 = axs[0].bar(algorithms, visited_nodes, color='salmon', edgecolor='black')
axs[0].set_title("Số node đã duyệt của các thuật toán")
axs[0].set_ylabel("Số node")
axs[0].set_xticklabels(algorithms, rotation=90)
axs[0].grid(axis='y', linestyle='--', alpha=0.6)

for bar, value in zip(bars1, visited_nodes):
    axs[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value}', ha='center', va='bottom')

# --- Biểu đồ thời gian ---
bars2 = axs[1].bar(algorithms, avg_times, color='skyblue', edgecolor='black')
axs[1].set_title("Thời gian trung bình của các thuật toán")
axs[1].set_ylabel("Thời gian (ms)")
axs[1].set_xticklabels(algorithms, rotation=90)
axs[1].grid(axis='y', linestyle='--', alpha=0.6)

for bar, time in zip(bars2, avg_times):
    if not np.isnan(time):
        axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{time:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
