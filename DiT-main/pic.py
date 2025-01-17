import re
import matplotlib.pyplot as plt

# 读取日志文件
log_file = "/lab/2025/zjl/code/DiT/results/001-DiT-XL-2/log.txt"
epoch_losses = {}

# 解析日志文件
with open(log_file, "r") as f:
    current_epoch = -1
    for line in f:
        # 检查 epoch 开始
        epoch_match = re.search(r"Beginning epoch (\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epoch_losses[current_epoch] = []
        # 检查 Train Loss 信息
        loss_match = re.search(r"Train Loss: ([\d.]+)", line)
        if loss_match and current_epoch != -1:
            loss = float(loss_match.group(1))
            epoch_losses[current_epoch].append(loss)

# 计算每个 epoch 的平均损失
epochs = sorted(epoch_losses.keys())
avg_losses = [sum(epoch_losses[epoch]) / len(epoch_losses[epoch]) for epoch in epochs]

# 设置论文风格
plt.style.use("seaborn-v0_8-paper")
plt.figure(figsize=(6, 4))

# 绘制损失曲线
plt.plot(
    epochs, avg_losses,
    linestyle='-', linewidth=1.5, markersize=5, color='b', label='Average Loss'
)

# 设置标签和标题
plt.xlabel("Epoch", fontsize=12, labelpad=10)
plt.ylabel("Loss", fontsize=12, labelpad=10)
plt.title("Loss vs. Epoch", fontsize=14, pad=15)
plt.legend(fontsize=10)

# 调整刻度和网格
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# 紧凑布局
plt.tight_layout()

# 保存为高分辨率图像
plt.savefig("loss_vs_epoch.png", dpi=300)
plt.show()
