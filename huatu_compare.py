import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 从 Excel 文件读取数据
data = pd.read_excel("D:\\simulation\\step1\\比较.xlsx")

# 架构类型
architectures = data['approach'].unique()

# 端口数量
port_counts = data['duankou'].unique()

# 为不同的架构定义颜色
colors = {
    'hybrid': 'blue',
    'rail-only': 'green',
    'spine-leaf': 'orange',
    'clos': 'red'
}

# 创建图表
fig, ax1 = plt.subplots(figsize=(12, 8))

# 设置 X 轴为端口数的索引 (0, 1, 2, 3)
x = np.arange(len(port_counts))

# 创建次 Y 轴
ax2 = ax1.twinx()

# 柱状图宽度
width = 0.2  # 更窄的宽度可以避免柱子重叠

# 循环绘制每种架构的柱状图和折线图
for i, arch in enumerate(architectures):
    # 提取当前架构的数据
    arch_data = data[data['approach'] == arch]
    
    # 将 port_counts 与 arch_data 的端口信息对齐
    arch_data = arch_data.set_index('duankou').reindex(port_counts).reset_index()

    # 计算柱状图的 X 轴位置，需要根据架构数量进行偏移，让所有柱子并排显示
    x_offset = x + width * (i - (len(architectures) - 1) / 2)

    # 绘制柱状图 (GPU 数量)
    ax1.bar(x_offset, arch_data['Ntotal'], width, label=f'{arch} (GPU)', color=colors[arch], alpha=0.7)

    # 绘制折线图 (EPS 成本)
    ax2.plot(x, arch_data['n_EPS'], marker='o', linestyle='-', color=colors[arch], label=f'{arch} (EPS)')

# 设置图表标题和轴标签
ax1.set_xlabel("number of ports", fontsize=20)
ax1.set_ylabel("num_GPUs", fontsize=20)
ax2.set_ylabel("num_EPS", fontsize=20)
plt.title("Different Architectures’ GPU Count and EPS Cost at Various Port Counts", fontsize=20)

# 设置 X 轴刻度
ax1.set_xticks(x)
ax1.set_xticklabels(port_counts, fontsize=20) # 设置 X 轴刻度标签字体大小

# 设置 Y 轴刻度字体大小
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

# 添加图例
# 将两个 Y 轴的图例合并
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2

# 显示图例
plt.legend(lines, labels, loc='upper left', fontsize=20)

# 调整布局，防止标签重叠
plt.tight_layout()

# 显示图表
plt.show()
