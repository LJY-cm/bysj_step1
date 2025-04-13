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

# 计算 GPU 数量 / EPS 成本
data['GPU_per_EPS'] = data['Ntotal'] / data['n_EPS']

# 创建图表
fig, ax = plt.subplots(figsize=(14, 8))

# 设置 X 轴为端口数的索引
x = np.arange(len(port_counts))

# 柱状图宽度
width = 0.2

# 循环绘制每种架构的柱状图
for i, arch in enumerate(architectures):
    # 提取当前架构的数据
    arch_data = data[data['approach'] == arch]

    #确保每个端口数的架构都有数据，进行对齐
    arch_data = arch_data.set_index('duankou').reindex(port_counts).reset_index()
    
    # 计算柱状图的 X 轴位置，需要根据架构数量进行偏移，让所有柱子并排显示
    x_offset = x + width * (i - (len(architectures) - 1) / 2)
    
    # 绘制柱状图 (GPU per EPS)
    ax.bar(x_offset, arch_data['GPU_per_EPS'], width, label=arch, color=colors[arch], alpha=0.7)

# 设置图表标题和轴标签
ax.set_xlabel("number of ports", fontsize=20)
ax.set_ylabel("GPU per EPS (Ntotal / n_EPS)", fontsize=20)
plt.title("GPU per EPS for Different Architectures at Various Port Counts", fontsize=20)

# 设置 X 轴刻度
ax.set_xticks(x)
ax.set_xticklabels(port_counts, fontsize=20)

# 设置 Y 轴刻度
ax.tick_params(axis='y', labelsize=20)

# 添加图例
plt.legend(loc='upper left', fontsize=20)

# 添加网格线
ax.grid(axis='y', linestyle='--')

# 调整布局，防止标签重叠
plt.tight_layout()

# 显示图表
plt.show()
