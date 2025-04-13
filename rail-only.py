import math
def calculate_rain_max_gpus(num_rails, M):
    """计算 Rain-only 架构支持的最大 GPU 数量。"""
    max_gpus = num_rails * M
    return math.floor(max_gpus)

# 示例用法：
num_rails = 8  # Rail 交换机数量
M = 16           # 每个 Rail 交换机的端口数量

max_gpus = calculate_rain_max_gpus(num_rails, M)
print(f"Rain-only 架构可以支持的最大 GPU 数量：{max_gpus}")
print(f"所需 EPS 数量：{num_rails}")