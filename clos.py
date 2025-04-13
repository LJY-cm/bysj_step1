import math
def calculate_clos_max_gpus(k):
    """
    计算 Clos/Fat-tree 架构支持的最大 GPU 数量
    Args:
        k (int):  端口数，也是 pod 的数量，需要是2的幂
    Returns:
        int: 支持的最大 GPU 数量
    """
    max_gpus = (k / 2) ** 2 * k  # 根据描述，每个 pod 有 (k/2)^2 个 GPU，共有 k 个 pod
    return round(max_gpus)

# Example Usage:
k = 128  # 端口数
n_EPS = (k/2)**2+k**2
max_gpus = calculate_clos_max_gpus(k)
print(f"Clos/Fat-tree 架构可以支持的最大 GPU 数量：{max_gpus}")
print('EPS的数量',n_EPS)
