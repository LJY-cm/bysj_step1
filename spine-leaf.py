import math
def calculate_spine_leaf_max_gpus(k):
    """计算 Spine-Leaf 架构支持的最大 GPU 数量"""
    max_gpus = (k ** 2) / 2
    return round(max_gpus)

def calculate_wasted_ports(L_actual, S_actual, p):
    """计算端口浪费数量"""
    Leaf_used = min(L_actual * p, S_actual * L_actual)
    Spine_used = min(S_actual * p, L_actual * S_actual)

    Leaf_wasted = L_actual * p - Leaf_used
    Spine_wasted = S_actual * p - Spine_used
    total_ports = L_actual * p + S_actual * p
    total_wasted = Leaf_wasted + Spine_wasted
    wasted_ratio = total_wasted/total_ports

    return total_wasted, wasted_ratio

def calculate_supported_gpus(L_actual, S_actual, p, ports_per_server, gpus_per_server):
    """
    计算在考虑端口浪费的情况下，Spine-Leaf 架构能支持的 GPU 总数。

    参数：
    L_actual: 实际 Leaf 交换机的数量。
    S_actual: 实际 Spine 交换机的数量。
    p: 每个交换机的端口数。
    ports_per_server: 每台服务器连接到 Leaf 交换机的端口数。
    gpus_per_server: 每台服务器配备的 GPU 数量。

    返回值：
    能支持的 GPU 总数。
    """

    # 计算 Leaf 交换机实际使用的总容量
    Leaf_used = min(L_actual * p, S_actual * L_actual)

    # 计算可以连接的服务器总数
    total_servers = Leaf_used / ports_per_server

    # 计算能支持的 GPU 总数
    total_gpus = int(total_servers * gpus_per_server) #保证为整数

    return total_gpus


# # 示例参数
# L_actual = 44  # Leaf 交换机数量
# S_actual = 20   # Spine 交换机数量
# p = 64        # 每台交换机的端口数
# ports_per_server = 1  # 每台服务器连接到 Leaf 交换机的端口数
# gpus_per_server = 8  # 每台服务器上的 GPU 数量

# # 计算可以支持的 GPU 总数
# supported_gpus = calculate_supported_gpus(L_actual, S_actual, p, ports_per_server, gpus_per_server)
# print(f"Spine-Leaf 架构存在端口浪费的情况下可以支持的最大 GPU 数量：{supported_gpus}")
# total_wasted, wasted_ratio = calculate_wasted_ports(L_actual, S_actual, p)
# print(f"总浪费端口数: {total_wasted}")
# print(f"端口浪费比例:{wasted_ratio}")
# 示例用法：
k = 64  # Leaf 交换机数量以及端口数量
max_gpus = calculate_spine_leaf_max_gpus(k)
print(f"Spine-Leaf 架构无端口浪费的情况下可以支持的最大 GPU 数量：{max_gpus}")
spine_switches = math.ceil(k/2)
leaf_switches = k

n_EPS = spine_switches + leaf_switches
print(f"所需 EPS 数量：{n_EPS}")