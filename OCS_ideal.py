import math

def calculate_ntotal(p, alpha, k, beta, L):
    """
    计算给定参数下的GPU总数。

    参数：
    p: 交换机的端口数。
    alpha: 参数，仅用于 Edge 层。
    k: 每台服务器的 GPU 数量。
    beta: 每台服务器平均接入的 EPS 端口数。
    L: 固定拓扑架构中的层数 (L >= 2)。

    返回值：
    Ntotal: GPU的总数。
    """

    if not isinstance(p, int) or p <= 0:
        raise ValueError("p 必须是正整数")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("alpha 必须是正数")
    if not isinstance(k, (int, float)) or k <= 0:
        raise ValueError("k 必须是正数")
    if not isinstance(beta, (int, float)) or beta <= 0:
        raise ValueError("beta 必须是正数")
    if not isinstance(L, int) or L < 2:
        raise ValueError("L 必须是大于等于 2 的整数。")
    # if p % (alpha + 1) != 0:
    #     raise ValueError("p 必须是 (alpha + 1) 的倍数。")

    # 更新后的公式
    term1 = (4 * alpha) / (alpha + 1)
    term2 = k / beta
    term3 = (p / 2) ** L  # 层数调整为 L-1
    ntotal = term1 * term2 * term3

    return int(ntotal)  # 通常 GPU 数量为整数

# 示例 (您可以根据需要更改这些值)
p = 64  # 示例端口数
print('端口数：',p)
#alpha = 21 #每台服务器的 GPU 数量
k = 8  # 每台服务器的 GPU 数量
beta = 1  # 每台服务器平均接入的 EPS 端口数
print('服务器平均接入EPS端口数：',beta)
L = 3  # 层数

# 循环计算并输出 alpha 从 0 到 21 的结果
for alpha in range(1, 22): # Changed range to 1 to 22 (exclusive) to include 1-21 since alpha can not be zero
    print('超额配置比率：',alpha)
    try:
        ntotal = calculate_ntotal(p, alpha, k, beta, L)
        print(f"GPU的总数 (Ntotal): {ntotal}")
        n_EPS = math.ceil(ntotal/8*beta/p)
        print('所需EPS数量：',n_EPS)
    except ValueError as e:
        print(f"错误：{e}")
    print("-" * 20)
