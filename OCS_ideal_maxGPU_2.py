import math
from decimal import Decimal, ROUND_HALF_UP

def precise_round(number, ndigits):
    """
    对数字进行指定位数的四舍五入。

    参数：
    number: 要四舍五入的数字。
    ndigits: 保留的小数位数。

    返回值：
    四舍五入后的数字。
    """
    # 将数字转换为 Decimal 类型
    d = Decimal(str(number))
    # 使用四舍五入策略
    rounded = d.quantize(Decimal(10) ** -ndigits, rounding=ROUND_HALF_UP)
    return float(rounded)

def calculate_max_gpus_with_eps(p, alpha, k, beta, num_eps,L):
    """
    根据给定的 EPS 数量计算最多支持的 GPU 数量。

    参数：
    p: 交换机的端口数。
    alpha: 参数，仅用于 Edge 层。
    k: 每台服务器的 GPU 数量。
    beta: 每台服务器平均接入的 EPS 端口数。
    num_eps: EPS 的数量。

    返回值：
    Ntotal: GPU 的总数。
    """

    # 参数检查
    if not isinstance(p, int) or p <= 0:
        raise ValueError("p 必须是正整数")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("alpha 必须是正数")
    if not isinstance(k, (int, float)) or k <= 0:
        raise ValueError("k 必须是正数")
    if not isinstance(beta, (int, float)) or beta <= 0:
        raise ValueError("beta 必须是正数")
    if not isinstance(num_eps, (int, float)) or num_eps <= 0:
        raise ValueError("EPS 数量必须是正数")

    # 根据 EPS 数量计算层数 L
    # 在 Spine-Leaf 架构中，EPS 数量大约为 (p/2) ** (L - 1)
    # 因此，L = log_base_(p/2)(num_eps) + 1  (修正)
    # L = math.log(num_eps, p / 2) 
    # L = math.ceil(L)  # 取整，确保层数为整数 (修正)
    #L = precise_round(L,0) #如果需要使用precise_round，请先测试其准确性
    print('层数：', L)

    # 计算 GPU 总数
    term1 = (4 * alpha) / (alpha + 1)
    term2 = k / beta
    term3 = (p / 2) ** L
    ntotal = term1 * term2 * term3

    return int(ntotal)  # GPU 数量为整数


# 示例
p = 32
print('端口数：', p)
alpha = 3
print('超额配置比率：', alpha)
k = 8
beta = 8
num_eps = 64
print('EPS 数量：', num_eps)
L = 1
try:
    ntotal = calculate_max_gpus_with_eps(p, alpha, k, beta, num_eps,L)
    print(f"最多支持的 GPU 数量 (Ntotal): {ntotal}")
except ValueError as e:
    print(f"错误：{e}")
