import math
import pandas as pd
import os  # 导入 os 模块

def calculate_ntotal(p, alpha, k, beta, L):
    """计算给定参数下的GPU总数"""
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

    term1 = (4 * alpha) / (alpha + 1)
    term2 = k / beta
    term3 = (p / 2) ** L
    ntotal = term1 * term2 * term3

    return int(ntotal)

p = 32
k = 8
L = 3


# 构建路径
base_path = "D:\\simulation\\step1\\端口"
output_path = os.path.join(base_path, str(p), str(L))
output_path1 = os.path.join(base_path, str(all))
# 检查路径是否存在，如果不存在则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"创建目录: {output_path}")

all_results = []
# 双层循环计算并输出 alpha 从 1 到 21，beta 从 1 到 8 的结果
for alpha in range(1, 22):
    print('超额配置比率：',alpha)
    for beta in range(1, 9):
        print('服务器平均接入EPS端口数：',beta)
        try:
            ntotal = calculate_ntotal(p, alpha, k, beta, L)
            print(f"GPU的总数 (Ntotal): {ntotal}")
            n_EPS_1stlayer = math.ceil(ntotal / 8 * beta / p)
            if(L==1):
                n_EPS = n_EPS_1stlayer
            elif(L==2):
                n_EPS = n_EPS_1stlayer*1.5
            elif(L==3):
                n_EPS = n_EPS_1stlayer+n_EPS_1stlayer+n_EPS_1stlayer/2
            n_EPS = math.ceil(n_EPS)
            print('所需EPS数量：',n_EPS)

            # 将结果添加到列表中
            all_results.append([alpha, beta, ntotal, n_EPS])

        except ValueError as e:
            print(f"错误：{e}")
        print("-" * 20)
    print("=" * 40) # Add a separator between beta loops

# 使用 pandas 创建 DataFrame
df = pd.DataFrame(all_results, columns=['Alpha', 'Beta', 'Ntotal', 'n_EPS'])

# 将 DataFrame 写入 Excel 文件
filename = f"all_alpha_beta_results_32_3layer.xlsx"
filepath = os.path.join("D:\\simulation\\step1", filename)
df.to_excel(filepath, index=False)
print("所有结果已保存到 all_alpha_beta_results_32_3layer.xlsx 文件中")

# "alpha=x, varying beta" 文件的循环
for alpha in range(1, 22):
    data = []  # 用于存储当前alpha的所有结果
    for beta in range(1, 9):
        try:
            ntotal = calculate_ntotal(p, alpha, k, beta, L)
            n_EPS_1stlayer = math.ceil(ntotal / 8 * beta / p)
            if(L==1):
                n_EPS = n_EPS_1stlayer
            elif(L==2):
                n_EPS = n_EPS_1stlayer*1.5
            elif(L==3):
                n_EPS = n_EPS_1stlayer+n_EPS_1stlayer+n_EPS_1stlayer/2
            n_EPS = math.ceil(n_EPS)
            data.append([ntotal, n_EPS, beta])  # 添加结果到data列表，包含beta值
        except ValueError as e:
            print(f"错误：{e}")

    # 创建 DataFrame 并保存到 Excel 文件
    df = pd.DataFrame(data, columns=['Ntotal', 'n_EPS', 'beta'])  # 列名包含beta
    filename = f"alpha={alpha}_varying_beta.xlsx"
    filepath = os.path.join(output_path, filename)
    df.to_excel(filepath, index=False)
    print(f"alpha={alpha}文件存储完毕")


# "beta=x, varying alpha" 文件的循环
for beta in range(1, 9):
    data = []  # 用于存储当前beta的所有结果
    for alpha in range(1, 22):
        try:
            ntotal = calculate_ntotal(p, alpha, k, beta, L)
            n_EPS_1stlayer = math.ceil(ntotal / 8 * beta / p)
            if(L==1):
                n_EPS = n_EPS_1stlayer
            elif(L==2):
                n_EPS = n_EPS_1stlayer*1.5
            elif(L==3):
                n_EPS = n_EPS_1stlayer+n_EPS_1stlayer+n_EPS_1stlayer/2
            n_EPS = math.ceil(n_EPS)
            data.append([ntotal, n_EPS, alpha])  # 添加结果到data列表,包含alpha值
        except ValueError as e:
            print(f"错误：{e}")

    # 创建 DataFrame 并保存到 Excel 文件
    df = pd.DataFrame(data, columns=['Ntotal', 'n_EPS', 'alpha'])  # 列名包含alpha
    filename = f"beta={beta}_varying_alpha.xlsx"
    filepath = os.path.join(output_path, filename)
    df.to_excel(filepath, index=False)
    print(f"beta={beta}文件存储完毕")
