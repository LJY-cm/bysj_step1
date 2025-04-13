import math

def calculate_tpu_ocs(N_gpu):
    """计算 TPUv4 架构需要的 Palomar OCS 数量"""
    GPUs_per_cube = 64           #每个4x4x4块的GPU数量
    links_per_cube = 16*6         #每个4x4x4块的链路数量
    palomar_ports = 128             #Palomar OCS 总端口数, 实际上还有8个容错
    
    cubes_per_palomar = palomar_ports / 2 #每个Palomar OCS 可以支持的 4x4x4 小块的数量, 每个小块需要2个端口来连接相对的面
    num_cubes = N_gpu / GPUs_per_cube   #所需的 4x4x4 小块的数量
    if(num_cubes <= cubes_per_palomar):
        num_palomar_ocs = links_per_cube/2
        return num_palomar_ocs
    else:
        print('超出了TPUv4的能力范围')
        return 0
    # num_palomar_ocs = math.ceil(num_cubes / cubes_per_palomar)

    


def calculate_rain_rails(N_gpu, k, M):
    """计算 Rain-only 架构需要的 Rail 交换机数量"""
    
    num_rails = N_gpu / M
    
    return math.ceil(num_rails)


def calculate_clos_switches(N_gpu):
    """计算 Clos/Fat-tree 架构需要的交换机数量"""
    k1 = pow(4 * N_gpu, 1/3)
    # 找到大于等于 k 的最小 2 的幂, 作为端口数
    k = 2 ** math.ceil(math.log2(k1))
    print('端口数：',k)

    n_pod = k
    print('pod数：',n_pod)

    n_gpu_per_pod = N_gpu/n_pod
    print('每pod的GPU数：',n_gpu_per_pod)

    num_edge_per_pod = math.ceil(math.sqrt(n_gpu_per_pod))
    print('每pod的edge交换机数：',num_edge_per_pod)

    num_aggregation_per_pod = math.ceil(num_edge_per_pod)
    print('每pod的agg交换机数：',num_aggregation_per_pod)

    num_edge_switches = num_edge_per_pod*n_pod
    num_aggregation_switches = num_aggregation_per_pod*n_pod
    num_core_switches = num_edge_per_pod*num_aggregation_per_pod
    N_EPS = num_edge_switches + num_aggregation_switches + num_core_switches
    print('三层clos/fat-tree支持',N_gpu,'个GPU','','共需',N_EPS,'个EPS')
    return math.ceil(num_edge_switches), math.ceil(num_aggregation_switches),  math.ceil(num_core_switches)


def calculate_spine_leaf(N_gpu):
    """计算 Spine-Leaf 架构需要的交换机数量"""
    k = math.sqrt(2 * N_gpu)
    # 找到大于等于 k 的最小 2 的幂
    k = 2 ** math.ceil(math.log2(k))
    print('端口数：',k)
    num_leaf_switches = k
    num_spine_switches = k / 2
    N_EPS = num_leaf_switches + num_spine_switches
    print('两层spine-leaf支持',N_gpu,'个GPU','','共需',N_EPS,'个EPS')
    return math.ceil(num_leaf_switches), math.ceil(num_spine_switches)


def calculate_all_architectures(N_gpu, rain_k=8, rain_M=32):
    """计算所有架构的交换机/OCS并输出"""

    print(f"目标GPU数量: {N_gpu}")

    # TPUv4
    num_palomar_ocs = calculate_tpu_ocs(N_gpu)
    print(f"TPUv4 架构需要 {num_palomar_ocs} 个 Palomar OCS")

    # Rain-only
    num_rails = calculate_rain_rails(N_gpu, rain_k, rain_M)
    print(f"Rain-only 架构需要 {num_rails} 个 Rail 交换机 (k={rain_k}, M={rain_M})")

    # Clos/Fat-tree
    num_edge_switches, num_aggregation_switches, num_core_switches = calculate_clos_switches(N_gpu)
    print(f"Clos/Fat-tree 架构需要 {num_edge_switches} 个 edge 交换机， {num_aggregation_switches} 个 aggregation 交换机，{num_core_switches}个core交换机")

    # Spine-Leaf
    num_leaf_switches, num_spine_switches = calculate_spine_leaf(N_gpu)
    print(f"Spine-Leaf 架构需要 {num_leaf_switches} 个 leaf 交换机， {num_spine_switches} 个 spine 交换机")


# 示例
N_gpu = 4096  # 假设要支持 64 个 blocks，即 262144 个 GPU
calculate_all_architectures(N_gpu)
