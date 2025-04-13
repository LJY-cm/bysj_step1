import math

def calculate_tpu_max_gpus(palomar_ports, gpus_per_cube, links_per_cube):
    """
    计算给定 Palomar OCS 端口数下 TPUv4 架构支持的最大 GPU 数量。
    Args:
        palomar_ports (int): Palomar OCS 总端口数。
        gpus_per_cube (int): 每个 4x4x4 块的 GPU 数量。
        links_per_cube (int): 每个 4x4x4 块的链路数量。
    Returns:
        int: 支持的最大 GPU 数量。
    """

    cubes_per_palomar = palomar_ports / 2
    max_gpus = cubes_per_palomar * gpus_per_cube
    return math.ceil(max_gpus)

# 示例用法：
palomar_ports = 128  # Palomar OCS 端口数
gpus_per_cube = 64   # 每个 4x4x4 小块的 GPU 数量
links_per_cube = 16*6 # 每个 4x4x4 小块的链路数量

max_gpus = calculate_tpu_max_gpus(palomar_ports, gpus_per_cube, links_per_cube)
print(f"TPUv4 架构可以支持的最大 GPU 数量：{max_gpus}")
