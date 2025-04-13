import networkx as nx
import numpy as np
import math
import itertools

import csv
import numpy as np
import networkx as nx
import math

class NetworkTopology:
    def __init__(self, p, alpha, k, beta, L, communication_matrix=None, link_capacity=10):
        self.p = p
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.L = L
        self.num_gpus = self.calculate_ntotal(p, alpha, k, beta, L)
        self.num_endpoints = self.num_gpus // self.k
        self.num_ocs_blocks = self.num_endpoints
        self.num_eps_layers = self.L - 1
        self.eps_ports = self.p
        self.num_leaf_switches = math.ceil(self.num_ocs_blocks * self.beta)
        self.num_spine_switches = self.p // 2
        self.link_capacity = link_capacity

        self.graph = nx.Graph()
        self.build_topology()
        self.ocs_connections = {}
        self.communication_matrix = communication_matrix
        if communication_matrix is None:
            self.communication_matrix = self.generate_default_communication_matrix()

        # 初始化所有边的流量为 0
        self.link_traffic = {tuple(sorted(edge)): 0 for edge in self.graph.edges()}

    def calculate_ntotal(self, p, alpha, k, beta, L):
        term1 = (4 * alpha) / (alpha + 1)
        term2 = k / beta
        term3 = (p / 2) ** L
        ntotal = term1 * term2 * term3
        return int(ntotal)

    def build_topology(self):
        # 添加节点
        for i in range(self.num_endpoints):
            self.graph.add_node(f"Endpoint_{i}")
        for i in range(self.num_ocs_blocks):
            self.graph.add_node(f"OCS_{i}")
        for i in range(self.num_leaf_switches):
            self.graph.add_node(f"Leaf_{i}")
        for i in range(self.num_spine_switches):
            self.graph.add_node(f"Spine_{i}")

        # 连接
        # 1. Endpoints -> OCS Blocks
        for i in range(self.num_endpoints):
            ocs_index = i // (self.num_endpoints // self.num_ocs_blocks)  # 计算对应的 OCS 块
            self.graph.add_edge(f"Endpoint_{i}", f"OCS_{ocs_index}", weight=0)

        # 2. OCS Blocks -> Leaf Switches
        for i in range(self.num_ocs_blocks):
            leaf_index = i % self.num_leaf_switches  # 均匀分布到 Leaf 交换机
            self.graph.add_edge(f"OCS_{i}", f"Leaf_{leaf_index}", weight=1)

        # 3. Leaf Switches -> Spine Switches
        for leaf in range(self.num_leaf_switches):
            for spine in range(self.num_spine_switches):
                self.graph.add_edge(f"Leaf_{leaf}", f"Spine_{spine}", weight=1)

    def set_ocs_connection(self, ocs_block_id, endpoint1_id, endpoint2_id):
        if ocs_block_id not in self.ocs_connections:
            self.ocs_connections[ocs_block_id] = []
        self.ocs_connections[ocs_block_id].append((endpoint1_id, endpoint2_id))

    def get_shortest_path_with_congestion(self, source_endpoint_id, target_endpoint_id):
        source = f"Endpoint_{source_endpoint_id}"
        target = f"Endpoint_{target_endpoint_id}"

        if source_endpoint_id == target_endpoint_id:
            return [source, target], 0, False

        source_ocs_id = source_endpoint_id // (self.num_endpoints // self.num_ocs_blocks)
        target_ocs_id = target_endpoint_id // (self.num_endpoints // self.num_ocs_blocks)
        if source_ocs_id == target_ocs_id:
            return [source, target], 0, False

        try:
            shortest_path = nx.dijkstra_path(self.graph, source=source, target=target, weight='weight')
            total_hops = 0
            congested = False

            for i in range(len(shortest_path) - 1):
                node1 = shortest_path[i]
                node2 = shortest_path[i + 1]
                edge = tuple(sorted((node1, node2)))  # 确保边的一致性
                edge_weight = self.graph.edges[node1, node2]['weight']
                total_hops += edge_weight

                if self.link_traffic[edge] + self.communication_matrix[source_endpoint_id, target_endpoint_id] > self.link_capacity:
                    congested = True
                    break

            return shortest_path, total_hops, congested

        except nx.NetworkXNoPath:
            return None, None, None

    def calculate_all_pairs_paths_and_congestion(self):
        all_pairs_results = {}
        self.link_traffic = {tuple(sorted(edge)): 0 for edge in self.graph.edges()}  # 重置流量

        for source_gpu in range(self.num_gpus):
            for target_gpu in range(self.num_gpus):
                if source_gpu != target_gpu:
                    path, hops, congested = self.get_shortest_path_with_congestion(source_gpu, target_gpu)
                    all_pairs_results[(source_gpu, target_gpu)] = (path, hops, congested)

                    if path:
                        for i in range(len(path) - 1):
                            node1 = path[i]
                            node2 = path[i + 1]
                            edge = tuple(sorted((node1, node2)))  # 确保边的一致性
                            self.link_traffic[edge] += self.communication_matrix[source_gpu, target_gpu]

        return all_pairs_results

    def generate_default_communication_matrix(self):
        """
        生成一个默认的混合 PP、DP、TP 通信矩阵。
        """
        matrix_size = self.num_gpus
        matrix = np.zeros((matrix_size, matrix_size))

        # PP 通信 (主对角线)
        for i in range(matrix_size - 1):
            matrix[i, i + 1] = 1  # 假设通信量为 1
            matrix[i + 1, i] = 1  # 双向通信

        # DP 通信 (块状, 假设每 4 个 GPU 为一组)
        dp_group_size = 4
        for i in range(0, matrix_size, dp_group_size):
            for j in range(i, min(i + dp_group_size, matrix_size)):
                for k in range(i, min(i + dp_group_size, matrix_size)):
                    if j != k:
                        matrix[j, k] = 0.5  # 假设 DP 组内通信量为 0.5

        # TP 通信 (对角线附近)
        for i in range(matrix_size - 2):
            matrix[i, i + 2] = 0.3
            matrix[i + 2, i] = 0.3
        return matrix

    def optimize_topology_for_communication(self):
        """
        根据通信矩阵优化 OCS 连接 (简化示例)。
        """
        for i in range(self.num_ocs_blocks):
            for j in range(self.num_ocs_blocks):
                if i != j:
                    start_index_i = i * self.k
                    end_index_i = (i + 1) * self.k
                    start_index_j = j * self.k
                    end_index_j = (j + 1) * self.k
                    # 判断 i 和 j 对应的 GPU 之间是否有通信
                    if np.any(self.communication_matrix[start_index_i:end_index_i, start_index_j:end_index_j]):
                        # 有的话，就在 OCS 内部建立连接
                        endpoint1_id = i  # 一个 ocs 对应一个 endpoint
                        endpoint2_id = j
                        self.set_ocs_connection(i, endpoint1_id, endpoint2_id)  # 这里简化了

    def save_communication_pairs_to_csv(self, file_path):
        """
        将通信对保存到 CSV 文件。
        """
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["src_pod", "dst_pod", "traffic"])  # 写入表头

            # 遍历所有 GPU 对
            for src_gpu in range(self.num_gpus):
                for dst_gpu in range(self.num_gpus):
                    if src_gpu != dst_gpu:
                        # 获取源 Pod 和目标 Pod
                        src_pod = src_gpu // self.k
                        dst_pod = dst_gpu // self.k
                        # 获取通信流量
                        traffic = self.communication_matrix[src_gpu, dst_gpu]
                        # 如果通信流量大于 0，写入文件
                        if traffic > 0:
                            writer.writerow([src_pod, dst_pod, traffic])

# 示例用法
p = 8
alpha = 1
k = 8
beta = 1
L = 2
link_capacity = 10  # 设置链路容量

topology = NetworkTopology(p, alpha, k, beta, L, link_capacity=link_capacity)

# 计算所有 GPU 对的路径和拥塞
all_pairs_results = topology.calculate_all_pairs_paths_and_congestion()

# 查看链路流量
print("\n链路流量:")
for edge, traffic in topology.link_traffic.items():
    print(f"{edge}: {traffic:.2f}")

# 将通信对保存到 CSV 文件
file_path = r"D:\\simulation\\step1\\通信对.csv"
topology.save_communication_pairs_to_csv(file_path)
print(f"通信对已保存到文件: {file_path}")
