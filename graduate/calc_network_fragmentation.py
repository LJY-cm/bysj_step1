import numpy as np


def get_fragmentation_from_csv(filename):
    with open(filename, 'r') as f:
        row_list = f.read().splitlines()

    network_fragmentation_list = []
    utilization_time_list = []
    product_sum = 0
    for i in range(1, len(row_list)):
        network_fragmentation, utilization_time = row_list[i].split(',')
        network_fragmentation = float(network_fragmentation)
        utilization_time = float(utilization_time)
        network_fragmentation_list.append(network_fragmentation)
        utilization_time_list.append(utilization_time)

        product_sum += network_fragmentation * utilization_time

    res = 1 - product_sum / sum(utilization_time_list)
    print(res, filename)
    return res


if __name__ == '__main__':
    base_path_512GPU = 'D:/simulation/rapidNetSim-master/large_exp_512GPU/5000_150'
    base_path_8192GPU = 'D:/simulation/rapidNetSim-master/large_exp_8192GPU'
    base_path_4096GPU = 'D:/simulation/rapidNetSim-master/large_exp_4096GPU'
    base_path_512GPU_all2all = 'D:/simulation/rapidNetSim-master/large_exp_512GPU_all2all'


    # get_fragmentation_from_csv(f'{base_path_4096GPU}/oxc_scheduler_noopt/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_4096GPU}/static_balance/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_4096GPU}/static_balance_greedy/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_4096GPU}/static_ecmp/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_4096GPU}/static_ecmp_greedy/network_fragmentation.log.csv')

    # print('----')
    # get_fragmentation_from_csv(f'{base_path_8192GPU}/oxc_scheduler_noopt/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_8192GPU}/static_balance/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_8192GPU}/static_balance_greedy/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_8192GPU}/static_ecmp/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_8192GPU}/static_ecmp_greedy/network_fragmentation.log.csv')

    # print('----')
    # get_fragmentation_from_csv(f'{base_path_512GPU_all2all}/oxc_scheduler_noopt/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_512GPU_all2all}/static_balance/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_512GPU_all2all}/static_balance_greedy/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_512GPU_all2all}/static_ecmp/network_fragmentation.log.csv')
    # get_fragmentation_from_csv(f'{base_path_512GPU_all2all}/static_ecmp_greedy/network_fragmentation.log.csv')

    print('---- 512GPU 5000_150 ----')
    get_fragmentation_from_csv(f'{base_path_512GPU}/oxc_scheduler_noopt/network_fragmentation.log.csv')
    get_fragmentation_from_csv(f'{base_path_512GPU}/oxc_scheduler_noopt_releax/network_fragmentation.log.csv')
    get_fragmentation_from_csv(f'{base_path_512GPU}/static_balance/network_fragmentation.log.csv')
    get_fragmentation_from_csv(f'{base_path_512GPU}/static_ecmp/network_fragmentation.log.csv')
    get_fragmentation_from_csv(f'{base_path_512GPU}/static_ecmp_random/network_fragmentation.log.csv')
    get_fragmentation_from_csv(f'{base_path_512GPU}/static_routing/network_fragmentation.log.csv')
    get_fragmentation_from_csv(f'{base_path_512GPU}/static_scheduler_locality/network_fragmentation.log.csv')
