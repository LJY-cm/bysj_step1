#-- coding:UTF-8 --
# from ctypes import sizeof # 不再需要 ctypes
# import matplotlib.pyplot as plt # 不再需要 matplotlib
# import matplotlib.ticker as mticker # 不再需要 matplotlib
import pandas as pd
import numpy as np
import os

np.set_printoptions(suppress=True)
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 绘图相关，不再需要
# plt.rcParams['axes.unicode_minus'] = False # 绘图相关，不再需要
# plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 绘图相关，不再需要

# --- 样式定义 (不再需要绘图样式) ---
# colors = ...
# markers = ...
# line_styles = ...


# --- 数据加载函数 (保持不变) ---
def load_csv_get_beta(filepath):
    try:
        df = pd.read_csv(filepath, header=None)
        df.columns = ['taskidname', 'taskid', 'type', 'value']
        if not all(col in df.columns for col in ['taskid', 'type', 'value']):
             print(f"错误: 文件 {filepath} 缺少必要的列 ('taskid', 'type', 'value')")
             return None
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 {filepath} 是空的。")
        return None
    except Exception as e:
        print(f"加载或处理文件 {filepath} 时出错: {e}")
        return None

# --- 指标计算函数 (保持不变, 但只用到 get_finish_time) ---
def get_valid_time(df_data, task_id, time_type):
    """辅助函数：安全地获取时间值"""
    if df_data is None: return np.nan
    task_df = df_data[df_data['taskid'] == task_id]
    if task_df.empty: return np.nan
    time_series = task_df.loc[task_df['type'] == time_type, 'value']
    if time_series.empty or pd.isna(time_series.iloc[0]):
        return np.nan
    return time_series.iloc[0]

# def get_completion_time(df_data, task_num): # JRT (不再需要)
#     ...

def get_finish_time(df_data, task_num): # JCT (需要)
    res_list = []
    if df_data is None: return []
    for i in range(task_num):
        arriving_time = get_valid_time(df_data, i, 'arriving_time')
        finish_time = get_valid_time(df_data, i, 'finish_time')
        if pd.isna(arriving_time) or pd.isna(finish_time):
            res_list.append(np.nan)
        else:
            completion_time = finish_time - arriving_time
            res_list.append(completion_time if completion_time >= 0 else np.nan)
    return [t for t in res_list if pd.notna(t)] # 只返回有效数值

# def get_wait_time(df_data, task_num): # JWT (不再需要)
#    ...


# --- 绘图函数 (不再需要) ---
# def draw_cdf_distribution_plot(...):
#    ...


# --- 主程序 ---
if __name__ == "__main__":
    task_nums = 5000
    # ---- 数据文件路径 ----
    base_path = 'D:\\simulation\\rapidNetSim-master\\large_exp_512GPU\\5000_150\\'
    scheduler_names_in_code = [
        'OXC NoOpt', 'Static Locality', 'Static Balance',
        'Static Routing', 'OXC Relax', 'Static ECMP', 'Static ECMP Random'
    ]
    file_paths = {
        'OXC NoOpt': os.path.join(base_path, 'oxc_scheduler_noopt', 'task_time.log'),
        'Static Locality': os.path.join(base_path, 'static_scheduler_locality', 'task_time.log'),
        'Static Balance': os.path.join(base_path, 'static_balance', 'task_time.log'),
        'Static Routing': os.path.join(base_path, 'static_routing', 'task_time.log'),
        'OXC Relax': os.path.join(base_path, 'oxc_scheduler_noopt_releax', 'task_time.log'),
        'Static ECMP': os.path.join(base_path, 'static_ecmp', 'task_time.log'),
        'Static ECMP Random': os.path.join(base_path, 'static_ecmp_random', 'task_time.log'),
    }

    # ---- 加载数据 ----
    all_dataframes = {}
    print("--- 开始加载数据 ---")
    for name in scheduler_names_in_code:
        path = file_paths.get(name)
        if path:
            # print(f"加载数据: {name} 从 {path}") # 减少打印信息
            all_dataframes[name] = load_csv_get_beta(path)
            if all_dataframes[name] is None:
                print(f"跳过 {name} 由于加载错误。")
        else:
            print(f"警告：未找到调度器 '{name}' 的文件路径配置。")
            all_dataframes[name] = None
    print("--- 数据加载完成 ---")


    # ---- 计算指标 ----
    # jrt_data = {} # 不再需要
    # jwt_data = {} # 不再需要
    jct_data = {}
    std_dev_jct = {} # 用于存储 Stability (JCT 标准差)

    print("\n--- 开始计算 JCT 和 Stability ---")
    # 确保按定义的顺序计算
    for name in scheduler_names_in_code:
        df = all_dataframes.get(name)
        if df is not None:
            # print(f"计算指标: {name}") # 减少打印信息
            # 计算 JCT 数据
            current_jct_data = get_finish_time(df, task_nums)
            jct_data[name] = current_jct_data # 存储 JCT 数据 (如果后续需要平均值等)

            # 计算 JCT 标准差 (Stability)
            valid_data = [d for d in current_jct_data if pd.notna(d)] # 确保过滤 NaN
            if valid_data: # 只有在有有效数据时才计算标准差
                std_dev_jct[name] = np.std(valid_data)
                # print(f"  -> {name}: JCT Count={len(valid_data)}, StdDev={std_dev_jct[name]:.2f}") # 详细打印
            else:
                std_dev_jct[name] = np.nan # 没有有效数据，标准差为 NaN
                print(f"警告: {name} 没有有效的 JCT 数据计算标准差。")
                # print(f"  -> {name}: JCT Count=0, StdDev=N/A") # 详细打印
        else:
            # 如果数据加载失败，用空列表填充
            jct_data[name] = []
            std_dev_jct[name] = np.nan # 标准差也为 NaN
            print(f"跳过计算: {name} (数据加载失败)")
    print("--- JCT 和 Stability 计算完成 ---")


    # ---- 绘制 CDF/Distribution 图 (不再需要) ----
    # print("\n绘制 JRT CDF/Distribution 图...")
    # ... (所有 draw_cdf_distribution_plot 调用都删除)


    # ---- 打印 Stability (JCT Standard Deviation) ----
    print("\n--- Stability (JCT Standard Deviation) ---")
    print(f"{'Scheduler':<20}\t{'Stability (Std Dev)':<10}")
    for name in scheduler_names_in_code:
        stability_val = std_dev_jct.get(name, np.nan)
        # 检查是否为 NaN 才打印，否则打印 'N/A'
        stability_str = f"{stability_val:<10.2f}" if pd.notna(stability_val) else f"{'N/A':<10}"
        print(f"{name:<20}\t{stability_str}")


    # ---- (可选) 打印平均值 (保持不变，但只打印 JCT 平均值) ----
    print("\n--- Average Values (JCT Only) ---")
    print(f"{'Scheduler':<20}\t{'Avg JCT':<10}")
    # 按定义的顺序打印
    for name in scheduler_names_in_code:
         # 使用 np.mean 计算，get 返回空列表以处理 KeyError
         avg_jct = np.mean(jct_data.get(name, [])) # 计算 JCT 平均值
         # 检查是否为 NaN 才打印，否则打印 'N/A'
         jct_str = f"{avg_jct:<10.2f}" if pd.notna(avg_jct) else f"{'N/A':<10}"
         print(f"{name:<20}\t{jct_str}")

