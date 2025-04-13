#-- coding:UTF-8 --
from ctypes import sizeof
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os

np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 或者 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['Times New Roman']

# --- 样式定义 (CDF/Distribution 图) ---
cdf_colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:brown", "mediumvioletred", "dodgerblue", "cyan"]
cdf_markers = ['o', 's', 'D', '^', 'p', '*', '>', '<', '+']
cdf_line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (5, 5))]

# --- 样式定义 (Queue Length 图) ---
queue_styles = ['-', '-.', '--', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
queue_markers = ['o', '>', '8', '*', 'x', '+', 'p', 'D']
queue_colors = ["red", "orange", "blue", "c", "m", "brown", "purple", "dodgerblue", "green"]


# --- 数据加载函数 (CDF/Distribution 图 和 Queue Length 图 共用) ---
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

def get_two_res_from_csv(filename):
    """Loads queue length and time data from a CSV file (for Queue Length plot)."""
    queue_length_list = []
    time_list = []
    try:
        df = pd.read_csv(filename, header=None, names=['queue_length', 'time'], comment='#')
        df['queue_length'] = pd.to_numeric(df['queue_length'], errors='coerce')
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df.dropna(subset=['queue_length', 'time'], inplace=True)
        df_filtered = df[df['time'] > 0].copy()
        queue_length_list = df_filtered['queue_length'].tolist()
        time_list = df_filtered['time'].tolist()
    except FileNotFoundError:
        print(f"Error: File not found - {filename}")
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty - {filename}")
    except Exception as e:
        print(f"Error reading or processing file {filename}: {e}")
    return queue_length_list, time_list

# --- 指标计算函数 (CDF/Distribution 图) ---
def get_valid_time(df_data, task_id, time_type):
    """辅助函数：安全地获取时间值 (CDF/Distribution 图)."""
    if df_data is None: return np.nan
    task_df = df_data[df_data['taskid'] == task_id]
    if task_df.empty: return np.nan
    time_series = task_df.loc[task_df['type'] == time_type, 'value']
    if time_series.empty or pd.isna(time_series.iloc[0]):
        return np.nan
    return time_series.iloc[0]

def get_completion_time(df_data, task_num): # JRT
    res_list = []
    if df_data is None: return []
    for i in range(task_num):
        start_time = get_valid_time(df_data, i, 'start_time')
        finish_time = get_valid_time(df_data, i, 'finish_time')
        if pd.isna(start_time) or pd.isna(finish_time):
            res_list.append(np.nan)
        else:
            runtime = finish_time - start_time
            res_list.append(runtime if runtime >= 0 else np.nan)
    return [t for t in res_list if pd.notna(t)]

def get_finish_time(df_data, task_num): # JCT
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
    return [t for t in res_list if pd.notna(t)]

def get_wait_time(df_data, task_num): # JWT
    res_list = []
    if df_data is None: return []
    for i in range(task_num):
        arriving_time = get_valid_time(df_data, i, 'arriving_time')
        start_time = get_valid_time(df_data, i, 'start_time')
        if pd.isna(arriving_time) or pd.isna(start_time):
            res_list.append(np.nan)
        else:
            wait_time = start_time - arriving_time
            res_list.append(wait_time if wait_time >= 0 else np.nan)
    return [t for t in res_list if pd.notna(t)]


# --- 绘图函数 (CDF/Distribution 图) ---
def draw_cdf_distribution_plot(ax, data_dict, x_label, title, cdf_ylim=(0.9, 1.005), pdf_ylim_max=None, num_bins=25, x_lim=None): # 添加 ax 参数
    """
    绘制包含 CDF 曲线和分布直方图的组合图到指定的子图 ax 上。
    柱状图更宽，有边框，可设置 X 轴范围。
    """
    ax2 = ax.twinx()

    # 过滤掉无效或空的数据系列，并获取有效的调度器名称列表
    valid_data_dict = {k: v for k, v in data_dict.items() if v and any(pd.notna(item) for item in v)}
    if not valid_data_dict:
        print(f"警告: {title} 没有有效数据可绘制。")
        return None
    scheduler_names_valid = list(valid_data_dict.keys())

    all_data_flat = [item for name in scheduler_names_valid for item in valid_data_dict[name] if pd.notna(item)]
    if not all_data_flat:
        print(f"警告: {title} 所有数据值都无效。")
        return None

    # 确定合适的X轴范围和bins (基于所有有效数据)
    x_min_data = min(all_data_flat)
    x_max_data = max(all_data_flat)

    # 如果设置了 x_lim，则使用 x_lim 来确定 bins 的范围
    if x_lim is not None and isinstance(x_lim, (list, tuple)) and len(x_lim) == 2:
        x_min_plot = x_lim[0]
        x_max_plot = x_lim[1]
        # 过滤数据以适应绘图范围 (影响 PDF 计算)
        filtered_data_dict = {}
        for k, data in valid_data_dict.items():
            filtered_data = [d for d in data if pd.notna(d) and x_min_plot <= d <= x_max_plot]
            if filtered_data: # 只保留过滤后仍有数据的系列
                filtered_data_dict[k] = filtered_data
        if not filtered_data_dict:
             print(f"警告: 在指定的 X 轴范围 {x_lim} 内，{title} 没有有效数据。")
             return None
        plot_data_dict = filtered_data_dict # 使用过滤后的数据进行 PDF 计算
        scheduler_names_valid = list(plot_data_dict.keys()) # 更新有效的调度器列表
    else:
        x_min_plot = x_min_data
        x_max_plot = x_max_data
        plot_data_dict = valid_data_dict # 使用原始有效数据

    if x_max_plot <= x_min_plot: x_max_plot = x_min_plot + 1 # 避免范围为0或负
    bin_width = (x_max_plot - x_min_plot) / num_bins
    bins = np.arange(x_min_plot, x_max_plot + bin_width, bin_width) if x_max_plot > x_min_plot else np.array([x_min_plot, x_min_plot + 1])
    if len(bins) < 2: bins = np.array([x_min_plot, x_min_plot + 1])


    max_pdf_val = 0

    # --- 绘制直方图 (PDF) ---
    bar_handles = []
    num_series = len(scheduler_names_valid) # 基于过滤后的有效调度器数量
    bin_diff = np.diff(bins)
    typical_bin_width = bin_diff[0] if len(bin_diff) > 0 else 1.0
    total_bar_width = typical_bin_width * 0.9 # 增加柱子宽度比例
    bar_width = total_bar_width / num_series if num_series > 0 else total_bar_width
    bin_centers = bins[:-1] + np.diff(bins) / 2 if len(bins) > 1 else np.array([bins[0] + 0.5])

    current_count = 0
    # 使用 plot_data_dict 进行循环
    for k in scheduler_names_valid: # 确保顺序一致性
        data = plot_data_dict[k] # 获取过滤后的数据
        # data = [d for d in data if pd.notna(d)] # 再次确认，虽然前面已过滤
        if not data: # 理论上不会发生，因为 plot_data_dict 只包含有数据的
            # current_count += 1 # 不需要增加，因为是基于有效列表循环
            continue

        hist, current_bins = np.histogram(data, bins=bins)
        total_count = len(data) if len(data) > 0 else 1
        freq_percent = (hist / total_count) * 100

        max_pdf_val = max(max_pdf_val, freq_percent.max() if len(freq_percent) > 0 else 0)

        offset = (current_count - (num_series - 1) / 2) * bar_width
        current_bin_centers = bin_centers[:len(freq_percent)] # 对齐长度

        # 获取原始索引以确保颜色/标记一致性
        original_index = list(data_dict.keys()).index(k) # 从原始 data_dict 获取索引

        bars = ax2.bar(current_bin_centers + offset, freq_percent, width=bar_width,
                       color=cdf_colors[original_index % len(cdf_colors)], # 使用原始索引获取颜色
                       alpha=0.75, # 调整透明度
                       label=f'{k} (PDF)',
                       edgecolor='grey', # 添加边框
                       linewidth=0.5)   # 设置边框宽度
        if bars:
            bar_handles.append(bars[0])

        current_count += 1 # 增加有效系列的计数


    ax2.set_ylabel("PDF (%)", fontsize=22)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    if pdf_ylim_max is None:
        pdf_ylim_max_auto = max_pdf_val * 1.15 if max_pdf_val > 0 else 1 # 留更多顶部空间
    else:
        pdf_ylim_max_auto = pdf_ylim_max
    ax2.set_ylim(0, pdf_ylim_max_auto)


    # --- 绘制 CDF (使用原始 valid_data_dict 绘制完整 CDF 曲线) ---
    count = 0
    cdf_handles = []
    cdf_labels = [] # 单独存储标签用于图例
    # 循环原始有效数据字典的键，保持顺序
    for k in list(valid_data_dict.keys()): # 使用原始有效数据
        data = valid_data_dict[k]
        # data = [d for d in data if pd.notna(d)] # 再次确认
        if not data:
            # count += 1 # 不需要，因为是基于有效列表循环
            continue

        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        if len(sorted_data) > 0:
            x_cdf = np.concatenate(([sorted_data[0]], sorted_data))
            y_cdf = np.concatenate(([0], y))
        else:
            x_cdf = np.array([0])
            y_cdf = np.array([0])

        # 获取原始索引以确保颜色/标记一致性
        original_index = list(data_dict.keys()).index(k) # 从原始 data_dict 获取索引

        line, = ax.plot(x_cdf, y_cdf, label=k,
                 linestyle=cdf_line_styles[original_index % len(cdf_line_styles)], # 使用原始索引
                 color=cdf_colors[original_index % len(cdf_colors)],             # 使用原始索引
                 marker=cdf_markers[original_index % len(cdf_markers)],             # 使用原始索引
                 markersize=5,
                 markevery=max(1, int(len(x_cdf) * 0.1)),
                 linewidth=2.5,
                 zorder=3) # 确保 CDF 线在柱状图之上
        cdf_handles.append(line)
        cdf_labels.append(k) # 添加标签
        count += 1

    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel("CDF", fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(cdf_ylim)
    ax.grid(axis='y', linestyle=':')

    # --- 图例和标题 ---
    if cdf_handles:
        legend = ax.legend(cdf_handles, cdf_labels, # 使用收集的句柄和标签
                   loc='upper right', # 图例位置改到左上角，避免遮挡柱状图
                   fontsize=12, ncol=1, frameon=True, title="Schedulers (CDF)") # 确保图例在最上层
        if legend and legend.get_frame():
            legend.get_frame().set_alpha(0.8) # 使图例背景稍微透明

    ax.set_title(title, fontsize=20, pad=15) # 标题字体和padding调整

    # --- 设置 X 轴范围 (移到最后) ---
    if x_lim is not None and isinstance(x_lim, (list, tuple)) and len(x_lim) == 2:
        ax.set_xlim(x_lim[0], x_lim[1])
        print(f"为图 '{title}' 设置 X 轴范围为: {x_lim}")
    elif x_lim is None and len(all_data_flat) > 0: # 如果没设置 x_lim，自动调整以包含所有数据
        ax.set_xlim(left=x_min_data, right=x_max_data) # 使用数据的实际范围


def draw_queue_length_plot(ax, data_sources, title):
    """绘制队列长度随时间变化的折线图到指定的子图 ax 上."""
    plot_count = 0
    for label, filepath in data_sources.items():
        print(f"加载队列长度数据: {label}")
        queue_length, time_list = get_two_res_from_csv(filepath)
        if time_list and queue_length: # 检查数据是否成功加载
            style = queue_styles[plot_count % len(queue_styles)]
            color = queue_colors[plot_count % len(queue_colors)]

            ax.plot(time_list, queue_length, linestyle=style, color=color, label=label, linewidth=1.5) # 调整线宽
            plot_count += 1
        else:
            print(f"  跳过 {label}，数据缺失或无效。")

    ax.set_xlabel("时间 (s)", fontsize=22) # X 轴标签
    ax.set_ylabel("队列长度", fontsize=22) # Y 轴标签
    ax.set_title(title, fontsize=18, pad=12) # 标题和 padding
    ax.tick_params(axis='x', labelsize=18) # 刻度标签字体大小
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', linestyle=':', linewidth=0.5) # 网格线
    ax.legend(fontsize=10, loc='upper right', frameon=True).get_frame().set_alpha(0.8) # 图例，右上角，背景透明


# --- 主程序 ---
if __name__ == "__main__":
    task_nums = 5000
    base_path_cdf = 'D:\\simulation\\rapidNetSim-master\\large_exp_2048GPU\\5000_15\\' # CDF/Distribution base path
    base_path_queue = 'D:\\simulation\\rapidNetSim-master\\large_exp_2048GPU\\5000_15\\' # Queue Length base path

    # --- 调度器名称 ---
    # scheduler_names_in_code = [
    #     'OXC-vClos', 'vClos', 'best',
    #     'Source Routing', 'OXC-vClos-Releax', 'Balanced ECMP', 'ECMP'
    # ]
    # scheduler_names_queue = [ # For Queue Length plot, adjust as needed
    #     'OXC-vClos', 'vClos', 'best',
    #     'Source Routing', 'OXC-vClos-Releax', 'Balanced ECMP', 'ECMP'
    # ]
    scheduler_names_in_code = [
        'OXC-vClos',  'best',
        'Source Routing',  'Balanced ECMP'
    ]
    scheduler_names_queue = [ # For Queue Length plot, adjust as needed
        'OXC-vClos',  'best',
        'Source Routing', 'Balanced ECMP'
    ]
    # --- 文件路径 (CDF/Distribution) ---
    file_paths_cdf = {
        'OXC-vClos': os.path.join(base_path_cdf, 'oxc_scheduler_noopt', 'task_time.log'),
        'vClos': os.path.join(base_path_cdf, 'static_scheduler_locality', 'task_time.log'),
        'best': os.path.join(base_path_cdf, 'static_balance', 'task_time.log'),
        'Source Routing': os.path.join(base_path_cdf, 'static_routing', 'task_time.log'),
        'OXC-vClos-Releax': os.path.join(base_path_cdf, 'oxc_scheduler_noopt_releax', 'task_time.log'),
        'Balanced ECMP': os.path.join(base_path_cdf, 'static_ecmp', 'task_time.log'),
        'ECMP': os.path.join(base_path_cdf, 'static_ecmp_random', 'task_time.log'),
    }

    # --- 文件路径 (Queue Length) ---
    file_paths_queue = {
        'OXC-vClos': os.path.join(base_path_queue, 'oxc_scheduler_noopt', 'queue_length.txt'),
        'vClos': os.path.join(base_path_queue, 'static_scheduler_locality', 'queue_length.txt'),
        'best': os.path.join(base_path_queue, 'static_balance', 'queue_length.txt'),
        'Source Routing': os.path.join(base_path_queue, 'static_routing', 'queue_length.txt'),
        'OXC-vClos-Releax': os.path.join(base_path_queue, 'oxc_scheduler_noopt_releax', 'queue_length.txt'),
        'Balanced ECMP': os.path.join(base_path_queue, 'static_ecmp', 'queue_length.txt'),
        'ECMP': os.path.join(base_path_queue, 'static_ecmp_random', 'queue_length.txt')
    }

    # ---- 加载数据 (CDF/Distribution) ----
    all_dataframes_cdf = {}
    print("--- 加载 CDF/Distribution 数据 ---")
    for name in scheduler_names_in_code:
        path = file_paths_cdf.get(name)
        if path:
            print(f"加载: {name}")
            all_dataframes_cdf[name] = load_csv_get_beta(path)
            if all_dataframes_cdf[name] is None:
                print(f"  -> 加载失败或文件无效: {name}")
        else:
            print(f"警告：未找到调度器 '{name}' 的文件路径配置。")
            all_dataframes_cdf[name] = None

    # ---- 计算指标 (CDF/Distribution) ----
    jrt_data = {}
    jwt_data = {}
    jct_data = {}
    print("\n--- 计算 CDF/Distribution 指标 ---")
    for name in scheduler_names_in_code:
        df = all_dataframes_cdf.get(name)
        if df is not None:
            print(f"计算: {name}")
            jrt_data[name] = get_completion_time(df, task_nums)
            jwt_data[name] = get_wait_time(df, task_nums)
            jct_data[name] = get_finish_time(df, task_nums)
        else:
            jrt_data[name] = []
            jwt_data[name] = []
            jct_data[name] = []

    # ---- 准备 Queue Length 数据源 ----
    data_sources_queue_plot = {name: file_paths_queue[name] for name in scheduler_names_queue if name in file_paths_queue}

    # ---- 创建子图 ----
    fig, axes = plt.subplots(2, 2, figsize=(20, 16)) # 创建 2x2 子图, 调整 figsize
    fig.suptitle(" ", fontsize=28, y=0.93) # 总标题

    # ---- 绘制 CDF/Distribution 图 ----
    print("\n绘制 JRT CDF/Distribution 图...")
    draw_cdf_distribution_plot(axes[0, 0], jrt_data, # ax=axes[0, 0]
                                 x_label="JRT (s)", title="(a) JRT 分布 (CDF & PDF)",
                                 cdf_ylim=(0.9, 1.005), pdf_ylim_max=5, num_bins=20) #  调整 num_bins 和 x_lim

    print("\n绘制 JWT CDF/Distribution 图...")
    draw_cdf_distribution_plot(axes[0, 1], jwt_data, # ax=axes[0, 1]
                                 x_label="JWT (s)", title="(b) JWT 分布 (CDF & PDF)",
                                 cdf_ylim=(0.8, 1.005), pdf_ylim_max=50, num_bins=20) # 调整 num_bins

    print("\n绘制 JCT CDF/Distribution 图...")
    draw_cdf_distribution_plot(axes[1, 0], jct_data, # ax=axes[1, 0]
                                 x_label="JCT (s)", title="(c) JCT 分布 (CDF & PDF)",
                                 cdf_ylim=(0.8, 1.005), pdf_ylim_max=10, num_bins=20) # 调整 num_bins 和 x_lim
    
    # ---- 绘制 Queue Length 图 ----
    print("\n绘制 Queue Length 图...")
    draw_queue_length_plot(axes[1, 1], data_sources_queue_plot, "(d) 队列长度随时间变化") # ax=axes[1, 1]

    # ---- 调整布局和显示 ----
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # 调整 tight_layout 以适应总标题
    plt.show()
