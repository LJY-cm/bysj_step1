
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

# --- 样式定义 ---
# 确保这个顺序与 scheduler_names_in_code 匹配
colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:brown", "mediumvioletred", "dodgerblue", "cyan"]
markers = ['o', 's', 'D', '^', 'p', '*', '>', '<', '+']
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (5, 5))]


# --- 数据加载函数 ---
def load_csv_get_beta(filepath):
    try:
        df = pd.read_csv(filepath, header=None)
        df.columns = ['taskidname', 'taskid', 'type', 'value']
        if not all(col in df.columns for col in ['taskid', 'type', 'value']):
             print(f"错误: 文件 {filepath} 缺少必要的列 ('taskid', 'type', 'value')")
             return None
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        # 不再打印 NaN 警告，让后续函数处理
        # if df['value'].isnull().any():
        #     print(f"警告: 文件 {filepath} 的 'value' 列包含无法转换为数值的值或原始空值。")
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

# --- 指标计算函数 (增加健壮性) ---
def get_valid_time(df_data, task_id, time_type):
    """辅助函数：安全地获取时间值"""
    if df_data is None: return np.nan
    # 优化查找：先按 taskid 过滤一次，减少后续查找范围
    task_df = df_data[df_data['taskid'] == task_id]
    if task_df.empty: return np.nan
    time_series = task_df.loc[task_df['type'] == time_type, 'value']
    if time_series.empty or pd.isna(time_series.iloc[0]):
        # print(f"警告: Task {task_id} 缺少或无效的 '{time_type}'") # 可选的调试信息
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


# --- 绘图函数 (整合所有修改) ---


# --- 绘图函数 (整合所有修改) ---
def draw_cdf_distribution_plot(data_dict, x_label, title, cdf_ylim=(0.9, 1.005), pdf_ylim_max=None, num_bins=25, x_lim=None): # 默认 num_bins=25, 添加 x_lim
    """
    绘制包含 CDF 曲线和分布直方图的组合图。
    柱状图更宽，有边框，可设置 X 轴范围。
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # 过滤掉无效或空的数据系列，并获取有效的调度器名称列表
    valid_data_dict = {k: v for k, v in data_dict.items() if v and any(pd.notna(item) for item in v)}
    if not valid_data_dict:
        print(f"警告: {title} 没有有效数据可绘制。")
        plt.close(fig)
        return None
    scheduler_names_valid = list(valid_data_dict.keys())

    all_data_flat = [item for name in scheduler_names_valid for item in valid_data_dict[name] if pd.notna(item)]
    if not all_data_flat:
        print(f"警告: {title} 所有数据值都无效。")
        plt.close(fig)
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
             plt.close(fig)
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
                       color=colors[original_index % len(colors)], # 使用原始索引获取颜色
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

        line, = ax1.plot(x_cdf, y_cdf, label=k,
                 linestyle=line_styles[original_index % len(line_styles)], # 使用原始索引
                 color=colors[original_index % len(colors)],             # 使用原始索引
                 marker=markers[original_index % len(markers)],             # 使用原始索引
                 markersize=5,
                 markevery=max(1, int(len(x_cdf) * 0.1)),
                 linewidth=2.5)
        cdf_handles.append(line)
        cdf_labels.append(k) # 添加标签
        count += 1

    ax1.set_xlabel(x_label, fontsize=24)
    ax1.set_ylabel("CDF", fontsize=24)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.set_ylim(cdf_ylim)
    ax1.grid(axis='y', linestyle=':')

    # --- 图例和标题 ---
    if cdf_handles:
        ax1.legend(cdf_handles, cdf_labels, # 使用收集的句柄和标签
                   loc='center left', bbox_to_anchor=(0.01, 0.7),
                   fontsize=14, ncol=1, frameon=True, title="Schedulers (CDF)")

    plt.title(title, fontsize=26, pad=20)

    # --- 设置 X 轴范围 (移到最后) ---
    if x_lim is not None and isinstance(x_lim, (list, tuple)) and len(x_lim) == 2:
        ax1.set_xlim(x_lim[0], x_lim[1])
        print(f"为图 '{title}' 设置 X 轴范围为: {x_lim}")
    elif x_lim is None and len(all_data_flat) > 0: # 如果没设置 x_lim，自动调整以包含所有数据
        ax1.set_xlim(left=x_min_data, right=x_max_data) # 使用数据的实际范围
        # 可选：添加一点边距
        # padding = (x_max_data - x_min_data) * 0.05
        # ax1.set_xlim(left=x_min_data - padding, right=x_max_data + padding)


    try:
      fig.tight_layout()
    except ValueError:
      print(f"警告：tight_layout 在 {title} 图中失败。")
    return plt


# --- 主程序 ---
# --- 主程序 ---
if __name__ == "__main__":
    task_nums = 5000
    # ---- 数据文件路径 ----
    base_path = 'D:\\simulation\\rapidNetSim-master\\large_exp_512GPU\\5000_165\\'
    # 使用您代码中定义的调度器名称，确保与加载数据的键一致
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
    # 使用定义的顺序加载，确保后续颜色/标记按此顺序分配
    for name in scheduler_names_in_code:
        path = file_paths.get(name) # 获取路径
        if path:
            print(f"加载数据: {name} 从 {path}")
            all_dataframes[name] = load_csv_get_beta(path)
            if all_dataframes[name] is None:
                print(f"跳过 {name} 由于加载错误。")
        else:
            print(f"警告：未找到调度器 '{name}' 的文件路径配置。")
            all_dataframes[name] = None # 标记为 None


    # ---- 计算指标 ----
    jrt_data = {}
    jwt_data = {}
    jct_data = {}

    # 确保按定义的顺序计算
    for name in scheduler_names_in_code:
        df = all_dataframes.get(name)
        if df is not None:
            print(f"计算指标: {name}")
            jrt_data[name] = get_completion_time(df, task_nums)
            jwt_data[name] = get_wait_time(df, task_nums)
            jct_data[name] = get_finish_time(df, task_nums) # JCT 数据在这里计算
        else:
            # 如果数据加载失败，用空列表填充，绘图函数能处理空列表
            jrt_data[name] = []
            jwt_data[name] = []
            jct_data[name] = []


    # ---- 绘制 JRT CDF/Distribution 图 (类似图 a) ----
    print("\n绘制 JRT CDF/Distribution 图...")
    jrt_plot = draw_cdf_distribution_plot(jrt_data,
                                         x_label="JRT (s)",
                                         title="CDF and Distribution of JRT",
                                         cdf_ylim=(0.9, 1.005),
                                         pdf_ylim_max=5, # JRT PDF Y 轴上限可能需要根据减少bins后调整
                                         num_bins=25) # <--- 修改这里
    if jrt_plot:
        jrt_plot.show()

    # ---- 绘制 JWT CDF/Distribution 图 (类似图 b) ----
    print("\n绘制 JWT CDF/Distribution 图...")
    jwt_plot = draw_cdf_distribution_plot(jwt_data,
                                         x_label="JWT (s)",
                                         title="CDF and Distribution of JWT",
                                         cdf_ylim=(0.8, 1.005),
                                         pdf_ylim_max=50, # JWT PDF Y 轴上限可能需要根据减少bins后调整
                                         num_bins=25) # <--- 修改这里
    if jwt_plot:
        jwt_plot.show()

    # ---- 绘制 JCT CDF/Distribution 图 (作为图 c) ----
    print("\n绘制 JCT CDF/Distribution 图...")
    jct_plot = draw_cdf_distribution_plot(jct_data,
                                         x_label="JCT (s)",
                                         title="CDF and Distribution of JCT",
                                         cdf_ylim=(0.8, 1.005),
                                         pdf_ylim_max=None,
                                         num_bins=25) # <--- 修改这里
    if jct_plot:
        jct_plot.show()


    # ---- 计算并绘制 Stability (JCT Standard Deviation) 图 ----
    # (这部分代码保持不变，但将其注释掉，因为之前讨论过它绘制的是不同类型的数据)
    # print("\n计算并绘制 JCT Stability 图...")
    # std_dev_jct = {}
    #  # 按定义的顺序计算标准差
    # for name in scheduler_names_in_code:
    #     data = jct_data.get(name, []) # 获取数据，默认为空列表
    #     valid_data = [d for d in data if pd.notna(d)] # 过滤掉 NaN 值
    #     if valid_data: # 只有在有有效数据时才计算
    #         std_dev_jct[name] = np.std(valid_data)
    #     else:
    #         std_dev_jct[name] = np.nan # 标记为 NaN
    #         print(f"警告: {name} 没有有效的 JCT 数据计算标准差。")
    #
    # print("JCT Standard Deviations:", std_dev_jct)
    #
    # # 绘制标准差折线图 (现在会画成水平线样式)
    # stability_plot = draw_std_line_chart(std_dev_jct,
    #                                      ylabel="Stability (JCT Standard Deviation)",
    #                                      title="Stability of JCT for Different Schedulers") # 更新标题以匹配图形样式
    # if stability_plot:
    #     stability_plot.show()

    # ---- (可选) 打印平均值 ----
    print("\n--- Average Values ---")
    print(f"{'Scheduler':<20}\t{'Avg JRT':<10}\t{'Avg JWT':<10}\t{'Avg JCT':<10}")
    # 按定义的顺序打印
    for name in scheduler_names_in_code:
         # 使用 nanmean 计算平均值，自动忽略 NaN 或空列表
         avg_jrt = np.nanmean(jrt_data.get(name, np.nan))
         avg_jwt = np.nanmean(jwt_data.get(name, np.nan))
         avg_jct = np.nanmean(jct_data.get(name, np.nan))
         # 检查是否为 NaN 才打印，否则打印 'N/A'
         jrt_str = f"{avg_jrt:<10.2f}" if not np.isnan(avg_jrt) else f"{'N/A':<10}"
         jwt_str = f"{avg_jwt:<10.2f}" if not np.isnan(avg_jwt) else f"{'N/A':<10}"
         jct_str = f"{avg_jct:<10.2f}" if not np.isnan(avg_jct) else f"{'N/A':<10}"
         print(f"{name:<20}\t{jrt_str}\t{jwt_str}\t{jct_str}")



# # 将数据放入字典中
#     data = {
#         "oxc_scheduler_noopt": oxc_scheduler_noopt2,
#         "oxc_scheduler_noopt_releax": oxc_scheduler_noopt_releax2,
#         "static_scheduler_locality": static_scheduler_locality_2,
#         "static_balance": static_balance2,
#         "static_routing": static_routing2,
#         "static_ecmp": static_ecmp2,
#         "static_ecmp_random": static_ecmp_random2
#     }

#     #-------------------- 2. 调用函数 --------------------
#     # 调用函数，并自定义标签和标题
#     plt = draw_cdf_from_dict(data)

#     #-------------------- 3. 显示图表 --------------------
#     plt.show()

    # print("JRT",end="&")
    # print(sum(oxc_scheduler_noopt1)*1/len(oxc_scheduler_noopt1),end="&")
    # print(sum(static_scheduler_locality_1)*1/len(static_scheduler_locality_1),end="&")
    # print(sum(static_balance1)*1/len(static_balance1),end="&")
    # print(sum(oxc_scheduler_noopt_releax1)*1/len(oxc_scheduler_noopt_releax1),end="&")
    # print(sum(static_routing1)*1/len(static_routing1))
    # print(sum(static_ecmp1)*1/len(static_ecmp1))
    # print(sum(static_ecmp_random1)*1/len(static_ecmp_random1))

    # print()
    # print("JWT",end="&")
    # print(sum(oxc_scheduler_noopt2)*1/len(oxc_scheduler_noopt2),end="&")
    # print(sum(static_scheduler_locality_2)*1/len(static_scheduler_locality_2),end="&")
    # print(sum(static_balance2)*1/len(static_balance2),end="&")
    # print(sum(oxc_scheduler_noopt_releax2)*1/len(oxc_scheduler_noopt_releax2),end="&")
    # print(sum(static_routing2)*1/len(static_routing2))
    # print(sum(static_ecmp2)*1/len(static_ecmp2))
    # print(sum(static_ecmp_random2)*1/len(static_ecmp_random2))

    # print()
    # print("JCT",end="&")
    # print(sum(oxc_scheduler_noopt3)*1/len(oxc_scheduler_noopt3),end="&")
    # print(sum(static_scheduler_locality_3)*1/len(static_scheduler_locality_3),end="&")
    # print(sum(static_balance3)*1/len(static_balance3),end="&")
    # print(sum(oxc_scheduler_noopt_releax3)*1/len(oxc_scheduler_noopt_releax3),end="&")
    # print(sum(static_routing3)*1/len(static_routing3))
    # print(sum(static_ecmp3)*1/len(static_ecmp3))
    # print(sum(static_ecmp_random3)*1/len(static_ecmp_random3))
    # print()
