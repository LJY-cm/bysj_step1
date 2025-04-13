import matplotlib.pyplot as plt
import pandas as pd
import os # Import the os module

# --- 图形和字体设置 ---
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用 SimHei 显示中文
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 或者保留 Times New Roman，如果中文能正常显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 增加样式、标记和颜色以容纳更多曲线
styles = ['-', '-.', '--', ':', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)), 'solid']
markers = [' ', '>', '8', '*', 'x', '+', 'p', 'D', 'o']
colors = ["red", "orange", "blue", "c", "cyan",
          "brown", "mediumvioletred", "dodgerblue", "green"]


# --- 数据加载函数 ---
def load_csv_get_beta(filepath):
    """从 CSV 文件加载任务时间数据。"""
    try:
        df = pd.read_csv(filepath, header=None)
        # 增加列名健壮性，处理可能的空文件或格式问题
        if df.shape[1] >= 4:
            df.columns = ['taskidname', 'taskid', 'type', 'value'][:df.shape[1]] # 只取前四列或实际列数
            return df
        else:
            print(f"警告: 文件 {filepath} 的列数少于预期 (需要 4 列)，跳过。")
            return None
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"警告: 文件 {filepath} 为空，跳过。")
        return None
    except Exception as e:
        print(f"加载 {filepath} 时出错: {e}")
        return None


# --- 计算完成时间函数 ---
def get_completion_time(df_data, task_num):
    """计算每个任务的完成时间。"""
    if df_data is None or df_data.empty:
        return []
    res_list = []
    # 确保 taskid 列存在且为数值类型，以便进行比较
    if 'taskid' not in df_data.columns or 'type' not in df_data.columns or 'value' not in df_data.columns:
        print(f"警告: 数据缺少必要的列 ('taskid', 'type', 'value')。")
        return []
    try:
        # 尝试将 taskid 转换为数值类型，忽略无法转换的错误
        df_data['taskid'] = pd.to_numeric(df_data['taskid'], errors='coerce')
        df_data.dropna(subset=['taskid'], inplace=True) # 删除 taskid 为 NaN 的行
        df_data['taskid'] = df_data['taskid'].astype(int) # 转换为整数
    except Exception as e:
        print(f"警告: 转换 'taskid' 列为数值类型时出错: {e}")
        return []

    valid_task_ids = set(df_data['taskid'].unique()) # 使用集合提高查找效率
    actual_tasks_processed = 0

    for i in range(task_num):
        if i not in valid_task_ids:
            # print(f"警告: 未在数据中找到任务 ID {i}。")
            continue

        # 提高查询效率
        task_data = df_data[df_data['taskid'] == i]
        start_time_series = task_data.loc[task_data['type'] == 'start_time', 'value']
        finish_time_series = task_data.loc[task_data['type'] == 'finish_time', 'value']

        # 检查是否成功找到开始和结束时间
        if not start_time_series.empty and not finish_time_series.empty:
            # 假设每个任务只有一个开始和结束时间，取第一个
            start_time = start_time_series.iloc[0]
            finish_time = finish_time_series.iloc[0]

            # 确保时间值为数值类型
            if pd.api.types.is_number(start_time) and pd.api.types.is_number(finish_time):
                if finish_time >= start_time: # 基本健全性检查
                     res_list.append(finish_time - start_time)
                     actual_tasks_processed += 1
                # else:
                #     print(f"警告: 任务 ID {i} 的完成时间 ({finish_time}) 小于开始时间 ({start_time})。")
            # else:
            #     print(f"警告: 任务 ID {i} 的开始或结束时间不是有效的数值。")
        # else:
            # print(f"警告: 任务 ID {i} 缺少开始或完成时间。")

    if actual_tasks_processed == 0 and task_num > 0:
         print(f"警告: 未能为任何任务计算完成时间。请检查数据格式和 task_num ({task_num})。")
    elif actual_tasks_processed < task_num:
        print(f"警告: 预期 {task_num} 个任务，但仅找到 {actual_tasks_processed} 个任务的有效完成时间。")
    return res_list


# --- CDF 绘图函数 ---
def draw_cdf_from_dict(data_dict, title="任务完成时间 CDF"):
    """绘制CDF图
    Input: 接受任意数量的数据，key充当画图的图例，value是画图用的原始数据
    """
    plt.figure(figsize=(8, 5)) # 稍微调整图形尺寸
    count = 0
    valid_method_count = len(data_dict) # 已经预先过滤

    if valid_method_count > len(styles) or valid_method_count > len(markers) or valid_method_count > len(colors):
        print("警告: 样式/标记/颜色不足以区分所有方法，可能存在复用。")

    # record_cdf = {} # 如果需要保存 CDF 数据，取消注释

    for k, data in data_dict.items():
        # 数据已在调用前检查，这里假设 data 是非空列表
        style_idx = count % len(styles)
        marker_idx = count % len(markers)
        color_idx = count % len(colors)

        data = list(data)
        x = sorted(data)
        y = []
        size = len(x)
        # if size == 0: continue # 理论上不会发生，因为已过滤

        for i in range(size):
            y.append((i + 1) / size) # 标准 CDF 计算

        # record_cdf[f'{k}_x'] = x # 如果需要保存 CDF 数据
        # record_cdf[f'{k}_y'] = y # 如果需要保存 CDF 数据

        # 绘制 CDF 线条
        plt.plot(x, y,
                 linestyle=styles[style_idx], color=colors[color_idx], linewidth=2.0) # 线宽稍减
        # 使用不可见点和标记来创建图例项
        plt.plot([], [], label=k, # 使用空列表创建不可见点
                 linestyle=styles[style_idx], color=colors[color_idx], marker=markers[marker_idx],
                 markersize=8, linewidth=2.0) # 调整标记大小

        count += 1

    # --- 图形定制 ---
    plt.ylim(0, 1.02) # Y 轴范围从 0 到 1.02，给顶部留点空间
    plt.xlim(left=0)  # X 轴从 0 开始
    # plt.xlim(0, 500) # 如果需要，设置特定的 X 轴上限
    plt.yticks(fontsize=18) # 调整字体大小
    plt.xticks(fontsize=18)
    plt.xlabel("任务完成时间 (秒)", fontsize=20) # X 轴标签
    plt.ylabel("CDF", fontsize=20)          # Y 轴标签
    plt.title(title, fontsize=22)           # 添加标题
    plt.grid(True, linestyle='--', alpha=0.7) # 网格线样式

    # 调整图例位置和样式
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3, # 将图例放在图下方
               fontsize=14, frameon=False, handletextpad=0.2, handlelength=1.5)

    # --- 可选：保存 CDF 数据到 CSV ---
    # if record_cdf:
    #     try:
    #         max_len = 0
    #         for key in record_cdf: max_len = max(max_len, len(record_cdf[key]))
    #         padded_data = {}
    #         for key in record_cdf:
    #             padding = [float('nan')] * (max_len - len(record_cdf[key]))
    #             padded_data[key] = record_cdf[key] + padding
    #         df_to_save = pd.DataFrame(padded_data)
    #         csv_filename = f'{title.replace(" ", "_").lower()}_cdf_data.csv'
    #         df_to_save.to_csv(csv_filename, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 避免 Excel 中文乱码
    #         print(f"CDF 数据已保存到 {csv_filename}")
    #     except Exception as e:
    #         print(f"保存 CDF 数据到 CSV 时出错: {e}")
    # --- 结束可选保存 ---

    return plt


# --- 主程序 ---
if __name__ == "__main__":
    # --- 配置区 ---
    # !! 重要：设置预期任务数量 !!
    task_nums = 200  # <<<<====== 请根据您的仿真实验调整此值 ======

    # 定义基础路径 (绝对路径，使用正斜杠)
    base_path_512GPU_5000_150 = 'D:/simulation/rapidNetSim-master/large_exp_512GPU/5000_150'

    # 根据图片中的文件夹定义方法列表
    methods = [
        'oxc_scheduler_noopt',
        'oxc_scheduler_noopt_releax', # 注意 'relax' 的拼写
        'static_balance',
        'static_ecmp',
        'static_ecmp_random',
        'static_routing',
        'static_scheduler_locality'
    ]

    # 定义图例标签 (可以自定义，保持与 methods 列表顺序一致)
    legend_labels = {
        'oxc_scheduler_noopt': 'OXC NoOpt',
        'oxc_scheduler_noopt_releax': 'OXC Relax',
        'static_balance': 'Static Balance',
        'static_ecmp': 'Static ECMP',
        'static_ecmp_random': 'Static ECMP Random',
        'static_routing': 'Static Routing',
        'static_scheduler_locality': 'Static Locality'
    }
    # --- 结束配置区 ---


    # --- 数据加载与处理 ---
    completion_times_all = {}
    for method in methods:
        data_path = os.path.join(base_path_512GPU_5000_150, method, 'task_time.log')
        print(f"正在加载: {method} 从 {data_path}")
        df_data = load_csv_get_beta(data_path)
        completion_times = get_completion_time(df_data, task_nums)
        if completion_times: # 只添加有有效数据的条目
            label = legend_labels.get(method, method) # 获取图例标签，如果未定义则使用方法名
            completion_times_all[label] = completion_times
        else:
            print(f"警告: 方法 '{method}' 没有有效的完成时间数据，将不在图中显示。")

    # --- 绘图 ---
    if not completion_times_all:
        print("\n错误：没有加载到任何有效的数据用于绘图。请检查：")
        print(f"  1. 基础路径 '{base_path_512GPU_5000_150}' 是否正确。")
        print(f"  2. 子文件夹 {methods} 是否都存在。")
        print(f"  3. 各子文件夹中的 'task_time.log' 文件是否存在且格式正确。")
        print(f"  4. 'task_nums' ({task_nums}) 是否设置正确。")
    else:
        print("\n开始绘图...")
        plot_title = f'任务完成时间 CDF ({os.path.basename(base_path_512GPU_5000_150)}, N={task_nums})'
        pt = draw_cdf_from_dict(completion_times_all, title=plot_title)

        # 调整布局以适应标题和图例
        plt.subplots_adjust(bottom=0.25, top=0.9) # 增加底部边距给图例，调整顶部边距给标题

        # --- 可选：保存图形 ---
        # graph_dir = './graph_output' # 保存图形的目录
        # if not os.path.exists(graph_dir):
        #     os.makedirs(graph_dir)
        # script_name = os.path.splitext(os.path.basename(__file__))[0]
        # file_name = os.path.join(graph_dir, f"{script_name}_512GPU_5000_150_cdf.pdf")
        # print(f"正在保存图形到: {file_name}")
        # pt.savefig(file_name, bbox_inches='tight', dpi=300) # 使用 bbox_inches='tight' 并指定分辨率
        # --- 结束可选保存 ---

        pt.show()
        print("\n绘图完成。")
