import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams['font.sans-serif'] = ['Times New Roman']

# Increased number of styles, markers, and colors to accommodate more lines
styles = ['-', '-.', '--', ':', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)), 'solid']
markers = [' ', '>', '8', '*', 'x', '+', 'p', 'D', 'o'] # Added 'o' marker
colors = ["red", "orange", "blue", "c", "cyan",
          "brown", "mediumvioletred", "dodgerblue", "green"]


def load_csv_get_beta(filepath):
    """Loads task time data from a CSV file."""
    try:
        df = pd.read_csv(filepath, header=None)
        df.columns = ['taskidname', 'taskid', 'type', 'value']
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_completion_time(df_data, task_num):
    """Calculates completion time for each task."""
    if df_data is None:
        return []
    res_list = []
    # Ensure task IDs are within the expected range and data exists
    valid_task_ids = df_data['taskid'].unique()
    actual_tasks_processed = 0
    for i in range(task_num):
        if i not in valid_task_ids:
            # print(f"Warning: Task ID {i} not found in the data.")
            continue # Skip if task ID doesn't exist

        start_time_series = df_data.loc[(df_data['taskid'] == i) & (df_data['type'] == 'start_time'), 'value']
        finish_time_series = df_data.loc[(df_data['taskid'] == i) & (df_data['type'] == 'finish_time'), 'value']

        if not start_time_series.empty and not finish_time_series.empty:
            start_time = start_time_series.values[0]
            finish_time = finish_time_series.values[0]
            res_list.append(finish_time - start_time)
            actual_tasks_processed += 1
        # else:
            # print(f"Warning: Missing start or finish time for Task ID {i}.")

    if actual_tasks_processed < task_num:
        print(f"Warning: Expected {task_num} tasks, but only found completion times for {actual_tasks_processed}.")
    return res_list


def draw_cdf_from_dict(data_dict):
    """绘制CDF图
    Input: 接受任意数量的数据，key充当画图的图例，value是画图用的原始数据
    """
    plt.figure(figsize=(6, 4))
    # 适配曲线数量
    count = 0
    valid_method_count = len(data_dict)
    if valid_method_count > len(styles) or valid_method_count > len(markers) or valid_method_count > len(colors):
        print("Warning: Not enough unique styles/markers/colors for all methods. Some may be reused.")

    for k, data in data_dict.items():
        if not data: # Skip if data list is empty (e.g., file not found or no tasks completed)
             print(f"Skipping plot for '{k}' due to missing data.")
             continue

        style_idx = count % len(styles)
        marker_idx = count % len(markers)
        color_idx = count % len(colors)

        data = list(data)
        x = sorted(data)
        # print(f"Plotting for {k}: {len(x)} data points") # Debug print
        y = []
        size = len(x)
        if size == 0:
             print(f"Skipping plot for '{k}' as data size is 0 after sorting.")
             continue # Skip if no valid completion times

        for i in range(size):
            # y.append(1 - i / size) # CCDF
            y.append((i + 1) / size) # CDF (start from 1/size, end at 1)
        
        # Plot the main CDF line
        plt.plot(x, y, # label=k, # Labeling done with the invisible line
                 linestyle=styles[style_idx], color=colors[color_idx], linewidth=2.5)
        
        # Plot an invisible line with marker for the legend
        plt.plot([-10], [-10], label=k, # Use negative coordinates to hide the point
                 linestyle=styles[style_idx], color=colors[color_idx], marker=markers[marker_idx], linewidth=2.5)
        
        # --- Scatter plot logic (currently disabled as scatter_value is empty) ---
        # scatter_x = []
        # scatter_y = []
        # scatter_value = [] # Define values like [0.9, 0.95, 0.99] if you want markers at specific CDF points
        # for val in scatter_value:
        #     found = False
        #     for i in range(size):
        #         if y[i] >= val:
        #             scatter_x.append(x[i])
        #             scatter_y.append(y[i])
        #             found = True
        #             break
        #     # if not found: # Optional: handle case where value is never reached
        #     #     scatter_x.append(x[-1])
        #     #     scatter_y.append(y[-1])

        # if scatter_x: # Only plot scatter if points were found
        #     plt.scatter(scatter_x, scatter_y,
        #                 marker=markers[marker_idx], s=100, color=colors[color_idx], zorder=5) # Ensure markers are on top
        # --- End Scatter plot logic ---

        count += 1

    # --- Plot Customization ---
    plt.ylim(0, 1) # Show the full CDF range 0 to 1
    plt.xlim(left=0) # Start x-axis at 0, adjust right limit if needed based on data
    # plt.xlim(0, 500) # Or set a specific upper limit like before if desired
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    # plt.yscale("symlog", linthreshy=0.0001) # Optional log scale
    plt.xlabel("Completion Time (s)", fontsize=24) # Added units
    plt.ylabel("CDF", fontsize=24)
    plt.grid(True) # Turn grid on
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc="lower center", fontsize=16, # Adjusted fontsize
               mode="expand", borderaxespad=0, ncol=3, frameon=False,
               handletextpad=0.1, handlelength=1.0) # Adjusted handlelength
    return plt


if __name__ == "__main__":
    base_path_512GPU_5000_150 = 'D:/simulation/rapidNetSim-master/large_exp_512GPU/5000_150'

    # List of methods based on the folder names in the image
    methods = [
        'oxc_scheduler_noopt',
        'oxc_scheduler_noopt_releax',
        'static_balance',
        'static_ecmp',
        'static_ecmp_random',
        'static_routing',
        'static_scheduler_locality'
    ]

    # Dictionary to hold completion time data for each method
    data_dict = {}

    # --- IMPORTANT: Set the expected number of tasks ---
    # This should match the number of tasks submitted in your simulation experiments.
    # If task IDs are 0 to N-1, task_num should be N.
    task_num = 200  # <<<<====== PLEASE ADJUST THIS VALUE BASED ON YOUR EXPERIMENT ======

    # Loop through each method, load data, and calculate completion times
    for method in methods:
        data_path = os.path.join(base_path_512GPU_5000_150, method, 'task_time.log')
        print(f"Loading data for: {method} from {data_path}")
        df_data = load_csv_get_beta(data_path)
        completion_times = get_completion_time(df_data, task_num)
        data_dict[method] = completion_times # Store the results

    # Draw the CDF plot using the collected data
    pt = draw_cdf_from_dict(data_dict)

    pt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to prevent legend overlap
    # --- Optional: Save the plot ---
    # graph_path = './graph/' # Save in a 'graph' subdirectory relative to the script
    # if not os.path.isdir(graph_path):
    #     os.makedirs(graph_path)
    # script_name = os.path.splitext(os.path.basename(__file__))[0] # Get script name without extension
    # file_name = os.path.join(graph_path, f"{script_name}_cdf.pdf")
    # print(f"Saving plot to: {file_name}")
    # pt.savefig(file_name, bbox_inches='tight')
    # --- End Optional Save ---

    pt.show()
