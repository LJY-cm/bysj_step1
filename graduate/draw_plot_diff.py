import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np # Import numpy for potential calculations like mean/median

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False # Good practice

# --- Styles, Markers, Colors (keep as is or expand if needed) ---
styles = ['-', '-.', '--', ':', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10)), 'solid']
markers = [' ', '>', '8', '*', 'x', '+', 'p', 'D', 'o'] # Add more if needed
colors = ["red", "orange", "blue", "green", "purple", "brown", "cyan", "magenta", "lime"] # Add more if needed


# --- Data Loading Function (keep as is) ---
def load_csv_get_beta(filepath):
    """Loads task time data from a CSV/log file."""
    try:
        df = pd.read_csv(filepath, header=None)
        df.columns = ['taskidname', 'taskid', 'type', 'value']
        # Ensure taskid is integer and value is numeric
        df['taskid'] = pd.to_numeric(df['taskid'], errors='coerce').astype('Int64') # Use Int64 to handle potential NaN
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['taskid', 'value'], inplace=True) # Remove rows where conversion failed
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

# --- Completion Time Calculation Function (Modified for Robustness) ---
def get_completion_time(df_data):
    """Calculates completion time (finish - start) for each task ID."""
    if df_data is None or df_data.empty:
        return [], 0 # Return empty list and 0 tasks if no data

    completion_times = {}
    # Find the maximum task ID present in the data to determine task_num dynamically
    if 'taskid' not in df_data.columns or df_data['taskid'].isnull().all():
        print("Warning: 'taskid' column missing or empty.")
        return [], 0
    try:
        # Filter out non-numeric taskids if any slipped through
        task_ids = pd.to_numeric(df_data['taskid'], errors='coerce').dropna().unique()
        if len(task_ids) == 0:
            print("Warning: No valid numeric task IDs found.")
            return [], 0
        max_task_id = int(max(task_ids))
        task_num = max_task_id + 1 # Assuming task IDs are 0-based index
    except Exception as e:
        print(f"Error determining task number: {e}")
        return [], 0


    print(f"  Expecting up to {task_num} tasks (max task ID found: {max_task_id}).")
    res_list = [np.nan] * task_num # Initialize with NaN

    grouped = df_data.groupby('taskid')

    for taskid, group in grouped:
        if taskid >= task_num or taskid < 0: # Safety check
             print(f"  Skipping invalid task ID: {taskid}")
             continue

        start_time_val = group.loc[group['type'] == 'start_time', 'value']
        finish_time_val = group.loc[group['type'] == 'finish_time', 'value']

        if not start_time_val.empty and not finish_time_val.empty:
            start_time = start_time_val.iloc[0]
            finish_time = finish_time_val.iloc[0]
            # Ensure times are valid numbers before subtracting
            if pd.notna(start_time) and pd.notna(finish_time):
                 completion_time = finish_time - start_time
                 if completion_time >= 0: # Basic sanity check
                     res_list[taskid] = completion_time
                 else:
                     print(f"  Warning: Negative completion time for task {taskid} ({finish_time} - {start_time}). Setting to NaN.")
            else:
                 print(f"  Warning: Missing start or finish time value for task {taskid}. Setting completion time to NaN.")
        else:
            # print(f"  Warning: Missing start_time or finish_time entry for task {taskid}.") # Optional: reduce verbosity
            pass # Keep NaN

    # Count how many valid completion times were calculated
    valid_times_count = sum(1 for t in res_list if pd.notna(t))
    print(f"  Calculated {valid_times_count} valid completion times out of {task_num} potential tasks.")

    # Check if the list is mostly NaNs, which might indicate a problem
    if valid_times_count < task_num * 0.5: # Example threshold: less than 50% valid
        print(f"  Warning: Many tasks ({task_num - valid_times_count}) seem to be missing completion times.")

    return res_list, task_num


# --- Plotting Function (Modified for Clarity and Flexibility) ---
def draw_plot_from_dict(data_dict, baseline_label):
    """Plots multiple slowdown curves."""
    plt.figure(figsize=(10, 6)) # Adjust size as needed
    count = 0

    if not data_dict:
        print("No slowdown data to plot.")
        # Optionally plot empty axes
        plt.xlabel("Task ID", fontsize=14)
        plt.ylabel(f"Completion Time Slowdown (vs {baseline_label})", fontsize=14)
        plt.title("Task Completion Time Slowdown (No Data)", fontsize=16)
        plt.grid(True)
        plt.show()
        return plt # Return the plt object

    print("\nPlotting Slowdown Curves:")
    max_tasks = 0 # Keep track of the maximum task index needed for x-axis

    for k, data in data_dict.items():
        # Filter out NaN values for plotting and analysis, but keep track of original indices
        y_raw = np.array(data)
        valid_indices = np.where(pd.notna(y_raw))[0]
        y_plot = y_raw[valid_indices]
        x_plot = valid_indices # Use the original task IDs as x-values

        if len(x_plot) == 0:
            print(f"- Skipping '{k}': No valid slowdown data points.")
            continue

        if len(x_plot) > 0:
             max_tasks = max(max_tasks, x_plot[-1]) # Update max task ID seen

        # Calculate mean/median slowdown for the label (optional, but informative)
        mean_slowdown = np.mean(y_plot) if len(y_plot) > 0 else np.nan
        median_slowdown = np.median(y_plot) if len(y_plot) > 0 else np.nan
        print(f"- Plotting '{k}'. Mean Slowdown: {mean_slowdown:.3f}, Median Slowdown: {median_slowdown:.3f}")

        label = f"{k} (Median: {median_slowdown:.2f})" # Use median in legend as it's less sensitive to outliers

        style_idx = count % len(styles)
        color_idx = count % len(colors)
        # marker_idx = count % len(markers) # Usually not needed for potentially many points

        plt.plot(x_plot, y_plot, label=label,
                 linestyle=styles[style_idx], color=colors[color_idx],
                 # marker=markers[marker_idx], markevery=max(1, len(x_plot)//20), markersize=4, # Optional markers
                 linewidth=1.5) # Adjust linewidth
        count += 1

    # --- Plot Customization ---
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--') # Line at y=0 (no slowdown)
    plt.xlim(left=-0.05 * max_tasks if max_tasks else 0, right=max_tasks * 1.05 if max_tasks else 100) # Set x-limit based on data
    # plt.ylim(bottom=XXX, top=YYY) # Set y-limits if needed, e.g., based on min/max slowdown observed
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Task ID", fontsize=14)
    plt.ylabel(f"Completion Time Slowdown (vs {baseline_label})", fontsize=14) # Make label dynamic
    plt.title("Task Completion Time Slowdown Comparison", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust legend positioning
    plt.legend(fontsize=10, loc='best') # 'best' tries to find a good spot
    # Or place outside:
    # plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(3, count), fancybox=True)

    return plt


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    base_data_dir = 'D:/simulation/rapidNetSim-master/large_exp_512GPU/5000_150/' # ADJUST THIS PATH if needed
    log_filename = 'task_time.log' # The name of the log file in each subfolder

    # Define the baseline method
    baseline_method_name = 'oxc_scheduler_noopt'

    # List of all methods directories (automatically find them or list manually)
    try:
        all_methods = sorted([d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))])
        print(f"Found methods: {all_methods}")
    except FileNotFoundError:
        print(f"Error: Base directory not found: {base_data_dir}")
        exit()

    # Optional: Define shorter labels for the legend
    method_labels = {
        'oxc_scheduler_noopt': 'OXC NoOpt (Baseline)',
        'oxc_scheduler_noopt_releax': 'OXC Relax',
        'static_balance': 'Static Balance',
        'static_ecmp': 'Static ECMP',
        'static_ecmp_random': 'Static ECMP Random',
        'static_routing': 'Static Routing',
        'static_scheduler_locality': 'Static Locality',
        # Add more if needed, or it will use the directory name
    }
    # --- End Configuration ---

    # --- Load Baseline Data ---
    baseline_label = method_labels.get(baseline_method_name, baseline_method_name)
    baseline_filepath = os.path.join(base_data_dir, baseline_method_name, log_filename)
    print(f"\nLoading baseline data: {baseline_label} from {baseline_filepath}")
    baseline_df = load_csv_get_beta(baseline_filepath)
    if baseline_df is None:
        print("FATAL: Could not load baseline data. Exiting.")
        exit()

    baseline_completion_times, baseline_task_num = get_completion_time(baseline_df)
    if not baseline_completion_times or baseline_task_num == 0:
        print("FATAL: Could not calculate baseline completion times. Exiting.")
        exit()
    print(f"Baseline '{baseline_label}' has {baseline_task_num} tasks.")


    # --- Load Comparison Data and Calculate Slowdown ---
    slowdown_data_dict = {}
    print("\nLoading comparison data and calculating slowdown:")

    for method_name in all_methods:
        # Skip comparing the baseline against itself for slowdown plot
        if method_name == baseline_method_name:
            continue

        label = method_labels.get(method_name, method_name)
        filepath = os.path.join(base_data_dir, method_name, log_filename)
        print(f"- Processing: {label} from {filepath}")

        comp_df = load_csv_get_beta(filepath)
        if comp_df is None:
            print(f"  Skipping {label}: Failed to load data.")
            continue

        comp_completion_times, comp_task_num = get_completion_time(comp_df)

        if not comp_completion_times or comp_task_num == 0:
            print(f"  Skipping {label}: Failed to calculate completion times.")
            continue

        # Ensure task numbers match or handle appropriately (e.g., align based on task ID)
        # For simplicity, assuming the number of tasks should ideally be the same.
        # We'll calculate slowdown for tasks present in both, up to the minimum length.
        num_tasks_to_compare = min(baseline_task_num, comp_task_num)
        if baseline_task_num != comp_task_num:
             print(f"  Warning: Task number mismatch! Baseline has {baseline_task_num}, {label} has {comp_task_num}. Comparing up to {num_tasks_to_compare} tasks.")


        slowdown = [np.nan] * num_tasks_to_compare # Initialize slowdown list for this method

        valid_slowdown_count = 0
        for i in range(num_tasks_to_compare):
            comp_time = comp_completion_times[i]
            base_time = baseline_completion_times[i]

            # Check if both times are valid numbers
            if pd.notna(comp_time) and pd.notna(base_time):
                if base_time > 1e-9: # Avoid division by zero or near-zero
                    slowdown[i] = (comp_time - base_time) / base_time
                    valid_slowdown_count += 1
                elif comp_time > 1e-9:
                     # Base time is zero/negligible, comp time is positive -> treat as large slowdown? Or skip?
                     # Option 1: Assign a large number (can skew plots)
                     # slowdown[i] = 1e6 # Example large number
                     # Option 2: Set to NaN (safer)
                     slowdown[i] = np.nan
                     print(f"  Task {i}: Baseline time is zero, comparison time is {comp_time}. Setting slowdown to NaN.")
                else:
                     # Both times are zero/negligible, difference is zero
                     slowdown[i] = 0.0
                     valid_slowdown_count += 1
            # Else: one or both times are NaN, keep slowdown[i] as NaN

        print(f"  Calculated {valid_slowdown_count} valid slowdown values for {label}.")
        if valid_slowdown_count > 0:
            slowdown_data_dict[label] = slowdown
        else:
            print(f"  Skipping {label}: No valid slowdown values could be calculated.")


    # --- Plot the results ---
    pt = draw_plot_from_dict(slowdown_data_dict, baseline_label) # Pass baseline label for axis title

    pt.tight_layout() # Adjust layout to prevent labels overlapping

    # --- Optional: Save Plot ---
    graph_dir = './graph_output' # Define output directory
    if not os.path.exists(graph_dir):
        try:
            os.makedirs(graph_dir)
        except OSError as e:
            print(f"Could not create graph directory {graph_dir}: {e}")
            graph_dir = '.' # Save in current directory as fallback

    file_name = os.path.join(graph_dir, "task_completion_slowdown_comparison.pdf")
    try:
        print(f"\nSaving plot to: {file_name}")
        pt.savefig(file_name, bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Error saving plot: {e}")
    # --- End Optional Save ---

    pt.show() # Display the plot

    print("\nScript finished.")
