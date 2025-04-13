# -- coding:UTF-8 --
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Only needed if you load data from pandas later

# --- Basic Matplotlib Setup ---
plt.rcParams['font.sans-serif'] = ['SimHei'] # Use SimHei font
plt.rcParams['axes.unicode_minus'] = False # Display minus sign correctly

# --- Data (FINAL REAL DATA) ---
# Use the same scheduler names as in your previous code
scheduler_names = [
    'OXC NoOpt', 'Static Locality', 'Static Balance',
    'Static Routing', 'OXC Relax', 'Static ECMP', 'Static ECMP Random'
]

# Metrics shown on the X-axis
metrics = ['JRT', 'JWT', 'JCT', 'Stability']

# Final real data including Stability (JCT Std Dev)
# Each inner list corresponds to a scheduler and contains values for [JRT, JWT, JCT, Stability]
# The order MUST match the order in scheduler_names
real_data = {
    'OXC NoOpt':          [5292.17, 3686.94,  8979.11, 20811.89],
    'Static Locality':    [5292.17, 4711.96, 10004.13, 21554.20],
    'Static Balance':     [5292.17, 3474.85,  8767.02, 20640.27],
    'Static Routing':     [5352.41, 4599.72,  9952.13, 21790.54],
    'OXC Relax':          [5768.57, 4854.77, 10623.34, 24451.53],
    'Static ECMP':        [5481.53, 4378.23,  9859.76, 22025.14],
    'Static ECMP Random': [5829.57, 9638.35, 15467.92, 27901.14]
    # Add more entries if you have more schedulers
}

# --- Style Definitions (Consistent with your previous code) ---
# Use the exact colors list you defined previously
# Ensure this list has enough colors for all your schedulers
colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:brown", "mediumvioletred", "dodgerblue", "cyan"]

# --- Plotting Logic ---
num_metrics = len(metrics)
# Use the number of keys in real_data to determine num_schedulers
num_schedulers = len(real_data)
# Ensure scheduler_names list matches the keys in real_data
if num_schedulers != len(scheduler_names):
    print("警告：scheduler_names 列表的长度与 real_data 中的调度器数量不匹配。")
    # Optionally adjust scheduler_names based on data keys or raise an error
    scheduler_names = list(real_data.keys()) # Use keys from data as names

# Positions for the groups on the x-axis
x_pos = np.arange(num_metrics)

# Width of a single bar
total_width = 0.8 # Total width allocated for the bars in one group
bar_width = total_width / num_schedulers

fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed

# Iterate through each scheduler to plot its bars
for i, scheduler_name in enumerate(scheduler_names):
    # Get the data for the current scheduler from real_data
    scheduler_values = real_data.get(scheduler_name, [0] * num_metrics) # Default to 0 if name not found

    # Calculate the offset for the bars of the current scheduler
    offset = (i - (num_schedulers - 1) / 2) * bar_width

    # Plot the bars for the current scheduler
    rects = ax.bar(x_pos + offset, scheduler_values, bar_width,
                   label=scheduler_name,
                   color=colors[i % len(colors)]) # Use colors from the list

# --- Customize the Plot ---
ax.set_ylabel('值 (s)', fontsize=14) # Set Y-axis label (Added units)
ax.set_title('(a) CLUSTER512', fontsize=16, y=-0.15) # Set title and move it down a bit

# Set the positions and labels for the x-axis ticks
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=14)

# Add a legend
# Place it above the plot, adjust ncol for better layout if many schedulers
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=10)

# Add horizontal grid lines (dotted style like the image)
ax.yaxis.grid(True, linestyle=':', linewidth=0.7)
ax.set_axisbelow(True) # Ensure grid is drawn behind bars

# Set Y-axis limits (adjust based on your actual data range, including stability)
max_y_value = 0
for data_list in real_data.values():
    # Make sure to handle potential non-numeric stability values if you haven't replaced them yet
    numeric_data = [v for v in data_list if isinstance(v, (int, float))]
    if numeric_data: # Check if list is not empty
      max_y_value = max(max_y_value, max(numeric_data))

# Set ylim based on calculated max_y_value, ensure it's not zero
# Add some padding at the top, minimum limit of 10 if all values are zero/negative
ax.set_ylim(0, max(max_y_value * 1.1, 10)) # Increased top padding slightly

# Adjust layout to prevent labels overlapping
plt.tight_layout(rect=[0, 0.05, 1, 0.9]) # Adjust rect to make space for legend/title

# --- Show the Plot ---
plt.show()
