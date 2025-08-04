import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read all sheets from Excel file
excel_path = 'Radar.xlsx'
sheets = pd.read_excel(excel_path, sheet_name=None)

# Function to plot radar chart (for subplots)
def plot_radar_subplot(ax, df, title):
    model_names = list(df.columns[1:])
    labels = [f'Question{model_map.get(name, name)}' for name in model_names]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    row = df.iloc[0]
    values = row[1:].tolist()
    values += values[:1]

    # More harmonious color palette
    COLOR_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]
    MODEL_COLORS = {
        "gpt-4.1": COLOR_PALETTE[1],
        "o3-mini": COLOR_PALETTE[7],
        "o1": COLOR_PALETTE[6],
        "claude-3-7-sonnet": COLOR_PALETTE[4],
        "deepseek-r1": COLOR_PALETTE[3],
        "deepseek-v3": COLOR_PALETTE[2],
        "qwen2.5-max": COLOR_PALETTE[5],
        "gemini-2.5-pro-exp": COLOR_PALETTE[0]
    }
    color_line = MODEL_COLORS.get(title, "#1f77b4")
    color_fill = color_line

    ax.plot(angles, values, linewidth=3, linestyle='solid', color=color_line, alpha=0.9)
    ax.fill(angles, values, color=color_fill, alpha=0.12)  # Lighter fill

    # Label text adjustment
    ax.set_thetagrids(np.degrees(angles[:-1]), [''] * num_vars)
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 6.5, label, size=22, ha='center', va='center', zorder=1000, fontname='DejaVu Sans')

    ax.set_ylim(0, 6)
    ax.set_yticks([2, 4, 6])
    ax.set_yticklabels([])  
    ax.set_title(title, y=1.13, fontsize=32, weight='bold', fontname='DejaVu Sans', color=color_line)

    ax.spines['polar'].set_visible(False)
    ax.xaxis.grid(True, color='#BBBBBB', linestyle='solid', linewidth=1.2)
    ax.yaxis.grid(True, color='#BBBBBB', linestyle='solid', linewidth=1.2)
    # Outer black circle
    rmax = ax.get_ylim()[1]
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(theta, [rmax]*len(theta), color='black', linewidth=1.3, zorder=10)

fig, axes = plt.subplots(2, 4, figsize=(22, 12), subplot_kw=dict(polar=True))
fig.subplots_adjust(wspace=0.45, hspace=0.6)

# Model number and name mapping
model_map = {
    "gpt-4.1": 1,
    "o3-mini": 2,
    "o1": 3,
    "claude-3-7-sonnet": 4,
    "deepseek-r1": 5,
    "deepseek-v3": 6,
    "qwen2.5-max": 7,
    "gemini-2.5-pro-exp": 8
}

for idx, (sheet_name, df) in enumerate(sheets.items()):
    ax = axes[idx // 4, idx % 4]
    plot_radar_subplot(ax, df, sheet_name)

# Output mapping relationship
mapping_items = [f"Question {num}: {name}" for name, num in model_map.items()]
mapping_lines = ['  '.join(mapping_items[i:i+5]) for i in range(0, len(mapping_items), 5)]
mapping_str = '\n'.join(mapping_lines)
plt.figtext(0.5, 0.01, mapping_str, ha='center', fontsize=22, linespacing=1.5, fontweight='bold')
plt.savefig('Radar.pdf', bbox_inches='tight', dpi=1000)
plt.show()
