import pandas as pd
import numpy as np
from itertools import combinations

# Read a single math experiment Excel file
excel_path = 'math_experiment_20250419_182416.xlsx'
xls = pd.ExcelFile(excel_path)

# Only process the first 8 sheets (Round 1~8)
round_sheets = [name for name in xls.sheet_names if name.startswith('Round')][:8]

k = 4  # top-k

avg_overlaps = []
for round_name in round_sheets:
    df = pd.read_excel(xls, sheet_name=round_name)
    # Only keep the model scoring part (remove Model name and last column)
    score_df = df.iloc[:, 1:-1]
    model_names = df['Model name'].tolist()
    # Each column is a rater's scores for all models
    # Calculate top-k overlap rate between all pairs of raters
    overlap_rates = []
    n = score_df.shape[1]
    for i, j in combinations(range(n), 2):
        # Get top-k indices for the i-th and j-th raters
        topk_i = set(score_df.iloc[:, i].nlargest(k).index)
        topk_j = set(score_df.iloc[:, j].nlargest(k).index)
        overlap = len(topk_i & topk_j) / k
        overlap_rates.append(overlap)
    avg_overlap = np.mean(overlap_rates)
    avg_overlaps.append(avg_overlap)
    print(f"{round_name} Average Top-{k} Overlap Rate: {avg_overlap:.4f}")

# Output the average Top-k Overlap Rate for all eight rounds
final_avg = np.mean(avg_overlaps)
print(f"Average Top-{k} Overlap Rate across all eight rounds: {final_avg:.4f}")
