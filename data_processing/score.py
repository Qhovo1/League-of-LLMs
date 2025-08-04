import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_excel('score.xlsx', header=None, skiprows=1)
df.columns = ['Model name', 'Answer Average Score', 'Question Average Score']

# Sort by Answer Score from high to low
df = df.sort_values('Answer Average Score', ascending=False).reset_index(drop=True)

# Vertical grouped bar chart
bar_width = 0.25  # Slightly wider bars
x = np.arange(len(df))

plt.figure(figsize=(10, 6))  # Appropriate canvas ratio
# Answer on left, Question on right
bars1 = plt.bar(
    x, 
    df['Answer Average Score'], 
    width=bar_width, 
    color='#9CBAE0', 
    edgecolor='black',      
    label="Own Average Score on Others' Questions"
)
bars2 = plt.bar(
    x + bar_width, 
    df['Question Average Score'], 
    width=bar_width, 
    color='#EBB882', 
    edgecolor='black',      
    label="Others' Average Score on Own Question"
)

# Label scores - staggered display
for bar, score in zip(bars1, df['Answer Average Score']):
    plt.text(bar.get_x() + bar.get_width()/2 - 0.01, score + 0.3, f'{score:.1f}', ha='center', va='bottom', fontsize=9.5)
for bar, score in zip(bars2, df['Question Average Score']):
    plt.text(bar.get_x() + bar.get_width()/2 + 0.01, score + 0.3, f'{score:.1f}', ha='center', va='bottom', fontsize=9.5)

# X-axis ticks set between two groups of bars, vertical display
plt.xticks(x + bar_width/2, df['Model name'], rotation=0, fontsize=9.5)


# Add dashed line to the right of each model name
for i in range(len(x)-1):
    plt.axvline(x[i] + bar_width*2.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.ylabel('Score', fontsize=10)
plt.ylim(50, 90)  # Y-axis range set to 50-90, leaving space at the top
plt.yticks(np.arange(50, 91, 10), fontsize=9)  # Adjust y-axis ticks
plt.legend(loc='upper right', fontsize=10, frameon=True)  # Larger legend font

# Remove grid lines
plt.grid(False)

plt.tight_layout()
plt.savefig('score.pdf')
plt.show()