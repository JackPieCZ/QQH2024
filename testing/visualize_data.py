import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('./data/season_1_matches.csv') 
# Create a DataFrame
df = pd.DataFrame(data)

# Map 'W' to colors: 1 -> Green (Win), 0 -> Red (Loss)
colors = df['W'].map({1: 'green', 0: 'red'})

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(df['SC'], df['ACC'], c=colors, s=50, edgecolor='black', alpha=0.8)

# Add labels and title
plt.xlabel('Score', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('2D Visualization of Team Performance', fontsize=14)

# Add a legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Win', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Loss', markerfacecolor='red', markersize=10)
]
plt.legend(handles=legend_elements, loc='upper left')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()