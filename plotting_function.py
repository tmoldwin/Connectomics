import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
print("Current working directory:", os.getcwd())

def dist_plot(df, ax, plot_type='mean_std'):
    # Melt the DataFrame to long-form for seaborn plotting
    df_melted = df.melt(var_name='Column', value_name='Value')

    if plot_type == 'mean_std':
        # Calculate mean and standard deviation
        stats = df_melted.groupby('Column')['Value'].agg(['mean', 'std']).reset_index()
        
        # Create the plot with error bars
        ax.errorbar(stats['Column'], stats['mean'], yerr=stats['std'], fmt='o', capsize=5)
    elif plot_type == 'box':
        sns.boxplot(x='Column', y='Value', data=df_melted, ax=ax)
    elif plot_type == 'swarm':
        sns.swarmplot(x='Column', y='Value', data=df_melted, size=5, ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

if __name__ == "__main__":
    np.random.seed(42)

    # Generate a large dataset with 30 rows and 5 columns
    data = {
        'AA': np.random.normal(loc=10, scale=2, size=30),  # Normal distribution, mean=10, std=2
        'BB': np.random.normal(loc=15, scale=3, size=30),  # Normal distribution, mean=15, std=3
        'C': np.random.normal(loc=20, scale=4, size=30),  # Normal distribution, mean=20, std=4
        'D': np.random.normal(loc=25, scale=5, size=30),  # Normal distribution, mean=25, std=5
        'E': np.random.normal(loc=30, scale=6, size=30)  # Normal distribution, mean=30, std=6
    }
    df = pd.DataFrame(data)
    print(df)
    fig , ax = plt.subplots(1, 1)
    dist_plot(df, ax)
    plt.show()