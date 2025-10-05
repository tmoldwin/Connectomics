import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Database path
db_path = "neuron_synapses_new.db"

# Define actual neuron types (not fragments or other cell types)
NEURON_TYPES = {
    'pyramidal neuron',
    'interneuron', 
    'excitatory/spiny neuron with atypical tree',
    'spiny stellate neuron',
    'unclassified neuron'
}

# Define consistent colors for each cell type
CELL_TYPE_COLORS = {
    'pyramidal neuron': '#1f77b4',  # Blue
    'interneuron': '#ff7f0e',  # Orange
    'excitatory/spiny neuron with atypical tree': '#2ca02c',  # Green
    'spiny stellate neuron': '#d62728',  # Red
    'unclassified neuron': '#9467bd'  # Purple
}

def load_indegree_data():
    """Load in-degree data with E/I breakdown"""
    print("Loading in-degree data with E/I breakdown...")
    
    conn = sqlite3.connect(db_path)
    
    # Get excitatory, inhibitory, and total in-degree for each neuron
    # ei_type = 2 is excitatory, ei_type = 1 is inhibitory
    query = """
    SELECT 
        post_seg_id,
        post_type,
        post_region,
        SUM(CASE WHEN ei_type = 2 THEN pair_count ELSE 0 END) as excitatory_indegree,
        SUM(CASE WHEN ei_type = 1 THEN pair_count ELSE 0 END) as inhibitory_indegree,
        SUM(pair_count) as total_indegree
    FROM edge_list_table
    WHERE post_type IN ('pyramidal neuron', 'interneuron', 'excitatory/spiny neuron with atypical tree', 'spiny stellate neuron', 'unclassified neuron')
    GROUP BY post_seg_id, post_type, post_region
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Calculate net in-degree (E - I)
    df['net_indegree'] = df['excitatory_indegree'] - df['inhibitory_indegree']
    df['type_layer'] = df['post_type'] + ' - ' + df['post_region']
    
    print(f"Loaded in-degree data for {len(df)} neurons")
    return df

def create_ei_indegree_plots(df):
    """Create KDE plots with E/I breakdown and heatmap summary"""
    print("Creating E/I in-degree KDE plots...")
    
    os.makedirs('Plots', exist_ok=True)
    plt.style.use('default')
    
    # Create mosaic layout with heatmap on top
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 2, 2], hspace=0.3, wspace=0.3)
    
    ax_heatmap = fig.add_subplot(gs[0, :])  # Heatmap spans full width at top
    ax1 = fig.add_subplot(gs[1, 0:2])  # Excitatory
    ax2 = fig.add_subplot(gs[1, 2:4])  # Inhibitory
    ax3 = fig.add_subplot(gs[2, 0:2])  # Total
    ax4 = fig.add_subplot(gs[2, 2:4])  # E/(E+I) ratio
    
    fig.suptitle('In-Degree Analysis: Excitatory, Inhibitory, Total, and E/(E+I) Ratio', fontsize=16, fontweight='bold')
    
    # Prepare data for plots with consistent colors
    cell_types = sorted(NEURON_TYPES)
    
    # Create heatmap data (rows = neuron types, columns = metrics)
    heatmap_data = []
    cell_labels = []
    for nt in cell_types:
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            # Clean labels: remove "neuron" and shorten
            clean_label = nt.replace(' neuron', '').replace('excitatory/spiny', 'exc/spiny').replace(' with atypical tree', '')
            clean_label = clean_label.title()
            cell_labels.append(clean_label)
            heatmap_data.append([
                len(subset),
                subset['excitatory_indegree'].mean(),
                subset['inhibitory_indegree'].mean(),
                subset['total_indegree'].mean(),
                subset['excitatory_indegree'].mean() / (subset['excitatory_indegree'].mean() + subset['inhibitory_indegree'].mean())
            ])
    
    heatmap_data = np.array(heatmap_data)
    metric_labels = ['N', 'Mean E', 'Mean I', 'Mean Total', 'E/(E+I)']
    
    # Normalize each metric (column) for heatmap coloring
    heatmap_normalized = heatmap_data.copy()
    for col in range(heatmap_normalized.shape[1]):
        col_min = heatmap_normalized[:, col].min()
        col_max = heatmap_normalized[:, col].max()
        if col_max > col_min:
            heatmap_normalized[:, col] = (heatmap_normalized[:, col] - col_min) / (col_max - col_min)
    
    # Plot heatmap (no transpose - rows are cell types, columns are metrics)
    im = ax_heatmap.imshow(heatmap_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels - x on top, horizontal
    ax_heatmap.xaxis.tick_top()
    ax_heatmap.xaxis.set_label_position('top')
    ax_heatmap.set_xticks(np.arange(len(metric_labels)))
    ax_heatmap.set_xticklabels(metric_labels, rotation=0, ha='center', fontsize=11, fontweight='bold')
    ax_heatmap.set_yticks(np.arange(len(cell_labels)))
    ax_heatmap.set_yticklabels(cell_labels, rotation=0, ha='right', fontsize=10)
    ax_heatmap.set_title('Summary Statistics by Cell Type', fontweight='bold', pad=20, fontsize=12)
    
    # Annotate with actual values
    for i in range(len(cell_labels)):
        for j in range(len(metric_labels)):
            if j == 0:
                text = f'{int(heatmap_data[i, j])}'
            elif j == 4:
                text = f'{heatmap_data[i, j]:.3f}'
            else:
                text = f'{int(heatmap_data[i, j])}'
            ax_heatmap.text(j, i, text, ha='center', va='center', 
                          color='black' if heatmap_normalized[i, j] < 0.5 else 'white',
                          fontweight='bold', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, orientation='vertical', pad=0.01, aspect=10)
    cbar.set_label('Normalized Value', rotation=270, labelpad=15)
    
    # Plot 1: Excitatory in-degree KDE
    for nt in cell_types:
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            clean_label = nt.replace('excitatory/spiny neuron with atypical tree', 'exc/atypical')
            
            # Plot excitatory KDE
            if len(subset) > 1:
                sns.kdeplot(data=subset, x='excitatory_indegree', ax=ax1, 
                           color=CELL_TYPE_COLORS[nt], 
                           label=f'{clean_label} (N={len(subset)})', 
                           linewidth=2.5, alpha=0.8)
    
    ax1.set_title('Excitatory In-Degree Distribution', fontweight='bold')
    ax1.set_xlabel('Excitatory In-Degree (Number of Synapses)')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='upper right')
    # Set x-axis limits starting from -1000
    exc_mean = df['excitatory_indegree'].mean()
    exc_std = df['excitatory_indegree'].std()
    ax1.set_xlim(-1000, exc_mean + 3 * exc_std)
    
    # Plot 2: Inhibitory In-Degree KDE
    for nt in cell_types:
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            clean_label = nt.replace('excitatory/spiny neuron with atypical tree', 'exc/atypical')
            
            # Plot inhibitory KDE
            if len(subset) > 1:
                sns.kdeplot(data=subset, x='inhibitory_indegree', ax=ax2, 
                           color=CELL_TYPE_COLORS[nt], 
                           label=f'{clean_label} (N={len(subset)})', 
                           linewidth=2.5, alpha=0.8)
    
    ax2.set_title('Inhibitory In-Degree Distribution', fontweight='bold')
    ax2.set_xlabel('Inhibitory In-Degree (Number of Synapses)')
    ax2.set_ylabel('Density')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='upper right')
    # Set x-axis limits starting from -1000
    inh_mean = df['inhibitory_indegree'].mean()
    inh_std = df['inhibitory_indegree'].std()
    ax2.set_xlim(-1000, inh_mean + 3 * inh_std)
    
    # Plot 3: Total In-Degree KDE
    for nt in cell_types:
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            clean_label = nt.replace('excitatory/spiny neuron with atypical tree', 'exc/atypical')
            
            # Plot total indegree KDE
            if len(subset) > 1:
                sns.kdeplot(data=subset, x='total_indegree', ax=ax3, 
                           color=CELL_TYPE_COLORS[nt], 
                           label=f'{clean_label} (N={len(subset)})', 
                           linewidth=2.5, alpha=0.8)
    
    ax3.set_title('Total In-Degree Distribution', fontweight='bold')
    ax3.set_xlabel('Total In-Degree (Number of Synapses)')
    ax3.set_ylabel('Density')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc='upper right')
    # Set x-axis limits starting from -1000
    total_mean = df['total_indegree'].mean()
    total_std = df['total_indegree'].std()
    ax3.set_xlim(-1000, total_mean + 3 * total_std)
    
    # Plot 4: E/(E+I) ratio KDE plot
    # Calculate E/(E+I) ratio for each neuron
    df['ei_ratio'] = df['excitatory_indegree'] / (df['excitatory_indegree'] + df['inhibitory_indegree'])
    df['ei_ratio'] = df['ei_ratio'].fillna(0)  # Handle division by zero
    
    for nt in cell_types:
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            clean_label = nt.replace('excitatory/spiny neuron with atypical tree', 'exc/atypical')
            
            # Plot E/(E+I) ratio KDE
            if len(subset) > 1:
                sns.kdeplot(data=subset, x='ei_ratio', ax=ax4, 
                           color=CELL_TYPE_COLORS[nt], 
                           label=f'{clean_label} (N={len(subset)})', 
                           linewidth=2.5, alpha=0.8)
    
    ax4.set_title('E/(E+I) Ratio Distribution', fontweight='bold')
    ax4.set_xlabel('E/(E+I) Ratio')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.legend(fontsize=8, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('Plots/kde_indegree_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("E/I IN-DEGREE STATISTICS")
    print("="*80)
    
    for nt in sorted(NEURON_TYPES):
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            
            print(f"\n{nt.upper()} (N={len(subset)}):")
            print(f"  Excitatory in-degree: {subset['excitatory_indegree'].mean():.1f} ± {subset['excitatory_indegree'].std():.1f}")
            print(f"    Median: {subset['excitatory_indegree'].median():.0f}")
            print(f"    Range: {subset['excitatory_indegree'].min():.0f} - {subset['excitatory_indegree'].max():.0f}")
            
            print(f"  Inhibitory in-degree: {subset['inhibitory_indegree'].mean():.1f} ± {subset['inhibitory_indegree'].std():.1f}")
            print(f"    Median: {subset['inhibitory_indegree'].median():.0f}")
            print(f"    Range: {subset['inhibitory_indegree'].min():.0f} - {subset['inhibitory_indegree'].max():.0f}")
            
            print(f"  Net in-degree (E-I): {subset['net_indegree'].mean():.1f} ± {subset['net_indegree'].std():.1f}")
            print(f"    Median: {subset['net_indegree'].median():.0f}")
            print(f"    Range: {subset['net_indegree'].min():.0f} - {subset['net_indegree'].max():.0f}")
            
            print(f"  E/(E+I) ratio: {subset['ei_ratio'].mean():.3f} ± {subset['ei_ratio'].std():.3f}")
            print(f"    E:I ratio: {subset['excitatory_indegree'].sum() / subset['inhibitory_indegree'].sum():.2f}")
    
    print("="*80)

def main():
    """Main function"""
    print("KDE Analysis of E/I In-Degree")
    print("=" * 50)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
    
    # Load in-degree data with E/I breakdown
    df = load_indegree_data()
    
    # Create E/I in-degree plots
    create_ei_indegree_plots(df)
    
    print("\nE/I in-degree analysis complete!")

if __name__ == "__main__":
    main()
