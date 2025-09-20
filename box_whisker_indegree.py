import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from collections import defaultdict

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

def load_and_calculate_indegree():
    """Load data and calculate in-degree for each neuron"""
    print("Loading data from database...")
    
    conn = sqlite3.connect(db_path)
    
    # Load all synaptic connections TO neurons
    query = """
    SELECT pre_seg_id, post_seg_id, pre_type, post_type, pre_region, post_region, pair_count
    FROM edge_list_table
    WHERE post_type IN ('pyramidal neuron', 'interneuron', 'excitatory/spiny neuron with atypical tree', 'spiny stellate neuron', 'unclassified neuron')
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} connections to neurons")
    
    # Calculate in-degree for each neuron
    indegree_data = []
    
    for post_id, group in df.groupby('post_seg_id'):
        neuron_type = group.iloc[0]['post_type']
        layer = group.iloc[0]['post_region']
        
        # Sum all incoming connections
        total_indegree = group['pair_count'].sum()
        
        indegree_data.append({
            'neuron_id': post_id,
            'neuron_type': neuron_type,
            'layer': layer,
            'indegree': total_indegree,
            'type_layer': f"{neuron_type} - {layer}"
        })
    
    indegree_df = pd.DataFrame(indegree_data)
    print(f"Calculated in-degree for {len(indegree_df)} neurons")
    
    return indegree_df

def create_box_whisker_plots(indegree_df):
    """Create box and whisker plots of in-degree per neuron"""
    print("Creating box and whisker plots...")
    
    # Create plots directory
    os.makedirs('Plots', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # 1. Box and whisker plot by neuron type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('In-Degree Distribution: Box and Whisker Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Basic box plot by neuron type
    ax1 = axes[0, 0]
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    for nt in sorted(NEURON_TYPES):
        if nt in indegree_df['neuron_type'].values:
            data = indegree_df[indegree_df['neuron_type'] == nt]['indegree'].values
            if len(data) > 0:
                box_data.append(data)
                n_neurons = len(data)
                box_labels.append(f"{nt}\n(N={n_neurons})")
    
    bp1 = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)
    
    # Style the box plot
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax1.set_title('In-Degree by Neuron Type')
    ax1.set_ylabel('In-Degree (Number of Synapses)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale box plot (to better see distributions)
    ax2 = axes[0, 1]
    bp2 = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)
    
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    ax2.set_title('In-Degree by Neuron Type (Log Scale)')
    ax2.set_ylabel('In-Degree (Log Scale)')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot by type-layer combinations (top combinations only)
    ax3 = axes[1, 0]
    
    # Get top 10 type-layer combinations by count
    type_layer_counts = indegree_df['type_layer'].value_counts().head(10)
    
    tl_data = []
    tl_labels = []
    for tl in type_layer_counts.index:
        data = indegree_df[indegree_df['type_layer'] == tl]['indegree'].values
        tl_data.append(data)
        n_neurons = len(data)
        tl_labels.append(f"{tl}\n(N={n_neurons})")
    
    bp3 = ax3.boxplot(tl_data, labels=tl_labels, patch_artist=True, showfliers=False)
    
    for patch in bp3['boxes']:
        patch.set_facecolor('lightyellow')
        patch.set_alpha(0.7)
    
    ax3.set_title('In-Degree by Type-Layer (Top 10)')
    ax3.set_ylabel('In-Degree (Number of Synapses)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison of quartiles across types
    ax4 = axes[1, 1]
    
    # Calculate quartiles for each neuron type
    quartile_data = []
    for nt in sorted(NEURON_TYPES):
        if nt in indegree_df['neuron_type'].values:
            data = indegree_df[indegree_df['neuron_type'] == nt]['indegree']
            if len(data) > 0:
                q1, q2, q3 = np.percentile(data, [25, 50, 75])
                quartile_data.append({
                    'type': nt,
                    'Q1': q1,
                    'Median': q2,
                    'Q3': q3,
                    'Mean': data.mean(),
                    'N': len(data)
                })
    
    quartile_df = pd.DataFrame(quartile_data)
    
    x_pos = np.arange(len(quartile_df))
    width = 0.2
    
    ax4.bar(x_pos - width, quartile_df['Q1'], width, label='Q1 (25th percentile)', alpha=0.7)
    ax4.bar(x_pos, quartile_df['Median'], width, label='Median (50th percentile)', alpha=0.7)
    ax4.bar(x_pos + width, quartile_df['Q3'], width, label='Q3 (75th percentile)', alpha=0.7)
    
    ax4.set_xlabel('Neuron Type')
    ax4.set_ylabel('In-Degree')
    ax4.set_title('Quartile Comparison Across Neuron Types')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{row['type']}\n(N={row['N']})" for _, row in quartile_df.iterrows()], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Plots/box_whisker_indegree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print box plot statistics
    print("\n" + "="*70)
    print("BOX AND WHISKER PLOT STATISTICS")
    print("="*70)
    
    for nt in sorted(NEURON_TYPES):
        if nt in indegree_df['neuron_type'].values:
            data = indegree_df[indegree_df['neuron_type'] == nt]['indegree']
            if len(data) > 0:
                q1, q2, q3 = np.percentile(data, [25, 50, 75])
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                outliers = data[(data < lower_fence) | (data > upper_fence)]
                
                print(f"\n{nt.upper()} (N={len(data)}):")
                print(f"  Q1 (25th percentile): {q1:.0f}")
                print(f"  Median (50th percentile): {q2:.0f}")
                print(f"  Q3 (75th percentile): {q3:.0f}")
                print(f"  IQR (Q3-Q1): {iqr:.0f}")
                print(f"  Lower fence: {max(lower_fence, data.min()):.0f}")
                print(f"  Upper fence: {min(upper_fence, data.max()):.0f}")
                print(f"  Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
                print(f"  Range: {data.min():.0f} - {data.max():.0f}")
    
    print("="*70)

def main():
    """Main function"""
    print("Box and Whisker Analysis of Neuron In-Degree")
    print("=" * 50)
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
    
    # Load data and calculate in-degree
    indegree_df = load_and_calculate_indegree()
    
    # Create box and whisker plots
    create_box_whisker_plots(indegree_df)
    
    print("\nBox and whisker analysis complete!")

if __name__ == "__main__":
    main()
