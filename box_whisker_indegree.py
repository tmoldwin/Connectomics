import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    """Create box and whisker plots with E/I breakdown"""
    print("Creating E/I in-degree box and whisker plots...")
    
    os.makedirs('Plots', exist_ok=True)
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('In-Degree Analysis: Excitatory, Inhibitory, and Net', fontsize=16, fontweight='bold')
    
    # Plot 1: E, I, and Net in-degree by cell type
    ax1 = axes[0, 0]
    
    # Prepare data for grouped box plots
    cell_types = sorted(NEURON_TYPES)
    x_positions = np.arange(len(cell_types))
    width = 0.25
    
    exc_data = []
    inh_data = []
    net_data = []
    labels = []
    
    for nt in cell_types:
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            exc_data.append(subset['excitatory_indegree'].values)
            inh_data.append(subset['inhibitory_indegree'].values)
            net_data.append(subset['net_indegree'].values)
            # Clean up long labels
            clean_label = nt.replace('excitatory/spiny neuron with atypical tree', 'exc/atypical')
            labels.append(f"{clean_label}\n(N={len(subset)})")
    
    # Create side-by-side box plots
    positions_exc = x_positions - width
    positions_inh = x_positions
    positions_net = x_positions + width
    
    bp1_exc = ax1.boxplot(exc_data, positions=positions_exc, widths=width*0.8, patch_artist=True, showfliers=False)
    bp1_inh = ax1.boxplot(inh_data, positions=positions_inh, widths=width*0.8, patch_artist=True, showfliers=False)
    bp1_net = ax1.boxplot(net_data, positions=positions_net, widths=width*0.8, patch_artist=True, showfliers=False)
    
    # Color the boxes
    for patch in bp1_exc['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.7)
    for patch in bp1_inh['boxes']:
        patch.set_facecolor('blue')
        patch.set_alpha(0.7)
    for patch in bp1_net['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_alpha(0.9)
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_title('E/I/Net In-Degree by Cell Type')
    ax1.set_ylabel('In-Degree (Number of Synapses)')
    ax1.grid(True, alpha=0.3)
    ax1.legend([bp1_exc['boxes'][0], bp1_inh['boxes'][0], bp1_net['boxes'][0]], 
              ['Excitatory', 'Inhibitory', 'Net (E-I)'], loc='upper right')
    
    # Plot 2: Same as Plot 1 but log scale
    ax2 = axes[0, 1]
    
    # Only plot positive values for log scale (excitatory and inhibitory only)
    bp2_exc = ax2.boxplot(exc_data, positions=positions_exc, widths=width*0.8, patch_artist=True, showfliers=False)
    bp2_inh = ax2.boxplot(inh_data, positions=positions_inh, widths=width*0.8, patch_artist=True, showfliers=False)
    
    for patch in bp2_exc['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.7)
    for patch in bp2_inh['boxes']:
        patch.set_facecolor('blue')
        patch.set_alpha(0.7)
    
    ax2.set_xticks(x_positions - width/2)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_title('E/I In-Degree by Cell Type (Log Scale)')
    ax2.set_ylabel('In-Degree (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend([bp2_exc['boxes'][0], bp2_inh['boxes'][0]], 
              ['Excitatory', 'Inhibitory'], loc='upper right')
    
    # Plot 3: E/(E+I) ratio box plot
    ax3 = axes[1, 0]
    
    # Calculate E/(E+I) ratio for each neuron
    df['ei_ratio'] = df['excitatory_indegree'] / (df['excitatory_indegree'] + df['inhibitory_indegree'])
    df['ei_ratio'] = df['ei_ratio'].fillna(0)  # Handle division by zero
    
    ratio_data = []
    for nt in cell_types:
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            ratio_data.append(subset['ei_ratio'].values)
    
    bp3 = ax3.boxplot(ratio_data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch in bp3['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    ax3.set_title('E/(E+I) Ratio by Cell Type')
    ax3.set_ylabel('E/(E+I) Ratio')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary statistics
    summary_data = []
    for nt in cell_types:
        if nt in df['post_type'].values:
            subset = df[df['post_type'] == nt]
            clean_name = nt.replace('excitatory/spiny neuron with atypical tree', 'exc/atypical')[:15]
            summary_data.append([
                clean_name,
                len(subset),
                f"{subset['excitatory_indegree'].mean():.0f}",
                f"{subset['inhibitory_indegree'].mean():.0f}",
                f"{subset['net_indegree'].mean():.0f}",
                f"{subset['ei_ratio'].mean():.3f}"
            ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Cell Type', 'N', 'Mean E', 'Mean I', 'Mean Net', 'E/(E+I)'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig('Plots/box_whisker_indegree.png', dpi=300, bbox_inches='tight')
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
    print("Box and Whisker Analysis of E/I In-Degree")
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
