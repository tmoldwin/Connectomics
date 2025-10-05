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

# Define consistent colors for each cell type (same as box_whisker_indegree.py)
CELL_TYPE_COLORS = {
    'pyramidal neuron': '#1f77b4',  # Blue
    'interneuron': '#ff7f0e',  # Orange
    'excitatory/spiny neuron with atypical tree': '#2ca02c',  # Green
    'spiny stellate neuron': '#d62728',  # Red
    'unclassified neuron': '#9467bd'  # Purple
}

def calculate_ei_ratio_fast():
    """Calculate E/(E+I) ratio using SQL aggregation for speed"""
    print("Calculating E/(E+I) ratios using SQL aggregation...")
    
    conn = sqlite3.connect(db_path)
    
    # CORRECTED: ei_type = 2 is excitatory, ei_type = 1 is inhibitory
    query = """
    SELECT 
        post_seg_id,
        post_type,
        post_region,
        SUM(CASE WHEN ei_type = 2 THEN pair_count ELSE 0 END) as excitatory_synapses,
        SUM(CASE WHEN ei_type = 1 THEN pair_count ELSE 0 END) as inhibitory_synapses,
        SUM(pair_count) as total_synapses,
        COUNT(*) as num_connections
    FROM edge_list_table
    WHERE post_type IN ('pyramidal neuron', 'interneuron', 'excitatory/spiny neuron with atypical tree', 'spiny stellate neuron', 'unclassified neuron')
    GROUP BY post_seg_id, post_type, post_region
    """
    
    print("Running SQL query...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded data for {len(df)} neurons")
    
    # Calculate E/(E+I) ratio
    df['ei_total'] = df['excitatory_synapses'] + df['inhibitory_synapses']
    df['ei_ratio'] = df['excitatory_synapses'] / df['ei_total']
    df['type_layer'] = df['post_type'] + ' - ' + df['post_region']
    
    # Handle division by zero (neurons with no E or I inputs)
    df['ei_ratio'] = df['ei_ratio'].fillna(0)
    
    print(f"Neurons with E or I inputs: {len(df[df['ei_total'] > 0])}")
    print(f"Mean E/(E+I) ratio: {df['ei_ratio'].mean():.3f}")
    
    return df

def create_ei_plots(df):
    """Create clean E/(E+I) ratio plots"""
    print("Creating clean E/(E+I) ratio plots...")
    
    os.makedirs('Plots', exist_ok=True)
    plt.style.use('default')
    
    # Filter out neurons with no E/I inputs
    df_valid = df[df['ei_total'] > 0].copy()
    
    # Classify neurons as excitatory or inhibitory based on type
    excitatory_types = {'pyramidal neuron', 'excitatory/spiny neuron with atypical tree', 'spiny stellate neuron'}
    inhibitory_types = {'interneuron'}
    
    df_valid['cell_class'] = df_valid['post_type'].apply(
        lambda x: 'Excitatory' if x in excitatory_types 
        else 'Inhibitory' if x in inhibitory_types 
        else 'Other'
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('E/(E+I) Ratio Analysis by Cell Type and Layer', fontsize=16, fontweight='bold')
    
    # Plot 1: KDE plot by cell type
    ax1 = axes[0, 0]
    
    for nt in sorted(NEURON_TYPES):
        if nt in df_valid['post_type'].values:
            subset = df_valid[df_valid['post_type'] == nt]
            if len(subset) > 1:
                # Clean up long labels
                clean_label = nt.replace('excitatory/spiny neuron with atypical tree', 'exc/atypical')
                sns.kdeplot(data=subset, x='ei_ratio', ax=ax1, 
                           color=CELL_TYPE_COLORS[nt], 
                           label=f"{clean_label} (N={len(subset)})", 
                           linewidth=2.5, alpha=0.8)
    
    ax1.set_title('E/(E+I) Ratio Distribution by Cell Type', fontweight='bold')
    ax1.set_xlabel('E/(E+I) Ratio')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=8, loc='upper left')
    
    # Plot 2: KDE plot by type-layer combinations (top 8 by count for clarity)
    ax2 = axes[0, 1]
    type_layer_counts = df_valid['type_layer'].value_counts().head(8)
    
    colors_tl = plt.cm.Set3(np.linspace(0, 1, len(type_layer_counts)))
    
    for idx, tl in enumerate(type_layer_counts.index):
        data = df_valid[df_valid['type_layer'] == tl]['ei_ratio']
        if len(data) > 1:
            # Clean up labels
            clean_label = tl.replace('excitatory/spiny neuron with atypical tree', 'exc/atyp')
            data.plot.kde(ax=ax2, color=colors_tl[idx], 
                         label=f"{clean_label} (N={len(data)})", 
                         linewidth=2, alpha=0.7)
    
    ax2.set_title('E/(E+I) Ratio Distribution by Type-Layer (Top 8)', fontweight='bold')
    ax2.set_xlabel('E/(E+I) Ratio')
    ax2.set_ylabel('Density')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=7, loc='upper left')
    
    # Plot 3: Excitatory vs Inhibitory cell classes KDE
    ax3 = axes[1, 0]
    exc_data = df_valid[df_valid['cell_class'] == 'Excitatory']['ei_ratio']
    inh_data = df_valid[df_valid['cell_class'] == 'Inhibitory']['ei_ratio']
    other_data = df_valid[df_valid['cell_class'] == 'Other']['ei_ratio']
    
    if len(exc_data) > 1:
        sns.kdeplot(data=df_valid[df_valid['cell_class'] == 'Excitatory'], x='ei_ratio', 
                   ax=ax3, color='red', label=f'Excitatory (N={len(exc_data)})', 
                   linewidth=3, alpha=0.7)
    if len(inh_data) > 1:
        sns.kdeplot(data=df_valid[df_valid['cell_class'] == 'Inhibitory'], x='ei_ratio', 
                   ax=ax3, color='blue', label=f'Inhibitory (N={len(inh_data)})', 
                   linewidth=3, alpha=0.7)
    if len(other_data) > 1:
        sns.kdeplot(data=df_valid[df_valid['cell_class'] == 'Other'], x='ei_ratio', 
                   ax=ax3, color='gray', label=f'Other (N={len(other_data)})', 
                   linewidth=3, alpha=0.7)
    
    ax3.set_title('E/(E+I) Ratio: Excitatory vs Inhibitory Cells', fontweight='bold')
    ax3.set_xlabel('E/(E+I) Ratio')
    ax3.set_ylabel('Density')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.legend(fontsize=10)
    
    # Plot 4: Distribution histogram with class separation
    ax4 = axes[1, 1]
    
    ax4.hist(exc_data, bins=30, alpha=0.6, color='red', label=f'Excitatory (N={len(exc_data)})', density=True)
    ax4.hist(inh_data, bins=30, alpha=0.6, color='blue', label=f'Inhibitory (N={len(inh_data)})', density=True)
    
    ax4.axvline(exc_data.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Exc Mean: {exc_data.mean():.3f}')
    ax4.axvline(inh_data.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Inh Mean: {inh_data.mean():.3f}')
    
    ax4.set_title('E/(E+I) Distribution by Cell Class')
    ax4.set_xlabel('E/(E+I) Ratio')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Plots/ei_ratio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n" + "="*80)
    print("E/(E+I) RATIO STATISTICS")
    print("="*80)
    
    print(f"Total neurons analyzed: {len(df_valid)}")
    print(f"Overall E/(E+I) ratio: {df_valid['ei_ratio'].mean():.3f} ± {df_valid['ei_ratio'].std():.3f}")
    print(f"Median E/(E+I) ratio: {df_valid['ei_ratio'].median():.3f}")
    
    # Stats by excitatory vs inhibitory classes
    exc_stats = df_valid[df_valid['cell_class'] == 'Excitatory']['ei_ratio']
    inh_stats = df_valid[df_valid['cell_class'] == 'Inhibitory']['ei_ratio']
    
    print(f"\nBY CELL CLASS:")
    print(f"EXCITATORY CELLS (N={len(exc_stats)}):")
    print(f"  E/(E+I): {exc_stats.mean():.3f} ± {exc_stats.std():.3f}")
    print(f"  Median: {exc_stats.median():.3f}")
    
    print(f"INHIBITORY CELLS (N={len(inh_stats)}):")
    print(f"  E/(E+I): {inh_stats.mean():.3f} ± {inh_stats.std():.3f}")
    print(f"  Median: {inh_stats.median():.3f}")
    
    print(f"\nBY CELL TYPE:")
    for nt in sorted(NEURON_TYPES):
        if nt in df_valid['post_type'].values:
            subset = df_valid[df_valid['post_type'] == nt]
            print(f"\n{nt.upper()}:")
            print(f"  N: {len(subset)}")
            print(f"  E/(E+I): {subset['ei_ratio'].mean():.3f} ± {subset['ei_ratio'].std():.3f}")
            print(f"  Median: {subset['ei_ratio'].median():.3f}")
            print(f"  Mean E: {subset['excitatory_synapses'].mean():.0f}")
            print(f"  Mean I: {subset['inhibitory_synapses'].mean():.0f}")
    
    print("="*80)

def main():
    """Main function"""
    print("E/(E+I) Ratio Analysis")
    print("ei_type 2 = Excitatory (asymmetric)")
    print("ei_type 1 = Inhibitory")
    print("=" * 50)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
    
    # Calculate ratios
    df = calculate_ei_ratio_fast()
    
    # Create plots
    create_ei_plots(df)
    
    print("\nE/(E+I) analysis complete!")

if __name__ == "__main__":
    main()
