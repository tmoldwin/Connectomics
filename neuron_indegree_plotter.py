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

def load_data_from_db():
    """Load synaptic connections from the new database"""
    print("Loading data from database...")
    
    conn = sqlite3.connect(db_path)
    
    # Load all synaptic connections
    query = """
    SELECT pre_seg_id, post_seg_id, pre_type, post_type, pre_region, post_region, pair_count
    FROM edge_list_table
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} synaptic connections")
    return df

def calculate_indegree_per_neuron(df):
    """Calculate in-degree for each neuron (counting ALL connections TO neurons)"""
    print("Calculating in-degree per neuron...")
    
    # Filter to include ALL connections TO actual neurons (from any source)
    connections_to_neurons = df[df['post_type'].isin(NEURON_TYPES)]
    
    print(f"Total connections to neurons: {len(connections_to_neurons):,}")
    
    # Also show breakdown by source type
    print("Breakdown by pre-synaptic type:")
    source_breakdown = connections_to_neurons['pre_type'].value_counts().head(10)
    for source_type, count in source_breakdown.items():
        print(f"  {source_type}: {count:,}")
    
    # Calculate in-degree: sum of pair_count for each post-synaptic neuron
    indegree_data = []
    
    # Group by post-synaptic neuron
    for post_id, group in connections_to_neurons.groupby('post_seg_id'):
        neuron_type = group.iloc[0]['post_type']
        layer = group.iloc[0]['post_region']
        
        # Sum all incoming connections (pair_count represents number of synapses)
        total_indegree = group['pair_count'].sum()
        
        # Count connections from different source types
        from_neurons = group[group['pre_type'].isin(NEURON_TYPES)]['pair_count'].sum()
        from_fragments = group[~group['pre_type'].isin(NEURON_TYPES)]['pair_count'].sum()
        
        indegree_data.append({
            'neuron_id': post_id,
            'neuron_type': neuron_type,
            'layer': layer,
            'indegree': total_indegree,
            'indegree_from_neurons': from_neurons,
            'indegree_from_fragments': from_fragments,
            'type_layer': f"{neuron_type}_{layer}"
        })
    
    indegree_df = pd.DataFrame(indegree_data)
    print(f"Calculated in-degree for {len(indegree_df)} neurons")
    
    return indegree_df

def create_indegree_plots(indegree_df):
    """Create B&W plots of in-degree per neuron"""
    print("Creating in-degree plots...")
    
    # Set style for B&W plots
    plt.style.use('default')
    sns.set_palette("gray")
    
    # Create plots directory
    os.makedirs('Plots', exist_ok=True)
    
    # 1. Distribution of in-degree by neuron type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('In-Degree Distribution by Neuron Type (Black & White)', fontsize=16, fontweight='bold')
    
    # Plot 1: Box plot by neuron type
    ax1 = axes[0, 0]
    box_data = [indegree_df[indegree_df['neuron_type'] == nt]['indegree'].values 
                for nt in NEURON_TYPES if nt in indegree_df['neuron_type'].values]
    box_labels = [nt for nt in NEURON_TYPES if nt in indegree_df['neuron_type'].values]
    
    bp1 = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightgray')
        patch.set_edgecolor('black')
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp1[element], color='black')
    
    ax1.set_title('In-Degree by Neuron Type')
    ax1.set_ylabel('In-Degree (Number of Connections)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of all in-degrees
    ax2 = axes[0, 1]
    ax2.hist(indegree_df['indegree'], bins=50, color='gray', edgecolor='black', alpha=0.7)
    ax2.set_title('Overall In-Degree Distribution')
    ax2.set_xlabel('In-Degree (Number of Connections)')
    ax2.set_ylabel('Number of Neurons')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean in-degree by type with error bars
    ax3 = axes[1, 0]
    type_stats = indegree_df.groupby('neuron_type')['indegree'].agg(['mean', 'std', 'count']).reset_index()
    
    bars = ax3.bar(range(len(type_stats)), type_stats['mean'], 
                   yerr=type_stats['std'], capsize=5, 
                   color='lightgray', edgecolor='black', alpha=0.8)
    ax3.set_xticks(range(len(type_stats)))
    ax3.set_xticklabels(type_stats['neuron_type'], rotation=45, ha='right')
    ax3.set_title('Mean In-Degree by Neuron Type')
    ax3.set_ylabel('Mean In-Degree ± SD')
    ax3.grid(True, alpha=0.3)
    
    # Add sample sizes as text
    for i, (bar, count) in enumerate(zip(bars, type_stats['count'])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + type_stats['std'].iloc[i] + 5,
                f'N={count}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: In-degree by type and layer
    ax4 = axes[1, 1]
    type_layer_stats = indegree_df.groupby('type_layer')['indegree'].agg(['mean', 'std', 'count']).reset_index()
    type_layer_stats = type_layer_stats.sort_values('mean', ascending=True)
    
    bars = ax4.barh(range(len(type_layer_stats)), type_layer_stats['mean'],
                    xerr=type_layer_stats['std'], capsize=3,
                    color='lightgray', edgecolor='black', alpha=0.8)
    ax4.set_yticks(range(len(type_layer_stats)))
    ax4.set_yticklabels([label.replace('_', ' - ') for label in type_layer_stats['type_layer']], fontsize=8)
    ax4.set_title('Mean In-Degree by Type-Layer')
    ax4.set_xlabel('Mean In-Degree ± SD')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Plots/indegree_distribution_bw.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 2. Detailed violin plot by neuron type
    fig2, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create violin plot data
    violin_data = []
    violin_labels = []
    for nt in NEURON_TYPES:
        if nt in indegree_df['neuron_type'].values:
            data = indegree_df[indegree_df['neuron_type'] == nt]['indegree'].values
            if len(data) > 0:
                violin_data.append(data)
                n_neurons = len(data)
                violin_labels.append(f"{nt}\n(N={n_neurons})")
    
    # Create violin plot
    parts = ax.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True)
    
    # Style violin plot in B&W
    for pc in parts['bodies']:
        pc.set_facecolor('lightgray')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)
    
    ax.set_xticks(range(len(violin_labels)))
    ax.set_xticklabels(violin_labels, rotation=45, ha='right')
    ax.set_title('In-Degree Distribution by Neuron Type (Detailed View)', fontsize=14, fontweight='bold')
    ax.set_ylabel('In-Degree (Number of Connections)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Plots/indegree_violin_bw.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("IN-DEGREE SUMMARY STATISTICS")
    print("="*60)
    
    overall_stats = indegree_df['indegree'].describe()
    print(f"Overall in-degree statistics (N={len(indegree_df)} neurons):")
    print(f"  Mean: {overall_stats['mean']:.1f}")
    print(f"  Median: {overall_stats['50%']:.1f}")
    print(f"  Std: {overall_stats['std']:.1f}")
    print(f"  Min: {overall_stats['min']:.0f}")
    print(f"  Max: {overall_stats['max']:.0f}")
    
    print(f"\nBy neuron type:")
    for nt in NEURON_TYPES:
        if nt in indegree_df['neuron_type'].values:
            subset = indegree_df[indegree_df['neuron_type'] == nt]['indegree']
            print(f"  {nt}:")
            print(f"    N: {len(subset)}")
            print(f"    Mean ± SD: {subset.mean():.1f} ± {subset.std():.1f}")
            print(f"    Median: {subset.median():.1f}")
            print(f"    Range: {subset.min():.0f} - {subset.max():.0f}")
    
    # Additional analysis: neuron vs fragment inputs
    print(f"\nSource breakdown (mean values):")
    print(f"  Mean inputs from neurons: {indegree_df['indegree_from_neurons'].mean():.1f}")
    print(f"  Mean inputs from fragments: {indegree_df['indegree_from_fragments'].mean():.1f}")
    print(f"  Fraction from fragments: {indegree_df['indegree_from_fragments'].sum() / indegree_df['indegree'].sum():.1%}")
    
    print("="*60)

def main():
    """Main function to run the analysis"""
    print("Neuron In-Degree Analysis")
    print("=" * 50)
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
    
    # Load data
    df = load_data_from_db()
    
    # Calculate in-degree
    indegree_df = calculate_indegree_per_neuron(df)
    
    # Create plots
    create_indegree_plots(indegree_df)
    
    print("\nAnalysis complete! Check the 'Plots' directory for output files.")

if __name__ == "__main__":
    main()
