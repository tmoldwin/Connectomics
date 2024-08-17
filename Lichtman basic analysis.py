import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
import seaborn as sns
import plotting_function as pf
import numpy as np
import os

# Read the csv file into a pandas object
df = pd.read_csv('synapses_toviah.csv')

cell_types = ['pyramidal neuron', 'interneuron', 'excitatory/spiny neuron with atypical tree', 'unclassified neuron', 'spiny stellate neuron']

class Neuron:
    def __init__(self, neuron_id, df):
        self.neuron_id = neuron_id
        self.neuron_type = None
        self.layer = None
        self.type_layer = None
        in_df = df[df['post_seg_id'] == self.neuron_id]
        out_df = df[df['pre_seg_id'] == self.neuron_id]
        in_df_full_neuron = df[df['pre_type'].isin(cell_types)]
        out_df_full_neuron = df[df['post_type'].isin(cell_types)]
        self.in_type_counts = in_df['pre_type_layer'].value_counts()
        self.out_type_counts = out_df['post_type_layer'].value_counts()

        if len(in_df) > 0:
            self.neuron_type = in_df.iloc[0]['post_type']
            self.layer = in_df.iloc[0]['post_region']
            self.type_layer = in_df.iloc[0]['post_type_layer']
        else:
            self.neuron_type = out_df.iloc[0]['pre_type']
            self.layer = out_df.iloc[0]['pre_region']
            self.type_layer = out_df.iloc[0]['pre_type_layer']


    def __str__(self):
        return (
            f'Neuron ID: {self.neuron_id}\n'
            f'Neuron Type: {self.neuron_type}\n'
            f'Layer: {self.layer}\n\n'
            f'Input DataFrame:\n{self.in_df}\n\n'
            f'Output DataFrame:\n{self.out_df}\n'
            f'InputNeuron DataFrame:\n{self.in_df_full_neuron}\n\n'
            f'OutputNeuron DataFrame:\n{self.out_df_full_neuron}\n'

        )

    def bar_plots(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot in_type_counts
        self.in_type_counts.plot(kind='bar', ax=axes[0], position=0, width=0.4, label='Input Types', color='blue',
                                 align='center')

        # Plot out_type_counts
        self.out_type_counts.plot(kind='bar', ax=axes[1], position=1, width=0.4, label='Output Types', color='orange',
                                  align='edge')
        for ax in axes:
            ax.set_xlabel('Type - Layer')
            ax.set_ylabel('Counts')
            ax.set_title(f'Neuron ID: {self.neuron_id} - Type Counts')
            ax.legend()
            plt.xticks(rotation=45)
        fig.suptitle(f'Neuron ID: {self.neuron_id}, Type: {self.neuron_type}, Layer: {self.layer}')
        plt.subplots_adjust(bottom=0.4)
        plt.show()

# New class GroupNeuronType

# key: id, value: neuron
# key: neuron_type value: list of all neurons of that type
class NeuronGroup:
    def __init__(self, name, neuron_list):
        self.name = name
        self.neuron_list = neuron_list
        self.in_matrix = pd.DataFrame()
        self.out_matrix = pd.DataFrame()
        self.N = len(neuron_list)


        for neuron in self.neuron_list:
            self.in_matrix = pd.concat([self.in_matrix, neuron.in_type_counts], axis=1)
            self.out_matrix = pd.concat([self.out_matrix, neuron.out_type_counts], axis=1)

        self.in_matrix = self.in_matrix.fillna(0).astype(int).T
        self.out_matrix = self.out_matrix.fillna(0).astype(int).T

        self.in_means = self.in_matrix.mean().reset_index()
        self.in_means.columns  = ['Column', 'Mean']

        self.out_means= self.out_matrix.mean().reset_index()
        self.out_means.columns = ['Column', 'Mean']

        self.in_variance = self.in_matrix.var()

        self.out_variance = self.out_matrix.var()

        #self.plotting()
    def plotting(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Input matrix plot
        pf.dist_plot(self.in_matrix, ax=axes[0])
        axes[0].set_title(f'{self.name} - Input Connections')
        axes[0].set_xlabel('Presynaptic Neuron Type')
        axes[0].set_ylabel('Number of Synapses')
        
        # Output matrix plot
        pf.dist_plot(self.out_matrix, ax=axes[1])
        axes[1].set_title(f'{self.name} - Output Connections')
        axes[1].set_xlabel('Postsynaptic Neuron Type')
        axes[1].set_ylabel('Number of Synapses')
        
        # Overall figure title
        fig.suptitle(f'Connection Distribution for {self.name} Neurons (N={self.N})', fontsize=16)
        
        plt.tight_layout()
        plt.show()

class Connectome:
    def __init__(self, df, cell_types):
        self.df = df
        self.cell_types = cell_types

        # Give counts of pre-type and post-type unique values
        self.pre_type_counts = df['pre_type'].value_counts()
        self.post_type_counts = df['post_type'].value_counts()

        df['pre_type_layer'] = df['pre_type'] + '_' + df['pre_region']
        df['post_type_layer'] = df['post_type'] + '_' + df['post_region']

        pre_is_neuron = df[df['pre_type'].isin(cell_types)]
        post_is_neuron = df[df['post_type'].isin(cell_types)]

        self.pre_neuron_ids = pre_is_neuron['pre_seg_id'].unique()
        self.post_neuron_ids = post_is_neuron['post_seg_id'].unique()

        self.union_neuron_set = set(self.pre_neuron_ids).union(set(self.post_neuron_ids))
        self.union_neuron_set = list(self.union_neuron_set)
        self.generate_neurons()

        self.neuron_groups = {}
        for type, neuron_list in self.neurons_by_type_layer.items():
            self.neuron_groups[type] = NeuronGroup(type, neuron_list)
        
        self.create_weight_matrices()

    def generate_neurons(self):
        self.neurons_by_id = {}
        self.neurons_by_type = {}
        self.neurons_by_type_layer = {}

        for index, id in enumerate(self.union_neuron_set):
            print(str(index) + " out of " + str(len(self.union_neuron_set)))
            neuron = Neuron(id, self.df)
            self.neurons_by_id[id] = neuron
            
            # Initialize the list for this neuron type if it doesn't exist
            if neuron.neuron_type not in self.neurons_by_type:
                self.neurons_by_type[neuron.neuron_type] = []
            
            # Append the neuron to the list for its type
            self.neurons_by_type[neuron.neuron_type].append(neuron)

            # Initialize the list for this neuron type-layer if it doesn't exist
            if neuron.type_layer not in self.neurons_by_type_layer:
                self.neurons_by_type_layer[neuron.type_layer] = []
            
            # Append the neuron to the list for its type-layer
            self.neurons_by_type_layer[neuron.type_layer].append(neuron)

    def create_weight_matrices(self):
        neuron_types = list(self.neuron_groups.keys())
        print(neuron_types)
        n_types = len(neuron_types)
        out_weight_matrix = np.zeros((n_types, n_types))
        in_weight_matrix = np.zeros((n_types, n_types))

        for i, pre_type in enumerate(neuron_types):
            pre_group = self.neuron_groups[pre_type]
            for j, post_type in enumerate(neuron_types):
                # Outgoing connections
                out_weight = pre_group.out_means.loc[pre_group.out_means['Column'] == post_type, 'Mean'].values
                out_weight_matrix[i, j] = out_weight[0] if len(out_weight) > 0 else 0

                # Incoming connections
                in_weight = pre_group.in_means.loc[pre_group.in_means['Column'] == post_type, 'Mean'].values
                in_weight_matrix[i, j] = in_weight[0] if len(in_weight) > 0 else 0

        self.out_weight_matrix = pd.DataFrame(out_weight_matrix, index=neuron_types, columns=neuron_types)
        self.in_weight_matrix = pd.DataFrame(in_weight_matrix, index=neuron_types, columns=neuron_types)
        return self.out_weight_matrix, self.in_weight_matrix

    def plot_weight_matrices(self):
        if not hasattr(self, 'out_weight_matrix') or not hasattr(self, 'in_weight_matrix'):
            self.create_weight_matrices()

        # Sort neuron types
        neuron_types = sorted(self.out_weight_matrix.index, 
                              key=lambda x: (x.split()[0], x.split()[-1]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

        sns.heatmap(self.out_weight_matrix.loc[neuron_types, neuron_types], 
                    annot=False, cmap='YlOrRd', fmt='.2f', ax=ax1)
        ax1.set_title('Average Number of Outgoing Connections')
        ax1.set_xlabel('Postsynaptic Neuron Type')
        ax1.set_ylabel('Presynaptic Neuron Type')

        sns.heatmap(self.in_weight_matrix.loc[neuron_types, neuron_types], 
                    annot=False, cmap='YlOrRd', fmt='.2f', ax=ax2)
        ax2.set_title('Average Number of Incoming Connections')
        ax2.set_xlabel('Presynaptic Neuron Type')
        ax2.set_ylabel('Postsynaptic Neuron Type')

        plt.tight_layout()

        # Ensure plots directory exists
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/weight_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_connection_distribution(self):
        # Prepare data
        data = []
        for type_layer, neurons in self.neurons_by_type_layer.items():
            for neuron in neurons:
                incoming = neuron.in_type_counts.sum()
                outgoing = neuron.out_type_counts.sum()
                neuron_type, layer = type_layer.split('_')
                data.append({'type': neuron_type, 'layer': layer, 'incoming': incoming, 'outgoing': outgoing})
        
        df = pd.DataFrame(data)
        
        # Sort by neuron type and layer
        df['type_layer'] = df['type'] + '-' + df['layer']
        df = df.sort_values(['type', 'layer'])
        
        # Calculate N for each type_layer
        type_counts = df['type_layer'].value_counts().sort_index()
        
        # Create labels with N
        labels = [f"{tl}\n(N={n})" for tl, n in type_counts.items()]
        
        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
        fig.suptitle('Distribution of Incoming and Outgoing Connections by Neuron Type-Layer', fontsize=16)
        
        # Plot incoming connections
        sns.boxplot(x='incoming', y='type_layer', data=df, ax=ax1, color='lightblue')
        ax1.set_title('Incoming Connections')
        ax1.set_xlabel('Number of Connections')
        ax1.set_yticklabels(labels)
        
        # Plot outgoing connections
        sns.boxplot(x='outgoing', y='type_layer', data=df, ax=ax2, color='lightgreen')
        ax2.set_title('Outgoing Connections')
        ax2.set_xlabel('Number of Connections')
        ax2.set_yticklabels([])  # Remove y-axis labels for the second plot
        
        plt.tight_layout()
        
        # Ensure plots directory exists
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/connection_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

print("len", len(df))
# len 273840
random_rows = df.sample(n=10000, random_state=42)
random_rows = df
my_connectome = Connectome(random_rows, cell_types)
my_connectome.plot_connection_distribution()
my_connectome.plot_weight_matrices()
plt.show()

# print(my_connectome.neurons_by_type)

# C.1. Create a dict of all unique pre and/or post IDs, mapped to type (e.g. pyramidal neuron, layer 4)
#   2. Find the IDs of things that are neurons
# D. For each UID that is a neuron (either pre-synaptic or post-synaptic) create an object which contains 1. Neuron type, 2. Layer,
#  3. a dataframe of its input synapses 4. a dataframe of its output synapses, also both 3 and 4, filtered by where the input/output is a neuron.
#E. Also make a dictionary with key (neuron type, neuron layer), and value is a list of neurons

# 
# #Next step: weight matrix - stack the mean counts on top of each other