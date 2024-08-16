import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
import seaborn as sns
import plotting_function as pf

from Lichtman2024.plotting_function import dist_plot

# Read the csv file into a pandas object
df = pd.read_csv('synapses_toviah.csv')

cell_types = ['pyramidal neuron', 'interneuron', 'excitatory/spiny neuron with atypical tree', 'unclassified neuron', 'spiny stellate neuron']

class Neuron:
    def __init__(self, neuron_id, df):
        self.neuron_id = neuron_id
        self.neuron_type = None
        self.layer = None
        in_df = df[df['post_seg_id'] == self.neuron_id]
        out_df = df[df['pre_seg_id'] == self.neuron_id]
        in_df_full_neuron = df[df['pre_type'].isin(cell_types)]
        out_df_full_neuron = df[df['post_type'].isin(cell_types)]
        self.in_type_counts = in_df['pre_type_layer'].value_counts()
        self.out_type_counts = out_df['post_type_layer'].value_counts()

        if len(in_df) > 0:
            self.neuron_type = in_df.iloc[0]['post_type']
            self.layer = in_df.iloc[0]['post_region']
        else:
            self.neuron_type = out_df.iloc[0]['pre_type']
            self.layer = out_df.iloc[0]['pre_region']

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
        print(self.name, self.N)
        print("in matrix")
        print(self.in_matrix)
        print(tabulate(self.in_matrix, headers='keys', tablefmt='pretty'))
        print("out matrix")
        print(self.out_matrix)


        self.in_means = self.in_matrix.mean().reset_index()
        self.in_means.columns  = ['Column', 'Mean']

        self.out_means= self.out_matrix.mean().reset_index()
        self.out_means.columns = ['Column', 'Mean']

        self.in_variance = self.in_matrix.var()

        self.out_variance = self.out_matrix.var()


        print("for type: " + self.name)
        print(self.in_means)
        print(self.out_means)
        self.plotting()
    def plotting(self):
        fig, axes = plt.subplots(1, 2)
        pf.dist_plot(self.in_matrix, ax=axes[0])
        pf.dist_plot(self.out_matrix, ax=axes[1])
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
        for type, neuron_list in self.neurons_by_type.items():
            self.neuron_groups[type] = NeuronGroup(type, neuron_list)


    def generate_neurons(self):
        self.neurons_by_id = {}
        self.neurons_by_type = {type: [] for type in self.cell_types}

        for index, id in enumerate(self.union_neuron_set):
            print(str(index) + " out of " + str(len(self.union_neuron_set)))
            neuron = Neuron(id, self.df)
            self.neurons_by_id[id] = neuron
            self.neurons_by_type[neuron.neuron_type].append(neuron)

random_rows = df.sample(n=1000, random_state=42)

my_connectome = Connectome(random_rows, cell_types)
# print(my_connectome.neurons_by_type)

# C.1. Create a dict of all unique pre and/or post IDs, mapped to type (e.g. pyramidal neuron, layer 4)
#   2. Find the IDs of things that are neurons
# D. For each UID that is a neuron (either pre-synaptic or post-synaptic) create an object which contains 1. Neuron type, 2. Layer,
#  3. a dataframe of its input synapses 4. a dataframe of its output synapses, also both 3 and 4, filtered by where the input/output is a neuron.
#E. Also make a dictionary with key (neuron type, neuron layer), and value is a list of neurons


