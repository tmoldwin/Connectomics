import pandas as pd
from matplotlib import pyplot as plt

# Read the csv file into a pandas object
df = pd.read_csv('synapses_toviah.csv')

# Give counts of pre-type and post-type unique values
pre_type_counts = df['pre_type'].value_counts()
post_type_counts = df['post_type'].value_counts()

#
cell_types = ['pyramidal neuron', 'interneuron', 'excitatory/spiny neuron with atypical tree', 'unclassified neuron', 'spiny stellate neuron']

df['pre_type_layer'] = df['pre_type'] + '_' + df['pre_region']
df['post_type_layer'] = df['post_type'] + '_' + df['post_region']

pre_is_neuron = df[df['pre_type'].isin(cell_types)]
post_is_neuron = df[df['post_type'].isin(cell_types)]

pre_neuron_ids = pre_is_neuron['pre_seg_id'].unique()
post_neuron_ids = post_is_neuron['post_seg_id'].unique()

union_neuron_set = set(pre_neuron_ids).union(set(post_neuron_ids))
union_neuron_set = list(union_neuron_set)

class Neuron:
    def __init__(self, neuron_id, df):
        self.neuron_id = neuron_id
        self.neuron_type = None
        self.layer = None
        self.in_df = df[df['post_seg_id'] == self.neuron_id]
        self.out_df = df[df['pre_seg_id'] == self.neuron_id]
        self.in_df_full_neuron = df[df['pre_type'].isin(cell_types)]
        self.out_df_full_neuron = df[df['post_type'].isin(cell_types)]
        self.in_type_counts = self.in_df['pre_type_layer'].value_counts()
        self.out_type_counts = self.out_df['post_type_layer'].value_counts()

        if len(self.in_df) > 0:
            self.neuron_type = self.in_df.iloc[0]['post_type']
            self.layer = self.in_df.iloc[0]['post_region']
        else:
            self.neuron_type = self.out_df.iloc[0]['pre_type']
            self.layer = self.out_df.iloc[0]['pre_region']

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


neuron = Neuron(union_neuron_set[5000], df)
neuron.bar_plots()

# C.1. Create a dict of all unique pre and/or post IDs, mapped to type (e.g. pyramidal neuron, layer 4)
#   2. Find the IDs of things that are neurons
# D. For each UID that is a neuron (either pre-synaptic or post-synaptic) create an object which contains 1. Neuron type, 2. Layer,
#  3. a dataframe of its input synapses 4. a dataframe of its output synapses, also both 3 and 4, filtered by where the input/output is a neuron.
#E. Also make a dictionary with key (neuron type, neuron layer), and value is a list of neurons


