import sqlite3
import pandas as pd

# Let's investigate the database structure and counts
conn = sqlite3.connect('neuron_synapses_new.db')
cursor = conn.cursor()

print('Database investigation:')
print('='*50)

# Check total synapses
cursor.execute('SELECT COUNT(*) FROM edge_list_table')
total = cursor.fetchone()[0]
print(f'Total synapses in database: {total:,}')

# Check neuron types
print('\nPost-synaptic types (top 10):')
cursor.execute('SELECT post_type, COUNT(*) FROM edge_list_table GROUP BY post_type ORDER BY COUNT(*) DESC LIMIT 10')
for ptype, count in cursor.fetchall():
    print(f'  {ptype}: {count:,}')

# Check what we're actually filtering for
neuron_types = ['pyramidal neuron', 'interneuron', 'excitatory/spiny neuron with atypical tree', 'spiny stellate neuron', 'unclassified neuron']
neuron_types_str = "'" + "', '".join(neuron_types) + "'"

print(f'\nFiltering for these neuron types: {neuron_types}')

# Count neuron-to-neuron connections
query1 = f'''
SELECT COUNT(*) FROM edge_list_table 
WHERE pre_type IN ({neuron_types_str}) AND post_type IN ({neuron_types_str})
'''
cursor.execute(query1)
neuron_to_neuron = cursor.fetchone()[0]
print(f'Neuron-to-neuron connections: {neuron_to_neuron:,}')

# Count unique post-synaptic neurons
query2 = f'''
SELECT COUNT(DISTINCT post_seg_id) FROM edge_list_table 
WHERE post_type IN ({neuron_types_str})
'''
cursor.execute(query2)
unique_post_neurons = cursor.fetchone()[0]
print(f'Unique post-synaptic neurons: {unique_post_neurons:,}')

# Check average connections per neuron
if unique_post_neurons > 0:
    print(f'Average connections per neuron: {neuron_to_neuron / unique_post_neurons:.1f}')

# Let's look at a specific neuron to see what's happening
query3 = f'''
SELECT post_seg_id, post_type, COUNT(*) as connection_count, SUM(pair_count) as total_synapses
FROM edge_list_table 
WHERE post_type IN ({neuron_types_str}) AND pre_type IN ({neuron_types_str})
GROUP BY post_seg_id, post_type
ORDER BY total_synapses DESC
LIMIT 10
'''
cursor.execute(query3)

print('\nTop 10 neurons by total synapses:')
for post_id, post_type, conn_count, total_synapses in cursor.fetchall():
    print(f'  Neuron {post_id} ({post_type}): {conn_count} connections, {total_synapses} total synapses')

# Let's also check what happens if we include ALL connections TO neurons (not just FROM neurons)
query4 = f'''
SELECT post_seg_id, post_type, COUNT(*) as connection_count, SUM(pair_count) as total_synapses
FROM edge_list_table 
WHERE post_type IN ({neuron_types_str})
GROUP BY post_seg_id, post_type
ORDER BY total_synapses DESC
LIMIT 10
'''
cursor.execute(query4)

print('\nTop 10 neurons by total synapses (including ALL inputs):')
for post_id, post_type, conn_count, total_synapses in cursor.fetchall():
    print(f'  Neuron {post_id} ({post_type}): {conn_count} connections, {total_synapses} total synapses')

# Check the pair_count distribution
print('\nPair count distribution:')
cursor.execute('SELECT pair_count, COUNT(*) FROM edge_list_table GROUP BY pair_count ORDER BY pair_count LIMIT 10')
for pair_count, freq in cursor.fetchall():
    print(f'  {pair_count} synapses per connection: {freq:,} connections')

conn.close()
