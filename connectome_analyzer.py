import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a Polars dataframe
df = pl.read_csv('connectome.csv')
print(df.head())
matrix = df.to_numpy()
matrix = matrix[:,1:].astype(np.float)
print(sum(sum(matrix)))
print(matrix)
# Perform mathematical operations
# For example, let's calculate the mean of each column
# df = df.select([pl.col(c).mean().alias(c) for c in df.columns])
#
# # Convert the Polars dataframe to a Pandas dataframe for plotting
# df_pandas = df.to_pandas()
#
# # Plot the results using imshowa
fig = plt.figure(figsize=(20,20))
plt.imshow(np.log(matrix), cmap='viridis', interpolation = 'nearest')
plt.show()