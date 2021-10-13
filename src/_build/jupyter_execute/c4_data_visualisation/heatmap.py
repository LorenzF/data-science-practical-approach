#!/usr/bin/env python
# coding: utf-8

# ## Heatmap plot
# 
# A heatmap also deals with 2 dimensional data and cares about the relation.
# Here instead of numerical data with dots, we are using categorical data where every combination of the 2 categories has a singular value.
# 
# This results into a matrix that we visualize where each index of the matrix has its own color based on a color gradient.
# This plot got its name as it is used to find 'hot spots' between combinations of 2 categorical features.

# In[1]:


import pandas as pd
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(16,12)})


# To make optimal use of this plot, we are going to take on a rather complex dataset, where we have measurements of brain networks.
# The idea is that we have several networks with several nodes in 2 hemispheres, the content of the data is not as important here, what matters is that we want to find correlations between different nodes in the brain.

# In[2]:


brain_df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
brain_df.head()


# luckily for us, the pandas library has an easy method of finding out what the correlation is between different columns of numerical data.
# These correlations are denoted between -1 (completely opposite) to 1 (completely related).
# Take a minute to understand how the columns and index changed using the operation, you can see that a node in a network and hemisphere has a correlation of 1.00 with itself.

# In[3]:


brain_df.corr()


# This result is way to much to see a pattern, yet if we add a color scale and give each a gradation, we can see some correlations.
# 
# Can you see how nodes from the same network are related with a more whitish color?
# The heatmap might be fairly intimidating at first but is a powerful tool when handling bigger datasets.

# In[4]:


sns.heatmap(data=brain_df.corr())


# Without going into the medical details we can also apply some machine learning to it and create a clustermap.
# This map is a way to group nodes from similar networks into clusters, an advances technique!
# 
# Gaze over the colors and look at the axi, notice how the computer figured out how to group the most similar nodes from networks.
# Also, I did not create this by myself, so don't give me credit for this!

# In[5]:


# Select a subset of the networks
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (brain_df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
brain_df = brain_df.loc[:, used_columns]

# Create a categorical palette to identify the networks
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# Convert the palette to vectors that will be drawn on the side of the matrix
networks = brain_df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=brain_df.columns).map(network_lut)

# Draw the full plot
g = sns.clustermap(brain_df.corr(), center=0, cmap="vlag",
                   row_colors=network_colors, col_colors=network_colors,
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))

g.ax_row_dendrogram.remove()


# In[ ]:




