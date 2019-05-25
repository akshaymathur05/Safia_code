# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:17:44 2019

@author: Akshay Mathur
"""


# Correction Matrix Plot
import matplotlib.pyplot as plt
import pandas
#import seaborn as sns
import numpy

#data = pandas.read_csv('onlyMobileDataset - Copy.csv')
#data = pandas.read_csv('onlyTraditionalDataset - Copy.csv')
data = pandas.read_csv('mixedDataset.csv')
corr = data.corr()
#'''
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,30,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#plt.show()
#'''
'''
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
'''
ax.set_xticklabels(
    ax.get_xticks(),
    fontsize = 6
)
ax.set_yticklabels(
    ax.get_yticks(),
    fontsize = 6
)
plt.show()        
#'''