import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
from collections import OrderedDict
import EnsembleAnalysis.utils.plot as jplt


datafile = 'contactmap.txt'
outpng = datafile.split(".")[0]+"_out"

# this is the actual residue index for the contactmap matrix
xstarts = 30
ystarts = 40


### whole map
data = np.loadtxt(datafile)
data = ma.masked_where(data == 0, data)
# Define labels for X and Y axis with an interval of 1
ylabels = [str(i) for i in np.arange(xstarts, xstarts + data.shape[0])]
xlabels = [str(i) for i in np.arange(ystarts, ystarts + data.shape[1])]

fig, ax = plt.subplots(1,1,figsize=(8,10))
im, cbar = jplt.pairwise_heatmap(data, ax,
                      x_tick_labels=xlabels,
                      y_tick_labels=ylabels,
                      x_tick_step=5,    # showing labels with an interval of 5
                      y_rotation=0,
                      cmap="inferno_r") # emphasize high contact frequencies with darker colors
# set the limits of color to be 0 ~ 1
im.set_clim(0, 1)
ax.set_xlabel('Residue ID_X')
ax.set_ylabel('Residue ID_Y')
cbar.set_label('Contact frequencies')
plt.tight_layout()
plt.savefig(f"{outpng}_whole.png")
plt.show()


### Zoom-in contactmap
# only want resid 130 ~ 150 for X; and all residues on Y
xzoomin_starts = 130
xzoomin_ends = 150

data = np.loadtxt(datafile)[xzoomin_starts-xstarts:xzoomin_ends-xstarts, :]
data = ma.masked_where(data == 0, data)
# define labels for X and Y axis
ylabels = [str(i) for i in np.arange(xzoomin_starts, xzoomin_starts + data.shape[0])]
xlabels = [str(i) for i in np.arange(ystarts, ystarts + data.shape[1])]

fig, ax = plt.subplots(1,1,figsize=(5,3))
im, cbar = jplt.pairwise_heatmap(data, ax,
                      x_tick_labels=xlabels,
                      y_tick_labels=ylabels,
                      x_tick_step=2, # make label interval smaller since we zoom in the contactmap
                      y_tick_step=1,
                      y_rotation=0,
                      cmap="inferno_r")
# set the limits of color to be 0 ~ 1
im.set_clim(0, 1)
ax.set_xlabel('Residue_ID_X')
ax.set_ylabel('Residue_ID_Y')
cbar.set_label('Contact frequencies')
plt.tight_layout()
plt.savefig(f"{outpng}_zoomin.png")
plt.show()
