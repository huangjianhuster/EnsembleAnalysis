import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
from collections import OrderedDict
import EnsembleAnalysis.utils.plot as jplt


datafile = './contactmap.txt'
outpng = datafile.split(".")[0]+"_out"

# this is the actual residue index for the contactmap matrix
xstarts = 105
ystarts = 1164

# whole map
data = np.loadtxt(datafile)
data = ma.masked_where(data == 0, data)
ylabels = [str(i) for i in np.arange(xstarts, xstarts + data.shape[0])]
xlabels = [str(i) for i in np.arange(1164, 1164 + data.shape[1])]

fig, ax = plt.subplots(1,1,figsize=(5,6))
im, cbar = jplt.pairwise_heatmap(data, ax,
                      x_tick_labels=xlabels,
                      y_tick_labels=ylabels,
                      x_tick_step=5,
                      y_rotation=0,
                      cmap="inferno_r")
im.set_clim(0, 1)
ax.set_xlabel('Rib helix')
ax.set_ylabel('MHR12')
cbar.set_label('Contact frequencies')
plt.tight_layout()
plt.savefig(f"{outpng}_whole.png")
plt.show()

# only want resid 205 ~ 220
data = np.loadtxt(datafile)[205-xstarts:220-xstarts, :]
data = ma.masked_where(data == 0, data)
ylabels = [str(i) for i in np.arange(205, 205 + data.shape[0])]
xlabels = [str(i) for i in np.arange(1164, 1164 + data.shape[1])]

fig, ax = plt.subplots(1,1,figsize=(5,6))
im, cbar = jplt.pairwise_heatmap(data, ax,
                      x_tick_labels=xlabels,
                      y_tick_labels=ylabels,
                      x_tick_step=5,
                      y_tick_step=1,
                      y_rotation=0,
                      cmap="inferno_r")
im.set_clim(0, 1)
ax.set_xlabel('Rib helix')
ax.set_ylabel('MHR12')
cbar.set_label('Contact frequencies')
plt.tight_layout()
plt.savefig(f"{outpng}_zoomin.png")
plt.show()
