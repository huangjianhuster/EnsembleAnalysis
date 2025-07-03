# Author: Jian Huang
# E-mail: huangjianhuster@gmail.com

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import MDAnalysis as mda

# Plot histogram
def hist_plot(arr, ax=None, bins=20, fit=True, **kwargs):
    """
    arr: numpy ndarray, 1 dimension
    ax: matplotlib.axes._axes.Axes
    fit: fit data with Gaussian distribution
    **kwargs: keyword arguments in ax.hist
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,4))

    counts, bins, patches = ax.hist(arr, bins=bins, density=True, \
                            facecolor='skyblue', edgecolor='white', **kwargs)

    if fit:
        mu, sigma = norm.fit(arr)
        print(f"Fitted Gaussian: μ = {mu:.2f}, σ = {sigma:.2f}")
        x = np.linspace(bins[0], bins[-1], 1000)
        pdf = norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, '--', linewidth=2, label=f'μ={mu:.2f}, σ={sigma:.2f}')
    return counts, bins, patches


# Plot line
def line_plot(arr1, arr2=None, ax=None, decoration=False, **kwargs):
    """
    arr1: x data (1-dimensional ndarray)
    arr2: y data (1-dimensional ndarray), optional.
    ax: matplotlib.axes._axes.Axes
    args, kwargs: arguments or keyword arguments in plt.plot
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,4))


    if arr2 is None:
        y_data = arr1
        x_data = np.arange(1, len(y_data)+1, 1)
    else:
        assert len(arr1) == len(arr2)
        y_data = arr2
        x_data = arr1

    ax.plot(x_data, y_data, **kwargs)

    if decoration:
        ax.vlines(x_data, 0, y_data, colors='grey', alpha=0.5)

    return ax

# Plot errorbar
def errorbar_plot(arr1, arr2, yerror=None, xerror=None, ax=None, **kwargs):
    """
    arr1: x data (1-dimensional ndarray)
    arr2: y data (1-dimensional ndarray)
    xerror: x error
    yerror: y error
    ax: matplotlib.axes._axes.Axes
    kwargs: keyword arguments in plt.errorbar
    """
    if (not xerror) and (not yerror):
        raise ValueError("either xerror or yerror should be given!")

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,4))

    # use keywords to control figure style
    line_width = kwargs.get('linewidth', 2)
    line_color = kwargs.get('color', 'skyblue')
    line_style = kwargs.get('linestyle', '-')
    eline_width = kwargs.get('elinewidth', 1.5)
    eline_color = kwargs.get('ecolor', 'grey')
    capsize = kwargs.get('capsize', 2)
    marker = kwargs.get('marker', 'o')
    marker_size = kwargs.get('markersize', 6)

    if xerror and (not yerror):
        ax.errorbar(arr1, arr2, xerr=xerror,
                    marker = marker,
                    markersize = marker_size,
                    color = line_color,
                    linestyle = line_style,
                    linewidth = line_width,
                    capsize = capsize,
                    ecolor = eline_color,
                    elinewidth = eline_width,
                    **kwargs)
    if yerror and (not xerror):
        ax.errorbar(arr1, arr2, yerr=yerror,
                    marker = marker,
                    markersize = marker_size,
                    color = line_color,
                    linestyle = line_style,
                    linewidth = line_width,
                    capsize = capsize,
                    ecolor = eline_color,
                    elinewidth = eline_width,
                    **kwargs)

    return ax
