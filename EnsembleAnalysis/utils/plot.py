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


# Plot Ramachandra plot
def plot_2d_histogram_angles(x, y, bins=50, log_scale=True, contours=True,
                            fig=None, ax=None, **kwargs):
    """
    Create a 2D histogram with contours for angle data (-180 to 180 degrees).

    Parameters:
    -----------
    x : array-like, shape (n_samples, n_features) or (n_samples,)
        X angle data in degrees (-180 to 180)
    y : array-like, shape (n_samples, n_features) or (n_samples,)
        Y angle data in degrees (-180 to 180)
    bins : int or array-like, default=50
        Number of bins for histogram
    log_scale : bool, default=True
        Whether to use log scale for colors
    contours : bool, default=True
        Whether to overlay contour lines
    fig : matplotlib.figure.Figure, optional
        Figure object to use. If None, creates new figure.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new axes.
    **kwargs : dict
        Additional customization options

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    hist : 2D histogram array
    xedges, yedges : bin edges
    cbar_ax : colorbar axes object (None if colorbar=False)
    """

    # Flatten arrays if they are 2D
    if x.ndim > 1:
        x_flat = x.flatten()
    else:
        x_flat = x.copy()

    if y.ndim > 1:
        y_flat = y.flatten()
    else:
        y_flat = y.copy()

    # Remove any NaN values
    mask = ~(np.isnan(x_flat) | np.isnan(y_flat))
    x_clean = x_flat[mask]
    y_clean = y_flat[mask]

    # Set up bins - ensure they cover the full range
    if isinstance(bins, int):
        x_bins = np.linspace(-180, 180, bins + 1)
        y_bins = np.linspace(-180, 180, bins + 1)
    else:
        x_bins = bins
        y_bins = bins

    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=[x_bins, y_bins], density=True)

    # Create figure and axes if not provided
    if fig is None or ax is None:
        figsize = kwargs.get('figsize', (10, 8))
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        created_fig = False

    # Handle log scale
    if log_scale:
        # Add small value to avoid log(0)
        hist_plot = hist.copy()
        hist_plot[hist_plot == 0] = np.nan  # Set zeros to NaN for white regions

        # Find minimum non-zero value for log scale
        min_val = np.nanmin(hist_plot[hist_plot > 0]) if np.any(hist_plot > 0) else 1
        max_val = np.nanmax(hist_plot)

        # Create log normalization
        if max_val > min_val:
            norm = LogNorm(vmin=min_val, vmax=max_val)
        else:
            norm = None
            log_scale = False
            warnings.warn("Not enough data variation for log scale, using linear scale")
    else:
        hist_plot = hist.copy()
        hist_plot[hist_plot == 0] = np.nan
        norm = None

    # Set colormap - use a colormap that shows white for NaN
    cmap = kwargs.get('cmap', 'viridis')
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    cmap.set_bad('white')  # Set NaN values to white

    # Create the histogram plot
    # For pcolormesh with shading='flat', we need the full edge arrays
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, hist_plot.T,
                       cmap=cmap,
                       norm=norm,
                       shading='flat',
                       alpha=kwargs.get('alpha', 1.0))

    # Add contours if requested
    if contours and np.any(hist > 0):
        # Create contour levels
        n_contours = kwargs.get('n_contours', 5)
        if log_scale and norm is not None:
            # Log-spaced contour levels
            contour_levels = np.logspace(np.log10(min_val), np.log10(max_val), n_contours)
        else:
            # Linear contour levels
            contour_levels = np.linspace(np.nanmin(hist_plot), np.nanmax(hist_plot), n_contours)

        # override contour levels
        if kwargs.get('override_contour_levels', None):
            contour_levels = override_contour_levels
        print(f"contour_levels: {contour_levels}")

        # Create contour coordinates
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        X_contour, Y_contour = np.meshgrid(x_centers, y_centers)

        # Add contour lines
        contour_colors = kwargs.get('contour_colors', 'black')
        contour_alpha = kwargs.get('contour_alpha', 0.6)
        contour_linewidths = kwargs.get('contour_linewidths', 1.0)

        cs = ax.contour(X_contour, Y_contour, hist_plot.T,
                       levels=contour_levels,
                       colors=contour_colors,
                       alpha=contour_alpha,
                       linewidths=contour_linewidths)

        # Add contour labels if requested
        if kwargs.get('contour_labels', False):
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Add grid if requested
    if kwargs.get('grid', True):
        ax.grid(True, alpha=0.3)

    # Add colorbar using make_axes_locatable or fig.add_axes
    cbar_ax = None
    if kwargs.get('colorbar', True):
        # Try to use make_axes_locatable first (works well with gridspec and layout constraints)
        try:

            # Create divider for the axes
            divider = make_axes_locatable(ax)

            # Get colorbar parameters
            cbar_size = kwargs.get('cbar_size', '3%')  # Width as percentage of parent
            cbar_pad = kwargs.get('cbar_pad', 0.1)    # Padding between plot and colorbar

            # Create colorbar axes on the right side
            cbar_ax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)

            # Add colorbar
            cbar = fig.colorbar(im, cax=cbar_ax)

        except Exception:
            # Fall back to manual positioning with fig.add_axes
            # Get the position of the main axes (need to draw first for accurate position)
            fig.canvas.draw()
            pos = ax.get_position()

            # Calculate colorbar position (to the right of the main plot)
            cbar_width = kwargs.get('cbar_width', 0.02)
            cbar_pad_manual = kwargs.get('cbar_pad', 0.02)

            # Colorbar axes position: [left, bottom, width, height]
            cbar_left = pos.x1 + cbar_pad_manual
            cbar_bottom = pos.y0
            cbar_height = pos.height

            # Create colorbar axes with same height as main plot
            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

            # Add colorbar
            cbar = fig.colorbar(im, cax=cbar_ax)

    # Only call tight_layout if we created the figure
    if created_fig:
        plt.tight_layout()

    return fig, ax, hist, xedges, yedges, cbar_ax

