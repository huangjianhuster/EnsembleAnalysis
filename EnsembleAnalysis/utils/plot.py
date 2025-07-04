# Author: Jian Huang
# E-mail: huangjianhuster@gmail.com

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm

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
    if xerror is None and yerror is None:
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
    marker_size = kwargs.get('markersize', 4)
    zorder = kwargs.get('zorder', 3)

    ax.errorbar(arr1, arr2, yerr=yerror, xerr=xerror,
                    marker = marker,
                    markersize = marker_size,
                    color = line_color,
                    linestyle = line_style,
                    linewidth = line_width,
                    capsize = capsize,
                    ecolor = eline_color,
                    elinewidth = eline_width,
                    zorder=zorder,
                    **kwargs)

    return ax


# Plot Ramachandra plot
def hist2d_contour_plot(x, y, bins=50, log_scale=True, contours=True,
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


# Some helper functon to generate synthetic data
np.random.seed(42)
def generate_line_plot_data():
    """Generate synthetic data for line plots"""
    
    # Dataset 1: Simple sine wave with noise
    x1 = np.linspace(0, 4*np.pi, 100)
    y1 = np.sin(x1) + 0.1 * np.random.randn(100)
    
    # Dataset 2: Exponential decay
    x2 = np.linspace(0, 5, 80)
    y2 = 2 * np.exp(-x2/2) + 0.05 * np.random.randn(80)
    
    # Dataset 3: Polynomial trend
    x3 = np.linspace(-2, 2, 60)
    y3 = x3**3 - 2*x3**2 + x3 + 0.2 * np.random.randn(60)
    
    # Dataset 4: Time series-like data
    x4 = np.arange(0, 100)
    y4 = np.cumsum(np.random.randn(100)) + 0.02 * x4
    
    return {
        'sine_wave': (x1, y1),
        'exponential_decay': (x2, y2),
        'polynomial': (x3, y3),
        'time_series': (x4, y4)
    }

def generate_histogram_data():
    """Generate synthetic data for 1D histograms"""
    
    # Dataset 1: Normal distribution
    normal_data = np.random.normal(50, 15, 1000)
    
    # Dataset 2: Bimodal distribution
    bimodal_data = np.concatenate([
        np.random.normal(30, 8, 500),
        np.random.normal(70, 12, 500)
    ])
    
    # Dataset 3: Skewed distribution (gamma)
    skewed_data = np.random.gamma(2, 2, 1000)
    
    # Dataset 4: Uniform distribution
    uniform_data = np.random.uniform(0, 100, 800)
    
    # Dataset 5: Exponential distribution
    exponential_data = np.random.exponential(2, 1000)
    
    # Dataset 6: Mixed distribution (realistic molecular dynamics data)
    md_like_data = np.concatenate([
        np.random.normal(10, 2, 300),    # Main population
        np.random.normal(15, 1, 150),    # Secondary population
        np.random.exponential(1, 50) + 20  # Rare events
    ])
    
    return {
        'normal': normal_data,
        'bimodal': bimodal_data,
        'skewed': skewed_data,
        'uniform': uniform_data,
        'exponential': exponential_data,
        'md_like': md_like_data
    }

def generate_errorbar_data():
    """Generate synthetic data for errorbar plots"""
    
    # Dataset 1: Experimental measurements with varying error
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y1 = 2 * x1 + 3 + np.random.randn(10) * 0.5
    yerr1 = np.random.uniform(0.2, 1.0, 10)
    
    # Dataset 2: Concentration vs response with error bars
    concentrations = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    response = 100 * concentrations / (concentrations + 2) + np.random.randn(7) * 2
    response_err = np.array([3, 2.5, 2, 1.5, 1.2, 1.0, 0.8])
    
    # Dataset 3: Time course with asymmetric errors
    time_points = np.array([0, 1, 2, 4, 8, 16, 24, 48])
    values = 10 * np.exp(-time_points/12) + np.random.randn(8) * 0.5
    err_low = np.random.uniform(0.3, 0.8, 8)
    err_high = np.random.uniform(0.5, 1.2, 8)
    
    # Dataset 4: Temperature vs property measurement
    temperatures = np.linspace(273, 373, 12)
    property_values = 0.02 * temperatures + np.random.randn(12) * 0.3
    temp_err = np.full(12, 2.0)  # ±2K temperature error
    prop_err = np.random.uniform(0.1, 0.5, 12)
    
    return {
        'linear_trend': {
            'x': x1, 'y': y1, 'yerr': yerr1, 'xerr': None
        },
        'dose_response': {
            'x': concentrations, 'y': response, 'yerr': response_err, 'xerr': None
        },
        'time_course': {
            'x': time_points, 'y': values, 
            'yerr': [err_low, err_high], 'xerr': None
        },
        'temperature_study': {
            'x': temperatures, 'y': property_values, 
            'yerr': prop_err, 'xerr': temp_err
        }
    }

# Generate Ramachandra plot data
def generate_example_angle_data(n_samples=1000, n_features=21):
    """Generate example angle data for testing"""
    # Create some clustered angle data
    np.random.seed(42)

    # Create multiple clusters
    cluster_centers = [(-90, -45), (0, 0), (45, 90), (-135, 135)]
    cluster_std = 20

    x_data = []
    y_data = []

    for i in range(n_samples):
        feature_data_x = []
        feature_data_y = []

        for j in range(n_features):
            # Randomly choose a cluster
            center_idx = np.random.randint(len(cluster_centers))
            cx, cy = cluster_centers[center_idx]

            # Generate points around the cluster center
            x_val = np.random.normal(cx, cluster_std)
            y_val = np.random.normal(cy, cluster_std)

            # Wrap to [-180, 180] range
            x_val = ((x_val + 180) % 360) - 180
            y_val = ((y_val + 180) % 360) - 180

            feature_data_x.append(x_val)
            feature_data_y.append(y_val)

        x_data.append(feature_data_x)
        y_data.append(feature_data_y)

    return np.array(x_data), np.array(y_data)


if __name__ == "__main__":
    # Generate all data
    line_data = generate_line_plot_data()
    hist_data = generate_histogram_data()
    errorbar_data = generate_errorbar_data()
    x_angles, y_angles = generate_example_angle_data(1000, 21)

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 1], 
                figure=fig, 
                height_ratios=[1, 1])	
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    # ax0 for line
    line_plot(line_data['sine_wave'][0], line_data['sine_wave'][1], ax=ax0, decoration=True)
    # ax1 for hist
    hist_plot(hist_data['normal'], ax=ax1, bins=50, fit=True)
    # ax2 for errorbar
    errorbar_plot(errorbar_data['linear_trend']['x'], errorbar_data['linear_trend']['y'], yerror=errorbar_data['linear_trend']['yerr'], ax=ax2)
    # ax3 for 2d histogram
    hist2d_contour_plot(x_angles, y_angles, bins=100, fig=fig, ax=ax3)

    plt.tight_layout()
    plt.show()