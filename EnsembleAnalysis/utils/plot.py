# Author: Jian Huang
# E-mail: huangjianhuster@gmail.com

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter


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


def hist2d_distXY_contour_plot(x, y, ax=None, fig=None, contour=True, contour_levels=None, contour_annotations=True,
                           bins=50, hist_bins=30, method='kde', kde_bandwidth=None, sigma=1.0, cmap='RdYlBu_r', colorbar=False):
    """
    Plot 2D density-contour map with side histograms.
    
    Parameters:
    -----------
    x, y : array-like
        2D data points coordinates
    ax : matplotlib.axes.Axes, optional
        Main plot axes. If None, will create new figure and axes.
    fig : matplotlib.figure.Figure, optional
        Figure object. If None, will create new figure.
    contour : bool, default True
        Whether to plot contour lines
    n_levels: int, default 5
        number of contour levels
    contour_levels : array-like, optional
        Custom contour levels. If None, automatically determined in log scale.
    bins : int, default 50
        Number of bins for density estimation grid
    hist_bins : int, default 30
        Number of bins for side histograms
    method : str, default 'kde'
        Method for density estimation: 'kde' or 'histogram'
        - 'kde': Kernel Density Estimation (smooth, slower)
        - 'histogram': 2D histogram (faster, shows actual bin counts)
    kde_bandwidth : float, optional
        Bandwidth for KDE. If None, uses scipy default. Only used when method='kde'.
    sigma : float, default 1.0
        Gaussian filter sigma for smoothing density map
    colorbar : bool, default False
        Whether to add a colorbar to the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax_main : matplotlib.axes.Axes
        Main density plot axes
    ax_histx : matplotlib.axes.Axes
        Top histogram axes (x-dimension)
    ax_histy : matplotlib.axes.Axes
        Right histogram axes (y-dimension)
    cbar_ax : matplotlib.axes.Axes or None
        Colorbar axes (None if colorbar=False)
    """
    
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Create figure and axes if not provided
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    
    if ax is None:
        # Create 2x2 gridspec for main plot and side histograms
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 4], width_ratios=[4, 1],
                      hspace=0.05, wspace=0.05)
        
        ax_main = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    else:
        # If axes provided, assume it's the main axes and create gridspec around it
        pos = ax.get_position()
        fig.delaxes(ax)
        
        # Create custom gridspec based on original axes position
        gs = GridSpec(2, 2, figure=fig, 
                      left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1,
                      height_ratios=[1, 4], width_ratios=[4, 1],
                      hspace=0.05, wspace=0.05)
        
        ax_main = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    # Create density estimation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = x_range * 0.1
    y_pad = y_range * 0.1
    
    if method.lower() == 'kde':
        # KDE method: smooth, continuous density estimation
        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, bins)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, bins)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Perform KDE
        if kde_bandwidth is None:
            kde = gaussian_kde(np.vstack([x, y]))
        else:
            kde = gaussian_kde(np.vstack([x, y]), bw_method=kde_bandwidth)
        
        # Evaluate KDE on grid
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
        
        # Apply Gaussian smoothing
        Z = gaussian_filter(Z, sigma=sigma)
        
        extent = [x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad]
        
    elif method.lower() == 'histogram':
        # 2D Histogram method: faster, shows actual data distribution
        Z, x_edges, y_edges = np.histogram2d(x, y, bins=bins, 
                                             range=[[x_min - x_pad, x_max + x_pad], 
                                                    [y_min - y_pad, y_max + y_pad]],
                                             density=True)
        Z = Z.T  # Transpose to match imshow orientation
        
        # Apply Gaussian smoothing
        Z = gaussian_filter(Z, sigma=sigma)
        
        # Create coordinate arrays for contour plotting
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    
    else:
        raise ValueError("Method must be 'kde' or 'histogram'")
    
    # Create custom colormap with white for zero values
    cmap = mpl.colormaps[cmap]
    # colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 256))
    colors = cmap(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]  # Set lowest value to white
    custom_cmap = mcolors.ListedColormap(colors)
    
    # Plot density map
    im = ax_main.imshow(Z, extent=extent, origin='lower', aspect='auto', cmap=custom_cmap)

    # Create colorbar if requested
    cbar_ax = None
    if colorbar:
        # Create colorbar without using divider to preserve sharex functionality
        # Manually create colorbar axes
        main_pos = ax_main.get_position()
        
        # Create colorbar axes manually
        cbar_width = 0.01
        cbar_pad = 0.05
        cbar_ax = fig.add_axes([main_pos.x1 + cbar_pad, main_pos.y0, 
                               cbar_width, main_pos.height])
        
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('', rotation=270, labelpad=15)
        
        # Adjust the right histogram position to account for colorbar
        cbar_pos = cbar_ax.get_position()
        hist_pos = ax_histy.get_position()
        new_left = cbar_pos.x1 + 0.07  # Add space between colorbar and histogram
        ax_histy.set_position([new_left, hist_pos.y0, hist_pos.width, hist_pos.height])


    # Handle aspect ratio adjustments - need to adjust histogram positions after aspect is set
    def adjust_histogram_positions():
        """Adjust histogram positions to match main plot after aspect changes"""
        # Force a draw to ensure aspect ratio is applied
        fig.canvas.draw_idle()
        
        # Get the actual position after aspect adjustment
        main_pos = ax_main.get_position()
        
        # Adjust top histogram to match main plot width
        histx_pos = ax_histx.get_position()
        ax_histx.set_position([main_pos.x0, histx_pos.y0, main_pos.width, histx_pos.height])
        
        # Adjust right histogram to match main plot height (if no colorbar)
        if not colorbar:
            hist_pos = ax_histy.get_position()
            ax_histy.set_position([hist_pos.x0, main_pos.y0, hist_pos.width, main_pos.height])
    
    # Store the adjustment function for later use if aspect is changed
    ax_main._adjust_histograms = adjust_histogram_positions
    
    # Plot contours if requested
    if contour:
        if contour_levels is None:
            # # Automatically determine contour levels in log scale
            # z_max = Z.max()
            # z_min = Z[Z > 0].min() if np.any(Z > 0) else z_max * 1e-6
            # # Create log-spaced contour levels
            # contour_levels = np.logspace(np.log10(z_min), np.log10(z_max), n_levels)[1:-1]

            # Use percentiles of the density distribution
            contour_levels = np.percentile(Z[Z > 0], [10, 25, 50, 75, 90, 95])
            
        print(f'contour levels: {contour_levels}')
        
        # Plot contour lines
        CS = ax_main.contour(X, Y, Z, levels=contour_levels, colors='k', 
                            linewidths=1, alpha=0.8)
        
        # Add contour labels
        if contour_annotations:
            ax_main.clabel(CS, inline=True, fontsize=8, fmt='%.1e')

    # Plot side histograms
    # X-dimension histogram (top)
    n_x, bins_x, patches_x = ax_histx.hist(x, bins=hist_bins, density=True, 
                                           alpha=0.5, color='grey', edgecolor='white')
    
    # Add envelope line for x histogram
    bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
    ax_histx.plot(bin_centers_x, n_x, color='grey', linewidth=2, alpha=0.5)
    
    # Y-dimension histogram (right)
    n_y, bins_y, patches_y = ax_histy.hist(y, bins=hist_bins, density=True, 
                                           alpha=0.5, color='grey', 
                                           edgecolor='white', orientation='horizontal')
    
    # Add envelope line for y histogram
    bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
    ax_histy.plot(n_y, bin_centers_y, color='grey', linewidth=2, alpha=0.5)
    
    # Clean up histogram axes
    # Remove unnecessary spines and ticks
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.tick_params(labelbottom=False, labelleft=False, left=False, top=False)
    
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.tick_params(labelbottom=False, labelleft=False, bottom=False, right=False)
    
    # Set histogram y-axis limits to match data range
    ax_histx.set_ylim(bottom=0)
    ax_histy.set_xlim(left=0)
    
    return fig, ax_main, ax_histx, ax_histy, cbar_ax


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


    # 2d density plot + colorbar + histogram in X and Y directions
    fig3, ax_main3, ax_histx3, ax_histy3,_ = hist2d_distXY_contour_plot(
    x_angles.flatten(), y_angles.flatten(), method='histogram', contour=True,
    bins=200, hist_bins=200, colorbar=True, contour_annotations=False,
    )
    # ax_main3.set_aspect('equal')
    ax_main3.set_xlabel('X Coordinate')
    ax_main3.set_ylabel('Y Coordinate')
    # ax_main3.grid()
    fig3.suptitle('Custom Contour Levels (Histogram Method)', fontsize=14)

    # Set equal aspect
    ax_main3.set_aspect('equal')
    ax_main3._adjust_histograms()
    plt.tight_layout()
    plt.show()

    # more test
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(2, 2, width_ratios=[0.5, 8], 
                figure=fig, 
                height_ratios=[0.5, 8])
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    fig, ax_main3, ax_histx3, ax_histy3,_ = hist2d_distXY_contour_plot(
    x_angles.flatten(), y_angles.flatten(), ax=ax3, fig=fig, method='histogram', contour=True,
    bins=200, hist_bins=200, colorbar=False, contour_annotations=True,
    )
    # ax_main3.set_aspect('equal')
    ax_main3.set_xlabel('X Coordinate')
    ax_main3.set_ylabel('Y Coordinate')
    # ax_main3.grid()
    fig3.suptitle('Custom Contour Levels (Histogram Method)', fontsize=14)

    # Set equal aspect
    ax_main3.set_aspect('equal')
    ax_main3._adjust_histograms()
    plt.tight_layout()
    plt.show()
