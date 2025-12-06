# Author: Jian Huang
# E-mail: huangjianhuster@gmail.com

# Dependencies
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# Create canvas
def make_canvas(n_plots, ncols=None):
    if ncols is None:
        ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()
    # Hide unused axes (if any)
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    return fig, axes


# Plot histogram 2D plot
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
    x_bins = y_bins = bins

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
            contour_levels = kwargs.get('override_contour_levels')
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


def pairwise_heatmap(matrix, ax, x_tick_labels=None, y_tick_labels=None, **kwargs):
    """
    Create a pairwise heatmap visualization with customizable tick labels.

    Parameters:
    -----------
    matrix : numpy.ndarray
        2D array representing the data to be visualized as a heatmap.
        Should be square for pairwise comparisons.
    ax : matplotlib.axes.Axes
        The axes object to plot the heatmap on.
    x_tick_labels : list or None, optional
        Custom labels for x-axis ticks. If None, automatic numeric labels will be generated.
        Length should match the number of columns or be None.
    y_tick_labels : list or None, optional
        Custom labels for y-axis ticks. If None, automatic numeric labels will be generated.
        Length should match the number of rows or be None.
    **kwargs : dict
        Additional keyword arguments for customization:
        - x_tick_step: int, automatically defined if not given
        - y_tick_step: int, automatically defined if not given
        - cmap : str, default 'RdYlBu_r'
            Colormap for the heatmap
        - grid_color : str, default 'white'
            Color of the grid lines
        - grid_linewidth : float, default 0.1
            Width of the grid lines
        - grid_interval : int, default 1
            Interval of the grid lines
        - colorbar_size : str, default '3%'
            Size of the colorbar relative to the main plot
        - colorbar_pad : float, default 0.1
            Padding between main plot and colorbar
        - x_rotation : float, default 0
            Rotation angle for x-axis tick labels
        - y_rotation : float, default 90
            Rotation angle for y-axis tick labels

    Returns:
    --------
    tuple
        (im, cbar) where im is the image object and cbar is the colorbar object

    Example:
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Create sample data
    >>> matrix = np.random.rand(10, 10)
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>>
    >>> # Basic usage
    >>> im, cbar = pairwise_heatmap(matrix, ax)
    >>>
    >>> # With custom labels and colormap
    >>> x_labels = [f'Gene_{i}' for i in range(10)]
    >>> y_labels = [f'Sample_{i}' for i in range(10)]
    >>> im, cbar = pairwise_heatmap(matrix, ax,
    ...                           x_tick_labels=x_labels,
    ...                           y_tick_labels=y_labels,
    ...                           cmap='viridis',
    ...                           x_rotation=45)
    """

    # Extract parameters with defaults
    cmap = kwargs.get('cmap', 'RdYlBu_r')
    grid_color = kwargs.get('grid_color', 'white')
    grid_linewidth = kwargs.get('grid_linewidth', 0.1)
    grid_interval = kwargs.get('grid_interval', 1)
    colorbar_size = kwargs.get('colorbar_size', '3%')
    colorbar_pad = kwargs.get('colorbar_pad', 0.1)
    x_rotation = kwargs.get('x_rotation', 0)
    y_rotation = kwargs.get('y_rotation', 90)
    x_tick_step = kwargs.get('x_tick_step', None)
    y_tick_step = kwargs.get('y_tick_step', None)

    num_rows, num_cols = matrix.shape

    # Create the main heatmap using imshow
    im = ax.imshow(matrix, cmap=cmap, aspect='equal')

    # Set axis limits
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(num_rows - 0.5, -0.5)

    # Determine tick positions based on matrix size
    def get_tick_step(size):
        if size <= 20:
            return 1
        elif size <= 50:
            return 5
        elif size <= 100:
            return 10
        else:
            return 20

    # Handle x-axis ticks and labels
    x_tick_step = get_tick_step(num_cols) if x_tick_step is None else x_tick_step
    x_tick_positions = list(range(0, num_cols, x_tick_step))

    if x_tick_labels is None:
        x_tick_labels = [str(i) for i in np.arange(1, num_cols + 1, x_tick_step)]
    else:
        # Validate length and filter if needed
        if len(x_tick_labels) == num_cols:
            x_tick_labels = [x_tick_labels[i] for i in x_tick_positions]
        elif len(x_tick_labels) != len(x_tick_positions):
            raise ValueError(f"x_tick_labels length ({len(x_tick_labels)}) must match either "
                           f"matrix columns ({num_cols}) or tick positions ({len(x_tick_positions)})")

    # Handle y-axis ticks and labels
    y_tick_step = get_tick_step(num_rows) if y_tick_step is None else y_tick_step
    y_tick_positions = list(range(0, num_rows, y_tick_step))

    if y_tick_labels is None:
        y_tick_labels = [str(i) for i in np.arange(1, num_rows + 1, y_tick_step)]
    else:
        # Validate length and filter if needed
        if len(y_tick_labels) == num_rows:
            y_tick_labels = [y_tick_labels[i] for i in y_tick_positions]
        elif len(y_tick_labels) != len(y_tick_positions):
            raise ValueError(f"y_tick_labels length ({len(y_tick_labels)}) must match either "
                           f"matrix rows ({num_rows}) or tick positions ({len(y_tick_positions)})")

    # Set ticks and labels
    ax.set_xticks(np.array(x_tick_positions))
    ax.set_xticklabels(x_tick_labels, rotation=x_rotation)
    ax.set_yticks(np.array(y_tick_positions))
    ax.set_yticklabels(y_tick_labels, rotation=y_rotation)

    ax.tick_params(direction='out')

    # turn off major grid
    ax.grid(False, which="major", axis="x")
    ax.grid(False, which="major", axis="y")

    # Add dashed grid lines for every row and column
    for i in range(1, num_rows, grid_interval):
        ax.axhline(i - 0.5, color=grid_color, linewidth=grid_linewidth, linestyle='--', alpha=0.6)
    for i in range(1, num_cols, grid_interval):
        ax.axvline(i - 0.5, color=grid_color, linewidth=grid_linewidth, linestyle='--', alpha=0.6)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
    cbar = plt.colorbar(im, cax=cax)

    return im, cbar

if __name__ == "__main__":
    # test heatmap
    test_matrix = np.random.rand(20, 20)
    fig, ax = plt.subplots(figsize=(8, 6))
    pairwise_heatmap(test_matrix, ax)
    ax.grid(visible=False)
    plt.tight_layout()
    plt.savefig("../../examples/plot_heatmap.png")
    plt.show()