# Author: Jian Huang
# E-mail: jianhuang@umass.edu

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from typing import Union, Dict, Any, Tuple
import warnings

def compare_distributions(
    dist1: Union[np.ndarray, list],
    dist2: Union[np.ndarray, list],
    bins: int = 50,
    normalize: bool = True,
    bin_strategy: str = 'union'
) -> Dict[str, Any]:
    """
    Compare two distributions using multiple statistical metrics.
    Handles distributions with different ranges by using a unified binning strategy.

    Parameters:
    -----------
    dist1, dist2 : array-like
        The two distributions to compare (can be raw data or probability distributions)
    bins : int, default=50
        Number of bins for histogram-based calculations (used for KL divergence, etc.)
    normalize : bool, default=True
        Whether to normalize histograms to probability distributions
    bin_strategy : str, default='union'
        How to handle different ranges:
        - 'union': Use the full range covering both distributions
        - 'intersection': Use only the overlapping range
        - 'individual': Use separate ranges (less comparable but preserves original shapes)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all comparison metrics and statistics
    """

    # Convert to numpy arrays
    dist1 = np.asarray(dist1)
    dist2 = np.asarray(dist2)

    # Remove NaN values
    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    # Calculate range information
    min1, max1 = np.min(dist1), np.max(dist1)
    min2, max2 = np.min(dist2), np.max(dist2)

    # Determine binning strategy
    if bin_strategy == 'union':
        bin_min = min(min1, min2)
        bin_max = max(max1, max2)
        bin_range = (bin_min, bin_max)
    elif bin_strategy == 'intersection':
        bin_min = max(min1, min2)
        bin_max = min(max1, max2)
        if bin_min >= bin_max:
            warnings.warn("No overlap between distributions. Using union strategy instead.")
            bin_min = min(min1, min2)
            bin_max = max(max1, max2)
        bin_range = (bin_min, bin_max)
        # Filter data to intersection range
        dist1 = dist1[(dist1 >= bin_min) & (dist1 <= bin_max)]
        dist2 = dist2[(dist2 >= bin_min) & (dist2 <= bin_max)]
    else:  # individual
        bin_range = None  # Will be handled separately for each distribution

    results = {
        'range_info': {
            'dist1_range': (min1, max1),
            'dist2_range': (min2, max2),
            'overlap_range': (max(min1, min2), min(max1, max2)),
            'union_range': (min(min1, min2), max(max1, max2)),
            'bin_strategy': bin_strategy,
            'bin_range_used': bin_range
        }
    }

    # === 1. WASSERSTEIN DISTANCE ===
    try:
        wasserstein_dist = wasserstein_distance(dist1, dist2)
        results['wasserstein_distance'] = wasserstein_dist
    except Exception as e:
        results['wasserstein_distance'] = f"Error: {str(e)}"

    # === 2. KOLMOGOROV-SMIRNOV TEST ===
    try:
        ks_statistic, ks_pvalue = ks_2samp(dist1, dist2)
        results['kolmogorov_smirnov'] = {
            'statistic': ks_statistic,
            'p_value': ks_pvalue,
            'significant_difference': ks_pvalue < 0.05
        }
    except Exception as e:
        results['kolmogorov_smirnov'] = f"Error: {str(e)}"

    # === 3. JENSEN-SHANNON DIVERGENCE ===
    try:
        # Create histograms for JS divergence
        if bin_strategy == 'individual':
            # Use separate ranges for each distribution
            hist1, _ = np.histogram(dist1, bins=bins)
            hist2, _ = np.histogram(dist2, bins=bins)
        else:
            # Use unified range
            hist1, _ = np.histogram(dist1, bins=bins, range=bin_range)
            hist2, _ = np.histogram(dist2, bins=bins, range=bin_range)

        if normalize:
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist1 = hist1 + eps
        hist2 = hist2 + eps

        # Normalize after adding epsilon
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)

        js_divergence = jensenshannon(hist1, hist2)
        results['jensen_shannon_divergence'] = js_divergence

    except Exception as e:
        results['jensen_shannon_divergence'] = f"Error: {str(e)}"

    # === 4. KL DIVERGENCE ===
    try:
        # Use same histograms as JS divergence
        if bin_strategy == 'individual':
            # Use separate ranges for each distribution
            hist1, _ = np.histogram(dist1, bins=bins)
            hist2, _ = np.histogram(dist2, bins=bins)
        else:
            # Use unified range
            hist1, _ = np.histogram(dist1, bins=bins, range=bin_range)
            hist2, _ = np.histogram(dist2, bins=bins, range=bin_range)

        if normalize:
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist1 = hist1 + eps
        hist2 = hist2 + eps

        # Normalize after adding epsilon
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)

        # KL divergence: D(P||Q) = sum(P * log(P/Q))
        kl_div_1_2 = np.sum(hist1 * np.log(hist1 / hist2))
        kl_div_2_1 = np.sum(hist2 * np.log(hist2 / hist1))

        results['kl_divergence'] = {
            'kl_div_1_to_2': kl_div_1_2,
            'kl_div_2_to_1': kl_div_2_1,
            'symmetric_kl': (kl_div_1_2 + kl_div_2_1) / 2
        }

    except Exception as e:
        results['kl_divergence'] = f"Error: {str(e)}"

    # === 5. TOTAL VARIATION DISTANCE ===
    try:
        # Use same histograms
        if bin_strategy == 'individual':
            # Use separate ranges for each distribution
            hist1, _ = np.histogram(dist1, bins=bins)
            hist2, _ = np.histogram(dist2, bins=bins)
        else:
            # Use unified range
            hist1, _ = np.histogram(dist1, bins=bins, range=bin_range)
            hist2, _ = np.histogram(dist2, bins=bins, range=bin_range)

        if normalize:
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)

        # Total variation distance = 0.5 * sum(|P - Q|)
        tv_distance = 0.5 * np.sum(np.abs(hist1 - hist2))
        results['total_variation_distance'] = tv_distance

    except Exception as e:
        results['total_variation_distance'] = f"Error: {str(e)}"

    # === 6. BASIC STATISTICS ===
    try:
        stats1 = {
            'mean': np.mean(dist1),
            'std': np.std(dist1),
            'median': np.median(dist1),
            'min': np.min(dist1),
            'max': np.max(dist1),
            'q25': np.percentile(dist1, 25),
            'q75': np.percentile(dist1, 75),
            'skewness': stats.skew(dist1),
            'kurtosis': stats.kurtosis(dist1),
            'n_samples': len(dist1)
        }

        stats2 = {
            'mean': np.mean(dist2),
            'std': np.std(dist2),
            'median': np.median(dist2),
            'min': np.min(dist2),
            'max': np.max(dist2),
            'q25': np.percentile(dist2, 25),
            'q75': np.percentile(dist2, 75),
            'skewness': stats.skew(dist2),
            'kurtosis': stats.kurtosis(dist2),
            'n_samples': len(dist2)
        }

        # Statistical tests
        t_stat, t_pval = stats.ttest_ind(dist1, dist2)
        mann_whitney_stat, mann_whitney_pval = stats.mannwhitneyu(dist1, dist2, alternative='two-sided')

        results['basic_statistics'] = {
            'distribution_1': stats1,
            'distribution_2': stats2,
            'difference_in_means': stats1['mean'] - stats2['mean'],
            'difference_in_medians': stats1['median'] - stats2['median'],
            'ratio_of_stds': stats1['std'] / stats2['std'] if stats2['std'] != 0 else np.inf,
            'welch_t_test': {
                'statistic': t_stat,
                'p_value': t_pval,
                'significant_difference': t_pval < 0.05
            },
            'mann_whitney_u_test': {
                'statistic': mann_whitney_stat,
                'p_value': mann_whitney_pval,
                'significant_difference': mann_whitney_pval < 0.05
            }
        }

    except Exception as e:
        results['basic_statistics'] = f"Error: {str(e)}"

    return results


def mean_squared_absolute_error(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute the Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    between two 1D arrays.

    Args:
        x: First 1D array.
        y: Second 1D array (same shape as x).

    Returns:
        A tuple (mse, mae) of floats.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")

    mse = np.mean((x - y) ** 2)
    mae = np.mean(np.abs(x - y))

    return {'MSE':mse, 'MAE':mae}

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from typing import Union, Dict, Any, Tuple
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def compare_distributions_2d(
    dist1: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    dist2: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    bins: Union[int, Tuple[int, int]] = 50,
    normalize: bool = True,
    bin_strategy: str = 'union',
    method: str = 'histogram'
) -> Dict[str, Any]:
    """
    Compare two 2D distributions using multiple statistical metrics.

    Parameters:
    -----------
    dist1, dist2 : array-like or tuple of arrays
        The two 2D distributions to compare. Can be:
        - 2D arrays of shape (n_samples, 2)
        - Tuples of (x_values, y_values) arrays
    bins : int or tuple of ints, default=50
        Number of bins for histogram-based calculations
    normalize : bool, default=True
        Whether to normalize histograms to probability distributions
    bin_strategy : str, default='union'
        How to handle different ranges: 'union', 'intersection', 'individual'
    method : str, default='histogram'
        Method for density estimation: 'histogram' or 'kde'

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all comparison metrics and statistics
    """

    # Convert inputs to standard format
    def standardize_input(dist):
        if isinstance(dist, tuple):
            x, y = dist
            return np.column_stack([np.asarray(x), np.asarray(y)])
        else:
            dist = np.asarray(dist)
            if dist.ndim == 1:
                raise ValueError("1D array provided. Use compare_distributions() for 1D data.")
            if dist.shape[1] != 2:
                raise ValueError("2D distributions must have exactly 2 columns.")
            return dist

    dist1 = standardize_input(dist1)
    dist2 = standardize_input(dist2)

    # Remove NaN values
    dist1 = dist1[~np.isnan(dist1).any(axis=1)]
    dist2 = dist2[~np.isnan(dist2).any(axis=1)]

    # Calculate range information
    min1 = np.min(dist1, axis=0)
    max1 = np.max(dist1, axis=0)
    min2 = np.min(dist2, axis=0)
    max2 = np.max(dist2, axis=0)

    # Determine binning strategy
    if bin_strategy == 'union':
        bin_min = np.minimum(min1, min2)
        bin_max = np.maximum(max1, max2)
        bin_range = [(bin_min[i], bin_max[i]) for i in range(2)]
    elif bin_strategy == 'intersection':
        bin_min = np.maximum(min1, min2)
        bin_max = np.minimum(max1, max2)
        if np.any(bin_min >= bin_max):
            warnings.warn("No overlap between distributions. Using union strategy instead.")
            bin_min = np.minimum(min1, min2)
            bin_max = np.maximum(max1, max2)
        bin_range = [(bin_min[i], bin_max[i]) for i in range(2)]
        # Filter data to intersection range
        mask1 = np.all((dist1 >= bin_min) & (dist1 <= bin_max), axis=1)
        mask2 = np.all((dist2 >= bin_min) & (dist2 <= bin_max), axis=1)
        dist1 = dist1[mask1]
        dist2 = dist2[mask2]
    else:  # individual
        bin_range = None

    # Handle bins parameter
    if isinstance(bins, int):
        bins = (bins, bins)

    results = {
        'range_info': {
            'dist1_range': (min1, max1),
            'dist2_range': (min2, max2),
            'overlap_range': (np.maximum(min1, min2), np.minimum(max1, max2)),
            'union_range': (np.minimum(min1, min2), np.maximum(max1, max2)),
            'bin_strategy': bin_strategy,
            'bin_range_used': bin_range,
            'method': method
        }
    }

    # === 1. 2D WASSERSTEIN DISTANCE ===
    try:
        # For 2D, we use Earth Mover's Distance (EMD)
        from scipy.spatial.distance import cdist

        # Sample points if distributions are too large
        max_samples = 1000
        if len(dist1) > max_samples:
            idx1 = np.random.choice(len(dist1), max_samples, replace=False)
            sample1 = dist1[idx1]
        else:
            sample1 = dist1

        if len(dist2) > max_samples:
            idx2 = np.random.choice(len(dist2), max_samples, replace=False)
            sample2 = dist2[idx2]
        else:
            sample2 = dist2

        # Calculate pairwise distances
        distances = cdist(sample1, sample2, metric='euclidean')

        # Simple approximation: average minimum distance
        wasserstein_2d = np.mean(np.min(distances, axis=1))
        results['wasserstein_distance_2d'] = wasserstein_2d

    except Exception as e:
        results['wasserstein_distance_2d'] = f"Error: {str(e)}"

    # === 2. 2D KOLMOGOROV-SMIRNOV TEST ===
    try:
        # Use Fasano-Franceschini test for 2D KS test
        def fasano_franceschini_2d(x1, x2):
            """2D Kolmogorov-Smirnov test"""
            n1, n2 = len(x1), len(x2)

            # Combine and sort data
            combined = np.vstack([x1, x2])

            # Create grid of test points
            n_test = min(100, len(combined))
            test_indices = np.random.choice(len(combined), n_test, replace=False)
            test_points = combined[test_indices]

            max_diff = 0
            for point in test_points:
                # Count points <= test point in each dimension
                f1 = np.mean(np.all(x1 <= point, axis=1))
                f2 = np.mean(np.all(x2 <= point, axis=1))
                diff = abs(f1 - f2)
                max_diff = max(max_diff, diff)

            # Approximate p-value
            ks_stat = max_diff
            effective_n = (n1 * n2) / (n1 + n2)
            p_value = 2 * np.exp(-2 * effective_n * ks_stat**2)

            return ks_stat, p_value

        ks_2d_stat, ks_2d_pval = fasano_franceschini_2d(dist1, dist2)
        results['kolmogorov_smirnov_2d'] = {
            'statistic': ks_2d_stat,
            'p_value': ks_2d_pval,
            'significant_difference': ks_2d_pval < 0.05
        }

    except Exception as e:
        results['kolmogorov_smirnov_2d'] = f"Error: {str(e)}"

    # === 3. 2D JENSEN-SHANNON DIVERGENCE ===
    try:
        # Create 2D histograms
        if bin_strategy == 'individual':
            hist1, _, _ = np.histogram2d(dist1[:, 0], dist1[:, 1], bins=bins)
            hist2, _, _ = np.histogram2d(dist2[:, 0], dist2[:, 1], bins=bins)
        else:
            hist1, _, _ = np.histogram2d(dist1[:, 0], dist1[:, 1], bins=bins,
                                       range=bin_range)
            hist2, _, _ = np.histogram2d(dist2[:, 0], dist2[:, 1], bins=bins,
                                       range=bin_range)

        if normalize:
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)

        # Flatten for JS divergence calculation
        hist1_flat = hist1.flatten()
        hist2_flat = hist2.flatten()

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist1_flat = hist1_flat + eps
        hist2_flat = hist2_flat + eps

        # Normalize after adding epsilon
        hist1_flat = hist1_flat / np.sum(hist1_flat)
        hist2_flat = hist2_flat / np.sum(hist2_flat)

        js_divergence_2d = jensenshannon(hist1_flat, hist2_flat)
        results['jensen_shannon_divergence_2d'] = js_divergence_2d

    except Exception as e:
        results['jensen_shannon_divergence_2d'] = f"Error: {str(e)}"

    # === 4. 2D KL DIVERGENCE ===
    try:
        # Use same histograms as JS divergence
        if bin_strategy == 'individual':
            hist1, _, _ = np.histogram2d(dist1[:, 0], dist1[:, 1], bins=bins)
            hist2, _, _ = np.histogram2d(dist2[:, 0], dist2[:, 1], bins=bins)
        else:
            hist1, _, _ = np.histogram2d(dist1[:, 0], dist1[:, 1], bins=bins,
                                       range=bin_range)
            hist2, _, _ = np.histogram2d(dist2[:, 0], dist2[:, 1], bins=bins,
                                       range=bin_range)

        if normalize:
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)

        # Flatten for KL divergence calculation
        hist1_flat = hist1.flatten()
        hist2_flat = hist2.flatten()

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist1_flat = hist1_flat + eps
        hist2_flat = hist2_flat + eps

        # Normalize after adding epsilon
        hist1_flat = hist1_flat / np.sum(hist1_flat)
        hist2_flat = hist2_flat / np.sum(hist2_flat)

        # KL divergence: D(P||Q) = sum(P * log(P/Q))
        kl_div_1_2 = np.sum(hist1_flat * np.log(hist1_flat / hist2_flat))
        kl_div_2_1 = np.sum(hist2_flat * np.log(hist2_flat / hist1_flat))

        results['kl_divergence_2d'] = {
            'kl_div_1_to_2': kl_div_1_2,
            'kl_div_2_to_1': kl_div_2_1,
            'symmetric_kl': (kl_div_1_2 + kl_div_2_1) / 2
        }

    except Exception as e:
        results['kl_divergence_2d'] = f"Error: {str(e)}"

    # === 5. 2D TOTAL VARIATION DISTANCE ===
    try:
        # Use same histograms
        if bin_strategy == 'individual':
            hist1, _, _ = np.histogram2d(dist1[:, 0], dist1[:, 1], bins=bins)
            hist2, _, _ = np.histogram2d(dist2[:, 0], dist2[:, 1], bins=bins)
        else:
            hist1, _, _ = np.histogram2d(dist1[:, 0], dist1[:, 1], bins=bins,
                                       range=bin_range)
            hist2, _, _ = np.histogram2d(dist2[:, 0], dist2[:, 1], bins=bins,
                                       range=bin_range)

        if normalize:
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)

        # Total variation distance = 0.5 * sum(|P - Q|)
        tv_distance_2d = 0.5 * np.sum(np.abs(hist1 - hist2))
        results['total_variation_distance_2d'] = tv_distance_2d

    except Exception as e:
        results['total_variation_distance_2d'] = f"Error: {str(e)}"

    # === 6. 2D BASIC STATISTICS ===
    try:
        # Calculate correlation matrices
        corr1 = np.corrcoef(dist1.T)
        corr2 = np.corrcoef(dist2.T)

        stats1 = {
            'mean': np.mean(dist1, axis=0),
            'std': np.std(dist1, axis=0),
            'median': np.median(dist1, axis=0),
            'min': np.min(dist1, axis=0),
            'max': np.max(dist1, axis=0),
            'correlation': corr1[0, 1],
            'covariance_matrix': np.cov(dist1.T),
            'n_samples': len(dist1)
        }

        stats2 = {
            'mean': np.mean(dist2, axis=0),
            'std': np.std(dist2, axis=0),
            'median': np.median(dist2, axis=0),
            'min': np.min(dist2, axis=0),
            'max': np.max(dist2, axis=0),
            'correlation': corr2[0, 1],
            'covariance_matrix': np.cov(dist2.T),
            'n_samples': len(dist2)
        }

        # 2D statistical tests
        # Hotelling's T² test for mean difference
        def hotelling_t2_test(x1, x2):
            n1, n2 = len(x1), len(x2)
            mean1, mean2 = np.mean(x1, axis=0), np.mean(x2, axis=0)
            cov1, cov2 = np.cov(x1.T), np.cov(x2.T)

            # Pooled covariance
            pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

            # T² statistic
            mean_diff = mean1 - mean2
            t2_stat = (n1 * n2) / (n1 + n2) * np.dot(mean_diff,
                      np.dot(np.linalg.inv(pooled_cov), mean_diff))

            # F statistic
            f_stat = ((n1 + n2 - 2 - 1) / (2 * (n1 + n2 - 2))) * t2_stat

            # P-value (approximate)
            p_val = 1 - stats.f.cdf(f_stat, 2, n1 + n2 - 2 - 1)

            return t2_stat, f_stat, p_val

        try:
            t2_stat, f_stat, t2_pval = hotelling_t2_test(dist1, dist2)
            hotelling_results = {
                't2_statistic': t2_stat,
                'f_statistic': f_stat,
                'p_value': t2_pval,
                'significant_difference': t2_pval < 0.05
            }
        except:
            hotelling_results = "Error in computation"

        results['basic_statistics_2d'] = {
            'distribution_1': stats1,
            'distribution_2': stats2,
            'difference_in_means': stats1['mean'] - stats2['mean'],
            'difference_in_correlations': stats1['correlation'] - stats2['correlation'],
            'hotelling_t2_test': hotelling_results
        }

    except Exception as e:
        results['basic_statistics_2d'] = f"Error: {str(e)}"

    # === 7. ADDITIONAL 2D METRICS ===
    try:
        # Energy distance (2D generalization of Cramér-von Mises)
        def energy_distance_2d(x1, x2, n_samples=500):
            """Calculate energy distance between two 2D distributions"""
            # Sample if too large
            if len(x1) > n_samples:
                idx1 = np.random.choice(len(x1), n_samples, replace=False)
                x1_sample = x1[idx1]
            else:
                x1_sample = x1

            if len(x2) > n_samples:
                idx2 = np.random.choice(len(x2), n_samples, replace=False)
                x2_sample = x2[idx2]
            else:
                x2_sample = x2

            # Calculate pairwise distances
            dist_12 = cdist(x1_sample, x2_sample, metric='euclidean')
            dist_11 = cdist(x1_sample, x1_sample, metric='euclidean')
            dist_22 = cdist(x2_sample, x2_sample, metric='euclidean')

            # Energy distance formula
            term1 = 2 * np.mean(dist_12)
            term2 = np.mean(dist_11)
            term3 = np.mean(dist_22)

            energy_dist = term1 - term2 - term3
            return energy_dist

        energy_dist = energy_distance_2d(dist1, dist2)

        results['additional_2d_metrics'] = {
            'energy_distance': energy_dist,
            'centroid_distance': np.linalg.norm(stats1['mean'] - stats2['mean']),
            'covariance_frobenius_distance': np.linalg.norm(
                stats1['covariance_matrix'] - stats2['covariance_matrix'], 'fro')
        }

    except Exception as e:
        results['additional_2d_metrics'] = f"Error: {str(e)}"

    return results


