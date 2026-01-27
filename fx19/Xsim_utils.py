from __future__ import division, unicode_literals, print_function
import numpy as np
from scipy.stats import wasserstein_distance

"""
Xsim_utils.py
-------------
Helper functions for Experimental Simulation modules.
This module handles:
1. File I/O (Reading .txt, .csv, GSAS outputs)
2. Data Normalization
3. Statistical Metrics (EMD, IAD)
"""

# ==============================================================================
# 1. FILE I/O
# ==============================================================================

def read_raw_txt(filename, background_subtracted=False):
    """
    Reads standard X Y text data.
    
    Args:
        filename (str): Path to the file.
        background_subtracted (bool): Placeholder for future logic.
        
    Returns:
        tuple: (q_array, y_obs_array)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    qs, ys_obs = [], []
    
    # Skip header (lines[1:]), read first two columns
    for l in lines[1:]:
        try:
            parts = l.split()
            if len(parts) >= 2:
                q = float(parts[0])
                y_obs = float(parts[1])
                qs.append(q)
                ys_obs.append(y_obs)
        except ValueError:
            continue
            
    return np.array(qs), np.array(ys_obs)


def read_gsas_histogram(filename):
    """
    Read the simulated histogram data from GSASII-generated CSV.
    
    Args:
        filename (str): Path to the GSAS .csv file.
        
    Returns:
        np.ndarray: Array containing [2theta, Intensity] columns.
    """
    if not filename:
        return np.array([])

    with open(filename, 'r') as f:
        data_raw = f.readlines()
    
    try:
        # Find the line containing 'weight' to skip headers (User logic)
        flag_indices = [i for i, l in enumerate(data_raw) if 'weight' in l]
        if not flag_indices:
            print(f"Warning: 'weight' keyword not found in {filename}")
            return np.array([])
            
        flag = flag_indices[0]
        data_body = data_raw[flag+1:]
        
        # Parse comma-separated values
        # Using float() instead of eval() for safety and speed
        parsed_data = []
        for l in data_body:
            clean_line = l.strip().split(',')
            if len(clean_line) >= 2:
                parsed_data.append([float(clean_line[0]), float(clean_line[1])])
                
        return np.array(parsed_data)
        
    except Exception as e:
        print(f"Error reading GSAS histogram {filename}: {e}")
        return np.array([])


# ==============================================================================
# 2. NORMALIZATION
# ==============================================================================

def normalize_to_100(arr, target_min=0, target_max=1):
    """
    Min-Max normalization scaled to a target range (default 0-1).
    """
    arr = np.array(arr)
    original_min = np.min(arr)
    original_max = np.max(arr)
    
    if original_max == original_min:
        return np.full_like(arr, (target_min + target_max) / 2)
    
    normalized_arr = (arr - original_min) / (original_max - original_min)
    scaled_arr = normalized_arr * (target_max - target_min) + target_min
    return scaled_arr


def _normalize_pmf(pmf):
    """
    Internal helper to ensure array sums to 1 (Probability Mass Function).
    Used for EMD and IAD calculations.
    """
    pmf = np.array(pmf)
    pmf_sum = np.sum(pmf)
    if pmf_sum == 0:
        return pmf.copy()
    return pmf / pmf_sum


# ==============================================================================
# 3. METRICS (Comparison Scoring)
# ==============================================================================

def difference_in_cum_sums(pmf_a, pmf_b):
    """
    Calculates the Integrated Absolute Difference (IAD) between CDFs.
    
    Args:
        pmf_a (np.array): Signal A (e.g., Simulated)
        pmf_b (np.array): Signal B (e.g., Experimental)
        
    Returns:
        float: Sum of absolute differences between the Cumulative Distribution Functions.
    """
    pmf_a = _normalize_pmf(pmf_a)
    pmf_b = _normalize_pmf(pmf_b)
    
    cdf_a = np.cumsum(pmf_a)
    cdf_b = np.cumsum(pmf_b)
    
    return np.sum(np.abs(cdf_a - cdf_b))


def get_emd_score(pmf_a, pmf_b):
    """
    Calculates the Earth Mover's Distance (EMD) / Wasserstein Distance.
    
    Args:
        pmf_a (np.array): Signal A
        pmf_b (np.array): Signal B
        
    Returns:
        float: The EMD score.
    """
    pmf_a = _normalize_pmf(pmf_a)
    pmf_b = _normalize_pmf(pmf_b)
    
    # Assuming indices represent the metric space (1D grid)
    positions = np.arange(len(pmf_a))
    
    return wasserstein_distance(
        u_values=positions, 
        v_values=positions, 
        u_weights=pmf_a, 
        v_weights=pmf_b
    )