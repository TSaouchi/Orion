import numpy as np
from scipy import stats
import dask.array as da
from dask.array import stats as dask_stats

def merge_points(data, axis=0, threshold=1e-5, aggregation_method='first'):

    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    
    if axis < 0 or axis >= data.shape[1]:
        raise ValueError(f"Invalid axis {axis} for data with shape {data.shape}.")
    
    valid_methods = {'mean', 'max', 'min', 'first', 'last'}
    if aggregation_method not in valid_methods:
        raise ValueError(f"Invalid aggregation method. Must be one of {valid_methods}.")

    if len(data) == 0:
        return np.array([])  # Return an empty array if there's no data

    # Sort the data along the specified axis
    sorted_indices = np.argsort(data[:, axis])
    sorted_data = data[sorted_indices]

    # Initialize an array to store merged data
    merged_data = []
    current_group = [sorted_data[0]]

    for i in range(1, len(sorted_data)):
        current_point = sorted_data[i]
        previous_point = current_group[-1]

        # Check if the difference along the specified axis is within the threshold
        if np.abs(current_point[axis] - previous_point[axis]) <= threshold:
            current_group.append(current_point)
        else:
            # Merge the current group based on the chosen aggregation method
            merged_data.append(aggregate_group(current_group, aggregation_method))
            # Start a new group with the current point
            current_group = [current_point]
    
    # Merge the last group
    if current_group:
        merged_data.append(aggregate_group(current_group, aggregation_method))

    return np.array(merged_data)

def aggregate_group(group, method):
    """
    Aggregates a group of points using the specified method.
    """
    if method == 'mean':
        return np.mean(group, axis=0)
    elif method == 'max':
        return np.max(group, axis=0)
    elif method == 'min':
        return np.min(group, axis=0)
    elif method == 'first':
        return group[0]
    elif method == 'last':
        return group[-1]

def get_closest_points(data, target_value=1, n_return_points=5, axis=0, 
                       sorted_axis = 0):
    """
    Find the n points where the values along a specified axis are closest to a target value,
    and return the corresponding data points.

    Parameters
    ----------
    data : numpy.ndarray
        Array of data points. Can be 1D, 2D, or 3D.
    target_value : float, optional
        The value to which the points should be closest (default is 1).
    n_return_points : int, optional
        The number of closest points to return (default is 5).
    axis : int, optional
        The axis along which to find the closest points (default is 0).

    Returns
    -------
    numpy.ndarray
        Array of data points that are closest to the target_value along the specified axis.
    """
    # Handle 1D data separately
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        axis = 0

    # Ensure n_return_points doesn't exceed the number of available points
    n_return_points = min(n_return_points, len(data))

    # Ensure axis is valid
    if axis >= data.shape[1]:
        raise ValueError(f"Axis {axis} is out of bounds for shape {data.shape}")

    # Calculate the absolute difference between the specified axis values and target_value
    diff = np.abs(data[:, axis] - target_value)
    
    # Get the indices of the n smallest differences
    closest_indices = np.argsort(diff)[:n_return_points]
    
    # Select the corresponding data points
    closest_points = data[closest_indices]

    # Sort by the values along the specified axis
    sorted_indices = np.argsort(closest_points[:, sorted_axis])
    
    # Return the points sorted according to the specified axis values
    return closest_points[sorted_indices]

def compute_stats(x):
    """
    Compute descriptive statistics for a given Dask or NumPy array efficiently.

    This function computes key statistics such as count, min, max, mean, variance, skewness, and kurtosis for a given 
    input array. The computation leverages Dask's lazy evaluation for efficiency when handling large arrays, ensuring 
    that only necessary parts of the array are computed in memory.

    Parameters
    ----------
    x : dask.array.Array or numpy.ndarray
        Input array for which to compute the statistics. If a Dask array is passed, the computation is done lazily to 
        handle large datasets without loading the entire array into memory. If a NumPy array is passed, it will be 
        computed directly.

    Returns
    -------
    stats_names : list
        List of strings representing the names of the computed statistics.
        These include 'count', 'min', 'max', 'mean', 'variance', 'skewness', and 'kurtosis'.
    stats_values : list
        List of computed values corresponding to the statistics in `stats_names`.
        These include the number of observations, minimum, maximum, mean, variance, skewness, and kurtosis, all returned 
        as regular Python floats.

    Example
    -------
    .. code-block:: python

        import dask.array as da
        from scipy import stats

        # For small data using NumPy
        data = np.array([1, 2, 3, 4, 5])
        stats_names, stats_values = compute_stats(data)
        print(stats_names, stats_values)
        
        # For large data using Dask
        dask_data = da.random.random(size=(10000,), chunks=(1000,))
        stats_names, stats_values = compute_stats(dask_data)
        print(stats_names, stats_values)

    Example Usage
    -------------
    >>> import dask.array as da
    >>> data = da.random.random(size=(10000,), chunks=(1000,))
    >>> stats_names, stats_values = compute_stats(data)
    >>> print(stats_names, stats_values)
    """
  
    count = x.size
    min_val = x.min()
    max_val = x.max()
    mean_val = x.mean()
    variance_val = x.var(ddof=1)  # ddof=1 for sample variance, like scipy's describe
    skewness_val = dask_stats.skew(x)
    kurtosis_val = dask_stats.kurtosis(x)
    
    # Compute only the statistics, not the entire array
    computed_stats = da.compute(count, min_val, max_val, mean_val, variance_val, skewness_val, kurtosis_val)
    
    # Define names and values of the statistics
    stats_names = ['count', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis']
    stats_values = [float(s) for s in computed_stats]  # Ensure all are regular floats
    
    return stats_names, stats_values