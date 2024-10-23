import numpy as np
from scipy import stats
import dask.array as da
from dask.array import stats as dask_stats

def get_unique_pairs(x, y, decimal_places=2):
    """
    Retrieve unique (x, y) pairs after rounding the x values to a specified number of decimal places.

    Parameters
    ----------
    x : numpy.ndarray
        Array of x values (e.g., time values).
    y : numpy.ndarray
        Array of y values corresponding to the x values (e.g., pressure values).
    decimal_places : int, optional
        The number of decimal places to which x values should be rounded (default is 2).

    Returns
    -------
    unique_x : numpy.ndarray
        Array of unique rounded x values.
    unique_y : numpy.ndarray
        Array of corresponding unique y values.

    Example
    -------
    
    .. code-block:: python
        
        x = base[0][0]["DeltaPressure"].get_attribute('TimeValue').compute()
        y = base[0][0]["DeltaPressure"].compute()
        
        unique_x, unique_y = get_unique_pairs(x, y, decimal_places=3)

    Example Usage
    -------------
    >>> unique_x, unique_y = get_unique_pairs(x, y, decimal_places=2)
    >>> print(unique_x, unique_y)
    """
    # Round x to the specified decimal places
    x_rounded = np.round(x, decimal_places)

    # Stack x and y together to form pairs, then use np.unique
    xy_pairs = np.column_stack((x_rounded, y))

    # Use np.unique to find unique rows and keep the first occurrence of each
    unique_pairs, indices = np.unique(xy_pairs, axis=0, return_index=True)

    # Sort the unique pairs by x values (first column)
    sorted_indices = np.argsort(unique_pairs[:, 0])

    # Extract the sorted unique x and y arrays
    unique_x = unique_pairs[sorted_indices, 0]
    unique_y = unique_pairs[sorted_indices, 1]

    return unique_x, unique_y

def get_closest_points(x, y, target_value=1, n_return_points=5):
    """
    Find the n points where the y values are closest to a specified target value, and return the corresponding x and y values.

    Parameters
    ----------
    x : numpy.ndarray
        Array of unique x values (e.g., time or rounded x values).
    y : numpy.ndarray
        Array of unique y values corresponding to the x values.
    target_value : float, optional
        The value to which the y values should be closest (default is 1).
    n : int, optional
        The number of closest points to return (default is 5).

    Returns
    -------
    x_closest : numpy.ndarray
        Array of x values corresponding to the closest y values.
    y_closest : numpy.ndarray
        Array of y values that are closest to the target_value.

    Example
    -------
    
    .. code-block:: python
        
        unique_x, unique_y = get_unique_pairs(x, y)
        
        # Find the 3 closest points to the target value 1
        x_closest, y_closest = get_closest_points(unique_x, unique_y, target_value=1, n=3)

    Example Usage
    -------------
    >>> x_closest, y_closest = get_closest_points(unique_x, unique_y, target_value=1, n=5)
    >>> print(x_closest, y_closest)
    """
    # Ensure n_return_points doesn't exceed the number of available points
    n_return_points = min(n_return_points, len(y))

    # Calculate the absolute difference between y and target_value
    diff = np.abs(y - target_value)
    
    # Get the indices of the n smallest differences
    closest_indices = np.argpartition(diff, n_return_points)[:n_return_points]
    
    # Select the corresponding x and y points
    closest_x = x[closest_indices]
    closest_y = y[closest_indices]

    # Sort by x values and get sorted indices
    sorted_indices = np.argsort(closest_x)
    
    # Return x and y sorted according to x values
    return closest_x[sorted_indices], closest_y[sorted_indices]

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
    