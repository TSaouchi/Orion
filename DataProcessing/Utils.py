import numpy as np

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