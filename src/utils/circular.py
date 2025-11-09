import numpy as np
# ============================================================================
# CIRCULAR STATISTICS FUNCTIONS
# ============================================================================

def circular_mean(angles_degrees):
    """
    Calculate circular mean of angles in degrees

    Handles the 0°/360° wraparound correctly using unit vectors

    Args:
        angles_degrees: Array-like of angles in degrees

    Returns:
        Circular mean in degrees (0-360)
    """
    if len(angles_degrees) == 0:
        return np.nan

    # Remove NaN values
    angles = np.array(angles_degrees)
    angles = angles[~np.isnan(angles)]

    if len(angles) == 0:
        return np.nan

    # Convert to radians
    angles_rad = np.deg2rad(angles)

    # Convert to unit vectors
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)

    # Average the vectors
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Convert back to angle
    mean_angle_rad = np.arctan2(mean_y, mean_x)
    mean_angle_deg = np.rad2deg(mean_angle_rad)

    # Normalize to 0-360
    if mean_angle_deg < 0:
        mean_angle_deg += 360

    return mean_angle_deg

def circular_std(angles_degrees):
    """
    Calculate circular standard deviation

    Returns value between 0 (all angles same) and 1 (angles uniformly distributed)
    """
    if len(angles_degrees) == 0:
        return np.nan

    angles = np.array(angles_degrees)
    angles = angles[~np.isnan(angles)]

    if len(angles) == 0:
        return np.nan

    angles_rad = np.deg2rad(angles)

    # Mean resultant length (R)
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)
    R = np.sqrt(np.mean(x)**2 + np.mean(y)**2)

    # Circular standard deviation
    circ_std = np.sqrt(-2 * np.log(R))

    return circ_std