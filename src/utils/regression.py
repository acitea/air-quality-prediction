
from typing import Callable
import numpy as np
import pandas as pd

global REGION_COORDS
REGION_COORDS = pd.DataFrame({
    "region": ["central", "north", "south", "east", "west"],
    "latitude": [1.3521, 1.4180, 1.2800, 1.3500, 1.3400],
    "longitude": [103.8198, 103.8270, 103.8500, 103.9400, 103.7000]
})

# Decorator to ensure region coordinates are loaded
def ensure_coords_set(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        global REGION_COORDS
        if REGION_COORDS is None:
            raise ValueError("REGION_COORDS is not set. Please load region coordinates mapping before calling this function.")
        return func(*args, **kwargs)
    return wrapper

@ensure_coords_set
def regression_features_pm25_daily(df: pd.DataFrame):
    """Prepare features for time series regression modeling"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # Lag features
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 28]:
        df[f'pm25_lag_{lag}d'] = df['pm25'].shift(lag)

    # Rolling statistics
    for window in [3, 7, 14, 28]:
        min_valid = max(1, window // 2)
        df[f'pm25_rolling_mean_{window}d'] = df['pm25'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'pm25_rolling_std_{window}d'] = df['pm25'].shift(1).rolling(window, min_periods=min_valid).std()
        df[f'pm25_rolling_min_{window}d'] = df['pm25'].shift(1).rolling(window, min_periods=min_valid).min()
        df[f'pm25_rolling_max_{window}d'] = df['pm25'].shift(1).rolling(window, min_periods=min_valid).max()

    df_clean = df.dropna()

    return df_clean

@ensure_coords_set
def regression_features_pm25_hourly(df: pd.DataFrame):
    """Prepare features for time series regression modeling"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'pm25_lag_{lag}h'] = df['pm25'].shift(lag)

    # Rolling statistics
    for window in [6, 12, 24, 72, 168]:
        min_valid = max(1, window // 2)
        df[f'pm25_rolling_mean_{window}h'] = df['pm25'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'pm25_rolling_std_{window}h'] = df['pm25'].shift(1).rolling(window, min_periods=min_valid).std()
        df[f'pm25_rolling_min_{window}h'] = df['pm25'].shift(1).rolling(window, min_periods=min_valid).min()
        df[f'pm25_rolling_max_{window}h'] = df['pm25'].shift(1).rolling(window, min_periods=min_valid).max()

    df_clean = df.dropna()

    return df_clean


@ensure_coords_set
def regression_features_wind_direction(df: pd.DataFrame):
    """Prepare features for time series regression modeling"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)


    # Convert direction to vector components (recommended for modeling)
    df['wind_u'] = -np.sin(np.deg2rad(df['wind_direction_avg']))
    df['wind_v'] = -np.cos(np.deg2rad(df['wind_direction_avg']))

    # Lag features (using vector components)
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f'wind_u_lag_{lag}h'] = df['wind_u'].shift(lag)
        df[f'wind_v_lag_{lag}h'] = df['wind_v'].shift(lag)
        df[f'wind_direction_lag_{lag}h'] = df['wind_direction_avg'].shift(lag)

    # Rolling statistics
    for window in [6, 12, 24, 72, 168]:
        min_valid = max(1, window // 2)
        df[f'wind_u_rolling_mean_{window}h'] = df['wind_u'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'wind_v_rolling_mean_{window}h'] = df['wind_v'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'direction_std_rolling_mean_{window}h'] = df['wind_direction_std'].shift(1).rolling(window, min_periods=min_valid).mean()

    df_clean = df.dropna()

    return df_clean


@ensure_coords_set
def regression_features_wind_speed(df: pd.DataFrame):
    """Prepare features for time series regression modeling"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)


    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f'wind_speed_lag_{lag}h'] = df['wind_speed_avg'].shift(lag)

    # Rolling statistics
    for window in [6, 12, 24, 72, 168]:
        min_valid = max(1, window // 2)
        df[f'wind_speed_rolling_mean_{window}h'] = df['wind_speed_avg'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'wind_speed_rolling_std_{window}h'] = df['wind_speed_avg'].shift(1).rolling(window, min_periods=min_valid).std()
        df[f'wind_speed_rolling_min_{window}h'] = df['wind_speed_avg'].shift(1).rolling(window, min_periods=min_valid).min()
        df[f'wind_speed_rolling_max_{window}h'] = df['wind_speed_avg'].shift(1).rolling(window, min_periods=min_valid).max()

    df_clean = df.dropna()

    return df_clean


@ensure_coords_set
def regression_features_air_temperature(df: pd.DataFrame):
    """Prepare features for time series regression modeling"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)


    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f'air_temperature_lag_{lag}h'] = df['air_temperature_avg'].shift(lag)

    # Rolling statistics
    for window in [6, 12, 24, 72, 168]:
        min_valid = max(1, window // 2)
        df[f'air_temperature_rolling_mean_{window}h'] = df['air_temperature_avg'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'air_temperature_rolling_std_{window}h'] = df['air_temperature_avg'].shift(1).rolling(window, min_periods=min_valid).std()
        df[f'air_temperature_rolling_min_{window}h'] = df['air_temperature_avg'].shift(1).rolling(window, min_periods=min_valid).min()
        df[f'air_temperature_rolling_max_{window}h'] = df['air_temperature_avg'].shift(1).rolling(window, min_periods=min_valid).max()

    df_clean = df.dropna()

    return df_clean

@ensure_coords_set
def regression_features_wind_direction_daily(df: pd.DataFrame):
    """Prepare features for time series regression modeling"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)


    # Convert direction to vector components (recommended for modeling)
    df['wind_u'] = -np.sin(np.deg2rad(df['wind_direction_avg_mean']))
    df['wind_v'] = -np.cos(np.deg2rad(df['wind_direction_avg_mean']))

    # Lag features (using vector components)
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 28]:
        df[f'wind_u_lag_{lag}d'] = df['wind_u'].shift(lag)
        df[f'wind_v_lag_{lag}d'] = df['wind_v'].shift(lag)
        df[f'wind_direction_lag_{lag}d'] = df['wind_direction_avg_mean'].shift(lag)

    # Rolling statistics
    for window in [3, 7, 14, 28]:
        min_valid = max(1, window // 2)
        df[f'wind_u_rolling_mean_{window}d'] = df['wind_u'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'wind_v_rolling_mean_{window}d'] = df['wind_v'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'direction_std_rolling_mean_{window}d'] = df['wind_direction_avg_std'].shift(1).rolling(window, min_periods=min_valid).mean()

    df_clean = df.dropna()

    return df_clean


@ensure_coords_set
def regression_features_wind_speed_daily(df: pd.DataFrame):
    """Prepare features for time series regression modeling"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)


    # Lag features
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 28]:
        df[f'wind_speed_lag_{lag}d'] = df['wind_speed_avg_mean'].shift(lag)

    # Rolling statistics
    for window in [3, 7, 14, 28]:
        min_valid = max(1, window // 2)
        df[f'wind_speed_rolling_mean_{window}d'] = df['wind_speed_avg_mean'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'wind_speed_rolling_std_{window}d'] = df['wind_speed_avg_mean'].shift(1).rolling(window, min_periods=min_valid).std()
        df[f'wind_speed_rolling_min_{window}d'] = df['wind_speed_avg_mean'].shift(1).rolling(window, min_periods=min_valid).min()
        df[f'wind_speed_rolling_max_{window}d'] = df['wind_speed_avg_mean'].shift(1).rolling(window, min_periods=min_valid).max()

    df_clean = df.dropna()

    return df_clean


@ensure_coords_set
def regression_features_air_temperature_daily(df: pd.DataFrame):
    """Prepare features for time series regression modeling"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)


    # Lag features
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 28]:
        df[f'air_temperature_lag_{lag}d'] = df['air_temperature_avg_mean'].shift(lag)

    # Rolling statistics
    for window in [3, 7, 14, 28]:
        min_valid = max(1, window // 2)
        df[f'air_temperature_rolling_mean_{window}d'] = df['air_temperature_avg_mean'].shift(1).rolling(window, min_periods=min_valid).mean()
        df[f'air_temperature_rolling_std_{window}d'] = df['air_temperature_avg_mean'].shift(1).rolling(window, min_periods=min_valid).std()
        df[f'air_temperature_rolling_min_{window}d'] = df['air_temperature_avg_mean'].shift(1).rolling(window, min_periods=min_valid).min()
        df[f'air_temperature_rolling_max_{window}d'] = df['air_temperature_avg_mean'].shift(1).rolling(window, min_periods=min_valid).max()

    df_clean = df.dropna()

    return df_clean


def _map_coords_to_region(df: pd.DataFrame, region_coords: pd.DataFrame) -> pd.Series:
    """
    For each row in df (with latitude/longitude) find the nearest region from region_coords.
    Returns a pandas Series of region names aligned with df.
    """
    # df: N x 2 (lat, lon)
    sensor_xy = df[["latitude", "longitude"]].to_numpy()  # shape (N, 2)
    # region_coords: M x 2 (lat, lon)
    region_xy = region_coords[["latitude", "longitude"]].to_numpy()  # shape (M, 2)

    # compute squared distances (N, M)
    # distance^2 = (lat1 - lat2)^2 + (lon1 - lon2)^2
    diff_lat = sensor_xy[:, [0]] - region_xy[:, 0]  # (N, 1) - (M,) -> (N, M)
    diff_lon = sensor_xy[:, [1]] - region_xy[:, 1]
    dist_sq = diff_lat**2 + diff_lon**2  # (N, M)

    # index of closest region for each sensor row
    nearest_idx = dist_sq.argmin(axis=1)  # (N,)

    # map to region names
    regions = region_coords["region"].to_numpy()
    return pd.Series(regions[nearest_idx], index=df.index, name="region")

def apply_func_to_groups(df: pd.DataFrame, group_col: list[str], func: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Apply a function to each group in the DataFrame and combine results"""
    grouped = df.groupby(group_col)
    processed_groups = []

    for name, group in grouped:
        processed_group = func(group)
        processed_groups.append(processed_group)

    return pd.concat(processed_groups).reset_index(drop=True)

def add_time_features(df: pd.DataFrame, add_hour: bool = False) -> pd.DataFrame:
    """Add time-based features to the DataFrame"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # Time-based features
    if add_hour:
        df['hour'] = (df['timestamp'].dt.hour).astype('int8')
    df['day_of_week'] = (df['timestamp'].dt.dayofweek).astype('int8')
    df['day_of_month'] = (df['timestamp'].dt.day).astype('int8')
    df['month'] = (df['timestamp'].dt.month).astype('int8')
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(bool)

    return df