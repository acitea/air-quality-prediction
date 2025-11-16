"""
Daily Weather Data Ingestion Script
====================================
Fetches current day's data from Singapore weather APIs and uploads to Hopsworks

Run schedule: Daily at 4:30pm UTC (12:30am SGT next day)
"""
from rich import print
from rich.console import Console
console = Console()

import sys
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import requests
import hopsworks
from typing import Dict, List, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "https://api-open.data.gov.sg/v2/real-time/api"

ENDPOINTS = {
    'pm25': f"{API_BASE_URL}/pm25",
    'wind_speed': f"{API_BASE_URL}/wind-speed",
    'wind_direction': f"{API_BASE_URL}/wind-direction",
    'air_temperature': f"{API_BASE_URL}/air-temperature"
}

REGION_COORDS = pd.DataFrame({
    "region": ["central", "north", "south", "east", "west"],
    "latitude": [1.3521, 1.4180, 1.2800, 1.3500, 1.3400],
    "longitude": [103.8198, 103.8270, 103.8500, 103.9400, 103.7000]
})

# Get D-1 date in Singapore 
CURRENT_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

from utils.circular import circular_mean, circular_std
from utils.regression import \
    regression_features_pm25_daily, \
    regression_features_pm25_hourly, \
    regression_features_wind_direction, \
    regression_features_wind_speed, \
    regression_features_air_temperature, \
    regression_features_wind_direction_daily, \
    regression_features_wind_speed_daily, \
    regression_features_air_temperature_daily, \
    apply_func_to_groups, \
    add_time_features

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


# ============================================================================
# API FETCHING WITH PAGINATION
# ============================================================================

def fetch_data_from_api(endpoint: str, date: str, max_retries: int = 3) -> Dict:
    """
    Fetch data from API with pagination handling

    Args:
        endpoint: API endpoint URL
        date: Date in YYYY-MM-DD format
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary with 'items' and 'stations' (if applicable)
    """
    all_items = []
    stations_metadata = {}
    pagination_token = None

    params = {'date': date}

    print(f"  Fetching from {endpoint.split('/')[-1]}...")

    retry_count = 0
    while retry_count < max_retries:
        try:
            if pagination_token:
                params['paginationToken'] = pagination_token

            response = requests.get(endpoint, params=params, timeout=30)

            data = response.json()

            if data.get('code', 0) != 0:
                error_msg = data.get('errorMsg', 'Unknown error')
                if 'not found' in error_msg.lower():
                    print(f"    No data available for {date}")
                    return {'items': [], 'stations': {}}
                raise Exception(f"API Error: {error_msg}")

            response_data = data.get('data', {})

            # Extract station metadata if available
            stations = response_data.get('stations', [])
            for station in stations:
                station_id = station.get('id')
                if station_id and station_id not in stations_metadata:
                    stations_metadata[station_id] = {
                        'id': station_id,
                        'name': station.get('name'),
                        'latitude': station.get('location', {}).get('latitude'),
                        'longitude': station.get('location', {}).get('longitude')
                    }

            # Extract items/readings
            items = response_data.get('items', []) or response_data.get('readings', [])
            all_items.extend(items)

            # Check for pagination
            pagination_token = response_data.get('paginationToken')

            if not pagination_token:
                break

        except requests.exceptions.RequestException as e:
            retry_count += 1
            print(f"    Request failed (attempt {retry_count}/{max_retries}): {e}")
            if retry_count >= max_retries:
                raise
            time.sleep(2 ** retry_count)  # Exponential backoff

    print(f"    Retrieved {len(all_items)} items")

    return {'items': all_items, 'stations': stations_metadata}

# ============================================================================
# PM2.5 PROCESSING
# ============================================================================

def process_pm25_data(items: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process PM2.5 data into hourly and daily dataframes

    Returns:
        Tuple of (hourly_df, daily_df)
    """
    if not items:
        print("  No PM2.5 data to process")
        return pd.DataFrame(), pd.DataFrame()

    # Flatten data
    flattened = []
    for item in items:
        timestamp = item.get('timestamp')
        readings = item.get('readings', {}).get('pm25_one_hourly', {})

        for region in ['east', 'west', 'north', 'south', 'central']:
            if region in readings:
                flattened.append({
                    'timestamp': timestamp,
                    'region': region,
                    'pm25': readings[region]
                })

    df = pd.DataFrame(flattened)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
   
    df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
    df = df.dropna(subset=['pm25'])

    # Hourly data (already hourly from API)
    hourly_df = df.copy()
    hourly_df = hourly_df.sort_values(['region', 'timestamp']).reset_index(drop=True)

    print(f"  Processed {len(hourly_df)} hourly PM2.5 records")

    # Daily aggregations
    df['date'] = df['timestamp'].dt.date
    daily_df = df.groupby(['region', 'date']).agg({
        'pm25': ['mean']
    }).reset_index()

    daily_df.columns = ['region', 'date', 'pm25']
    daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.drop('date', axis=1)

    print(f"  Processed {len(daily_df)} daily PM2.5 records")

    return hourly_df, daily_df

# ============================================================================
# WIND SPEED PROCESSING
# ============================================================================

def process_wind_speed_data(items: List[Dict], stations: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process wind speed data into hourly and daily dataframes

    Returns:
        Tuple of (hourly_df, daily_df)
    """
    if not items:
        print("  No wind speed data to process")
        return pd.DataFrame(), pd.DataFrame()

    # Flatten data
    flattened = []
    for item in items:
        timestamp = item.get('timestamp')
        data_points = item.get('data', [])

        for point in data_points:
            station_id = point.get('stationId')
            value = point.get('value')

            flattened.append({
                'timestamp': timestamp,
                'station_id': station_id,
                'wind_speed': value
            })

    df = pd.DataFrame(flattened)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
    df = df.dropna(subset=['wind_speed'])

    # Add station metadata
    df['station_name'] = df['station_id'].map(lambda x: stations.get(x, {}).get('name', 'Unknown'))
    df['latitude'] = df['station_id'].map(lambda x: stations.get(x, {}).get('latitude'))
    df['longitude'] = df['station_id'].map(lambda x: stations.get(x, {}).get('longitude'))

    # Aggregate to hourly
    df['hour'] = df['timestamp'].dt.floor('H')
    hourly_df = df.groupby(['station_name', 'hour']).agg({
        'wind_speed': ['mean', 'count'],
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

    hourly_df.columns = ['station_name', 'timestamp', 
                         'wind_speed_avg',
                         'reading_count',
                         'latitude', 'longitude']
    print(f"  Processed {len(hourly_df)} hourly wind speed records")

    # Daily aggregations
    df['date'] = df['timestamp'].dt.date
    daily_df = df.groupby(['station_name', 'date']).agg({
        'wind_speed': ['mean', 'min', 'max', 'std'],
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

    daily_df.columns = ['station_name', 'date',
                        'wind_speed_avg_mean', 'wind_speed_avg_min', 'wind_speed_avg_max',
                        'wind_speed_avg_std',
                        'latitude', 'longitude']
    daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
    daily_df['region'] = _map_coords_to_region(daily_df, REGION_COORDS)
    daily_df = daily_df.drop('date', axis=1)

    print(f"  Processed {len(daily_df)} daily wind speed records")

    return hourly_df, daily_df

# ============================================================================
# WIND DIRECTION PROCESSING (with Circular Statistics)
# ============================================================================

def process_wind_direction_data(items: List[Dict], stations: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process wind direction data with circular mean into hourly and daily dataframes

    Returns:
        Tuple of (hourly_df, daily_df)
    """
    if not items:
        print("  No wind direction data to process")
        return pd.DataFrame(), pd.DataFrame()

    # Flatten data
    flattened = []
    for item in items:
        timestamp = item.get('timestamp')
        data_points = item.get('data', [])

        for point in data_points:
            station_id = point.get('stationId')
            value = point.get('value')

            flattened.append({
                'timestamp': timestamp,
                'station_id': station_id,
                'wind_direction': value
            })

    df = pd.DataFrame(flattened)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['wind_direction'] = pd.to_numeric(df['wind_direction'], errors='coerce')
    df = df.dropna(subset=['wind_direction'])

    # Validate 0-360 range
    df = df[(df['wind_direction'] >= 0) & (df['wind_direction'] <= 360)]

    # Add station metadata
    df['station_name'] = df['station_id'].map(lambda x: stations.get(x, {}).get('name', 'Unknown'))
    df['latitude'] = df['station_id'].map(lambda x: stations.get(x, {}).get('latitude'))
    df['longitude'] = df['station_id'].map(lambda x: stations.get(x, {}).get('longitude'))

    # Aggregate to hourly with circular mean
    df['hour'] = df['timestamp'].dt.floor('H')

    hourly_data = []
    for (station_name, hour), group in df.groupby(['station_name', 'hour']):
        directions = group['wind_direction'].values

        hourly_data.append({
            'station_name': station_name,
            'timestamp': hour,
            'wind_direction_avg': circular_mean(directions),
            'wind_direction_std': circular_std(directions),
            'reading_count': len(directions),
            'latitude': group['latitude'].iloc[0],
            'longitude': group['longitude'].iloc[0]
        })

    hourly_df = pd.DataFrame(hourly_data)

    print(f"  Processed {len(hourly_df)} hourly wind direction records")

    # Daily aggregations with circular mean
    df['date'] = df['timestamp'].dt.date

    daily_data = []
    for (station_name, date), group in df.groupby(['station_name', 'date']):
        directions = group['wind_direction'].values

        daily_data.append({
            'station_name': station_name,
            'date': date,
            'wind_direction_avg_mean': circular_mean(directions),
            'wind_direction_avg_std': circular_std(directions),
            'wind_direction_avg_min': np.min(directions),
            'wind_direction_avg_max': np.max(directions),
            'latitude': group['latitude'].iloc[0],
            'longitude': group['longitude'].iloc[0]
        })

    daily_df = pd.DataFrame(daily_data)
    daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
    daily_df['region'] = _map_coords_to_region(daily_df, REGION_COORDS)
    daily_df = daily_df.drop('date', axis=1)

    print(f"  Processed {len(daily_df)} daily wind direction records")

    return hourly_df, daily_df

# ============================================================================
# AIR TEMPERATURE PROCESSING
# ============================================================================

def process_air_temperature_data(items: List[Dict], stations: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process air temperature data into hourly and daily dataframes

    Returns:
        Tuple of (hourly_df, daily_df)
    """
    if not items:
        print("  No air temperature data to process")
        return pd.DataFrame(), pd.DataFrame()

    # Flatten data
    flattened = []
    for item in items:
        timestamp = item.get('timestamp')
        data_points = item.get('data', [])

        for point in data_points:
            station_id = point.get('stationId')
            value = point.get('value')

            flattened.append({
                'timestamp': timestamp,
                'station_id': station_id,
                'air_temperature': value
            })

    df = pd.DataFrame(flattened)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['air_temperature'] = pd.to_numeric(df['air_temperature'], errors='coerce')
    df = df.dropna(subset=['air_temperature'])

    # Add station metadata
    df['station_name'] = df['station_id'].map(lambda x: stations.get(x, {}).get('name', 'Unknown'))
    df['latitude'] = df['station_id'].map(lambda x: stations.get(x, {}).get('latitude'))
    df['longitude'] = df['station_id'].map(lambda x: stations.get(x, {}).get('longitude'))

    # Aggregate to hourly
    df['hour'] = df['timestamp'].dt.floor('H')
    hourly_df = df.groupby(['station_name', 'hour']).agg({
        'air_temperature': ['mean', 'count'],
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

    hourly_df.columns = ['station_name', 'timestamp',
                         'air_temperature_avg',
                         'reading_count',
                         'latitude', 'longitude']

    print(f"  Processed {len(hourly_df)} hourly air temperature records")

    # Daily aggregations
    df['date'] = df['timestamp'].dt.date
    daily_df = df.groupby(['station_name', 'date']).agg({
        'air_temperature': ['mean', 'min', 'max', 'std'],
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

    daily_df.columns = ['station_name', 'date',
                        'air_temperature_avg_mean', 'air_temperature_avg_min', 'air_temperature_avg_max',
                        'air_temperature_avg_std',
                        'latitude', 'longitude']
    daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
    daily_df['region'] = _map_coords_to_region(daily_df, REGION_COORDS)
    daily_df = daily_df.drop('date', axis=1)

    print(f"  Processed {len(daily_df)} daily air temperature records")

    return hourly_df, daily_df

# ============================================================================
# HOPSWORKS UPLOAD
# ============================================================================

class Hopsworks:

    def __init__(self):
        from hsfs.feature import Feature
        import os
        self.Feature = Feature
        print("\nConnecting to Hopsworks...")
        self.project = hopsworks.login(engine="python", project=os.getenv("HOPSWORKS_PROJECT_NAME"), api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        self.fs = self.project.get_feature_store()

    def retrieve_from_hopsworks_from_time(self,
        feature_group: str,
        feature_names: List[str],
        version: int,
        time_threshold: datetime
    ) -> pd.DataFrame:
        fg = self.fs.get_feature_group(feature_group, version=version)

        df = fg.select(feature_names) \
            .filter(self.Feature('timestamp', 'timestamp') >= time_threshold) \
            .read()

        return df

    def append_to_hopsworks(self,
        data: dict[str, pd.DataFrame],
        version: int
    ):
        """
        Append dataframes to Hopsworks feature groups

        Args:
            data: Dictionary of dataframes to upload. Keys are: Feature Group Name, and Values are: DataFrame
        """

        idx = 0
        print("\nUploading to Hopsworks feature store...")
        for fg_name, df in data.items():
            if df.empty:
                print(f"  Skipping empty feature group: {fg_name}")
                continue

            print(f"  Uploading to feature group: {fg_name} ({len(df)} records)")

            try:
                if idx == 5: 
                    print("    Pausing for 150 seconds to avoid rate limits...")
                    time.sleep(150)
                    idx = 0
                self.fs.get_feature_group(fg_name, version=version).insert(df)
                idx += 1
            except Exception as e:
                print(f"    ✗ Failed to upload to {fg_name}: {e}")
                console.print_exception()
                raise

        print("\n✓ All data uploaded successfully!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main ingestion pipeline"""
    print("=" * 70)
    print("Daily Feature Data Ingestion Pipeline")
    print("=" * 70)
    print(f"\nRun time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Processing for Date: {CURRENT_DATE}")
    start_time = time.time()

    try:
        # Fetch data from all endpoints
        print("\n[1/3] Fetching data from APIs...")
        pm25_data = fetch_data_from_api(ENDPOINTS['pm25'], CURRENT_DATE)
        wind_speed_data = fetch_data_from_api(ENDPOINTS['wind_speed'], CURRENT_DATE)
        wind_direction_data = fetch_data_from_api(ENDPOINTS['wind_direction'], CURRENT_DATE)
        air_temperature_data = fetch_data_from_api(ENDPOINTS['air_temperature'], CURRENT_DATE)

        # Process data
        print("\n[2/3] Processing data...")
        print("\nProcessing PM2.5...")
        pm25_hourly_df, pm25_daily_df = process_pm25_data(pm25_data['items'])

        print("\nProcessing wind speed...")
        wind_speed_df, wind_speed_daily_df = process_wind_speed_data(
            wind_speed_data['items'], wind_speed_data['stations']
        )

        print("\nProcessing wind direction...")
        wind_direction_df, wind_direction_daily_df = process_wind_direction_data(
            wind_direction_data['items'], wind_direction_data['stations']
        )

        print("\nProcessing air temperature...")
        air_temperature_df, air_temperature_daily_df = process_air_temperature_data(
            air_temperature_data['items'], air_temperature_data['stations']
        )

            
        if pm25_hourly_df.empty and \
           wind_speed_df.empty and wind_speed_daily_df.empty and \
           wind_direction_df.empty and wind_direction_daily_df.empty and \
           air_temperature_df.empty and air_temperature_daily_df.empty:
            print("\n✗ No data available to process. Exiting pipeline.")
            return 0
        
        hopsworks_client = Hopsworks()



        fg_data = {
            "pm25_daily": pm25_daily_df, 
            "pm25_hourly": pm25_hourly_df, 
            "wind_speed_hourly": wind_speed_df, 
            "wind_speed_daily": wind_speed_daily_df, 
            "wind_direction_hourly": wind_direction_df, 
            "wind_direction_daily": wind_direction_daily_df, 
            "air_temperature_hourly": air_temperature_df, 
            "air_temperature_daily": air_temperature_daily_df
        }

        # [df.to_csv(f"src/cached_data/{name}.csv", index=False) for name, df in fg_data.items()] # Uncomment to "attempt to cache" locally

        # should be retrieving up to 29 days back of data, combine them, and compute rolling and lag
        for fg, df in fg_data.items():
            if 'hourly' in fg:
                time_threshold = datetime.fromisoformat(CURRENT_DATE) - timedelta(days=9*2)
                # time_threshold = datetime.now(timezone.utc) - timedelta(days=9)
            else:
                time_threshold = datetime.fromisoformat(CURRENT_DATE) - timedelta(days=29*2)
                # time_threshold = datetime.now(timezone.utc) - timedelta(days=29)
            days_lagged_df = hopsworks_client.retrieve_from_hopsworks_from_time(
                fg, 
                df.columns.tolist(), 
                version=4,
                time_threshold=time_threshold
                )
            # days_lagged_df.to_csv(f"src/hopsworks/{fg}_lagged.csv", index=False) # Uncomment to "attempt to cache" locally
            fg_data[fg] = pd.concat([df, days_lagged_df]).drop_duplicates().reset_index(drop=True)
            fg_data[fg]['timestamp'] = pd.to_datetime(fg_data[fg]['timestamp'], utc=True)

        # [df.to_csv(f"src/data_temp/{name}.csv", index=False) for name, df in fg_data.items()] # Uncomment to "attempt to cache" locally
        # [print(name, df.info()) for name, df in fg_data.items()] # Uncomment to debug data types & stats

        fg_data["pm25_daily"] = apply_func_to_groups(add_time_features(fg_data["pm25_daily"], add_hour=False), 'region', regression_features_pm25_daily)
        fg_data["pm25_hourly"] = apply_func_to_groups(add_time_features(fg_data["pm25_hourly"], add_hour=True), 'region', regression_features_pm25_hourly)
        fg_data["wind_speed_hourly"] = apply_func_to_groups(add_time_features(fg_data["wind_speed_hourly"], add_hour=True), 'station_name', regression_features_wind_speed)
        fg_data["wind_speed_daily"] = apply_func_to_groups(add_time_features(fg_data["wind_speed_daily"], add_hour=False), 'station_name', regression_features_wind_speed_daily)
        fg_data["wind_direction_hourly"] = apply_func_to_groups(add_time_features(fg_data["wind_direction_hourly"], add_hour=True), 'station_name', regression_features_wind_direction)
        fg_data["wind_direction_daily"] = apply_func_to_groups(add_time_features(fg_data["wind_direction_daily"], add_hour=False), 'station_name', regression_features_wind_direction_daily)
        fg_data["air_temperature_hourly"] = apply_func_to_groups(add_time_features(fg_data["air_temperature_hourly"], add_hour=True), 'station_name', regression_features_air_temperature)
        fg_data["air_temperature_daily"] = apply_func_to_groups(add_time_features(fg_data["air_temperature_daily"], add_hour=False), 'station_name', regression_features_air_temperature_daily)

        # ===================================================================
        # [print(name + f" | Rows: {len(df)}", df.head()) for name, df in fg_data.items()] # Uncomment to debug data types & stats

        # Upload to Hopsworks
        print("\n[3/3] Uploading to Hopsworks...")
        hopsworks_client.append_to_hopsworks(fg_data, version=4)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("✓ Pipeline completed successfully!")
        print(f"End time:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Total runtime: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        console.print_exception()
        return 1

if __name__ == "__main__":
    sys.exit(main())
