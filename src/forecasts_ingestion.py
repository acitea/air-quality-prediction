from datetime import datetime, timezone, timedelta
import zoneinfo
import openmeteo_requests

from utils.circular import circular_mean, circular_std
from utils.regression import add_time_features, apply_func_to_groups, regression_features_wind_direction, regression_features_wind_direction_daily, regression_features_wind_speed, regression_features_wind_speed_daily, regression_features_air_temperature, regression_features_air_temperature_daily
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

REGION_COORDS = pd.DataFrame({
    "region": ["central", "north", "south", "east", "west"],
    "latitude": [1.3521, 1.4180, 1.2800, 1.3500, 1.3400],
    "longitude": [103.8198, 103.8270, 103.8500, 103.9400, 103.7000]
})

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
CURRENT_DATETIME = datetime.now(zoneinfo.ZoneInfo("Asia/Singapore"))
CURRENT_DATE = CURRENT_DATETIME.strftime("%Y-%m-%d")
print("Fetching forecast data from Open-Meteo API for dates:", CURRENT_DATE , "to", (CURRENT_DATETIME + timedelta(days=7)).strftime("%Y-%m-%d"))

params = {
    "latitude": REGION_COORDS['latitude'].tolist(),
    "longitude": REGION_COORDS['longitude'].tolist(),
    "hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m"],
    "timezone": "Asia/Singapore",
    "wind_speed_unit": "kn",
    "start_date": CURRENT_DATE,
    "end_date": (CURRENT_DATETIME + timedelta(days=7)).strftime("%Y-%m-%d"),
}
responses = openmeteo.weather_api(url, params=params)

compiled_df = []
for response, region in zip(responses, REGION_COORDS['region'].tolist()):
    hourly = response.Hourly()

    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(2).ValuesAsNumpy()


    hourly_data = {
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "air_temperature_avg": hourly_temperature_2m,
        "wind_speed_avg": hourly_wind_speed_10m,
        "wind_direction_avg": hourly_wind_direction_10m,
        'reading_count': np.full(len(hourly_temperature_2m), 60), # placeholder for insertion
        "region": np.repeat(region, len(hourly_temperature_2m)),
        "latitude": np.repeat(REGION_COORDS.loc[REGION_COORDS['region'] == region, 'latitude'].values[0], len(hourly_temperature_2m)),
        "longitude": np.repeat(REGION_COORDS.loc[REGION_COORDS['region'] == region, 'longitude'].values[0], len(hourly_temperature_2m)),
    }

    hourly_dataframe = pd.DataFrame(hourly_data)
    compiled_df.append(hourly_dataframe)

compiled_df = pd.concat(compiled_df, ignore_index=True)

def aggregate_daily_scalar(df: pd.DataFrame, column: str) -> pd.DataFrame:

    # Daily aggregations
    df['date'] = df['timestamp'].dt.date
    daily_df = df.groupby(['region', 'date']).agg({
        column: ['mean', 'min', 'max', 'std'],
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

    daily_df.columns = ['region', 'date',
                        f'{column}_mean', f'{column}_min', f'{column}_max',
                        f'{column}_std',
                        'latitude', 'longitude']
    daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.drop('date', axis=1)

    return daily_df

def aggregate_daily_angular(df: pd.DataFrame) -> pd.DataFrame:

    # Daily aggregations with circular mean
    df['date'] = df['timestamp'].dt.date

    daily_data = []
    for (region, date), group in df.groupby(['region', 'date']):
        directions = group['wind_direction_avg'].values

        daily_data.append({
            'region': region,
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
    daily_df = daily_df.drop('date', axis=1)

    return daily_df


wind_speed_df = compiled_df[['timestamp', 'region', 'latitude', 'longitude', 'wind_speed_avg', 'reading_count']].copy()
wind_speed_daily_df = aggregate_daily_scalar(wind_speed_df.copy(), 'wind_speed_avg')

wind_direction_df = compiled_df[['timestamp', 'region', 'latitude', 'longitude', 'wind_direction_avg', 'reading_count']].copy()
wind_direction_df['wind_direction_std'] = circular_std(wind_direction_df['wind_direction_avg'].values)
wind_direction_daily_df = aggregate_daily_angular(wind_direction_df.copy())

air_temperature_df = compiled_df[['timestamp', 'region', 'latitude', 'longitude', 'air_temperature_avg', 'reading_count']].copy()
air_temperature_daily_df = aggregate_daily_scalar(air_temperature_df.copy(), 'air_temperature_avg')

fg_data = {
    "wind_speed_hourly": wind_speed_df, 
    "wind_speed_daily": wind_speed_daily_df, 
    "wind_direction_hourly": wind_direction_df, 
    "wind_direction_daily": wind_direction_daily_df, 
    "air_temperature_hourly": air_temperature_df, 
    "air_temperature_daily": air_temperature_daily_df
}

from utils.hopsworks import Hopsworks
hopsworks_client = Hopsworks()


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


fg_data["wind_speed_hourly"] = apply_func_to_groups(add_time_features(fg_data["wind_speed_hourly"], add_hour=True), 'region', regression_features_wind_speed)
fg_data["wind_speed_daily"] = apply_func_to_groups(add_time_features(fg_data["wind_speed_daily"], add_hour=False), 'region', regression_features_wind_speed_daily)
fg_data["wind_direction_hourly"] = apply_func_to_groups(add_time_features(fg_data["wind_direction_hourly"], add_hour=True), 'region', regression_features_wind_direction)
fg_data["wind_direction_daily"] = apply_func_to_groups(add_time_features(fg_data["wind_direction_daily"], add_hour=False), 'region', regression_features_wind_direction_daily)
fg_data["air_temperature_hourly"] = apply_func_to_groups(add_time_features(fg_data["air_temperature_hourly"], add_hour=True), 'region', regression_features_air_temperature)
fg_data["air_temperature_daily"] = apply_func_to_groups(add_time_features(fg_data["air_temperature_daily"], add_hour=False), 'region', regression_features_air_temperature_daily)


hopsworks_client.append_to_hopsworks(
    fg_data,
    version = 4
)

