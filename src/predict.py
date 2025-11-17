import datetime
import pandas as pd
from xgboost import XGBRegressor
import hopsworks
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
import os
root_dir = Path().absolute()
images_dir = os.path.join(root_dir, "docs", "outputs")
from datetime import timedelta
import numpy as np

def retrieve_model(REGION, project):
    mr = project.get_model_registry()

    retrieved_model = mr.get_model(
        name="air_quality_xgboost_model_" + REGION,
        version=1,
    )

    fv = retrieved_model.get_feature_view()
    saved_model_dir = retrieved_model.download()
    retrieved_xgboost_model = XGBRegressor()
    retrieved_xgboost_model.load_model(saved_model_dir + "/model.json")
    return retrieved_xgboost_model

def retrieve_feature_groups(fs):
    pm25_daily_fg = fs.get_feature_group(
        name="pm25_daily",
        version=4
    )
    wind_direction_daily_fg = fs.get_feature_group(
        name="wind_direction_daily",
        version=4
    )

    wind_speed_daily_fg = fs.get_feature_group(
        name="wind_speed_daily",
        version=4
    )

    air_temperature_daily_fg = fs.get_feature_group(
        name="air_temperature_daily",
        version=4
    )
    print("pm25_daily_fg:", "Loaded" if pm25_daily_fg is not None else "Not Loaded")
    print("wind_direction_daily_fg:", "Loaded" if wind_direction_daily_fg is not None else "Not Loaded")
    print("wind_speed_daily_fg:", "Loaded" if wind_speed_daily_fg is not None else "Not Loaded")
    print("air_temperature_daily_fg:", "Loaded" if air_temperature_daily_fg is not None else "Not Loaded")
    return pm25_daily_fg, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg


def retrieve_weather_forecast(REGION, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg, today):
    base_query = wind_direction_daily_fg.select_all().filter(wind_direction_daily_fg.timestamp >= today).filter(wind_direction_daily_fg.region == REGION)
    joined_query = (
        base_query
        .join(wind_speed_daily_fg.select_all(), on=["timestamp", "region"])
        .join(air_temperature_daily_fg.select_all(), on=["timestamp", "region"])
    )
    future_weather = joined_query.read()
    future_weather["timestamp"] = pd.to_datetime(future_weather["timestamp"])
    future_weather = future_weather.sort_values("timestamp").reset_index(drop=True)

    return future_weather.head(7)

def get_pm25_history(REGION, pm25_daily_fg, today):
    hist_pm25 = (
        pm25_daily_fg
        .select(["region", "timestamp", "pm25"])
        .filter(pm25_daily_fg.region == REGION)
        .filter(pm25_daily_fg.timestamp < today)
        .read()
    )

    hist_pm25["timestamp"] = pd.to_datetime(hist_pm25["timestamp"])
    hist_pm25 = hist_pm25.sort_values("timestamp").reset_index(drop=True)
    return hist_pm25


def forecast_next_n_days(model, hist_pm25, future_weather, n_days=7):
    """
    Recursive multi-step forecast:
    - hist_pm25: DataFrame with columns ['region','timestamp','pm25'] for past days
    - future_weather: DataFrame with 7 future rows (region, timestamp, weather features)
    - Returns forecast_df with ['region','timestamp','predicted_pm25']
    """
    booster = model.get_booster()
    model_features = booster.feature_names

    work_hist = hist_pm25.copy()

    forecasts = []

    lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 28]
    rolling_windows = [3, 7, 14, 28]

    for i in range(min(n_days, len(future_weather))):
        row_weather = future_weather.loc[[i]].copy()
        ts = row_weather["timestamp"].iloc[0]

        for lag in lag_days:
            lag_ts = ts - timedelta(days=lag)
            match = work_hist[work_hist["timestamp"] == lag_ts]
            if len(match) == 0:
                val = np.nan
            else:
                val = match["pm25"].iloc[-1]
            row_weather[f"pm25_lag_{lag}d"] = val

        for window in rolling_windows:
            start_ts = ts - timedelta(days=window)
            mask = (work_hist["timestamp"] > start_ts) & (work_hist["timestamp"] <= ts - timedelta(days=1))
            window_vals = work_hist.loc[mask, "pm25"]

            row_weather[f"pm25_rolling_mean_{window}d"] = window_vals.mean() if not window_vals.empty else np.nan
            row_weather[f"pm25_rolling_min_{window}d"]  = window_vals.min()  if not window_vals.empty else np.nan
            row_weather[f"pm25_rolling_max_{window}d"]  = window_vals.max()  if not window_vals.empty else np.nan
            row_weather[f"pm25_rolling_std_{window}d"]  = window_vals.std()  if not window_vals.empty else np.nan

        ts_dt = pd.to_datetime(ts)
        row_weather["day_of_week"] = ts_dt.dayofweek
        row_weather["day_of_month"] = ts_dt.day
        row_weather["month"] = ts_dt.month
        row_weather["year"] = ts_dt.year
        row_weather["is_weekend"] = ts_dt.dayofweek >= 5

        for f in model_features:
            if f not in row_weather.columns:
                row_weather[f] = np.nan

        X = row_weather[model_features].astype(float)

        y_hat = model.predict(X)[0]

        forecasts.append({
            "region": row_weather["region"].iloc[0],
            "timestamp": ts,
            "days_before_forecast_day": i + 1,
            "predicted_pm25": y_hat,
        })

        work_hist = pd.concat([
            work_hist,
            pd.DataFrame([{"region": row_weather["region"].iloc[0], "timestamp": ts, "pm25": y_hat}])
        ], ignore_index=True)

    forecast_df = pd.DataFrame(forecasts).sort_values("timestamp").reset_index(drop=True)
    return forecast_df

def get_weekly_pm25_forecast(model, hist_pm25, future_weather, pm25_daily_fg, REGION, today):
    hist_pm25 = (
        pm25_daily_fg
        .select(["region", "timestamp", "pm25"])
        .filter(pm25_daily_fg.region == REGION)
        .filter(pm25_daily_fg.timestamp < today)
        .read()
    )

    hist_pm25["timestamp"] = pd.to_datetime(hist_pm25["timestamp"])
    hist_pm25 = hist_pm25.sort_values("timestamp").reset_index(drop=True)

    future_weather["timestamp"] = pd.to_datetime(future_weather["timestamp"])
    future_weather = future_weather.sort_values("timestamp").reset_index(drop=True)
    future_weather = future_weather.head(7)

    forecast_7d = forecast_next_n_days(
        model=model,
        hist_pm25=hist_pm25,
        future_weather=future_weather,
        n_days=7,
    )
    return forecast_7d

def plot_air_quality_forecast(city: str, street: str, df: pd.DataFrame, file_path: str, hindcast=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    day = pd.to_datetime(df['timestamp']).dt.date
    # Plot each column separately in matplotlib
    ax.plot(day, df['predicted_pm25'], label='Predicted PM2.5', color='red', linewidth=2, marker='o', markersize=5, markerfacecolor='blue')

    # Set the y-axis to a logarithmic scale
    ax.set_yscale('log')
    ax.set_yticks([0, 10, 25, 50, 100, 250, 500])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=1)

    # Set the labels and title
    ax.set_xlabel('Date')
    ax.set_title(f"PM2.5 Predicted (Logarithmic Scale) for {city}, {street}")
    ax.set_ylabel('PM2.5')

    colors = ['green', 'yellow', 'orange', 'red', 'purple', 'darkred']
    labels = ['Good', 'Moderate', 'Unhealthy for Some', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    ranges = [(0, 49), (50, 99), (100, 149), (150, 199), (200, 299), (300, 500)]
    for color, (start, end) in zip(colors, ranges):
        ax.axhspan(start, end, color=color, alpha=0.3)

    # Add a legend for the different Air Quality Categories
    patches = [Patch(color=colors[i], label=f"{labels[i]}: {ranges[i][0]}-{ranges[i][1]}") for i in range(len(colors))]
    legend1 = ax.legend(handles=patches, loc='upper right', title="Air Quality Categories", fontsize='x-small')

    # Aim for ~10 annotated values on x-axis, will work for both forecasts ans hindcasts
    if len(df.index) > 11:
        every_x_tick = len(df.index) / 10
        ax.xaxis.set_major_locator(MultipleLocator(every_x_tick))

    plt.xticks(rotation=45)

    if hindcast == True:
        ax.plot(day, df['pm25'], label='Actual PM2.5', color='black', linewidth=2, marker='^', markersize=5, markerfacecolor='grey')
        legend2 = ax.legend(loc='upper left', fontsize='x-small')
        ax.add_artist(legend1)

    # Ensure everything is laid out neatly
    plt.tight_layout()

    # # Save the figure, overwriting any existing file with the same name
    plt.savefig(file_path)
    return plt

def main():
    REGIONS = ["west", "east", "central", "north", "south"]

    today = datetime.datetime.now() - datetime.timedelta(1)
    print("Making predictions from date:", today.strftime("%Y-%m-%d"))

    project = hopsworks.login(project='akeelaf')
    fs = project.get_feature_store() 

    for REGION in REGIONS:
        model = retrieve_model(REGION, project)
        pm25_daily_fg, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg = retrieve_feature_groups(fs)
        forecast = retrieve_weather_forecast(REGION, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg, today)
        hist_pm25 = get_pm25_history(REGION, pm25_daily_fg, today)
        weekly_forecast = get_weekly_pm25_forecast(model, hist_pm25, forecast, pm25_daily_fg, REGION, today)

        print(f"7-day PM2.5 forecast for region {REGION}:")

        os.makedirs(images_dir, exist_ok=True)
        pred_file_path = f"{images_dir}/pm25_forecast_{REGION}.png"
        plt = plot_air_quality_forecast('Singapore', REGION, weekly_forecast, pred_file_path)
        # plt.show()

        # Get or create feature group
        monitor_fg = fs.get_or_create_feature_group(
            name='aq_predictions',
            description='Air Quality prediction monitoring',
            version=9,
            primary_key=['region','timestamp','days_before_forecast_day'],
            event_time="timestamp"
        )
        monitor_fg.insert(weekly_forecast, wait=True)

        monitoring_df = monitor_fg.read()
        pm25_daily_df = pm25_daily_fg.read()
        outcome_df = pm25_daily_df[['timestamp', 'pm25', 'region']]
        preds_df =  monitoring_df[['timestamp', 'predicted_pm25', 'region']]

        hindcast_df = pd.merge(preds_df, outcome_df, on=["timestamp", "region"])
        hindcast_df = hindcast_df.sort_values(by=['timestamp'])
        hindcast_df = hindcast_df[hindcast_df['region'] == REGION]
        
        hindcast_path = f"{images_dir}/pm25_hindcast_{REGION}_1day.png"
        plt = plot_air_quality_forecast('Singapore', REGION, hindcast_df, hindcast_path, hindcast=True)
        # plt.show()

        dataset_api = project.get_dataset_api()
        str_today = today.strftime("%Y-%m-%d")
        if dataset_api.exists("Resources/airquality") == False:
            dataset_api.mkdir("Resources/airquality")
        dataset_api.upload(pred_file_path, f"Resources/airquality/singapore_{REGION}_{str_today}", overwrite=True)
        dataset_api.upload(hindcast_path, f"Resources/airquality/singapore_{REGION}_{str_today}", overwrite=True)

        proj_url = project.get_url()
        print(f"See images in Hopsworks here: {proj_url}/settings/fb/path/Resources/airquality")

if __name__ == "__main__":
    main()