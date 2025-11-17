import os
from datetime import datetime, timedelta
from zipfile import Path
import dotenv
import pandas as pd
from xgboost import XGBRegressor

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from xgboost import plot_importance
from matplotlib.patches import Patch

import hopsworks
from hsfs.feature import Feature
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def retrieve_feature_stores(fs):
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

    print("pm_25_daily_fg:", "Loaded" if pm25_daily_fg is not None else "Not Loaded")
    print("wind_direction_daily_fg:", "Loaded" if wind_direction_daily_fg is not None else "Not Loaded")
    print("wind_speed_daily_fg:", "Loaded" if wind_speed_daily_fg is not None else "Not Loaded")
    print("air_temperature_daily_fg:", "Loaded" if air_temperature_daily_fg is not None else "Not Loaded")
    return pm25_daily_fg, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg

def get_hopsworks_project():
    dotenv.load_dotenv()
    project = hopsworks.login(engine="python", project="akeelaf")
    fs = project.get_feature_store()
    return fs

def create_feature_view(fs, pm25_daily_fg, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg, REGION="west"):
    selected_features = pm25_daily_fg.select_all().filter(pm25_daily_fg.region == REGION).join(
        wind_direction_daily_fg.select_all(), on=["region", "timestamp"]
    ).join(
        wind_speed_daily_fg.select_all(), on=["region", "timestamp"]
    ).join(
        air_temperature_daily_fg.select_all(), on=["region", "timestamp"]
    )
    
    feature_view = fs.get_or_create_feature_view(
        name="air_quality_wind_temperature_features_daily_new",
        version=3,
        description="Feature view with air quality and weather features (wind speed/direction and temperature)",
        labels=['pm25'],
        query=selected_features
    )

    return feature_view

def split_data(feature_view, START_DATE):
    test_start = datetime.strptime(START_DATE, "%Y-%m-%d")
    X_train, X_test, y_train, y_test = feature_view.train_test_split(
        test_start=test_start
    )
    print("Test data starts from:", test_start.date())

    X_features = X_train.drop(columns=['timestamp'])
    X_test_features = X_test.drop(columns=['timestamp'])
    X_features.dropna(inplace=True)
    mask = X_features.dropna().index
    X_features = X_features.loc[mask].reset_index(drop=True)
    y_train = y_train.loc[mask].reset_index(drop=True)

    X_features = X_features.select_dtypes(include=["number"]).copy()
    X_test_features = X_test_features.select_dtypes(include=["number"]).copy()

    return X_features, y_train, X_test_features, y_test, X_test

def calculate_metrics(y_pred, y_test):
    # Calculating Mean Squared Error (MSE) using sklearn
    mse = mean_squared_error(y_test.iloc[:,0], y_pred)

    # Calculating R squared using sklearn
    r2 = r2_score(y_test.iloc[:,0], y_pred)
    return mse, r2

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

def train(X_features, X_test_features, y_train, y_test, X_test, n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, tree_method, eval_metric, random_state):
    xgb_regressor = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth, 
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method=tree_method,
        eval_metric=eval_metric,
        random_state=random_state,
    )

    print("Training using XGBRegressor model with params: ")
    print("Learning rate: ", learning_rate)
    print("Max depth: ", max_depth)
    print("Estimators: ", n_estimators)

    # Fitting the XGBoost Regressor to the training data
    xgb_regressor.fit(X_features, y_train)
    y_pred = xgb_regressor.predict(X_test_features)

    # Calculating metrics
    mse, r2 = calculate_metrics(y_pred, y_test)
    print("MSE:", mse)
    print("R squared:", r2)

    df = y_test
    df['predicted_pm25'] = y_pred

    df['timestamp'] = X_test['timestamp']
    df = df.sort_values(by=['timestamp'])
    df.head(15)

    return xgb_regressor, df, mse, r2

def plot(model, REGION, df):
    # Creating a directory for the model artifacts if it doesn't exist
    root_dir = Path().absolute()
    images_dir = os.path.join(root_dir, "docs", "outputs")
    
    file_path = images_dir + "/pm25_train_test_hindcast.png"
    plt = plot_air_quality_forecast('Singapore', REGION, df, file_path, hindcast=True) 

    # Plotting feature importances using the plot_importance function from XGBoost
    plot_importance(model.get_booster(), max_num_features=15)
    feature_importance_path = images_dir + "/feature_importance.png"
    plt.savefig(feature_importance_path)


def save(model, mse, r2, feature_view, REGION):
    project = hopsworks.login(engine="python", project="akeelaf")
    model_dir = "air_quality_model"
    model.save_model(model_dir + "/model.json")
    res_dict = { 
        "MSE": str(mse),
        "R squared": str(r2),
    }
    mr = project.get_model_registry()
    aq_model = mr.python.create_model(
        name=f"air_quality_xgboost_model_{REGION}", 
        metrics= res_dict,
        feature_view=feature_view,
        description="Air Quality (PM2.5) predictor for region " + REGION,
    )

    # Saving the model artifacts to the 'air_quality_model' directory in the model registry
    aq_model.save(model_dir)


def main():
    START_DATE = "2025-10-01"

    fs = get_hopsworks_project()

    # Creating an instance of the XGBoost Regressor
    n_estimators = 3000
    learning_rate = 0.02
    max_depth = 5
    min_child_weight = 4
    subsample=0.8
    colsample_bytree=0.8
    reg_alpha=0.5
    reg_lambda=1.5
    tree_method="hist"
    eval_metric="mae"
    random_state=42

    REGIONS = ["west", "east", "central", "north", "south"]
    for REGION in REGIONS:
        print(f"Training model for region: {REGION}")
        pm25_daily_fg, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg = retrieve_feature_stores(fs)
        feature_view = create_feature_view(fs, pm25_daily_fg, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg, REGION=REGION)
        print("Feature view created:", feature_view.name)
        X_features, y_train, X_test_features, y_test, X_test = split_data(feature_view, START_DATE)
        model, df, mse, r2 = train(X_features, X_test_features, y_train, y_test, X_test, n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, tree_method, eval_metric, random_state)
        plot(model, REGION, df)
        save(model, mse, r2, feature_view, REGION)


if __name__ == "__main__":
    main()
