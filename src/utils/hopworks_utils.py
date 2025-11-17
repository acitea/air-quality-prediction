def retrieve_feature_stores(fs):
    pm25_daily_fg = fs.get_feature_group(
        name="pm25_daily",
        version=2
    )

    wind_direction_daily_fg = fs.get_feature_group(
        name="wind_direction_daily",
        version=3
    )

    wind_speed_daily_fg = fs.get_feature_group(
        name="wind_speed_daily",
        version=3
    )

    air_temperature_daily_fg = fs.get_feature_group(
        name="air_temperature_daily",
        version=3
    )

    print("pm_25_daily_fg:", "Loaded" if pm25_daily_fg is not None else "Not Loaded")
    print("wind_direction_daily_fg:", "Loaded" if wind_direction_daily_fg is not None else "Not Loaded")
    print("wind_speed_daily_fg:", "Loaded" if wind_speed_daily_fg is not None else "Not Loaded")
    print("air_temperature_daily_fg:", "Loaded" if air_temperature_daily_fg is not None else "Not Loaded")
    return pm25_daily_fg, wind_direction_daily_fg, wind_speed_daily_fg, air_temperature_daily_fg