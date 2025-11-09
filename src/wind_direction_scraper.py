"""
Wind Direction Data Scraper with Circular Mean Aggregation
===========================================================
Scrapes per-minute wind direction data and aggregates to hourly circular means
"""
from utils.date import date_range_generator
from utils.circular import circular_mean, circular_std
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import time
import os
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

API_ENDPOINT = "https://api-open.data.gov.sg/v2/real-time/api/wind-direction"
START_DATE = "2025-01-01"
END_DATE = "2025-11-01"

# Directory configuration
DATA_DIR = "data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints", "wind-direction")
OUTPUT_DIR = os.path.join(DATA_DIR, "wind-direction")

SAVE_FREQUENCY = 10  # Save checkpoint every N days

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(items, stations_metadata, last_date, checkpoint_file="checkpoint.json"):
    """Save progress checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)

    checkpoint = {
        'last_date': last_date,
        'items_count': len(items),
        'stations_count': len(stations_metadata),
        'saved_at': datetime.now().isoformat()
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)

    # Save items
    items_file = checkpoint_path.replace('.json', '_items.json')
    with open(items_file, 'w') as f:
        json.dump(items, f)

    # Save stations metadata
    stations_file = checkpoint_path.replace('.json', '_stations.json')
    with open(stations_file, 'w') as f:
        json.dump(stations_metadata, f)

    print(f"  ✓ Checkpoint: {len(items)} readings | Last date: {last_date}")

def load_checkpoint(checkpoint_file="checkpoint.json"):
    """Load previous checkpoint if exists"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
    items_file = checkpoint_path.replace('.json', '_items.json')
    stations_file = checkpoint_path.replace('.json', '_stations.json')

    if os.path.exists(checkpoint_path) and os.path.exists(items_file):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        with open(items_file, 'r') as f:
            items = json.load(f)

        stations_metadata = {}
        if os.path.exists(stations_file):
            with open(stations_file, 'r') as f:
                stations_metadata = json.load(f)

        print(f"✓ Resumed from checkpoint:")
        print(f"  Last date: {checkpoint['last_date']}")
        print(f"  Readings collected: {len(items)}")
        print(f"  Stations: {len(stations_metadata)}")

        return items, stations_metadata, checkpoint['last_date']

    return [], {}, None

def clear_checkpoints():
    """Clear all checkpoint files"""
    if os.path.exists(CHECKPOINT_DIR):
        for file in os.listdir(CHECKPOINT_DIR):
            os.remove(os.path.join(CHECKPOINT_DIR, file))
        print("✓ Checkpoints cleared")

# ============================================================================
# DATA SCRAPING WITH DATE ITERATION
# ============================================================================

def scrape_date(api_endpoint, date_str):
    """
    Scrape wind direction data for a specific date with pagination handling

    Args:
        api_endpoint: The API endpoint URL
        date_str: Date in YYYY-MM-DD format

    Returns:
        Tuple of (readings_list, stations_dict)
    """
    readings = []
    stations_metadata = {}
    pagination_token = None
    params = {'date': date_str}

    while True:
        if pagination_token:
            params['paginationToken'] = pagination_token

        try:
            response = requests.get(api_endpoint, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('code', 0) != 0:
                error_msg = data.get('errorMsg', 'Unknown error')
                if 'Data not found' in error_msg or 'not found' in error_msg.lower():
                    break
                print(f"    API Error: {error_msg}")
                break

            response_data = data.get('data', {})

            # Extract and store station metadata (only once per station)
            stations = response_data.get('stations', [])
            for station in stations:
                station_id = station.get('id')
                if station_id and station_id not in stations_metadata:
                    stations_metadata[station_id] = {
                        'id': station_id,
                        'name': station.get('name'),
                        'device_id': station.get('deviceId'),
                        'latitude': station.get('location', {}).get('latitude'),
                        'longitude': station.get('location', {}).get('longitude')
                    }

            # Extract readings
            page_readings = response_data.get('readings', [])
            readings.extend(page_readings)

            # Check for pagination
            pagination_token = response_data.get('paginationToken')

            if not pagination_token:
                break

            time.sleep(0.3)  # Rate limiting between pages

        except requests.exceptions.RequestException as e:
            print(f"    Request failed: {e}")
            raise

    return readings, stations_metadata

def scrape_wind_direction_data(api_endpoint, start_date, end_date, save_frequency=10, resume=True):
    """
    Scrape wind direction data by iterating through dates with checkpoint system

    Args:
        api_endpoint: The API endpoint URL
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        save_frequency: Save checkpoint every N days
        resume: Whether to resume from previous checkpoint

    Returns:
        Tuple of (all_readings, stations_metadata)
    """
    # Try to resume from checkpoint
    if resume:
        all_readings, all_stations, last_date = load_checkpoint()
        if last_date:
            last_datetime = datetime.strptime(last_date, "%Y-%m-%d")
            next_datetime = last_datetime + timedelta(days=1)
            start_date = next_datetime.strftime("%Y-%m-%d")
    else:
        all_readings = []
        all_stations = {}
        last_date = None
        clear_checkpoints()

    print(f"\nScraping wind direction data from {start_date} to {end_date}")
    print(f"Checkpoint frequency: every {save_frequency} days")
    print("-" * 70)

    days_processed = 0

    try:
        for current_date in date_range_generator(start_date, end_date):
            # Scrape data for this date
            date_readings, date_stations = scrape_date(api_endpoint, current_date)

            # Update stations metadata
            all_stations.update(date_stations)

            if date_readings:
                all_readings.extend(date_readings)
                print(f"{current_date}: {len(date_readings):5d} readings | Total: {len(all_readings):7d} | Stations: {len(all_stations)}")
            else:
                print(f"{current_date}:     0 readings | Total: {len(all_readings):7d} (no data)")

            last_date = current_date
            days_processed += 1

            # Periodic checkpoint save
            if days_processed % save_frequency == 0:
                save_checkpoint(all_readings, all_stations, last_date)

            # Rate limiting between dates
            time.sleep(0.5)

        # Final checkpoint save
        if all_readings:
            save_checkpoint(all_readings, all_stations, last_date)

        print("-" * 70)
        print(f"✓ Collection complete: {len(all_readings)} total readings from {len(all_stations)} stations")

    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user")
        print(f"Saving checkpoint at {last_date}...")
        if all_readings and last_date:
            save_checkpoint(all_readings, all_stations, last_date)
        print(f"Run again to resume from {last_date}")
        raise

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"Saving checkpoint at {last_date}...")
        if all_readings and last_date:
            save_checkpoint(all_readings, all_stations, last_date)
        raise

    return all_readings, all_stations

# ============================================================================
# DATA PROCESSING WITH CIRCULAR MEAN AGGREGATION
# ============================================================================

def aggregate_to_hourly(readings, stations_metadata, output_dir=OUTPUT_DIR):
    """
    Process minute-level readings and aggregate to hourly circular means per station

    Args:
        readings: List of reading objects from API
        stations_metadata: Dictionary of station metadata
        output_dir: Directory to save CSV files

    Returns:
        Dictionary of dataframes by station
    """
    if not readings:
        print("No readings to process")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {len(readings)} minute-level readings...")

    # Flatten the nested structure
    flattened_data = []

    for reading in readings:
        timestamp = reading.get('timestamp')
        data_points = reading.get('data', [])

        for point in data_points:
            flattened_data.append({
                'timestamp': timestamp,
                'station_id': point.get('stationId'),
                'wind_direction': point.get('value')
            })

    print(f"  Flattened to {len(flattened_data)} data points")

    # Create dataframe
    df = pd.DataFrame(flattened_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['wind_direction'] = pd.to_numeric(df['wind_direction'], errors='coerce')

    # Remove null values
    df = df.dropna(subset=['wind_direction'])

    # Validate direction values (should be 0-360)
    initial_count = len(df)
    df = df[(df['wind_direction'] >= 0) & (df['wind_direction'] <= 360)]
    invalid_count = initial_count - len(df)
    if invalid_count > 0:
        print(f"  Removed {invalid_count:,} readings outside 0-360° range")

    print(f"  Processing {len(df)} valid data points")

    # Create hour column for aggregation
    df['hour'] = df['timestamp'].dt.floor('H')

    # Aggregate to hourly circular means by station
    print(f"\nAggregating to hourly circular means...")
    print(f"  (This uses circular statistics to handle 0°/360° wraparound)")

    hourly_data = []
    grouped = df.groupby(['station_id', 'hour'])

    total_groups = len(grouped)
    for i, ((station_id, hour), group) in enumerate(grouped, 1):
        if i % 1000 == 0:
            print(f"    Processed {i:,}/{total_groups:,} station-hour combinations...")

        directions = group['wind_direction'].values
        circ_mean = circular_mean(directions)
        circ_std_val = circular_std(directions)

        hourly_data.append({
            'timestamp': hour,
            'station_id': station_id,
            'wind_direction_avg': circ_mean,
            'wind_direction_std': circ_std_val,
            'reading_count': len(directions)
        })

    hourly_df = pd.DataFrame(hourly_data)

    print(f"  ✓ Aggregated to {len(hourly_df):,} hourly records")

    # Save separate file for each station
    print(f"\nSaving station files to: {os.path.abspath(output_dir)}")
    station_dfs = {}

    stations_processed = 0
    for station_id in sorted(hourly_df['station_id'].unique()):
        station_df = hourly_df[hourly_df['station_id'] == station_id].copy()
        station_df = station_df.sort_values('timestamp').reset_index(drop=True)

        # Add station metadata
        station_info = stations_metadata.get(station_id, {})
        station_df['station_name'] = station_info.get('name', 'Unknown')
        station_df['latitude'] = station_info.get('latitude')
        station_df['longitude'] = station_info.get('longitude')

        # Reorder columns
        station_df = station_df[['timestamp', 'station_id', 'station_name', 
                                 'wind_direction_avg', 'wind_direction_std', 'reading_count',
                                 'latitude', 'longitude']]

        # Save to CSV
        filename = os.path.join(output_dir, f"{station_id}-wind-direction-hourly-230701-251101.csv")
        station_df.to_csv(filename, index=False)

        # Calculate date range
        date_range = f"{station_df['timestamp'].min().date()} to {station_df['timestamp'].max().date()}"
        station_name = station_info.get('name', 'Unknown')

        print(f"  ✓ {os.path.basename(filename)}")
        print(f"    {station_name} | {len(station_df)} records | {date_range}")

        station_dfs[station_id] = station_df
        stations_processed += 1

    print(f"\n✓ Processed {stations_processed} stations")

    # Also save a stations metadata file
    stations_file = os.path.join(output_dir, "stations_metadata.csv")
    stations_df = pd.DataFrame(stations_metadata.values())
    stations_df.to_csv(stations_file, index=False)
    print(f"\n✓ Saved station metadata: {stations_file}")

    print(f"\nNote: wind_direction_std is circular standard deviation")
    print(f"  • Low values = consistent direction")
    print(f"  • High values = variable direction")

    return station_dfs

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Wind Direction Data Scraper with Circular Mean Aggregation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {os.path.abspath(DATA_DIR)}")
    print(f"  Checkpoints: {os.path.abspath(CHECKPOINT_DIR)}")
    print(f"  Output CSVs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"\nData will be aggregated using CIRCULAR MEAN (handles 0°/360° wraparound)")

    try:
        # Scrape data (will resume from checkpoint if exists)
        print("\n[1/2] Scraping per-minute data from API...")
        print("  (Press Ctrl+C to pause - progress will be saved)")
        readings, stations = scrape_wind_direction_data(API_ENDPOINT, START_DATE, END_DATE, 
                                                         save_frequency=SAVE_FREQUENCY, resume=True)

        # Process and aggregate to hourly with circular mean
        print("\n[2/2] Aggregating to hourly circular means and saving by station...")
        station_dfs = aggregate_to_hourly(readings, stations)

        print("\n" + "=" * 70)
        print("✓ COMPLETE!")
        print("=" * 70)

        # Clean up checkpoints after successful completion
        clear_checkpoints()

    except KeyboardInterrupt:
        print("\n\n⚠ Scraping paused. Run again to resume from checkpoint.")
    except Exception as e:
        print(f"\n\n✗ Error occurred: {e}")
        print("Progress has been saved. Run again to resume.")
