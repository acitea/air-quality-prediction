"""
Wind Speed CSV Processor - Hourly Aggregation
==============================================
Processes compiled yearly CSV with wind speed data and aggregates to hourly means
"""

import pandas as pd
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output paths
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "wind-speed-raw/HistoricalWindSpeedacrossSingapore2024.csv")  # Your CSV file
OUTPUT_DIR = os.path.join(DATA_DIR, "wind-speed-2024")

# Columns to drop
COLUMNS_TO_DROP = [
    'reading_type',
    'reading_unit',
    'station_id',
    'station_device_id',
    'update_timestamp'
]

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_wind_speed_csv(input_file, output_dir=OUTPUT_DIR):
    """
    Process wind speed CSV and aggregate to hourly means

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed files

    Returns:
        Dictionary of dataframes by station
    """
    print("=" * 70)
    print("Wind Speed CSV Processor")
    print("=" * 70)

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"\nReading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  ✓ Columns: {list(df.columns)}")

    # Drop unnecessary columns
    print(f"\nDropping unnecessary columns...")
    columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    print(f"  ✓ Dropped: {columns_to_drop}")
    print(f"  ✓ Remaining columns: {list(df.columns)}")

    # Convert timestamp to datetime
    print(f"\nProcessing timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])
    if 'reading_update_timestamp' in df.columns:
        df['reading_update_timestamp'] = pd.to_datetime(df['reading_update_timestamp'])

    # Convert reading_value to numeric
    df['reading_value'] = pd.to_numeric(df['reading_value'], errors='coerce')

    # Remove null values
    initial_count = len(df)
    df = df.dropna(subset=['reading_value'])
    dropped_nulls = initial_count - len(df)
    if dropped_nulls > 0:
        print(f"  ✓ Removed {dropped_nulls:,} rows with null reading values")

    print(f"  ✓ Processing {len(df):,} valid readings")

    # Create hour column for aggregation
    print(f"\nAggregating to hourly means...")
    df['hour'] = df['timestamp'].dt.floor('H')

    # Group by station and hour, calculate mean
    hourly_df = df.groupby(['station_name', 'hour']).agg({
        'reading_value': 'mean',
        'timestamp': 'count',  # Count of readings per hour
        'location_latitude': 'first',
        'location_longitude': 'first',
        'reading_update_timestamp': 'max' if 'reading_update_timestamp' in df.columns else 'first'
    }).reset_index()

    # Rename columns
    hourly_df = hourly_df.rename(columns={
        'hour': 'timestamp',
        'reading_value': 'wind_speed_avg',
        'timestamp': 'reading_count',
        'location_latitude': 'latitude',
        'location_longitude': 'longitude',
        'reading_update_timestamp': 'last_update'
    })

    print(f"  ✓ Aggregated to {len(hourly_df):,} hourly records")
    print(f"  ✓ Stations found: {hourly_df['station_name'].nunique()}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save separate file for each station
    print(f"\nSaving station files to: {os.path.abspath(output_dir)}")
    station_dfs = {}

    for station_name in sorted(hourly_df['station_name'].unique()):
        station_df = hourly_df[hourly_df['station_name'] == station_name].copy()
        station_df = station_df.sort_values('timestamp').reset_index(drop=True)

        # Create safe filename (replace spaces and special chars)
        safe_name = station_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = os.path.join(output_dir, f"{safe_name}-wind-speed-hourly.csv")

        station_df.to_csv(filename, index=False)

        # Calculate date range
        date_range = f"{station_df['timestamp'].min().date()} to {station_df['timestamp'].max().date()}"

        print(f"  ✓ {os.path.basename(filename)}")
        print(f"    {len(station_df):,} records | {date_range}")

        station_dfs[station_name] = station_df

    # Save a summary/metadata file
    summary_file = os.path.join(output_dir, "stations_summary.csv")
    summary_df = hourly_df.groupby('station_name').agg({
        'timestamp': ['min', 'max', 'count'],
        'latitude': 'first',
        'longitude': 'first',
        'wind_speed_avg': ['mean', 'min', 'max']
    }).reset_index()

    summary_df.columns = ['station_name', 'first_timestamp', 'last_timestamp', 
                          'record_count', 'latitude', 'longitude',
                          'avg_wind_speed', 'min_wind_speed', 'max_wind_speed']

    summary_df.to_csv(summary_file, index=False)
    print(f"\n  ✓ Saved summary: {os.path.basename(summary_file)}")

    print("\n" + "=" * 70)
    print("✓ Processing complete!")
    print("=" * 70)

    return station_dfs


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    # Allow custom input file as command line argument
    input_file = INPUT_FILE
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    print(f"Input file: {input_file}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    try:
        station_dfs = process_wind_speed_csv(input_file, OUTPUT_DIR)

        print(f"\nProcessed {len(station_dfs)} stations:")
        for station_name, df in station_dfs.items():
            print(f"  • {station_name}: {len(df):,} hourly records")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nPlease ensure your CSV file is at: {INPUT_FILE}")
        print(f"Or provide the path as argument: python wind_speed_processor.py /path/to/file.csv")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
