
import os
import pandas as pd
import glob
from collections import defaultdict

# Configuration
BASE_DIR = './data'
DIRECTORIES = {
    '2023': f'{BASE_DIR}/wind-speed-2023',
    '2024': f'{BASE_DIR}/wind-speed-2024',
    '2025': f'{BASE_DIR}/wind-speed-2025'
}
OUTPUT_DIR = f'{BASE_DIR}/wind-speed'
DESIRED_COLUMNS = [
    'timestamp',
    'station_name',
    'wind_speed_avg',
    # 'wind_speed_std',
    'reading_count',
    'latitude', 'longitude',
    'last_update'
]

def get_station_name(file_path):
    """Extract station_name from CSV file."""
    try:
        df = pd.read_csv(file_path, nrows=1)
        return df['station_name'].iloc[0] if 'station_name' in df.columns else None
    except Exception as e:
        print(f"  Warning: Could not read {file_path}: {e}")
        return None

def load_and_process_df(file_path):
    """Load CSV and remove station_id if present."""
    df = pd.read_csv(file_path)
    if 'station_id' in df.columns:
        df = df.drop('station_id', axis=1)
    return df

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Group files by station_name across all directories
print("Grouping files by station...")
station_files = defaultdict(list)

for year_label, directory in DIRECTORIES.items():
    if not os.path.exists(directory):
        print(f"  Warning: {directory} not found, skipping")
        continue

    for file_path in glob.glob(f"{directory}/*.csv"):
        station_name = get_station_name(file_path)
        if station_name:
            station_files[station_name].append((year_label, file_path))
            print(f"  Added: {os.path.basename(file_path)} [{year_label}] -> {station_name}")

# Merge files for each station
print(f"\nMerging {len(station_files)} stations...")
year_order = {'2023': 0, '2024': 1, '2025': 2}

for station_name, files_list in station_files.items():
    print(f"\n  {station_name}: {len(files_list)} files")

    # Sort by year and load all dataframes
    files_list.sort(key=lambda x: year_order[x[0]])
    all_dfs = []

    for year_label, file_path in files_list:
        try:
            df = load_and_process_df(file_path)
            all_dfs.append(df)
            print(f"    {year_label}: {len(df)} rows")
        except Exception as e:
            print(f"    Error loading {file_path}: {e}")

    if not all_dfs:
        continue

    # Merge, sort, and deduplicate
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')

    # Reorder columns
    merged_df = merged_df[[col for col in DESIRED_COLUMNS if col in merged_df.columns]]

    # Save
    output_file = f"{OUTPUT_DIR}/{station_name.replace(' ', '_').replace('/', '_')}-wind-speed-hourly-merged.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"    ✓ Saved {len(merged_df)} rows ({merged_df['timestamp'].min()} to {merged_df['timestamp'].max()})")

print(f"\nComplete! Files saved to: {OUTPUT_DIR}")
