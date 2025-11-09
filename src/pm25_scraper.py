"""
PM2.5 Data Scraper with Date Iteration and Checkpoint System
=============================================================
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import json
from utils.date import date_range_generator
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

API_ENDPOINT = "https://api-open.data.gov.sg/v2/real-time/api/pm25"  # Update this
START_DATE = "2023-07-01"
END_DATE = "2025-11-01"
REGIONS = ['east', 'west', 'north', 'south', 'central']
CHECKPOINT_DIR = "data/checkpoints"
SAVE_FREQUENCY = 10  # Save checkpoint every N days

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(items, last_date, checkpoint_file="checkpoint.json"):
    """Save progress checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)

    checkpoint = {
        'last_date': last_date,
        'items_count': len(items),
        'saved_at': datetime.now().isoformat()
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)

    # Save items
    items_file = checkpoint_path.replace('.json', '_items.json')
    with open(items_file, 'w') as f:
        json.dump(items, f)

    print(f"  ✓ Checkpoint: {len(items)} items | Last date: {last_date}")

def load_checkpoint(checkpoint_file="checkpoint.json"):
    """Load previous checkpoint if exists"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
    items_file = checkpoint_path.replace('.json', '_items.json')

    if os.path.exists(checkpoint_path) and os.path.exists(items_file):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        with open(items_file, 'r') as f:
            items = json.load(f)

        print(f"✓ Resumed from checkpoint:")
        print(f"  Last date: {checkpoint['last_date']}")
        print(f"  Items collected: {len(items)}")
        print(f"  Saved at: {checkpoint['saved_at']}")

        return items, checkpoint['last_date']

    return [], None

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
    Scrape PM2.5 data for a specific date with pagination handling

    Args:
        api_endpoint: The API endpoint URL
        date_str: Date in YYYY-MM-DD format

    Returns:
        List of items for that date
    """
    items = []
    pagination_token = None
    params = {'date': date_str}

    while True:
        if pagination_token:
            params['paginationToken'] = pagination_token

        try:
            response = requests.get(api_endpoint, params=params, timeout=30)
            # response.raise_for_status()

            data = response.json()

            if data.get('code', 0) != 0:
                error_msg = data.get('errorMsg', 'Unknown error')
                if 'Data not found' in error_msg or 'not found' in error_msg.lower():
                    # No data for this date, not an error
                    break
                print(f"    API Error: {error_msg}")
                break

            page_items = data.get('data', {}).get('items', [])
            items.extend(page_items)

            # Check for pagination
            pagination_token = data.get('data', {}).get('paginationToken')

            if not pagination_token:
                break

            # time.sleep(0.3)  # Rate limiting between pages

        except requests.exceptions.RequestException as e:
            print(f"    Request failed: {e}")
            raise

    return items

def scrape_pm25_data(api_endpoint, start_date, end_date, save_frequency=10, resume=True):
    """
    Scrape PM2.5 data by iterating through dates with checkpoint system

    Args:
        api_endpoint: The API endpoint URL
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        save_frequency: Save checkpoint every N days
        resume: Whether to resume from previous checkpoint

    Returns:
        List of all data items
    """
    # Try to resume from checkpoint
    if resume:
        all_items, last_date = load_checkpoint()
        if last_date:
            # Calculate next date to start from
            last_datetime = datetime.strptime(last_date, "%Y-%m-%d")
            next_datetime = last_datetime + timedelta(days=1)
            start_date = next_datetime.strftime("%Y-%m-%d")
    else:
        all_items = []
        last_date = None
        clear_checkpoints()

    print(f"\nScraping PM2.5 data from {start_date} to {end_date}")
    print(f"Checkpoint frequency: every {save_frequency} days")
    print("-" * 70)

    days_processed = 0

    try:
        for current_date in date_range_generator(start_date, end_date):
            # Scrape data for this date
            date_items = scrape_date(api_endpoint, current_date)

            if date_items:
                all_items.extend(date_items)
                print(f"{current_date}: {len(date_items):4d} items | Total: {len(all_items):6d}")
            else:
                print(f"{current_date}:    0 items | Total: {len(all_items):6d} (no data)")

            last_date = current_date
            days_processed += 1

            # Periodic checkpoint save
            if days_processed % save_frequency == 0:
                save_checkpoint(all_items, last_date)

            # Rate limiting between dates
            # time.sleep(0.5)

        # Final checkpoint save
        if all_items:
            save_checkpoint(all_items, last_date)

        print("-" * 70)
        print(f"✓ Collection complete: {len(all_items)} total items")

    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user")
        print(f"Saving checkpoint at {last_date}...")
        if all_items and last_date:
            save_checkpoint(all_items, last_date)
        print(f"Run again to resume from {last_date}")
        raise

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"Saving checkpoint at {last_date}...")
        if all_items and last_date:
            save_checkpoint(all_items, last_date)
        raise

    return all_items

# ============================================================================
# PROGRESSIVE PROCESSING AND SAVING
# ============================================================================

def process_and_save_by_region(items, output_prefix="pm25"):
    """
    Process API items and save separate CSV files for each region

    Args:
        items: List of API response items
        output_prefix: Prefix for output filenames

    Returns:
        Dictionary of dataframes by region
    """
    if not items:
        print("No items to process")
        return {}

    processed_data = []

    print(f"\nProcessing {len(items)} items...")

    for i, item in enumerate(items):
        timestamp = item.get('timestamp')
        date = item.get('date')
        updated = item.get('updatedTimestamp')

        readings = item.get('readings', {}).get('pm25_one_hourly', {})

        row = {
            'timestamp': timestamp,
            'date': date,
            'updated_timestamp': updated,
            'east': readings.get('east'),
            'west': readings.get('west'),
            'north': readings.get('north'),
            'south': readings.get('south'),
            'central': readings.get('central'),
            'national': readings.get('national')
        }
        processed_data.append(row)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(items)} items...")

    df = pd.DataFrame(processed_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\nSaving regional files...")
    region_dfs = {}

    for region in REGIONS:
        region_df = df[['timestamp', 'date', region]].copy()
        region_df = region_df.rename(columns={region: 'pm25'})
        region_df = region_df.dropna(subset=['pm25'])

        filename = f"{region}-pm2.5-hourly-230701-251101.csv"
        # save_filepath = os.path.join(CHECKPOINT_DIR, filename)
        region_df.to_csv(filename, index=False)

        # Calculate date range
        date_range = f"{region_df['date'].min().date()} to {region_df['date'].max().date()}"
        print(f"  ✓ {filename}")
        print(f"    {len(region_df)} records | {date_range}")

        region_dfs[region] = region_df

    return region_dfs


def create_multi_horizon_targets(df, horizons=[24, 48, 72, 96, 120, 144, 168]):
    """Create target variables for multiple forecast horizons (1-7 days ahead)"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Lag features
    for lag in [1, 3, 6, 12, 24, 48, 72, 168]:
        df[f'pm25_lag_{lag}h'] = df['pm25'].shift(lag)

    # Rolling statistics
    for window in [24, 72, 168]:
        df[f'pm25_rolling_mean_{window}h'] = df['pm25'].shift(1).rolling(window).mean()
        df[f'pm25_rolling_std_{window}h'] = df['pm25'].shift(1).rolling(window).std()

    # Multiple target columns
    for hours in horizons:
        days = hours // 24
        df[f'target_{days}d'] = df['pm25'].shift(-hours)

    df_clean = df.dropna()

    return df_clean

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PM2.5 Data Scraper with Date Iteration")
    print("=" * 70)

    try:
        # Scrape data (will resume from checkpoint if exists)
        print("\n[1/2] Scraping data from API...")
        print("  (Press Ctrl+C to pause - progress will be saved)")
        items = scrape_pm25_data(API_ENDPOINT, START_DATE, END_DATE, 
                                 save_frequency=SAVE_FREQUENCY, resume=True)

        # Process and save
        print("\n[2/2] Processing and saving by region...")
        region_dfs = process_and_save_by_region(items)

        print("\n" + "=" * 70)
        print("✓ COMPLETE! Files created:")
        for region in REGIONS:
            filename = f"{region}-pm2.5-hourly-230701-251101.csv"
            if os.path.exists(filename):
                size = os.path.getsize(filename) / 1024 / 1024  # MB
                print(f"  {filename} ({size:.2f} MB)")
        print("=" * 70)

        # Clean up checkpoints after successful completion
        clear_checkpoints()

    except KeyboardInterrupt:
        print("\n\n⚠ Scraping paused. Run again to resume from checkpoint.")
    except Exception as e:
        print(f"\n\n✗ Error occurred: {e}")
        print("Progress has been saved. Run again to resume.")
