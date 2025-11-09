from datetime import datetime, timedelta

# ============================================================================
# DATE UTILITIES
# ============================================================================

def date_range_generator(start_date_str, end_date_str):
    """
    Generate dates from start to end, one day at a time

    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format

    Yields:
        Date strings in YYYY-MM-DD format
    """
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")

    current = start
    while current <= end:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)