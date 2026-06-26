import aiohttp
import certifi
import ssl
import json
from datetime import datetime, timedelta, timezone

async def get_jsonparsed_data(request_url, timeout=60):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context), timeout=timeout_obj) as session:
        async with session.get(request_url) as response:
            response.raise_for_status()
            text = await response.text()
            return json.loads(text)

def generate_intervals(start_date, end_date, interval_level='year', right_closed=False):
    """Generate time intervals between start_date and end_date.
    
    All returned datetime objects are guaranteed to be in UTC timezone.
    If input datetime objects don't have timezone info, they are treated as UTC.
    
    Args:
        start_date: Start datetime (will be converted to UTC if naive)
        end_date: End datetime (will be converted to UTC if naive)
        interval_level: 'year', 'month', or 'day'
        right_closed: If True, intervals are right-closed [start, end], 
                     otherwise left-closed [start, end)
    
    Returns:
        List of (start, end) datetime tuples, all in UTC timezone
    """
    intervals = []
    
    # Ensure start_date and end_date are in UTC timezone
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    else:
        start_date = start_date.astimezone(timezone.utc)
    
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    else:
        end_date = end_date.astimezone(timezone.utc)

    def right_endpoint(current, next):
        if right_closed:
            return next
        else:
            return next - timedelta(days=1)

    if interval_level == 'year':
        current_date = start_date
        while current_date < end_date:
            try:
                next_year = current_date.replace(year=current_date.year + 1)
            except ValueError:
                next_year = current_date.replace(month=3, day=1, year=current_date.year + 1)
            if next_year > end_date:
                next_year = end_date
            intervals.append((current_date, right_endpoint(current_date, next_year)))
            current_date = next_year
    elif interval_level == 'month':
        current_date = start_date
        while current_date < end_date:
            year, month = current_date.year, current_date.month
            if month == 12:
                next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
            if next_month > end_date:
                next_month = end_date
            intervals.append((current_date, right_endpoint(current_date, next_month)))
            current_date = next_month
    elif interval_level == 'day':
        current_date = start_date
        while current_date < end_date:
            next_day = current_date + timedelta(days=1)
            if next_day > end_date:
                next_day = end_date
            intervals.append((current_date, right_endpoint(current_date, next_day)))
            current_date = next_day
    else:
        return None

    return intervals