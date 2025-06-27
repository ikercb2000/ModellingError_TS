# Package Modules

from data_apis.interfaces import *
from data_apis.enums import *

# Other Modules

import os
import re
import time
from datetime import timedelta
import requests


# Timestamp functions


def get_timestamp_8601(year: int, month: int, day: int, hour: int, minutes: int, seconds: int, tz_sign: str = "+", tz_hour: int = 0) -> str:

    list_times = [month, day, hour, minutes, seconds, tz_hour]
    list_new = []

    for i in list_times:
        if i >= 10:
            list_new.append(f"{i}")
        else:
            list_new.append(f"0{i}")

    if tz_sign != "+" and tz_sign != "-":
        raise ValueError("Choose between '+' and '-'")

    return f"{year}-{list_new[0]}-{list_new[1]}T{list_new[2]}:{list_new[3]}:{list_new[4]}{tz_sign}{list_new[5]}:00"

# Create directory function


def create_directory(directory: str):

    if directory is None:
        raise ValueError("Please specify a directory to save the data.")
    else:
        user_dir = os.path.abspath(directory)

    folder_dir = os.path.join(user_dir, "data")

    if not os.path.isdir(folder_dir):

        os.makedirs(folder_dir)

        if os.path.isdir(folder_dir):
            print(f"{directory} directory created.")

        else:
            raise LookupError("Folder not created. Check")

# Frequency conversion function


def parse_freq(time_str: str):

    match = re.match(r"(\d+)([smhd])", time_str.strip().lower())

    if not match:
        raise ValueError("Invalid format. Use '1s', '5m', '2h', '3d', etc.")

    value, unit = int(match.group(1)), match.group(2)

    time_mapping = {
        's': timedelta(seconds=value),
        'm': timedelta(minutes=value),
        'h': timedelta(hours=value),
        'd': timedelta(days=value)
    }

    return time_mapping[unit]

# Rate Limits Handling Function


def rate_limited_request(request_func, *args, **kwargs):
    """Wrapper to handle rate limits by adding a delay."""
    while True:
        response = request_func(*args, **kwargs)

        if isinstance(response, requests.Response):
            used_weight = int(response.headers.get("X-MBX-USED-WEIGHT-1M", 0))
            if used_weight > 1100:
                print("Rate limit approaching, sleeping for 60 seconds...")
                time.sleep(60)
                continue

        return response
