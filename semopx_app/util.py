import re
from pathlib import Path
from typing import List

import pandas as pd


def date_to_str(date, format="%Y%m%d") -> str:
    """
    Convert a date to a formatted string.

    Args:
        date: Date object, string, or Timestamp to convert
        format: strftime format string (default: "%Y%m%d")

    Returns:
        Formatted date string
    """
    return pd.Timestamp(str(date)).strftime(format)


class DayDelta:
    """
    Utility class for adding/subtracting days from dates.

    Supports arithmetic operations with dates, strings, and integers.
    Automatically preserves the type of the input (string format, int, or Timestamp).
    """

    def __init__(self, n: int = 1):
        """
        Initialize DayDelta.

        Args:
            n: Number of days to add (positive) or subtract (negative)
        """
        self.n = n

    def __add__(self, other):
        result = pd.Timestamp(str(other)) + pd.Timedelta(self.n, "Day")
        if isinstance(other, str):
            return date_to_str(
                result, format="%Y%m%d" if len(other) == 8 else "%Y-%m-%d"
            )
        elif isinstance(other, int):
            return int(date_to_str(result))
        else:
            return result

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return DayDelta(-self.n) + other

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        return DayDelta(other * self.n)

    def __rmul__(self, other):
        return self * other


def convert_timestamps(timestrings):
    """
    Convert timestamp strings to pandas datetime with UTC timezone.

    Args:
        timestrings: String or Series of ISO8601 formatted timestamps

    Returns:
        Datetime or DatetimeIndex with UTC timezone
    """
    return pd.to_datetime(timestrings, format="ISO8601", utc=True)
