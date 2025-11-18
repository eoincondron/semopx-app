from pathlib import Path
from typing import Callable, Literal
from warnings import warn
import io
import os

import pandas as pd
from tqdm import tqdm
import requests

from .client import SEMOAPIClient
from .util import date_to_str


CACHE_PATH = Path(__file__).parent / ".data_cache"
CACHE_PATH.mkdir(exist_ok=True)

# GitHub cache configuration
# When running on Streamlit Cloud, falls back to GitHub for cached files
GITHUB_REPO = os.environ.get("GITHUB_REPO", "eoincondron/semopx-app")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_CACHE_BASE = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/.data_cache"
)

# Enable GitHub fallback on Streamlit Cloud
USE_GITHUB_CACHE = (
    os.environ.get("STREAMLIT_SHARING_MODE", None) is not None
    or os.environ.get("USE_GITHUB_CACHE", "false").lower() == "true"
)


def cache_path_for_date(date: str, prefix: str) -> Path:
    """
    Generate a cache file path for a given date and prefix.

    Args:
        date: Date string in various formats (YYYYMMDD, YYYY-MM-DD, etc.)
        prefix: Prefix for the cache file name (e.g., 'day_ahead_prices')

    Returns:
        Path object for the cache file (e.g., .data_cache/day_ahead_prices_20240115.pq)
    """
    date = date_to_str(date)
    cache_file = f"{prefix}_{date}.pq"
    cache_path = CACHE_PATH / cache_file
    return cache_path


def download_from_github(filename: str) -> pd.DataFrame | None:
    """
    Try to download a parquet file from GitHub cache.

    Args:
        filename: Name of the parquet file to download

    Returns:
        DataFrame if successful, None if file not found or error occurs
    """
    if not USE_GITHUB_CACHE:
        return

    url = f"{GITHUB_CACHE_BASE}/{filename}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return pd.read_parquet(io.BytesIO(response.content))
        elif response.status_code == 404:
            # File doesn't exist in GitHub cache yet
            return None
        else:
            warn(f"GitHub cache returned status {response.status_code} for {filename}")
            return None
    except requests.exceptions.Timeout:
        warn(f"Timeout downloading {filename} from GitHub cache")
        return None
    except Exception as e:
        warn(f"Error downloading {filename} from GitHub cache: {e}")
        return None


def cache_daily_data(file_name_from_args: Callable):
    """
    Decorator that adds three-tier caching to data fetching functions.

    Implements a caching strategy:
    1. Local cache (.data_cache/) - fast, ephemeral on Streamlit Cloud
    2. GitHub cache - persistent across deployments
    3. API fetch - last resort

    Args:
        file_name_from_args: Function that generates cache filename from the decorated
                           function's arguments (e.g., lambda date: f"prices_{date}.pq")

    Returns:
        Decorator function that wraps data fetching functions with caching logic

    Example:
        @cache_daily_data(file_name_from_args=lambda date: f"prices_{date}.pq")
        def get_prices(date, cache_only=False, try_cache=True):
            return client.fetch_prices(date)
    """

    def wrapped(func):

        def inner(
            date, *args, cache_only: bool = False, try_cache: bool = True, **kwargs
        ) -> pd.DataFrame:
            date = date_to_str(date)
            filename = file_name_from_args(date, *args, **kwargs)
            cache_path = CACHE_PATH / filename

            # Step 1: Try local cache first (fast)
            if try_cache and cache_path.exists():
                return pd.read_parquet(cache_path)

            # Step 2: Try GitHub cache (persistent, slower)
            if try_cache:
                df = download_from_github(filename)
                if df is not None:
                    # Save to local cache for this session
                    try:
                        df.to_parquet(cache_path)
                    except Exception as e:
                        warn(f"Could not save to local cache: {e}")
                    return df

            # Step 3: If cache_only mode, raise error
            if cache_only:
                raise FileNotFoundError(f"Cache file not found: {cache_path}")

            # Step 4: Fetch from API as last resort
            df = func(date, *args, **kwargs)

            # Save to local cache
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                warn(f"Could not save to local cache: {e}")

            return df

        return inner

    return wrapped


def _load_date_range(
    func,
    start_date: str,
    end_date: str,
    cache_only: bool = False,
    try_cache: bool = True,
    on_missing: Literal["warn", "raise", "ignore"] = "warn",
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Load and concatenate data for a date range using a single-date function.

    Args:
        func: Single-date data fetching function to call for each date
        start_date: Start date (inclusive) in YYYYMMDD or YYYY-MM-DD format
        end_date: End date (inclusive) in YYYYMMDD or YYYY-MM-DD format
        cache_only: If True, only load from cache, don't fetch from API
        try_cache: If True, try to load from cache before fetching
        on_missing: How to handle missing data - 'warn', 'raise', or 'ignore'
        verbose: If True, show progress bar with tqdm
        **kwargs: Additional arguments to pass to func

    Returns:
        Concatenated DataFrame for all dates in the range

    Raises:
        ValueError: If no data found in the specified date range
    """
    dfs = []
    dates = pd.date_range(start_date, end_date)
    if verbose:
        dates = tqdm(dates, desc=f"Loading {func.__name__} data")

    for date in dates:
        try:
            dfs.append(func(date, cache_only=cache_only, try_cache=try_cache, **kwargs))
        except (FileNotFoundError, ValueError):
            if on_missing == "raise":
                raise
            elif on_missing == "warn":
                warn(f"{func.__name__} Data missing for date {date}, skipping")
    if not dfs:
        raise ValueError("No data found in the specified date range")

    return pd.concat(dfs)


@cache_daily_data(file_name_from_args=lambda date: f"day_ahead_prices_{date}.pq")
def get_day_ahead_prices_single_date(
    date, cache_only: bool = False, try_cache: bool = True
) -> pd.DataFrame:
    """
    Get day-ahead electricity prices for a single date.

    Fetches day-ahead market prices from SEMO API and standardizes column names
    by removing frequency suffixes (_30, _60) to facilitate data concatenation.

    Args:
        date: Date in YYYYMMDD or YYYY-MM-DD format
        cache_only: If True, only load from cache, don't fetch from API
        try_cache: If True, try to load from cache before fetching

    Returns:
        DataFrame with day-ahead prices indexed by timestamp. Contains columns like
        Index_prices_EUR with standardized naming (frequency suffixes removed).

    Raises:
        FileNotFoundError: If cache_only=True and data not in cache
    """
    prices = SEMOAPIClient().get_day_ahead_prices(date)
    # Column names can have mixed frequency suffixes, e.g. Index_prices_30_EUR and Index_prices_60_EUR
    # We standardize to just Index_prices_EUR to facilitate concatenation
    prices.columns = prices.columns.str.replace("_30|_60", "", regex=True)

    return prices


def get_day_ahead_prices_date_range(
    start_date: str,
    end_date: str,
    cache_only: bool = False,
    try_cache: bool = True,
    on_missing: Literal["warn", "raise", "ignore"] = "warn",
):
    """
    Get day-ahead electricity prices for a date range.

    Args:
        start_date: Start date (inclusive) in YYYYMMDD or YYYY-MM-DD format
        end_date: End date (inclusive) in YYYYMMDD or YYYY-MM-DD format
        cache_only: If True, only load from cache, don't fetch from API
        try_cache: If True, try to load from cache before fetching
        on_missing: How to handle missing data - 'warn', 'raise', or 'ignore'

    Returns:
        Concatenated DataFrame with day-ahead prices for all dates in range

    Raises:
        ValueError: If no data found in the specified date range
    """
    return _load_date_range(get_day_ahead_prices_single_date, **locals())


@cache_daily_data(
    file_name_from_args=lambda date, session_number: f"intraday_prices_{session_number}_{date}.pq"
)
def get_intraday_prices_single_date(
    date,
    session_number: int,
    cache_only: bool = False,
    try_cache: bool = True,
) -> pd.DataFrame:
    """
    Get intraday electricity prices for a single date and session.

    Args:
        date: Date in YYYYMMDD or YYYY-MM-DD format
        session_number: Intraday session number (typically 1-3)
        cache_only: If True, only load from cache, don't fetch from API
        try_cache: If True, try to load from cache before fetching

    Returns:
        DataFrame with intraday prices for the specified session indexed by timestamp

    Raises:
        FileNotFoundError: If cache_only=True and data not in cache
    """
    return SEMOAPIClient().get_intraday_prices(date, session_number=session_number)


def get_intraday_prices_date_range(
    start_date: str,
    end_date: str,
    session_number: int,
    cache_only: bool = False,
    on_missing: Literal["warn", "raise", "ignore"] = "warn",
):
    return _load_date_range(get_intraday_prices_single_date, **locals())


def _get_forecast_single_date(
    date,
    resource_name: str,
    cache_only: bool = False,
    try_cache: bool = True,
    as_of: str = "12:00",
) -> pd.DataFrame:
    client = SEMOAPIClient()
    reports = client.get_reports_date_range(date, resource_name=resource_name)
    if reports.empty:
        raise FileNotFoundError(f"No wind forecast report found for date {date}")

    reports.sort_values("PublishTime", inplace=True)

    if as_of:
        resources = reports.ResourceName[reports.PublishTime < f"{date}T{as_of}"].iloc[
            -1:
        ]
    else:
        resources = reports.ResourceName.unique()

    df = pd.concat(
        [client.download_XML(resource, as_tree=False) for resource in resources]
    )

    return df


@cache_daily_data(file_name_from_args=lambda date, as_of: f"load_forecast_{date}.pq")
def get_load_forecast_single_date(
    date, cache_only: bool = False, try_cache: bool = True, as_of: str = "12:00"
) -> pd.DataFrame:
    return _get_forecast_single_date(resource_name="PUB_DailyLoadFcst", **locals())


def get_load_forecast_date_range(
    start_date: str,
    end_date: str,
    as_of: str = "12:00",
    try_cache: bool = True,
    cache_only: bool = False,
    on_missing: Literal["warn", "raise", "ignore"] = "warn",
):
    return _load_date_range(get_load_forecast_single_date, **locals())


@cache_daily_data(
    file_name_from_args=lambda date, *args, **kwargs: f"wind_forecast_{date}.pq"
)
def get_wind_forecast_single_date(
    date, as_of: str = "12:00", cache_only: bool = False, try_cache: bool = True
) -> pd.DataFrame:
    return _get_forecast_single_date(
        resource_name="PUB_4DayAggRollWindUnitFcst", **locals()
    )


def get_wind_forecast_date_range(
    start_date: str,
    end_date: str,
    as_of: str = "12:00",
    try_cache: bool = True,
    cache_only: bool = False,
    on_missing: Literal["warn", "raise", "ignore"] = "warn",
):
    return _load_date_range(get_wind_forecast_single_date, **locals())


def get_all_market_price_data_date_range(
    start_date: str,
    end_date: str,
    cache_only: bool = False,
    on_missing: Literal["warn", "raise", "ignore"] = "warn",
) -> pd.DataFrame:
    """
    Get all market price data (day-ahead and intraday sessions 1-3) for a date range.

    Fetches day-ahead prices and all three intraday sessions, combining them into
    a single DataFrame for easy comparison and analysis.

    Args:
        start_date: Start date (inclusive) in YYYYMMDD or YYYY-MM-DD format
        end_date: End date (inclusive) in YYYYMMDD or YYYY-MM-DD format
        cache_only: If True, only load from cache, don't fetch from API
        on_missing: How to handle missing data - 'warn', 'raise', or 'ignore'

    Returns:
        DataFrame indexed by timestamp with columns:
        - day_ahead_price: Day-ahead market prices
        - intraday_price_1: Intraday session 1 prices
        - intraday_price_2: Intraday session 2 prices
        - intraday_price_3: Intraday session 3 prices

    Raises:
        ValueError: If no data found in the specified date range
    """
    kwargs = locals().copy()
    prices = get_day_ahead_prices_date_range(**kwargs)
    prices = prices.filter(like="_EUR").squeeze().to_frame("day_ahead_price")
    for n in range(1, 4):
        prices[f"intraday_price_{n}"] = get_intraday_prices_date_range(
            **kwargs,
            session_number=n,
        ).Index_prices_30_EUR

    return prices


def get_combined_forecasts_single_date(
    publish_date,
    days_ahead: int = 0,
    as_of: str = "12:00",
    try_cache: bool = True,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    Get combined wind and load forecasts for a given date.
    The forecasts cover a 24-hour period from 21:30 on the given date.
    The returned DataFrame has a 30-minute frequency and contains columns for wind and load forecasts.
    """
    publish_date = date_to_str(publish_date)

    kwargs = dict(
        date=publish_date, as_of=as_of, try_cache=try_cache, cache_only=cache_only
    )
    load = get_load_forecast_single_date(**kwargs)
    wind = get_wind_forecast_single_date(**kwargs)

    period_start = (
        pd.Timestamp(publish_date, tz="utc")
        + pd.Timedelta(days_ahead, "d")
        + pd.Timedelta("22h")
    )
    # 24-hour period from 21:30 on the given date, left inclusive
    indexer = slice(period_start, period_start + pd.Timedelta(0.999, "D"))

    wind = (
        wind.set_index("EndTime")
        .filter(regex="Forecast|PublishTime")
        .resample("30min")
        .mean()
    ).loc[indexer]
    load = load.set_index("EndTime").filter(regex="Forecast|PublishTime").loc[indexer]

    wind.columns = wind.columns.str.replace("Load", "").str.replace("Forecast", "Wind")
    load.columns = load.columns.str.replace("Forecast", "")
    load = load.rename(columns={"Aggregated": "LoadTotal"})
    wind = wind.rename(columns={"AggregatedWind": "WindTotal"})

    forecasts = wind.join(load, lsuffix="Wind", rsuffix="Load", how="inner")

    if len(forecasts) < 48 or forecasts.isna().any().any():
        raise ValueError(
            f"data incomplete for {publish_date} with {days_ahead} day prediction"
        )

    return forecasts


def get_combined_forecasts_date_range(
    start_date: str,
    end_date: str,
    days_ahead: int = 0,
    as_of: str = "12:00",
    cache_only: bool = False,
    on_missing: Literal["warn", "raise", "ignore"] = "warn",
) -> pd.DataFrame:
    """
    Get combined wind and load forecasts for a date range.

    Fetches and combines wind generation and load forecasts for multiple dates.
    Each date covers a 24-hour period from 21:30 with 30-minute frequency.

    Args:
        start_date: Start date (inclusive) in YYYYMMDD or YYYY-MM-DD format
        end_date: End date (inclusive) in YYYYMMDD or YYYY-MM-DD format
        days_ahead: Number of days ahead to forecast from publish date (default: 0)
        as_of: Time string (HH:MM) to get forecasts published before this time (default: "12:00")
        cache_only: If True, only load from cache, don't fetch from API
        on_missing: How to handle missing data - 'warn', 'raise', or 'ignore'

    Returns:
        Concatenated DataFrame indexed by timestamp with wind and load forecast columns
        for all regions (Total, ROI, NI) at 30-minute intervals

    Raises:
        ValueError: If no data found in the specified date range
    """
    return _load_date_range(
        get_combined_forecasts_single_date,
        **locals(),
    )
