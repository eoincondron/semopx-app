from sys import path

import numpy as np
import pandas as pd
from groupby_lib.groupby import GroupBy

from .data_cache import get_combined_forecasts_single_date

path.insert(0, "/Users/eoincondron/git/groupby-lib")


def load_forecast_data(alert_date: str, days_ahead: int = 3) -> pd.DataFrame:
    combined_forecasts = pd.concat(
        [
            get_combined_forecasts_single_date(alert_date, days_ahead=d)
            for d in range(days_ahead)
        ]
    )
    combined_forecasts["date"] = combined_forecasts.index.floor("D")  # type: ignore
    combined_forecasts["time_of_day"] = (
        combined_forecasts.index - combined_forecasts.date
    )
    overnight_evening = combined_forecasts.time_of_day > "23h"
    overnight_morning = combined_forecasts.time_of_day <= "8h"

    combined_forecasts["tariff_zone"] = pd.Categorical.from_codes(
        np.select(  # type: ignore
            [
                overnight_morning | overnight_evening,
                combined_forecasts.time_of_day.between("8h", "17h", inclusive="right"),
                combined_forecasts.time_of_day.between("17h", "19h", inclusive="right"),
            ],
            [0, 1, 2],
            3,
        ),
        ["Overnight", "DayTime", "Peak", "Evening"],  # type: ignore
    )
    date_offset = combined_forecasts.date - np.where(
        overnight_morning, 1, 0
    ) * pd.Timedelta(1, "d")
    combined_forecasts["day_relative_to_tariff_zone"] = (
        date_offset.astype("category").map(lambda d: d.strftime("%A")).rename("Day")
    )

    return combined_forecasts


def wind_percentage_forecast_table(combined_forecasts: pd.DataFrame, region: str = "Total") -> pd.DataFrame:
    """
    Generate a table of wind percentage forecasts categorized by day of the week and tariff zone.

    Parameters:
        combined_forecasts (pd.DataFrame): DataFrame containing combined load and wind forecasts.

    Returns:
        A DataFrame containing the mean wind percentage forecasts for three days
        categorized by day of the week and tariff zone (DayTime, Peak, Evening, Overnight).

    """
    gb = GroupBy(
        dict(
            Day=combined_forecasts.day_relative_to_tariff_zone,
            Period=combined_forecasts,
        )
    )
    wind_col, load_col = f"Wind{region}", f"Load{region}"
    return gb.mean(combined_forecasts[[wind_col, load_col]]).eval(
        f"WindPc = {wind_col} / {load_col}"
    )  # type: ignore
