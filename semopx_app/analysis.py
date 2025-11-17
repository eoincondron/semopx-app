from sys import path

import numpy as np
import pandas as pd
from groupby_lib.groupby import GroupBy

from .data_cache import get_combined_forecasts_single_date

path.insert(0, "/Users/eoincondron/git/groupby-lib")


def wind_percentage_forecast_table(alert_date: str) -> pd.DataFrame:
    """
    Generate a table of wind percentage forecasts categorized by day of the week and tariff zone.

    Parameters:
        alert_date: The date for which to generate the forecasts.

    Returns:
        A DataFrame containing the mean wind percentage forecasts for three days
        categorized by day of the week and tariff zone (DayTime, Peak, Evening, Overnight).

    """
    combined_forecasts = pd.concat(
        [get_combined_forecasts_single_date(alert_date, days_ahead=d) for d in range(3)]
    )
    combined_forecasts["time_of_day"] = (
        combined_forecasts.index - combined_forecasts.index.floor("D")
    )
    overnight_evening = combined_forecasts.time_of_day > "23h"
    overnight_morning = combined_forecasts.time_of_day <= "8h"

    combined_forecasts["tariff_zone"] = pd.Categorical.from_codes(
        np.select(
            [
                overnight_morning | overnight_evening,
                combined_forecasts.time_of_day.between("8h", "17h", inclusive="right"),
                combined_forecasts.time_of_day.between("17h", "19h", inclusive="right"),
            ],
            [0, 1, 2],
            3,
        ),
        ["Overnight", "DayTime", "Peak", "Evening"],
    )

    combined_forecasts["date"] = combined_forecasts.index.floor("D")

    bucket_date = combined_forecasts.date - np.where(
        overnight_morning, 1, 0
    ) * pd.Timedelta(1, "d")
    bucket_date = (
        bucket_date.astype("category").map(lambda d: d.strftime("%A")).rename("Day")
    )

    gb = GroupBy([bucket_date, combined_forecasts.tariff_zone.rename("Period")])
    return gb.mean(combined_forecasts[["AggregatedWind", "LoadTotal"]]).eval(
        "WindPc = AggregatedWind / LoadTotal"
    )
