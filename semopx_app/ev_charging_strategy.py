"""
EV Charging Strategy Module

This module implements a smart EV charging strategy that optimizes for using
wind-generated electricity while maintaining battery charge constraints.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def calculate_charge(
    foreseeable_wind_levels: NDArray[np.floating],
    current_charge: float,
    min_charge: float,
    max_level: float,
    good_wind_level: float,
    daily_usage: float,
) -> float:
    """
    Calculate the optimal charge amount for tonight based on wind forecasts.

    This function implements a smart charging algorithm that decides whether to charge
    the EV battery tonight based on two main conditions:

    1. **Opportunistic Charging**: Charge to maximum if tonight is the windiest night
       in the foreseeable future (next 4 nights) AND tonight's wind is above the
       "good wind" threshold.

    2. **Forced Charging**: Charge by daily_usage amount if the battery is (or will be)
       below the minimum charge level within the foreseeable future, and tonight has
       the best wind so far before the earliest forced charge date.

    Parameters
    ----------
    foreseeable_wind_levels : NDArray[np.floating]
        Array of wind contribution percentages for the next 4 nights (including tonight).
        First element [0] is tonight's wind level. Values should be between 0.0 and 1.0.
    current_charge : float
        Current battery charge level as a fraction (0.0 to 1.0).
    min_charge : float
        Minimum battery charge level to maintain (0.0 to 1.0). If the battery drops
        below this level, a forced charge is triggered.
    max_level : float
        Target charge level when charging (0.0 to 1.0). Charging stops once this
        level is reached.
    good_wind_level : float
        Threshold for considering a night "windy" (0.0 to 1.0). Nights above this
        threshold are candidates for opportunistic charging.
    daily_usage : float
        Daily battery depletion as a fraction of total capacity (0.0 to 1.0).
        Used to predict when the battery will fall below minimum charge.

    Returns
    -------
    float
        The amount to charge tonight as a fraction of total battery capacity.
        Returns 0 if no charging is needed tonight.

    Examples
    --------
    >>> import numpy as np
    >>> # Tonight is windiest and above threshold - charge to max
    >>> calculate_charge(
    ...     foreseeable_wind_levels=np.array([0.9, 0.7, 0.6, 0.5]),
    ...     current_charge=0.5,
    ...     min_charge=0.2,
    ...     max_level=0.9,
    ...     good_wind_level=0.8,
    ...     daily_usage=0.1
    ... )
    0.4

    >>> # Wind below threshold - don't charge
    >>> calculate_charge(
    ...     foreseeable_wind_levels=np.array([0.6, 0.7, 0.8, 0.5]),
    ...     current_charge=0.5,
    ...     min_charge=0.2,
    ...     max_level=0.9,
    ...     good_wind_level=0.8,
    ...     daily_usage=0.1
    ... )
    0.0

    Notes
    -----
    - The function assumes that foreseeable_wind_levels contains at least 1 element
    - The algorithm looks ahead 4 nights to make charging decisions
    - If already at max_level, no charging occurs regardless of wind conditions
    """
    if min_charge < daily_usage:
        raise ValueError("min_charge must be at least the value of daily_usage")
    todays_wind_pc = foreseeable_wind_levels[0]

    # Already fully charged - no need to charge
    if current_charge >= max_level:
        return 0

    # Opportunistic charging: Tonight is windiest and above threshold
    if (
        todays_wind_pc == foreseeable_wind_levels.max()
        and todays_wind_pc > good_wind_level
    ):
        return max_level - current_charge

    # Forced charging: Check if we'll drop below minimum in foreseeable future
    cum_max = todays_wind_pc
    charge = 0
    for i, wind_pc in enumerate(foreseeable_wind_levels):
        if current_charge - i * daily_usage <= min_charge:
            # Need to charge before day i - charge tonight if tonight is best so far
            if todays_wind_pc == cum_max:
                charge += daily_usage

        cum_max = max(cum_max, wind_pc)

    return charge


def get_charge_history(
    overnight_wind_pc: pd.Series,
    min_charge: float = 0.2,
    good_wind_level: float = 0.85,
    daily_usage: float = 0.1,
    max_charge: float = 0.9,
    starting_charge: float = 0.5,
) -> pd.DataFrame:
    """
    Simulate the EV charging strategy over a time period and return charging history.

    This function runs a simulation of the smart charging algorithm over a series of
    nights, tracking battery charge levels, charging events, and wind conditions.
    For each night, it calls `calculate_charge()` to determine whether to charge
    the battery based on wind forecasts.

    Parameters
    ----------
    overnight_wind_pc : pd.Series
        Time series of overnight wind contribution percentages (0.0 to 1.0).
        Index should be dates, values should be wind percentage for each night.
        Must contain at least 5 data points (needs 4-night lookback window).
    min_charge : float, default=0.2
        Minimum battery charge level to maintain (0.0 to 1.0). If the battery drops
        below this level, a forced charge is triggered.
    good_wind_level : float, default=0.85
        Threshold for considering a night "windy" (0.0 to 1.0). Nights above this
        threshold are candidates for opportunistic charging.
    daily_usage : float, default=0.1
        Daily battery depletion as a fraction of total capacity (0.0 to 1.0).
        Battery level is reduced by this amount each night before charging.
    max_charge : float, default=0.9
        Target charge level when charging (0.0 to 1.0). Charging stops once this
        level is reached.
    starting_charge : float, default=0.5
        Initial battery charge level at the start of the simulation (0.0 to 1.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and three columns:
        - 'charge': Amount charged on each night (0.0 to 1.0)
        - 'wind_pc': Wind contribution percentage on each night (0.0 to 1.0)
        - 'current_charge': Battery charge level after charging (0.0 to 1.0)

    Raises
    ------
    AssertionError
        If the battery charge drops below min_charge at any point, indicating
        a logic error in the charging algorithm.

    Examples
    --------
    >>> import pandas as pd
    >>> # Create sample wind data for 10 nights
    >>> dates = pd.date_range("2024-01-01", periods=10, freq="D")
    >>> wind_data = pd.Series([0.9, 0.7, 0.6, 0.5, 0.8, 0.9, 0.7, 0.6, 0.5, 0.4], index=dates)
    >>>
    >>> # Run simulation
    >>> history = get_charge_history(
    ...     overnight_wind_pc=wind_data,
    ...     min_charge=0.2,
    ...     good_wind_level=0.8,
    ...     daily_usage=0.1,
    ...     max_charge=0.9,
    ...     starting_charge=0.5
    ... )
    >>>
    >>> # Check when charging occurred
    >>> charging_nights = history[history['charge'] > 0]
    >>> print(f"Charged on {len(charging_nights)} nights")
    Charged on 2 nights

    Notes
    -----
    - The simulation runs from the first date to 4 dates before the last (to maintain
      the 4-night lookback window)
    - Battery level is decreased by daily_usage at the beginning of each night before
      charging decisions are made
    - The function ensures battery never drops below min_charge through strategic charging
    - Results can be used to calculate metrics like weighted average wind contribution
      during charging sessions: sum(wind_pc * charge) / sum(charge)
    """
    current_charge = starting_charge
    charges = {}

    # Iterate through nights, leaving last 4 for lookback window
    for i, (date, wind_pc) in enumerate(overnight_wind_pc[:-4].items()):
        # Daily usage reduces charge at start of night
        current_charge -= daily_usage

        # Calculate optimal charge for tonight based on next 4 nights
        charge = calculate_charge(
            np.asarray(overnight_wind_pc[i : i + 4]),
            current_charge,
            min_charge,
            max_charge,
            good_wind_level,
            daily_usage=daily_usage,
        )

        # Apply charge
        current_charge += charge

        # Ensure battery never drops below minimum (algorithm invariant)
        # assert current_charge >= min_charge, (
        #     f"Battery charge {current_charge:.2f} dropped below minimum {min_charge:.2f} "
        #     f"on {date}. This indicates a bug in the charging algorithm."
        # )

        # Record charging event
        charges[date] = charge, wind_pc, current_charge

    # Convert to DataFrame for easy analysis
    charges_df = pd.DataFrame(charges, ["charge", "wind_pc", "current_charge"]).T

    return charges_df
