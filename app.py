"""
SEMO Energy Market Analysis Dashboard

A Streamlit app for visualizing Irish energy market data including:
- Wind Contribution forecasts by time of day and tariff zone
- Day-ahead and intraday electricity prices
- Load and wind generation forecasts
"""

from typing import Optional
from warnings import warn
from functools import cached_property

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt

from semopx_app.util import DayDelta, date_to_str
from semopx_app import data_cache
from semopx_app.ev_charging_strategy import get_charge_history


TARIFF_ZONES = {
    "DayTime": "08:00:00 - 17:00:00",
    "Peak": "17:00:00 - 19:00:00",
    "Evening": "19:00:00 - 23:00:00",
    "Overnight": "23:00:00 - 08:00:00",
}


@st.cache_data(ttl=3600)
def load_forecast_data(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """ """
    if end_date is None:
        end_date = start_date

    if start_date > end_date:
        raise ValueError("start_date must be less than or equal to end_date")

    elif start_date != end_date:
        dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")
        dfs, missing = [], []
        for date in dates:
            try:
                dfs.append(load_forecast_data(date))
            except FileNotFoundError:
                missing.append(date)

        if missing:
            warn(
                f"missing forecast data on dates between {missing[0]} and {missing[-1]}"
            )

        dfs = pd.concat(dfs)
        # Each date covers 4 days so take most recent
        return dfs.groupby("EndTime").nth(-1)  # type: ignore

    dfs = []
    for name in ["load", "wind"]:
        df = getattr(data_cache, f"get_{name}_forecast_single_date")(
            start_date, as_of="23:00"
        ).set_index("EndTime")
        df = df.filter(like="Forecast")
        df.columns = name.title() + df.columns.str.replace(
            "AggregatedForecast", "Total"
        ).str.replace("LoadForecast", "")
        dfs.append(df)

    forecasts = pd.concat(dfs, axis=1, join="inner").loc[start_date:]

    forecasts["date"] = forecasts.index.floor("D")  # type: ignore
    forecasts["time_of_day"] = forecasts.index - forecasts.date

    tariff_zone_masks = []
    for period in ["DayTime", "Peak", "Evening"]:
        times = pd.TimedeltaIndex(TARIFF_ZONES[period].split(" - "))
        tariff_zone_masks.append(
            forecasts.time_of_day.between(*times, inclusive="right")
        )

    forecasts["tariff_zone"] = pd.Categorical.from_codes(
        np.select(tariff_zone_masks, [0, 1, 2], 3),
        ["DayTime", "Peak", "Evening", "Overnight"],  # type: ignore
    )

    return forecasts


class SEMODataLoader:
    """Main dashboard class for SEMO Energy Market Analysis."""

    REGION_MAP = {
        "All Ireland": "Total",
        "Republic of Ireland": "ROI",
        "Northern Ireland": "NI",
    }

    def __init__(
        self, date: str, n_days_lookback: int = 1, region: str = "All Ireland"
    ):
        """Initialize the dashboard with page configuration."""
        st.set_page_config(
            page_title="SEMO Energy Dashboard",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self.date = date_to_str(date)
        self.region = region
        self.forecast_data: pd.DataFrame
        self.n_days_lookback = n_days_lookback
        self.forecast_data = load_forecast_data(
            start_date=self.start_date, end_date=self.date
        )

    @property
    def start_date(self) -> str:
        return self.date - DayDelta(self.n_days_lookback - 1)

    @property
    def _regional_data(self):
        wind_col, load_col = f"Wind{self.region_suffix}", f"Load{self.region_suffix}"
        return pd.DataFrame(
            {
                "WindForecast": self.forecast_data[wind_col],
                "LoadForecast": self.forecast_data[load_col],
            },
            copy=False,
        )

    def wind_percentage_forecast_table(
        self, asof: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Generate a table of Wind Contribution forecasts categorized by day of the week and tariff zone.

        Parameters:
            combined_forecasts (pd.DataFrame): DataFrame containing combined load and wind forecasts.

        Returns:
            A DataFrame containing the mean Wind Contribution forecasts for three days
            categorized by day of the week and tariff zone (DayTime, Peak, Evening, Overnight).

        """
        if asof is None:
            asof = pd.Timestamp(self.date, tz="UTC")

        df = self.forecast_data.loc[asof:].copy()

        morning = df.time_of_day <= "8h"
        date_offset = df.date - np.where(morning, 1, 0) * pd.Timedelta(1, "d")
        df["day_relative_to_tariff_zone"] = (
            date_offset.astype("category").map(lambda d: d.strftime("%A")).rename("Day")
        )
        group_keys = ["day_relative_to_tariff_zone", "tariff_zone"]
        df = df[group_keys].join(self._regional_data)
        averages = df.groupby(group_keys, observed=True).mean()
        averages: pd.DataFrame = averages.eval("WindPc = WindForecast / LoadForecast")  # type: ignore
        averages.index.names = ["Day", "Period"]

        return averages

    @property
    def region_suffix(self) -> str:
        """
        Map user-friendly region name to data column suffix.

        Returns:
            Column suffix for the region
        """
        return self.REGION_MAP.get(self.region, "Total")

    def wind_load_forecast_plot(
        self, sampling_frequency: Optional[str] = None, n_days_lookback: int = 0
    ):
        """
        Render interactive time series plot of wind and load forecasts with Wind Contribution using Plotly.
        """
        if sampling_frequency:
            df = self._regional_data.resample(sampling_frequency).mean()
        else:
            df = self._regional_data

        from_date = self.date - DayDelta(n_days_lookback)
        if n_days_lookback > self.n_days_lookback:
            self.forecast_data = load_forecast_data(
                start_date=from_date, end_date=self.date
            )

        df = df.loc[from_date:]

        # Calculate Wind Contribution
        wind_pc = df.eval("WindForecast / LoadForecast") * 100  # type: ignore

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add Wind Generation trace
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["WindForecast"],
                name="Wind Generation",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="<b>Wind</b>: %{y:,.0f} MW<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Add Load trace
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["LoadForecast"],
                name="Total Demand",
                line=dict(color="#ff7f0e", width=2),
                hovertemplate="<b>Load</b>: %{y:,.0f} MW<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Add Wind Contribution trace on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=wind_pc,
                name="Wind Contribution",
                line=dict(color="cyan", width=2, dash="dash"),
                hovertemplate="<b>Wind %%</b>: %{y:.1f}%%<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            ),
            secondary_y=True,
        )

        # Add vertical line for current time
        fig.add_vline(
            x=pd.Timestamp.now().value,
            line_dash="dash",
            line_color="magenta",
            line_width=2,
            annotation_text="Now",
            annotation_position="top",
        )

        # Update layout
        fig.update_layout(
            title={
                "text": f"Wind & Load Forecast - {self.region}",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18, "color": "#1f77b4"},
            },
            xaxis_title="Time",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Power (MW)</b>", secondary_y=False)
        fig.update_yaxes(
            title_text="<b>Wind Contribution (%)</b>",
            secondary_y=True,
            range=[0, 100],
        )

        return fig

    def hourly_averages_plot(self):
        grouper = self.forecast_data.index.strftime("%H:%M").rename("Time of Day")
        hourly_averages = self._regional_data.groupby(grouper).mean()
        wind_contrib = hourly_averages.eval("100 * WindForecast/ LoadForecast")
        hourly_averages = hourly_averages.rename(
            columns=SEMODashboard.COLUMN_DISPLAY_NAME_MAP
        )
        ax = hourly_averages.plot(
            figsize=(10, 4),
            title="Hourly Averages of Wind, Demand",
            ylabel="Generation (MW)",
            rot=45,
        )
        ax.legend(loc="center left")
        ax = wind_contrib.to_frame("Wind Contribution").plot(
            ax=ax.twinx(), ylabel="Wind Contribution %", color="magenta", style="--o"
        )
        ax.legend(loc="upper right")

        return ax

    def graph_average_wind_contribution_for_ev_charging_strategy(
        self,
        min_charge=0.2,
        daily_usage=0.1,
        max_charge=0.9,
        starting_charge=0.5,
    ):
        """
        Calculate weighted average wind contribution for different good wind level thresholds.

        This method simulates a smart EV charging strategy that optimizes for wind-generated
        electricity. It evaluates charging decisions across a range of "good wind" thresholds
        (0% to 100% in 5% increments) and calculates the weighted average wind contribution
        during actual charging sessions.

        The charging strategy charges when:
        i) Tonight is windy (above threshold) AND tonight is the windiest in the next 4 nights, OR
        ii) Battery is/will be below minimum charge and tonight has better wind than days
            before the earliest forced charge date.

        Parameters
        ----------
        min_charge : float, default=0.2
            Minimum battery charge level to maintain (0.0-1.0). If the battery drops below
            this level, a forced charge is triggered.
        daily_usage : float, default=0.1
            Daily battery depletion as a fraction of total capacity (0.0-1.0). Used to
            predict when the battery will fall below minimum charge.
        max_charge : float, default=0.9
            Target charge level when charging (0.0-1.0). Charging stops once this level
            is reached.
        starting_charge : float, default=0.5
            Initial battery charge level at the start of the simulation period (0.0-1.0).

        Returns
        -------
        pd.Series
            A Series indexed by good_wind_level thresholds (0.0 to 1.0 in 0.05 increments),
            with values representing the weighted average wind contribution (as a fraction)
            during charging sessions for each threshold.

        Notes
        -----
        - The method uses overnight wind forecasts (tariff zone 'Overnight')
        - Wind contribution is calculated as WindForecast / LoadForecast
        - The weighted average is: sum(wind_pc * charge) / sum(charge)

        Examples
        --------
        >>> loader = SEMODataLoader("2024-01-15", region="All Ireland")
        >>> results = loader.graph_average_wind_contribution_for_ev_charging_strategy(
        ...     min_charge=0.3, daily_usage=0.15, max_charge=0.85
        ... )
        >>> optimal_threshold = results.idxmax()
        >>> print(f"Optimal threshold: {optimal_threshold:.0%}")
        """
        kwargs = locals().copy()
        del kwargs["self"]
        mask = self.forecast_data.tariff_zone == "Overnight"
        means = self._regional_data[mask].groupby(self.forecast_data.date).mean()
        overnight_wind_pc = means.eval("WindForecast/ LoadForecast").clip(0, 1)  # type: ignore

        results = {}
        for good_wind_level in np.arange(0, 1.05, 0.05):
            charges_df = get_charge_history(
                overnight_wind_pc, good_wind_level=good_wind_level, **kwargs
            )
            results[good_wind_level] = (
                charges_df.eval("wind_pc * charge").sum() / charges_df.charge.sum()
            )

        return pd.Series(results)


class SEMODashboard:
    """Main dashboard class for SEMO Energy Market Analysis."""

    REGION_MAP = {
        "All Ireland": "Total",
        "Republic of Ireland": "ROI",
        "Northern Ireland": "NI",
    }

    COLUMN_DISPLAY_NAME_MAP = {
        "WindForecast": "Wind Generation (MW)",
        "LoadForecast": "Total Demand (MW)",
        "WindPc": "Wind Contribution",
    }

    def __init__(self):
        """Initialize the dashboard with page configuration."""
        st.set_page_config(
            page_title="SEMO Energy Dashboard",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self.now = pd.Timestamp.now(tz="UTC")
        self.selected_region: str = "All Ireland"
        self._apply_custom_css()

    def _apply_custom_css(self):
        """Apply custom CSS styling to the dashboard."""
        st.markdown(
            """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1f77b4;
                margin-bottom: 0.5rem;
            }
            .sub-header {
                font-size: 1.2rem;
                color: #666;
                margin-bottom: 2rem;
            }
            div[data-testid="stDataFrame"] {
                font-size: 1rem;
            }

            /* Mobile responsive adjustments */
            @media (max-width: 768px) {
                .main-header {
                    font-size: 1.8rem;
                }
                .sub-header {
                    font-size: 1rem;
                }
                div[data-testid="stDataFrame"] {
                    font-size: 0.85rem;
                }
                /* Make dataframes scrollable horizontally on mobile */
                div[data-testid="stDataFrame"] > div {
                    overflow-x: auto;
                }
                /* Reduce padding on mobile */
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def render_header(self):
        """Render the main header section."""
        st.markdown(
            '<div class="main-header">⚡ SEMO Energy Market Dashboard</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Irish Electricity Market Analysis</div>',
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        """Render the sidebar with region selection and info."""
        st.sidebar.header("Settings")
        st.sidebar.markdown("---")

        # Region selector
        st.sidebar.subheader("🌍 Region Selection")
        region_options = ["All Ireland", "Republic of Ireland", "Northern Ireland"]
        self.selected_region = st.sidebar.selectbox(
            "Region",
            options=region_options,
            index=0,  # Default to "All Ireland"
            help="Select the region for wind and load forecasts",
        )

        # Info section
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info(
            """
            This dashboard displays energy market data from SEMO (Single Electricity Market Operator)
            and SEMOpx (Exchange Market) for Ireland and Northern Ireland.

            **Data Sources:**
            - SEMO Balancing Market
            - SEMOpx Exchange Market
            - Wind & Load Forecasts
            """
        )

    def _style_forecast_dataframe(self, df: pd.DataFrame) -> Styler:
        """
        Apply heatmap styling to the forecast dataframe.

        Args:
            df_data: DataFrame with forecast data

        Returns:
            Styled DataFrame
        """
        # Rename columns
        df = df.rename(columns=self.COLUMN_DISPLAY_NAME_MAP)
        styler = df.style

        for col_name, color_map, vmin, vmax in [
            ("Wind Generation (MW)", "Greens", 0, 6000),
            ("Total Demand (MW)", "Blues", 4000, 8000),
            ("Wind Contribution", "RdYlGn", 0, 1),
        ]:

            # Apply background gradient to Wind Generation
            styler = styler.background_gradient(
                subset=[col_name],
                cmap=color_map,
                vmin=vmin,
                vmax=vmax,
            )

        styler = styler.format(
            {
                "Wind Generation (MW)": "{:,.0f}",
                "Total Demand (MW)": "{:,.0f}",
                "Wind Contribution": "{:.1%}",
            },
        )

        return styler

    def render_wind_load_forecast_plot(self):
        """
        Render time series plot of wind and load forecasts with Wind Contribution.
        """
        st.subheader(f"📊 Wind & Load Forecast Time Series - {self.selected_region}")
        fig = self.data_loader.wind_load_forecast_plot()
        st.plotly_chart(fig, use_container_width=True)

    @cached_property
    def wind_contribution_table(self):
        return self.data_loader.wind_percentage_forecast_table(asof=self.now)

    def wind_contribution_table_styled(self):
        df = self.wind_contribution_table.reset_index()
        day_shade_key = (df.Day != df.Day.shift()).cumsum() % 2 == 1

        # Blank duplicate day names for visual grouping
        df["Day"] = df["Day"].astype(str).mask(df["Day"].duplicated(), "")

        styler = self._style_forecast_dataframe(df)
        styler = styler.apply(
            func=lambda s: [
                f"background-color: {'rgb(200, 200, 200)' if gray else 'white'}"
                for gray in day_shade_key
            ],
            subset=["Day", "Period"],
        )
        return styler

    def render_main_tab(self):
        """Render the wind forecast table with heatmap styling."""
        st.header(f"Wind Contribution Forecast by Tariff Zone - {self.selected_region}")

        st.markdown("---")

        st.markdown(
            """
            This table shows the forecasted wind generation as a percentage of total load,
            broken down by day of the week and tariff zone for the next 3 days.
            """
        )

        st.markdown(
            "\n**Tariff Zones:**\n"
            + "".join(
                [
                    f"\n- **{key}**: {times}".replace("00:00", "00")
                    for key, times in TARIFF_ZONES.items()
                ]
            )
        )

        # Display styled table
        st.dataframe(
            self.wind_contribution_table_styled(),
            hide_index=True,
            use_container_width=False,
            height="stretch",
            column_config={
                "Day": st.column_config.TextColumn(
                    "Day",
                    width="medium",
                ),
            },
        )

        self.render_summary_statistics()
        self.render_wind_load_forecast_plot()

    def render_summary_statistics(self):
        """Render summary statistics for the forecast data."""
        st.subheader("📈 Summary Statistics")
        df = self.wind_contribution_table

        col1, col2, col3 = st.columns(3)

        # Average Wind %
        with col1:
            avg_wind_pct = df["WindPc"].mean() * 100
            st.metric(
                label="Average Wind %",
                value=f"{avg_wind_pct:.1f}%",
                help="Average Wind Contribution across all time periods",
            )

        # Maximum Wind %
        with col2:
            max_wind_pct = df["WindPc"].max() * 100
            st.metric(
                label="Maximum Wind %",
                value=f"{max_wind_pct:.1f}%",
                help="Highest Wind Contribution",
            )

        # Minimum Wind %
        with col3:
            min_wind_pct = df["WindPc"].min() * 100
            st.metric(
                label="Minimum Wind %",
                value=f"{min_wind_pct:.1f}%",
                help="Lowest Wind Contribution",
            )

        # Best and worst periods
        self._render_best_worst_periods()

    def _render_best_worst_periods(self):
        """Render best and worst wind periods."""
        st.markdown("---")
        col_best, col_worst = st.columns(2)
        wind_pc = self.wind_contribution_table.WindPc

        with col_best:
            st.success("**🌬️ Best Wind Period**")
            best_day, best_time = wind_pc.idxmax()  # type: ignore
            st.write(f"**Day and Time:** {best_day} {best_time}")
            st.write(f"**Wind %:** {wind_pc.max():.1%}")

        with col_worst:
            st.warning("**🔻 Lowest Wind Period**")
            worst_day, worst_time = wind_pc.idxmin()  # type: ignore
            st.write(f"**Day and Time:** {worst_day} {worst_time}")
            st.write(f"**Wind %:** {wind_pc.min():.1%}")

    def _render_download_button(self):
        """Render CSV download button."""
        st.markdown("---")
        csv = self.data_loader.forecast_data.to_csv()
        selected_date = self.now.strftime("%Y-%m-%d")
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"wind_forecast_{selected_date}.csv",
            mime="text/csv",
        )

    def render_historical_data_tab(self):
        """Render the historical data tab with wind and load trends."""
        st.header(f"📊 Historical Data Analysis - {self.selected_region}")

        # Historical data settings
        st.subheader("📈 Settings")
        col1, col2 = st.columns(2)

        with col1:
            n_days_lookback = st.number_input(
                "Days to Look Back",
                min_value=1,
                max_value=365,
                value=90,
                step=1,
                help="Number of days of historical data to display",
            )

        with col2:
            frequency_options = ["day", "week", "month"]
            sampling_frequency = st.selectbox(
                "Sampling Frequency",
                options=frequency_options,
                index=1,  # Default to "week"
                help="Frequency for sampling historical data",
            )

        sampling_frequency_pandas = {"day": "d", "week": "W", "month": "m"}.get(
            sampling_frequency
        )

        st.markdown("---")

        selected_date = self.now.strftime("%Y-%m-%d")
        n_days = (
            self.data_loader.forecast_data.loc[:selected_date]
            .index.floor("D")
            .nunique()
        )

        if n_days < n_days_lookback:
            st.warning(
                f"Not all requested dates are available. Earliest date is {self.data_loader.forecast_data.index[0].date()}"
            )

        st.markdown(
            f"""
        Showing historical wind and load data for the past **{n_days} days**,
        grouped by **{sampling_frequency}**.
        """
        )
        fig = self.data_loader.wind_load_forecast_plot(
            n_days_lookback=n_days_lookback,
            sampling_frequency=sampling_frequency_pandas,
        )
        st.plotly_chart(fig, use_container_width=True)

        self._render_monthly_averages()
        self._render_tariff_zone_averages()
        self._render_hourly_averages_plot()

    def _render_monthly_averages(self):
        st.subheader("Monthly Averages")
        df = self.data_loader._regional_data
        monthly_summary = df.resample("ME").mean()
        monthly_counts = df.resample("D").size().resample("ME").size().tolist()
        n_days = ["" for _ in range(len(monthly_summary))]
        start, end = df.index[[0, -1]]
        if start.day > 1:
            n_days[0] = f" ({monthly_counts[0]} Days)"

        if end != end + pd.offsets.MonthEnd():
            n_days[-1] = f" ({monthly_counts[-1]} Days)"

        monthly_summary.index = (
            monthly_summary.index.strftime("%b").rename("Month") + n_days
        )
        monthly_summary.loc["Total"] = monthly_summary.mean()
        monthly_summary["Wind Contribution"] = (
            monthly_summary.WindForecast / monthly_summary.LoadForecast
        )

        st.dataframe(self._style_forecast_dataframe(monthly_summary))

    def _render_tariff_zone_averages(self):
        st.subheader("Tariff Zone Averages")

        tariff_zone_summary = (
            self.data_loader._regional_data.groupby(
                self.data_loader.forecast_data.tariff_zone.rename("Period")
            )
            .mean()
            .eval("WindPc = WindForecast/ LoadForecast")
        )  # type: ignore
        tariff_zone_summary.index = tariff_zone_summary.index.astype(str)
        times = tariff_zone_summary.index.map(TARIFF_ZONES).str.replace("00:00", "00")
        tariff_zone_summary.index = tariff_zone_summary.index + [
            f" ({t})" for t in times
        ]

        st.dataframe(self._style_forecast_dataframe(tariff_zone_summary))

    def _render_hourly_averages_plot(self):
        st.subheader("Hourly Averages")
        self.data_loader.hourly_averages_plot()
        plt.tight_layout()
        st.pyplot(plt.gcf())

    def render_ev_charging_strategy_tab(self):
        """Render the EV charging strategy tab with interactive parameters."""
        st.header(f"🔋 EV Charging Strategy Analysis - {self.selected_region}")

        st.markdown(
            """
        This analysis evaluates a smart EV charging strategy that maximizes the use of wind-generated electricity.
        The strategy charges the EV overnight either opportunistically or when it is forced to:

         - **Type I - Opportunistic:** 
        Tonight is predicted to be windy (above the "good wind" threshold) **AND** tonight is the windiest
        night in the foreseeable future (next 4 nights)

         - **Type II - Forced Charging:** 
                    The battery is (or will be) below the minimum charge level within the foreseeable future,
        and tonight has better wind than any day before the earliest "forced" charge date.

        The chart below shows how different "good wind" thresholds affect the weighted average wind contribution
        during charging sessions. Adjust the parameters to see how charging behavior and wind utilization change.
        """
        )

        st.markdown("---")
        st.subheader("⚙️ Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            min_charge = st.slider(
                "Minimum Charge Level (5% - 50%)",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Minimum battery charge level to maintain. If the battery drops (or will drop) below this level, a 'forced' charge is triggered.",
            )

        with col2:
            daily_usage = st.slider(
                "Daily Usage (5% - 50%)",
                min_value=0.05,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Daily battery depletion as a fraction of total capacity. Used to predict when the battery will fall below minimum charge.",
            )

        with col3:
            max_charge = st.slider(
                "Maximum Charge Level (70% - 100%)",
                min_value=0.7,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Target charge level when charging. Charging stops once this level is reached.",
            )

        # Validate parameters
        if min_charge >= max_charge:
            st.error("⚠️ Minimum charge must be less than maximum charge")
            return

        starting_charge = max_charge

        st.markdown("---")

        # Calculate and plot results
        with st.spinner("Calculating wind contribution analysis..."):
            results = self.data_loader.graph_average_wind_contribution_for_ev_charging_strategy(
                min_charge=min_charge,
                daily_usage=daily_usage,
                max_charge=max_charge,
                starting_charge=starting_charge,
            )

            # Display summary statistics
            st.subheader("📊 Summary Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Maximum Weighted Avg",
                    f"{float(results.max()) * 100:.1f}%",
                    help="Highest weighted average wind contribution achieved across all threshold values",
                )

            with col2:
                st.metric(
                    "Optimal Threshold",
                    f"{float(results.idxmax()) * 100:.1f}%",
                    help="The 'good wind' threshold that maximizes wind contribution during charging. This is the sweet spot between charging frequently (low threshold) and waiting for very windy nights (high threshold).",
                )

            with col3:
                st.metric(
                    "Range",
                    f"{float(results.max() - results.min()) * 100:.1f}%",
                    help="Difference between maximum and minimum weighted averages, showing how much the threshold selection impacts wind utilization",
                )

            # Create plotly figure
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=(results.index * 100).astype(float),  # Convert to percentage
                    y=(results.values * 100).astype(float),  # Convert to percentage
                    mode="lines+markers",
                    name="Weighted Average Wind %",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(size=8),
                    hovertemplate="<b>Good Wind Threshold</b>: %{x:.1f}%<br><b>Weighted Avg Wind</b>: %{y:.1f}%<extra></extra>",
                )
            )

            fig.update_layout(
                title={
                    "text": "Weighted Average Wind Contribution vs. Good Wind Threshold",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 18, "color": "#1f77b4"},
                },
                xaxis_title="Good Wind Threshold (%)",
                yaxis_title="Weighted Average Wind Contribution (%)",
                template="plotly_white",
                height=500,
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            with st.expander("📋 View Detailed Data"):
                results_df = pd.DataFrame(
                    {
                        "Good Wind Threshold (%)": (results.index * 100).astype(float),
                        "Weighted Avg Wind (%)": (results.values * 100).astype(float),
                    }
                )
                st.dataframe(
                    results_df.style.format(
                        {
                            "Good Wind Threshold (%)": "{:.1f}",
                            "Weighted Avg Wind (%)": "{:.2f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            # Show charge history for optimal threshold
            st.markdown("---")
            st.subheader("🔋 Charging History at Optimal Threshold")

            optimal_threshold = float(results.idxmax())
            st.markdown(
                f"Showing charging behavior using the optimal threshold of **{optimal_threshold * 100:.1f}%**"
            )

            # Get charge history for optimal threshold
            df = (
                self.data_loader._regional_data.groupby(
                    [
                        self.data_loader.forecast_data.date,
                        self.data_loader.forecast_data.tariff_zone,
                    ]
                )
                .mean()
                .eval("WindForecast/ LoadForecast")
            )
            overnight_wind_pc = df.xs("Overnight", 0, 1).clip(0, 1)

            charge_history = get_charge_history(
                overnight_wind_pc,
                min_charge=min_charge,
                good_wind_level=optimal_threshold,
                daily_usage=daily_usage,
                max_charge=max_charge,
                starting_charge=starting_charge,
            )

            # Create figure with secondary y-axis
            fig_history = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    "Battery Charge Level Over Time",
                    "Nightly Wind % and Charging Events",
                ),
                vertical_spacing=0.15,
                row_heights=[0.5, 0.5],
            )

            # Plot 1: Battery charge level
            fig_history.add_trace(
                go.Scatter(
                    x=charge_history.index,
                    y=charge_history["current_charge"] * 100,
                    name="Battery Level",
                    line=dict(color="#2ecc71", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(46, 204, 113, 0.2)",
                    hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Charge</b>: %{y:.1f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Add min/max charge reference lines
            fig_history.add_hline(
                y=min_charge * 100,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Min ({min_charge * 100:.0f}%)",
                row=1,
                col=1,
            )
            fig_history.add_hline(
                y=max_charge * 100,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Max ({max_charge * 100:.0f}%)",
                row=1,
                col=1,
            )

            # Plot 2: Wind percentage and charging events
            fig_history.add_trace(
                go.Scatter(
                    x=charge_history.index,
                    y=charge_history["wind_pc"] * 100,
                    name="Wind %",
                    line=dict(color="#3498db", width=2),
                    hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Wind</b>: %{y:.1f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Add bars for charging events
            charging_events = charge_history[charge_history["charge"] > 0]
            fig_history.add_trace(
                go.Bar(
                    x=charging_events.index,
                    y=charging_events["charge"] * 100,
                    name="Charge Amount",
                    marker_color="#9b59b6",
                    opacity=0.6,
                    yaxis="y3",
                    hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Charged</b>: %{y:.1f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Add good wind threshold line
            fig_history.add_hline(
                y=optimal_threshold * 100,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Good Wind Threshold ({optimal_threshold * 100:.0f}%)",
                row=2,
                col=1,
            )

            # Update axes
            fig_history.update_xaxes(title_text="Date", row=2, col=1)
            fig_history.update_yaxes(
                title_text="Battery Level (%)", row=1, col=1, range=[0, 100]
            )
            fig_history.update_yaxes(
                title_text="Wind Contribution (%)", row=2, col=1, range=[0, 100]
            )

            fig_history.update_layout(
                height=800,
                template="plotly_white",
                showlegend=True,
                hovermode="x unified",
            )

            st.plotly_chart(fig_history, use_container_width=True)

            # Show charging statistics
            col1, col2, col3 = st.columns(3)
            total_charges = (charge_history["charge"] > 0).sum()
            total_energy = charge_history["charge"].sum()
            avg_wind_during_charging = (
                charge_history.loc[charge_history["charge"] > 0, "wind_pc"].mean()
                if total_charges > 0
                else 0
            )

            with col1:
                st.metric(
                    "Total Charging Sessions",
                    f"{total_charges}",
                    help="Number of nights the EV was charged",
                )

            with col2:
                st.metric(
                    "Total Energy Charged",
                    f"{total_energy * 100:.1f}%",
                    help="Total battery capacity charged over the period",
                )

            with col3:
                st.metric(
                    "Avg Wind During Charging",
                    f"{avg_wind_during_charging * 100:.1f}%",
                    help="Average wind contribution during charging sessions",
                )

    def render_price_analysis_tab(self):
        """Render the price analysis tab (placeholder)."""
        st.header("Price Analysis")
        st.info(
            "🚧 Coming soon: Day-ahead and intraday price analysis with historical comparisons"
        )

    def render_footer(self):
        """Render the footer section."""
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.9rem;'>
                Built with Streamlit | Data from SEMO & SEMOpx |
                <a href='https://github.com/eoincondron/semopx-app' target='_blank'>View on GitHub</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def run(self):
        """Main method to run the dashboard."""
        # Render header and sidebar
        self.render_header()
        self.render_sidebar()

        selected_date = self.now.strftime("%Y-%m-%d")

        try:
            with st.spinner(f"Loading forecast data for {selected_date}..."):
                self.data_loader = SEMODataLoader(
                    selected_date,
                    n_days_lookback=90,
                    region=self.selected_region,
                )
        except Exception as e:
            st.error(f"Error loading forecast data: {str(e)}")
            st.info(
                "Please check that the data is available for the selected date and that the cache directory is accessible."
            )
            with st.expander("Show error details"):
                st.exception(e)
            return

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Wind Forecast Analysis",
                "📈 Historical Data",
                "🔋 EV Charging Strategy",
                "💰 Price Analysis (Coming Soon)",
            ]
        )

        # Render tab content
        with tab1:
            self.render_main_tab()

        with tab2:
            self.render_historical_data_tab()

        with tab3:
            self.render_ev_charging_strategy_tab()

        with tab4:
            self.render_price_analysis_tab()

        # Render footer
        self.render_footer()


def main():
    """Entry point for the Streamlit app."""
    dashboard = SEMODashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
