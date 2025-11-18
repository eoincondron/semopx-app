"""
SEMO Energy Market Analysis Dashboard

A Streamlit app for visualizing Irish energy market data including:
- Wind Contribution forecasts by time of day and tariff zone
- Day-ahead and intraday electricity prices
- Load and wind generation forecasts
"""

from typing import Optional
from datetime import datetime, timedelta
from warnings import warn

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from semopx_app.util import DayDelta, date_to_str
from semopx_app import data_cache


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

    def wind_percentage_forecast_table(self) -> pd.DataFrame:
        """
        Generate a table of Wind Contribution forecasts categorized by day of the week and tariff zone.

        Parameters:
            combined_forecasts (pd.DataFrame): DataFrame containing combined load and wind forecasts.

        Returns:
            A DataFrame containing the mean Wind Contribution forecasts for three days
            categorized by day of the week and tariff zone (DayTime, Peak, Evening, Overnight).

        """
        df = self.forecast_data.loc[self.date :].copy()

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


class SEMODashboard:
    """Main dashboard class for SEMO Energy Market Analysis."""

    REGION_MAP = {
        "All Ireland": "Total",
        "Republic of Ireland": "ROI",
        "Northern Ireland": "NI",
    }

    def __init__(self):
        """Initialize the dashboard with page configuration."""
        st.set_page_config(
            page_title="SEMO Energy Dashboard",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self.selected_date: str
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
        """Render the sidebar with date selection and info."""
        st.sidebar.header("Settings")
        st.sidebar.markdown("---")

        # Date selector
        st.sidebar.subheader("📅 Date Selection")
        default_date = datetime.now()
        self.selected_date = st.sidebar.date_input(
            "Forecast Date",
            value=default_date,
            min_value=datetime(2020, 1, 1),
            max_value=datetime.now() + timedelta(days=30),
            help="Select the date for which to generate forecasts",
        ).strftime("%Y-%m-%d")

        # Region selector
        st.sidebar.subheader("🌍 Region Selection")
        region_options = ["All Ireland", "Republic of Ireland", "Northern Ireland"]
        self.selected_region = st.sidebar.selectbox(
            "Region",
            options=region_options,
            index=0,  # Default to "All Ireland"
            help="Select the region for wind and load forecasts",
        )

        # Historical data settings
        st.sidebar.subheader("📈 Historical Data Settings")
        self.n_days_lookback = st.sidebar.number_input(
            "Days to Look Back",
            min_value=1,
            max_value=365,
            value=90,
            step=1,
            help="Number of days of historical data to display",
        )

        frequency_options = ["day", "week", "month"]
        self.sampling_frequency = st.sidebar.selectbox(
            "Sampling Frequency",
            options=frequency_options,
            index=1,  # Default to "day"
            help="Frequency for sampling historical data",
        )
        self._sampling_frequency_pandas = {"day": "d", "week": "W", "month": "m"}.get(
            self.sampling_frequency
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
        rename_map = {
            "WindForecast": "Wind Generation (MW)",
            "LoadForecast": "Total Demand (MW)",
            "WindPc": "Wind Contribution",
        }
        df = df.rename(columns=rename_map).reset_index()
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

    def render_main_tab(self):
        """
        Render the wind forecast table with heatmap styling.

        Args:
            date_str: Date string in YYYY-MM-DD format
            region_name: User-friendly region name
        """
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

        df = self.data_loader.wind_percentage_forecast_table().reset_index()

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

        # Display styled table
        st.dataframe(
            styler,
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
        """
        Render summary statistics for the forecast data.

        Args:
            df: Forecast dataframe
            date_str: Date string for CSV export
        """
        st.subheader("📈 Summary Statistics")
        df = self.data_loader.wind_percentage_forecast_table()

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
        self._render_best_worst_periods(max_wind_pct, min_wind_pct)

    def _render_best_worst_periods(self, max_wind_pct: float, min_wind_pct: float):
        """
        Render best and worst wind periods.

        Args:
            df: Forecast dataframe
            max_wind_pct: Maximum Wind Contribution
            min_wind_pct: Minimum Wind Contribution
        """
        st.markdown("---")
        col_best, col_worst = st.columns(2)
        df = self.data_loader.wind_percentage_forecast_table()

        with col_best:
            st.success("**🌬️ Best Wind Period**")
            best_day, best_time = df["WindPc"].idxmax()  # type: ignore
            st.write(f"**Day:** {best_day}")
            st.write(f"**Time:** {best_time}")
            st.write(f"**Wind %:** {max_wind_pct:.1f}%")

        with col_worst:
            st.warning("**🔻 Lowest Wind Period**")
            worst_day, worst_time = df["WindPc"].idxmin()  # type: ignore
            st.write(f"**Day:** {worst_day}")
            st.write(f"**Time:** {worst_time}")
            st.write(f"**Wind %:** {min_wind_pct:.1f}%")

    def _render_download_button(self):
        """
        Render CSV download button.

        Args:
            df: Forecast dataframe
            date_str: Date string for filename
        """
        st.markdown("---")
        csv = self.data_loader.forecast_data.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"wind_forecast_{self.selected_date}.csv",
            mime="text/csv",
        )

    def render_historical_data_tab(
        self,
    ):
        """
        Render the historical data tab with wind and load trends.

        Args:
            region_name: User-friendly region name
            n_days_lookback: Number of days to look back
            frequency: Grouping frequency ('hour', 'day', 'week')
        """
        st.header(f"📊 Historical Data Analysis - {self.selected_region}")

        n_days = (
            self.data_loader.forecast_data.loc[: self.selected_date]
            .index.floor("D")
            .nunique()
        )

        if n_days < self.n_days_lookback:
            st.warning(
                f"Not all requested dates are available. Earliest date is {self.data_loader.forecast_data.index[0].date()}"
            )

        st.markdown(
            f"""
        Showing historical wind and load data for the past **{n_days} days**,
        grouped by **{self.sampling_frequency}**.
        """
        )
        fig = self.data_loader.wind_load_forecast_plot(
            n_days_lookback=self.n_days_lookback,
            sampling_frequency=self._sampling_frequency_pandas,
        )
        st.plotly_chart(fig, use_container_width=True)

        self._render_monthly_averages()
        self._render_tariff_zone_averages()

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

        # # Download button

        # st.markdown("---")
        # csv = df_resampled.to_csv()
        # st.download_button(
        #     label="📥 Download Historical Data as CSV",
        #     data=csv,
        #     file_name=f"historical_data_{.strftime('%Y%m%start_dated')}_{end_date.strftime('%Y%m%d')}.csv",
        #     mime="text/csv",
        # )

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

        # Get selected date (set default if None)
        if self.selected_date is None:
            self.selected_date = datetime.now().strftime("%Y%m%d")

        try:
            with st.spinner(f"Loading forecast data for {self.selected_date}..."):
                self.data_loader = SEMODataLoader(
                    self.selected_date,
                    n_days_lookback=self.n_days_lookback,
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
        tab1, tab2, tab3 = st.tabs(
            [
                "📊 Wind Forecast Analysis",
                "📈 Historical Data",
                "💰 Price Analysis (Coming Soon)",
            ]
        )

        # Render tab content
        with tab1:
            self.render_main_tab()

        with tab2:
            self.render_historical_data_tab()

        with tab3:
            self.render_price_analysis_tab()

        # Render footer
        self.render_footer()


def main():
    """Entry point for the Streamlit app."""
    dashboard = SEMODashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
