"""
SEMO Energy Market Analysis Dashboard

A Streamlit app for visualizing Irish energy market data including:
- Wind percentage forecasts by time of day and tariff zone
- Day-ahead and intraday electricity prices
- Load and wind generation forecasts
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional, Union

from semopx_app.analysis import wind_percentage_forecast_table


class SEMODashboard:
    """Main dashboard class for SEMO Energy Market Analysis."""

    def __init__(self):
        """Initialize the dashboard with page configuration."""
        st.set_page_config(
            page_title="SEMO Energy Dashboard",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self.selected_date: Optional[Union[datetime, date]] = None
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

    def _style_forecast_dataframe(self, df_data: pd.DataFrame) -> pd.io.formats.style.Styler:
        """
        Apply heatmap styling to the forecast dataframe.

        Args:
            df_data: DataFrame with forecast data

        Returns:
            Styled DataFrame
        """
        styler = df_data.style

        for col_name, color_map, num_format in [
            ("Wind Generation (MW)", "Greens", "{:,.0f}"),
            ("Total Load (MW)", "Blues", "{:,.0f}"),
            ("Wind %", "RdYlGn", "{:.1%}"),
        ]:

            # Apply background gradient to Wind Generation
            styler = styler.background_gradient(
                subset=[col_name],
                cmap=color_map,
                vmin=df_data[col_name].min(),
                vmax=df_data[col_name].max(),
            ).format({col_name: num_format})

        return styler

    def _prepare_forecast_display_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare forecast data for display by formatting and blanking duplicate days.

        Args:
            df: Raw forecast dataframe with multi-index

        Returns:
            Formatted DataFrame ready for display
        """
        # Reset index and convert to strings
        df_styled = df.copy().reset_index()
        for col in df.index.names:
            df_styled[col] = df_styled[col].astype(str)

        # Rename columns
        rename_map = {
            "AggregatedWind": "Wind Generation (MW)",
            "LoadTotal": "Total Load (MW)",
            "WindPc": "Wind %",
        }
        df_styled = df_styled.rename(columns=rename_map)

        # Blank duplicate day names for visual grouping
        df_styled["Day"] = df_styled["Day"].mask(df_styled["Day"].duplicated(), "")

        return df_styled

    def render_wind_forecast_table(self, date_str: str):
        """
        Render the wind forecast table with heatmap styling.

        Args:
            date_str: Date string in YYYY-MM-DD format
        """
        st.header("Wind Percentage Forecast by Tariff Zone")

        st.markdown(
            """
        This table shows the forecasted wind generation as a percentage of total load,
        broken down by day of the week and tariff zone for the next 3 days.

        **Tariff Zones:**
        - **Overnight**: 23:00 - 08:00
        - **DayTime**: 08:00 - 17:00
        - **Peak**: 17:00 - 19:00
        - **Evening**: 19:00 - 23:00
        """
        )

        with st.spinner(f"Loading forecast data for {date_str}..."):
            try:
                # Get forecast data
                forecasts = wind_percentage_forecast_table(date_str)

            except Exception as e:
                st.error(f"Error loading forecast data: {str(e)}")
                st.info(
                    "Please check that the data is available for the selected date and that the cache directory is accessible."
                )
                with st.expander("Show error details"):
                    st.exception(e)
                return

            # Prepare display data
            df_display = self._prepare_forecast_display_data(forecasts)

            # Display styled table
            st.dataframe(
                self._style_forecast_dataframe(df_display),
                use_container_width=False,
                height=400,
                column_config={
                    "Day": st.column_config.TextColumn(
                        "Day",
                        width="medium",
                    ),
                },
            )

            # Render summary statistics
            self.render_summary_statistics(forecasts, date_str)

    def render_summary_statistics(self, df: pd.DataFrame, date_str: str):
        """
        Render summary statistics for the forecast data.

        Args:
            df: Forecast dataframe
            date_str: Date string for CSV export
        """
        st.subheader("📈 Summary Statistics")

        col1, col2, col3 = st.columns(3)

        # Average Wind %
        with col1:
            avg_wind_pct = df["WindPc"].mean() * 100
            st.metric(
                label="Average Wind %",
                value=f"{avg_wind_pct:.1f}%",
                help="Average wind percentage across all time periods",
            )

        # Maximum Wind %
        with col2:
            max_wind_pct = df["WindPc"].max() * 100
            st.metric(
                label="Maximum Wind %",
                value=f"{max_wind_pct:.1f}%",
                help="Highest wind percentage",
            )

        # Minimum Wind %
        with col3:
            min_wind_pct = df["WindPc"].min() * 100
            st.metric(
                label="Minimum Wind %",
                value=f"{min_wind_pct:.1f}%",
                help="Lowest wind percentage",
            )

        # Best and worst periods
        self._render_best_worst_periods(df, max_wind_pct, min_wind_pct)

        # Download button
        self._render_download_button(df, date_str)

    def _render_best_worst_periods(
        self, df: pd.DataFrame, max_wind_pct: float, min_wind_pct: float
    ):
        """
        Render best and worst wind periods.

        Args:
            df: Forecast dataframe
            max_wind_pct: Maximum wind percentage
            min_wind_pct: Minimum wind percentage
        """
        st.markdown("---")
        col_best, col_worst = st.columns(2)

        with col_best:
            st.success("**🌬️ Best Wind Period**")
            best_day, best_time = df["WindPc"].idxmax()
            st.write(f"**Day:** {best_day}")
            st.write(f"**Time:** {best_time}")
            st.write(f"**Wind %:** {max_wind_pct:.1f}%")

        with col_worst:
            st.warning("**🔻 Lowest Wind Period**")
            worst_day, worst_time = df["WindPc"].idxmin()
            st.write(f"**Day:** {worst_day}")
            st.write(f"**Time:** {worst_time}")
            st.write(f"**Wind %:** {min_wind_pct:.1f}%")

    def _render_download_button(self, df: pd.DataFrame, date_str: str):
        """
        Render CSV download button.

        Args:
            df: Forecast dataframe
            date_str: Date string for filename
        """
        st.markdown("---")
        csv = df.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"wind_forecast_{date_str}.csv",
            mime="text/csv",
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

        # Get selected date (set default if None)
        if self.selected_date is None:
            self.selected_date = datetime.now().date()

        date_str = self.selected_date.strftime("%Y-%m-%d")

        # Create tabs
        tab1, tab2 = st.tabs(
            ["📊 Wind Forecast Analysis", "📈 Price Analysis (Coming Soon)"]
        )

        # Render tab content
        with tab1:
            self.render_wind_forecast_table(date_str)

        with tab2:
            self.render_price_analysis_tab()

        # Render footer
        self.render_footer()


def main():
    """Entry point for the Streamlit app."""
    dashboard = SEMODashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
