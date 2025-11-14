import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Literal, Optional

import pandas as pd
import requests

from .util import DayDelta, date_to_str, convert_timestamps


class SEMOAPIClient:
    """
    Complete client for SEMO (Balancing Market) and SEMOpx (Exchange Market) data.

    Supports:
    - Same-day market results (using ExcludeDelayedPublication=0)
    - Historical market data (D+1 and older)
    - Both sem-o.com and semopx.com endpoints
    """

    def __init__(self):
        # SEMO (sem-o.com) - Balancing market AND same-day exchange data
        self.semo_api_url = "https://reports.sem-o.com/api/v1/documents/static-reports"
        self.semo_download_url = "https://reports.sem-o.com/api/v1/documents/"

        # SEMOpx (semopx.com) - Historical exchange data only (D+1)
        self.semopx_api_url = (
            "https://reports.semopx.com/api/v1/documents/static-reports"
        )
        self.semopx_download_url = "https://reports.semopx.com/documents/"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "SEMO-API-Client/1.0",
                "Accept": "application/json",
            }
        )

    def get_reports_single_page(
        self,
        page: int = 1,
        dpug_id: Optional[str] = None,
        report_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        date_filter: Optional[str] = None,
        page_size: int = 500,
        exclude_delayed: bool = False,
        use_semopx: bool = False,
    ) -> List[Dict]:
        """
        Query the API for reports.

        Args:
            page: Page number (0-indexed)
            dpug_id: Report identifier (e.g., 'EA-001')
            report_name: Report name
            resource_name: Pattern to match in ResourceName
            date_filter: Date filter (e.g., '>=2025-10-01')
            page_size: Results per page (max 500)
            exclude_delayed: If True, excludes same-day data. Set to False to get today's data!
            use_semopx: Use SEMOpx endpoint instead of SEMO

        Returns:
            API response dictionary
        """
        base_url = self.semopx_api_url if use_semopx else self.semo_api_url

        params = {
            "page": page,
            "page_size": page_size,
            "sort_by": "Date",
            "order_by": "DESC",
            "ExcludeDelayedPublication": 1 if exclude_delayed else 0,  # KEY PARAMETER!
        }

        if dpug_id:
            params["DPuG_ID"] = dpug_id
        if report_name:
            params["ReportName"] = report_name
        if resource_name:
            params["ResourceName"] = resource_name
        if date_filter:
            params["Date"] = date_filter

        response = self.session.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()["items"]

        total = len(data)
        print(f"Found {total} items (page {page})")

        return data

    def get_reports_multi_page(
        self,
        max_pages: Optional[int] = None,
        dpug_id: Optional[str] = None,
        report_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        date_filter: Optional[str] = None,
        page_size: int = 500,
        exclude_delayed: bool = False,
        use_semopx: bool = False,
    ) -> pd.DataFrame:
        """
        Get all reports with automatic pagination.

        Args:
            max_pages: Maximum number of pages to retrieve (None for all)
            dpug_id: Report identifier filter
            report_name: Report name filter
            resource_name: Resource name pattern filter
            date_filter: Date filter string (e.g., '>=2025-10-01')
            page_size: Results per page (max 500)
            exclude_delayed: Exclude same-day data if True
            use_semopx: Use SEMOpx endpoint instead of SEMO

        Returns:
            DataFrame containing all reports from all pages
        """
        kwargs = locals().copy()
        del kwargs["self"], kwargs["max_pages"]

        all_items = []
        page = 1

        while True:
            result = self.get_reports_single_page(page=page, **kwargs)
            if not result:
                break

            result = self.reports_to_dataframe(result)
            result["page_number"] = page

            all_items.append(result)
            page += 1
            if max_pages and page >= max_pages + 1:
                break
            time.sleep(0.3)

        print(f"Retrieved {len(all_items)} total reports from {page} page(s)")

        if not all_items:
            return pd.DataFrame()

        all_items = pd.concat(all_items)

        return all_items

    def get_reports_date_range(
        self,
        start_date=str | pd.Timestamp,
        end_date: Optional[str | pd.Timestamp] = None,
        dpug_id: Optional[str] = None,
        report_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        page_size: int = 500,
        exclude_delayed: bool = False,
        use_semopx: bool = False,
    ):
        """
        Get reports for a specific date range.

        Args:
            start_date: Start date for reports (string or Timestamp)
            end_date: End date for reports (defaults to start_date if None)
            dpug_id: Report identifier filter
            report_name: Report name filter
            resource_name: Resource name pattern filter
            page_size: Results per page (max 500)
            exclude_delayed: Exclude same-day data if True (must be False for today's data)
            use_semopx: Use SEMOpx endpoint instead of SEMO

        Returns:
            DataFrame containing reports within the date range
        """
        if end_date is None:
            end_date = start_date

        start_date = date_to_str(start_date, format="%Y-%m-%d")
        end_date = date_to_str(start_date, format="%Y-%m-%d")
        upper_bound = end_date + DayDelta()
        reports = self.get_reports_multi_page(
            date_filter=f">={start_date}<{upper_bound}",
            dpug_id=dpug_id,
            report_name=report_name,
            resource_name=resource_name,
            exclude_delayed=exclude_delayed,  # CRITICAL: Must be False to get today's data!
            use_semopx=use_semopx,
            page_size=page_size,
        )

        return reports

    def reports_to_dataframe(self, reports: List[Dict]) -> pd.DataFrame:
        """
        Convert API response to DataFrame.

        Args:
            reports: List of report dictionaries from API response

        Returns:
            DataFrame with reports data, including download URLs and converted timestamps
        """
        if not reports:
            return pd.DataFrame()

        df = pd.DataFrame(reports)

        # Add download URLs
        if "ResourceName" in df.columns:
            # For SEMO reports, use sem-o.com
            df["download_url_semo"] = df["ResourceName"].apply(
                lambda x: f"{self.semo_download_url}{x}" if x else None
            )
            # For SEMOpx reports, use semopx.com
            df["download_url_semopx"] = df["ResourceName"].apply(
                lambda x: f"{self.semopx_download_url}{x}" if x else None
            )

        # Convert dates
        for col in ["Date", "PublishTime"]:
            if col in df.columns:
                df[col] = convert_timestamps(df[col])

        if "PublishTime" in df.columns:
            df = df.sort_values("PublishTime")

        return df

    def download_and_parse_auction_data(
        self, resource_name: str, use_semopx: bool = False
    ) -> pd.DataFrame:
        """
        Download and parse a CSV file.

        Args:
            resource_name: Filename from API
            use_semopx: Use semopx.com endpoint (for historical data)

        Returns:
            DataFrame if successful
        """
        base_url = self.semopx_download_url if use_semopx else self.semo_download_url
        url = f"{base_url}{resource_name}"

        print(f"Downloading {resource_name}...")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        response = self.session.get(url)
        json = response.json()

        def parse_rows(rows):
            market = rows[0][1]
            cols = {}
            for i in range(1, 13, 3):
                times, values = rows[i + 1], rows[i + 2]
                name = "_".join(map(str, rows[i])).replace(" ", "_")
                cols[name] = pd.Series(values, pd.to_datetime(times))

            return market, pd.DataFrame(cols)

        df = pd.concat(dict(map(parse_rows, json["rows"])), names=["Market", "EndTime"])

        print(f"Downloaded: {len(df)} rows")
        return df

    def _get_auction_prices(
        self, date, session: Literal["DA", "IDA1", "IDA2", "IDA3"], market: str = "ROI"
    ) -> pd.DataFrame:
        reports = self.get_reports_date_range(
            date,
            use_semopx=True,
            dpug_id="EA-001",
            resource_name=f"MarketResult_SEM-{session}",
            exclude_delayed=False,
            page_size=500,
        )
        if reports.empty:
            raise ValueError(f"No reports found for {date} session {session}")

        reports = reports.sort_values("PublishTime")
        assert len(reports) == 1, f"Expected 1 report, got {len(reports)}"
        resource_name, publish_time = reports.iloc[0][["ResourceName", "PublishTime"]]
        df = self.download_and_parse_auction_data(resource_name)
        df["PublishTime"] = publish_time
        df = df.loc[f"{market}-{session}"]

        return df

    def get_day_ahead_prices(self, date):
        """
        Get day-ahead auction prices for a specific date.

        Args:
            date: Date string or Timestamp for the target date

        Returns:
            DataFrame with day-ahead prices for ROI market
        """
        return self._get_auction_prices(date, session="DA", market="ROI")

    def get_intraday_prices(
        self, date, session_number: Literal[1, 2, 3], market: str = "ROI"
    ):
        """
        Get intraday auction prices for a specific session.

        Args:
            date: Date string or Timestamp for the target date
            session_number: Intraday session number (1, 2, or 3)
            market: Market identifier (default "ROI")

        Returns:
            DataFrame with intraday prices for the specified session
        """
        return self._get_auction_prices(
            date, session=f"IDA{session_number}", market=market
        )

    def download_XML(
        self, resource_name, as_tree: bool = False
    ) -> pd.DataFrame | ET.Element:
        """
        Download and parse an XML resource.

        Args:
            resource_name: Name of the XML resource to download
            as_tree: If True, return ElementTree; if False, return DataFrame

        Returns:
            ElementTree or DataFrame depending on as_tree parameter
        """
        response = self.session.get(f"{self.semopx_download_url}/{resource_name}")
        tree = ET.fromstring(response.text)

        if as_tree:
            return tree

        elements = list(tree)

        if len(elements[0].attrib) > 1:
            df = pd.DataFrame([element.attrib for element in elements])
        else:
            df = pd.DataFrame(
                [pd.Series(element_tags_to_text(element)) for element in elements]
            )

        for key, col in df.items():
            key = str(key)
            if "Time" in key or "Date" in key:
                df[key] = convert_timestamps(col)
            else:
                try:
                    df[key] = pd.to_numeric(col)
                except TypeError:
                    continue

        df["PublishTime"] = pd.Timestamp(tree.attrib["PublishTime"], tz="UTC")

        return df


def element_tags_to_text(element):
    """
    Convert XML element to dictionary mapping tag names to text values.

    Args:
        element: XML Element to convert

    Returns:
        Dictionary mapping tag names to text content
    """
    return {e.tag: e.text for e in element}
