# SEMO Energy Market Analysis

Python client for SEMO (Single Electricity Market Operator) and SEMOpx (Exchange Market) energy market data with price forecasting, EV charging alerts, and interactive dashboard.

## Features

- 📊 **Data Access**: Fetch day-ahead and intraday electricity prices from SEMO/SEMOpx APIs
- 🌬️ **Forecasting**: Wind and load generation forecasts with ML-based price predictions
- ⚡ **EV Charging Alerts**: Smart alerts for optimal EV charging times based on electricity prices
- 📈 **Interactive Dashboard**: Streamlit web app for visualizing market data and forecasts
- 💾 **Caching**: Built-in parquet-based caching for fast data access

## Installation

### Using pip

```bash
# Basic installation
pip install -e .

# With optional features
pip install -e ".[alerts]"      # EV charging alerts
pip install -e ".[streamlit]"   # Interactive dashboard
pip install -e ".[dev]"         # Development tools
pip install -e ".[alerts,streamlit,dev]"  # All features
```

### Using conda

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate semopx-app

# Install the package
pip install -e .
```

## Quick Start

### Python API

```python
from semopx_app.client import SEMOAPIClient
from semopx_app.data_cache import get_day_ahead_prices_single_date
from semopx_app.analysis import wind_percentage_forecast_table
from semopx_app.model import generate_price_prediction

# Initialize client
client = SEMOAPIClient()

# Get day-ahead prices
prices = get_day_ahead_prices_single_date("2024-01-15")

# Get wind percentage forecasts
wind_forecast = wind_percentage_forecast_table("2024-01-15")

# Generate price prediction
prediction, scores = generate_price_prediction(
    prediction_date="2024-01-15",
    days_ahead=1,
    n_training_days=90
)
```

### Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

The dashboard features:
- 📊 Wind percentage forecasts by tariff zone and day of week
- 📈 Summary statistics and best/worst periods
- 💾 CSV export functionality
- 🎨 Clean, responsive UI

### EV Charging Alerts

Run the EV charging alert system:

```bash
semopx-alert
```

Or use it programmatically:

```python
from semopx_app.alert import EVChargingAlertSystem

system = EVChargingAlertSystem(alert_date="2024-01-15")
system.run()
```

## Project Structure

```
semopx-app/
├── app.py                      # Streamlit dashboard application
├── semopx_app/
│   ├── client.py              # API client for SEMO/SEMOpx
│   ├── data_cache.py          # Caching utilities for data fetching
│   ├── analysis.py            # Analysis functions (wind forecasts, etc)
│   ├── model.py               # ML models for price prediction
│   ├── alert.py               # EV charging alert system
│   └── util.py                # Utility functions
├── setup.py                   # Package setup (setuptools)
├── pyproject.toml            # Modern Python packaging config
└── environment.yml           # Conda environment specification
```

## Data Sources

This package interfaces with:

- **SEMO (sem-o.com)**: Balancing market data and same-day exchange data
- **SEMOpx (semopx.com)**: Historical exchange market data (D+1 and older)

## Configuration

### Cache Directory

By default, data is cached to `/Users/eoincondron/ipynb/Energy Data/cache`.

You can modify this by changing `CACHE_PATH` in:
- `semopx_app/util.py`
- `semopx_app/data_cache.py`

### Alert Configuration

Configure EV charging alerts by editing the default config in `semopx_app/alert.py`:

```python
config = {
    "thresholds": {
        "very_low": 30,   # €/MWh
        "low": 50,        # €/MWh
        "moderate": 70    # €/MWh
    },
    "charging": {
        "duration_hours": 4,
        "preferred_start": "23:00",
        "preferred_end": "07:00"
    },
    "alerts": {
        "email_enabled": False,
        "desktop_enabled": True
    }
}
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/eoincondron/semopx-app.git
cd semopx-app

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy semopx_app
```

### Running Tests

```bash
pytest --cov=semopx_app --cov-report=term-missing
```

## API Reference

### SEMOAPIClient

Main client for accessing SEMO and SEMOpx APIs.

```python
client = SEMOAPIClient()

# Get reports for a date range
reports = client.get_reports_date_range(
    start_date="2024-01-01",
    end_date="2024-01-31",
    resource_name="MarketResult_SEM-DA"
)

# Get day-ahead prices
prices = client.get_day_ahead_prices("2024-01-15")

# Get intraday prices
intraday = client.get_intraday_prices("2024-01-15", session_number=1)
```

### Data Caching Functions

```python
from semopx_app.data_cache import (
    get_day_ahead_prices_date_range,
    get_intraday_prices_date_range,
    get_wind_forecast_single_date,
    get_load_forecast_single_date,
    get_combined_forecasts_date_range
)

# Fetch data with automatic caching
prices = get_day_ahead_prices_date_range(
    start_date="2024-01-01",
    end_date="2024-01-31",
    try_cache=True  # Use cache if available
)
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Eoin Condron (econdr@gmail.com)

## Links

- Repository: https://github.com/eoincondron/semopx-app
- Issues: https://github.com/eoincondron/semopx-app/issues
