from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semopx-app",
    version="0.1.0",
    author="Eoin Condron",
    author_email="econdr@gmail.com",
    description="Python client for SEMO and SEMOpx energy market data with price forecasting and EV charging alerts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eoincondron/semopx-app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "lightgbm>=4.0.0",
        "tqdm>=4.65.0",
        "groupby-lib>=0.1.0",
        "pyarrow>=12.0.0",  # For parquet file support
    ],
    extras_require={
        "alerts": [
            "plyer>=2.1.0",  # Cross-platform desktop notifications
        ],
        "streamlit": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",  # For interactive charts
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "semopx-alert=semopx_app.alert:main",
        ],
    },
    package_data={
        "semopx_app": ["*.css"],
    },
    include_package_data=True,
)
