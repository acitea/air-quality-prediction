# Air Quality Prediction 🌍

Predicts US AQI PM2.5 air quality levels using environmental data retrieved from [NEA](https://data.gov.sg). The project features automated daily training via GitHub Actions and real-time results visualization through GitHub Pages.

## Installation

### Prerequisites

- Python 3.12+
- [UV package manager](https://github.com/astral-sh/uv)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/acitea/air-quality-prediction.git
cd air-quality-prediction
```

2. Install UV (if not already installed):
```bash
pip install uv
```
OR refer to [Official Docs](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)

3. Install dependencies:
```bash
uv sync
```

## Workflow

### Initial Data Scraping
-  ```*_scraper.py``` scripts to pull data and mass historical data downloaded as dataset from [NEA](https://data.gov.sg) 
-  ```*_processor.py``` scripts to process data from the previous step
-  ```merge_winds.py``` script to merge data
- ```notebooks/3-data-processing.ipynb``` to transform and perform backfilling

### Training the Model

Run the training pipeline:
```bash
uv run -m src/train.py
```

### View Results

After training, check the generated files:
- `air_quality_model_/images/feature_importance.png`
- `air_quality_model_/images/pm25_forecast.png`
- `air_quality_model_/images/pm25_hindcast.png`
- `air_quality_model_/images/pm25_hindcast_1day.png`

### GitHub Pages Dashboard

Visit the GitHub Pages site to see latest predictions and model performance: [View Dashboard](https://acitea.github.io/air-quality-prediction/)

## Project Structure

```
air-quality-prediction/
├── .github/
│   └── workflows/
│       ├── ingest.yml          # Daily data ingestion workflow
│       ├── train.yml           # Daily training workflow
│       └── deploy-pages.yml    # GitHub Pages deployment
├── notebooks/
│       └── *.ipynb             # Various notebooks for eda and development
├── src/
│   └── uitls/
│       └── *.py                # Short scripts for general utils
│   ├── __init__.py             # Package entry point
│   ├── daily_ingestion.py             # Main python script to process daily ingestion
│   ├── <other files>.py        # Other scripts used
│   └── train.py                # Training pipeline
├── data/                       # Data directory
├── air_quality_model/          # Model + Outputs
├── docs/                       # GitHub Pages site
│   ├── index.html              # Dashboard HTML
│   └── outputs/                # Copied outputs for web display
├── pyproject.toml              # UV project configuration
├── uv.lock                     # UV project configuration
└── README.md                   # This file
```
