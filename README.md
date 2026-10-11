# Household Electricity Consumption Forecasting Using Machine Learning

## Overview
This project develops an end-to-end predictive analytics model to forecast household electricity consumption one hour ahead using historical electricity measurements and time-based features. It follows the Knowledge Discovery in Databases (KDD) process, from data understanding and preprocessing to model development, evaluation, and interpretation.

## Problem Statement
Household electricity usage varies with time and consumption patterns. The objective is to build and compare machine learning regression models that predict the next hour's mean active power consumption using historical measurements and engineered time-series features.

## Dataset
**Name:** Individual Household Electric Power Consumption  
**Source:** UCI Machine Learning Repository  
**Dataset page:** https://archive.ics.uci.edu/dataset/235/individualhouseholdelectricpowerconsumption

The dataset contains minute-level household electricity measurements from December 2006 to November 2010. The notebook aggregates the readings into hourly observations and uses `Global_active_power` as the prediction target.

The notebook attempts to download the dataset automatically when it runs. If the download fails, download the dataset from the official UCI page, extract `household_power_consumption.txt`, and place it in a folder named `data/` next to the notebook.

> Dataset note: The data represents a single household and a historical period. Results should not be assumed to generalize to all households or current consumption patterns.

## Objectives
- Understand and inspect the household electricity dataset.
- Check missing values, data types, duplicates, and date/time coverage.
- Aggregate minute-level readings into hourly observations.
- Explore consumption patterns by time and day.
- Create time-based, lag, and rolling-window features.
- Compare regression models using a chronological train/validation/test split.
- Evaluate predictions and interpret feature importance.

## Methodology / KDD Workflow
1. **Problem definition:** Forecast the next hour's mean active power consumption.
2. **Data selection and understanding:** Load the UCI dataset and inspect its structure and quality.
3. **Preprocessing:** Parse timestamps, convert invalid markers to missing values, and aggregate observations hourly.
4. **Exploratory data analysis:** Visualize electricity consumption over time and compare average usage by hour and day of week.
5. **Feature engineering:** Create calendar features, lag features, and rolling statistics from historical consumption.
6. **Model development:** Compare a mean baseline, Linear Regression, Ridge Regression, Random Forest, Extra Trees, and Hist Gradient Boosting.
7. **Evaluation:** Select a model based on validation RMSE, then evaluate it on a held-out future period using MAE, RMSE, MAPE (excluding near-zero actual values), and R².
8. **Interpretation:** Plot actual versus predicted values, inspect residuals, and calculate permutation feature importance.

The data is split chronologically rather than randomly to reduce the risk of training on future observations.

## Models
- Baseline: `DummyRegressor` using the training mean
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Hist Gradient Boosting Regressor

## Evaluation Metrics
- **MAE (Mean Absolute Error):** Average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more strongly.
- **MAPE (Mean Absolute Percentage Error):** Reported only for actual values greater than 0.05 kW to avoid instability near zero.
- **R² (Coefficient of Determination):** Measures the proportion of target variance explained by the model.

## Results
In the current notebook run, **Hist Gradient Boosting** was selected using validation RMSE.

| Test metric | Value |
|---|---:|
| MAE | 0.3958 |
| RMSE | 0.5455 |
| MAPE (actual > 0.05 kW) | 57.1879% |
| R² | 0.4088 |

These are results from the current run and may vary slightly across environments or library versions. The MAPE value should be interpreted carefully because percentage errors can become large when actual consumption is low.

## Repository Structure
```text
.
├── Household_Electricity_Consumption_Forecasting.ipynb
├── README.md
├── data/
│   └── household_power_consumption.txt   # downloaded dataset; optional if notebook downloads it
└── outputs/
    ├── validation_model_comparison.csv
    ├── heldout_test_metrics.csv
    ├── test_predictions.csv
    └── permutation_feature_importance.csv
```

The `data/` and `outputs/` folders are created when the notebook is run. You may choose not to upload the raw dataset file if the repository would become too large; keep the official dataset link and download instructions in this README.

## How to Run
### Option 1: Google Colab
1. Upload `Household_Electricity_Consumption_Forecasting.ipynb` to Google Colab.
2. Run the cells from top to bottom.
3. Ensure the runtime has internet access so the dataset can be downloaded.

### Option 2: Local Jupyter
1. Install Python 3 and Jupyter Notebook or JupyterLab.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Household_Electricity_Consumption_Forecasting.ipynb
   ```
4. Run the cells from top to bottom. The notebook downloads the dataset into `data/` if it is not already present.

## Limitations
- The dataset represents one household, not a representative sample of all households.
- Data is historical (2006–2010), so present-day usage patterns may differ.
- The current model uses historical consumption and time-based features, without external weather, occupancy, or tariff data.
- Prediction errors remain significant; the model is an academic forecasting experiment, not a production energy-management system.
- The final next-hour estimate printed by the notebook is based on the last timestamp in the historical dataset, not a live forecast.

## Future Improvements
- Compare against a persistence baseline.
- Tune model hyperparameters and evaluate additional lag features.
- Investigate error patterns during peak and low-consumption periods.
- Incorporate weather, tariff, or occupancy data if available.
- Test on data from additional households before making broader claims.

## References
1. Hebrail, G., & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54
2. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/235/individualhouseholdelectricpowerconsumption

## License
The dataset is provided by the UCI Machine Learning Repository under the terms stated on its official dataset page. Add a repository license only if you intend to license your own code; verify dataset redistribution terms before committing the raw data.
