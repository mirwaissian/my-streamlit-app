#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures

# ----- NOAA API Configuration & Function -----
# Use the environment variable NCEI_API_TOKEN if available, else default
NOAA_API_TOKEN = os.getenv("NCEI_API_TOKEN", "CCSqekrmuGGNtWRrttiTKdVifdRuJpSi")
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
NOAA_DATASET_ID = "GHCND"
NOAA_LOCATION_ID = "FIPS:FI"
NOAA_DATATYPE_ID = ["TAVG", "AWND", "PRCP"]
NOAA_UNITS = "metric"
NOAA_LIMIT = 1000

def fetch_noaa_data(start_date, end_date, token=NOAA_API_TOKEN,
                    datasetid=NOAA_DATASET_ID, location_id=NOAA_LOCATION_ID,
                    datatypeid=NOAA_DATATYPE_ID, units=NOAA_UNITS, limit=NOAA_LIMIT):
    """
    Fetch NOAA weather data over a date range in chunks.
    The data is aggregated (mean per date & datatype), pivoted to wide format,
    and columns are renamed for clarity.
    """
    headers = {"token": token}
    df_noaa = pd.DataFrame()
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=364), end_date)
        params = {
            "datasetid": datasetid,
            "locationid": location_id,
            "startdate": current_start.strftime("%Y-%m-%d"),
            "enddate": current_end.strftime("%Y-%m-%d"),
            "datatypeid": datatypeid,
            "limit": limit,
            "units": units
        }
        response = requests.get(NOAA_BASE_URL, headers=headers, params=params)
        print(f"Fetching NOAA data from {params['startdate']} to {params['enddate']}: Status {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                df_chunk = pd.DataFrame(data["results"])
                df_noaa = pd.concat([df_noaa, df_chunk], ignore_index=True)
            else:
                print(f"No results for period {params['startdate']} to {params['enddate']}")
        else:
            print(f"Error {response.status_code} for period {params['startdate']} to {params['enddate']}")
        current_start = current_end + timedelta(days=1)
    if not df_noaa.empty:
        # Process NOAA data: convert dates, aggregate duplicates, pivot, and rename columns
        df_noaa["date"] = pd.to_datetime(df_noaa["date"])
        df_noaa = df_noaa.groupby(["date", "datatype"])["value"].mean().reset_index()
        df_noaa = df_noaa.pivot(index="date", columns="datatype", values="value").reset_index()
        df_noaa.columns.name = None  # Remove any MultiIndex name
        df_noaa.rename(columns={"TAVG": "temperature_C", "AWND": "wind_speed", "PRCP": "precipitation"}, inplace=True)
    return df_noaa

# ----- Load Electricity Prices Data -----
# (Ensure that "fmi_weather_and_price.csv" is in the current folder)
electricity_file = "fmi_weather_and_price.csv"
if not os.path.exists(electricity_file):
    raise FileNotFoundError(f"File '{electricity_file}' not found.")

df_electricity = pd.read_csv(electricity_file)
df_electricity.rename(columns={"Time": "date"}, inplace=True)
df_electricity["date"] = pd.to_datetime(df_electricity["date"], errors="coerce").dt.date

# Determine the full date range from the electricity dataset
elec_min_date = pd.to_datetime(df_electricity["date"]).min().date()
elec_max_date = pd.to_datetime(df_electricity["date"]).max().date()
print(f"Electricity dataset date range: {elec_min_date} to {elec_max_date}")

# ----- Fetch NOAA Weather Data for the Full Electricity Date Range -----
start_date = datetime(elec_min_date.year, elec_min_date.month, elec_min_date.day)
end_date = datetime(elec_max_date.year, elec_max_date.month, elec_max_date.day)
df_noaa = fetch_noaa_data(start_date, end_date)
if not df_noaa.empty:
    # Convert NOAA dates to plain date format to match electricity data
    df_noaa["date"] = df_noaa["date"].dt.date
    print(f"NOAA dataset date range: {df_noaa['date'].min()} to {df_noaa['date'].max()}")
else:
    print("No NOAA data fetched.")

# ----- Merge Electricity Data with NOAA Weather Data -----
df_merged = pd.merge(df_electricity, df_noaa, on="date", how="left")
print("Merged dataset preview:")
print(df_merged.head())
merged_csv = "merged_weather_electricity.csv"
df_merged.to_csv(merged_csv, index=False)
print(f"Merged dataset saved as '{merged_csv}'.")

# For visualization and further analysis, work with the merged data
df = df_merged.copy()

# =============================
# VISUALIZATIONS & ANALYSIS
# =============================

# 1. Basic Data Inspection
print("First few rows:")
print(df.head())
df.info()
print("Missing values by column:")
print(df.isnull().sum())
print("Summary statistics:")
print(df.describe())

# 2. Electricity Price & Temperature Over Time
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(df["date"]), df["Price"], label="Electricity Price (€)", color="red")
if "temperature_C" in df.columns:
    plt.plot(pd.to_datetime(df["date"]), df["temperature_C"], label="Temperature (°C)", color="blue", alpha=0.6)
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Electricity Price and Temperature Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 3. Scatter Plot: Temperature vs Electricity Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["temperature_C"], y=df["Price"])
plt.xlabel("Temperature (°C)")
plt.ylabel("Electricity Price (€)")
plt.title("Temperature vs Electricity Price")
plt.grid()
plt.show()

# 4. Correlation Matrix (Temperature, Wind, Price)
corr_matrix = df[['temperature_C', 'Wind', 'Price']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 5. Wind Transformation & Scatter Plot
correlation = df[["Wind", "Price"]].corr()
print("Correlation between Wind and Price:")
print(correlation)
df["Wind_log"] = np.log1p(df["Wind"])  # log(1+Wind) transformation
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Wind_log"], y=df["Price"])
plt.xlabel("Log Transformed Wind Speed (m/s)")
plt.ylabel("Electricity Price (€)")
plt.title("Log Transformed Wind Speed vs Electricity Price")
plt.grid()
plt.show()

# ...
# 6. Multiple Regression (OLS) Summary
df_reg = df.dropna(subset=["temperature_C", "Wind", "Price"])
X = df_reg[["temperature_C", "Wind"]]
X = sm.add_constant(X)
y = df_reg["Price"]
model = sm.OLS(y, X).fit()
print("OLS Regression Summary:")
print(model.summary())

# 7. Polynomial Features Addition on Valid Data Subset
# Create a subset of df with non-missing values in the required columns
df_poly = df.dropna(subset=["temperature_C", "Wind", "Price"]).copy()

# Generate polynomial features from temperature_C and Wind
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df_poly[["temperature_C", "Wind"]])

# Create a DataFrame for these polynomial features; ensure the index aligns with df_poly
poly_feature_names = poly.get_feature_names_out(["temperature_C", "Wind"])
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=df_poly.index)

# Concatenate the polynomial features back to the subset DataFrame
df_poly = pd.concat([df_poly, X_poly_df], axis=1)
print("Polynomial features added successfully (subset with no missing temperature, wind, or Price).")
print("First few rows of the subset with polynomial features:")
print(df_poly.head())

# 7b. Polynomial Regression on the Subset
# Prepare the exogenous variables (only the polynomial features) with a constant
X_poly_for_reg = sm.add_constant(df_poly[poly_feature_names])
y_poly_for_reg = df_poly["Price"]

# Check for missing values in X_poly_for_reg (should be 0)
print("Number of missing values in polynomial features:", X_poly_for_reg.isnull().sum().sum())

# Fit the OLS model
model_poly = sm.OLS(y_poly_for_reg, X_poly_for_reg).fit()
print("Polynomial OLS Regression Summary:")
print(model_poly.summary())

# 8. Residuals Diagnostics for the polynomial model
residuals_poly = model_poly.resid
sm.qqplot(residuals_poly, line='s')
plt.title("QQ Plot of Residuals (Polynomial Model)")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(residuals_poly, bins=20, edgecolor='k')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals (Polynomial Model)")
plt.show()

# 9. Check DataFrame Structure
print("DataFrame head after adding polynomial features:")
print(df.head())
print("DataFrame columns:")
print(df.columns)

# 10. Remove Duplicate Columns (if any)
df = df.loc[:, ~df.columns.duplicated()]
print("Columns after removing duplicates:")
print(df.columns)

# 11. Create Lagged Features
df = df.copy()  # Avoid SettingWithCopyWarning
df["temp_lag1"] = df["temperature_C"].astype(float).shift(1)
df["wind_lag1"] = df["Wind"].astype(float).shift(1)
df.dropna(inplace=True)
print("Lagged features added successfully.")
print(df.head())

# 12. Updated Correlation Matrix with Lagged Features
df_numeric = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
plt.title("Updated Correlation Matrix with Lagged Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# 13. Electricity Price Rolling Mean
df["Price_RollingMean"] = df["Price"].rolling(window=30, min_periods=1).mean()
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(df["date"]), df["Price"], label="Daily Prices", color="red", alpha=0.5)
plt.plot(pd.to_datetime(df["date"]), df["Price_RollingMean"], label="30-Day Rolling Mean", color="black")
plt.xlabel("Date")
plt.ylabel("Electricity Price (€)")
plt.title("Electricity Price Trend with 30-Day Rolling Mean")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 14. Distribution of Electricity Prices
plt.figure(figsize=(10, 6))
sns.histplot(df["Price"], bins=50, kde=True, color="red")
plt.xlabel("Electricity Price (€)")
plt.ylabel("Frequency")
plt.title("Distribution of Electricity Prices")
plt.grid()
plt.show()

# 15. Scatter Plot: Temperature vs Electricity Price (again)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["temperature_C"], y=df["Price"], color="blue")
plt.xlabel("Temperature (°C)")
plt.ylabel("Electricity Price (€)")
plt.title("Temperature vs Electricity Price")
plt.grid()
plt.show()

# 16. Scatter Plot: Wind Speed vs Electricity Price
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Wind"], y=df["Price"], color="green")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Electricity Price (€)")
plt.title("Wind Speed vs Electricity Price")
plt.grid()
plt.show()

# 17. Correlation Matrix (Numeric Features Only)
numeric_df = df.select_dtypes(include=["number"])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Numeric Features Only)")
plt.show()

# 18. Log Transformation of Electricity Price & Its Distribution
df["log_Price"] = np.log1p(df["Price"])
plt.figure(figsize=(8, 5))
sns.histplot(df["log_Price"], bins=30, kde=True, color="purple")
plt.xlabel("Log of Electricity Price (€)")
plt.ylabel("Frequency")
plt.title("Distribution of Log-Transformed Electricity Price")
plt.grid()
plt.show()

# 19. Updated Correlation Matrix (Only Numeric Columns)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Updated Correlation Matrix (Only Numeric Columns)")
plt.show()

# 20. Missing Values, Data Types, & Summary Statistics (Printed)
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing Values in Each Column:\n", missing_values)
print("Data Types:\n", df.dtypes)
print("Summary Statistics:\n", df.describe())

# 21. Boxplots for Filtered Data (After Outlier Removal)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.boxplot(y=df["temperature_C"])
plt.title("Boxplot of Temperature (After Outlier Removal)")
plt.subplot(1, 3, 2)
sns.boxplot(y=df["Wind"])
plt.title("Boxplot of Wind Speed (After Outlier Removal)")
plt.subplot(1, 3, 3)
sns.boxplot(y=df["Price"])
plt.title("Boxplot of Price (After Outlier Removal)")
plt.show()

# 22. Interaction Model: Residual Diagnostics
df["temp_wind_interaction"] = df["temperature_C"] * df["Wind"]
# Ensure polynomial terms exist; if not, add them:
if "temperature_C^2" not in df.columns:
    df["temperature_C^2"] = df["temperature_C"] ** 2
if "Wind^2" not in df.columns:
    df["Wind^2"] = df["Wind"] ** 2

X_interaction = df[["temperature_C", "temperature_C^2", "Wind", "Wind^2", "temp_lag1", "wind_lag1", "temp_wind_interaction"]]
y = df["Price"]
X_interaction = sm.add_constant(X_interaction)
model_interaction = sm.OLS(y, X_interaction).fit()
print("Interaction Model Regression Summary:")
print(model_interaction.summary())

plt.figure(figsize=(8, 6))
plt.scatter(model_interaction.fittedvalues, model_interaction.resid, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

sm.qqplot(model_interaction.resid, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(model_interaction.resid, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

shapiro_test = stats.shapiro(model_interaction.resid)
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")

# (Optional: Further ARIMA modeling or other analyses could follow here.)