#!/usr/bin/env python3
import streamlit as st

# 1) Make sure set_page_config is the FIRST Streamlit call
st.set_page_config(page_title="A3 Analysis", layout="wide")

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
import re
import io
import pyarrow as pa  # For potential Arrow conversions

# -----------------------------------------------------------------------------
# Helper Function: Highlight Significant P-values
# -----------------------------------------------------------------------------
def highlight_pval(val):
    """
    Returns a background style if a p-value is significant (i.e., less than 0.05).
    """
    if isinstance(val, float) and val < 0.05:
        return 'background-color: rgba(0, 255, 0, 0.2)'
    return ''

# -----------------------------------------------------------------------------
# Helper Function: Safe DataFrame Display
# -----------------------------------------------------------------------------
def safe_dataframe_display(df, columns=None, num_rows=5):
    """
    Safely displays a DataFrame in Streamlit by:
      1) Optionally selecting a subset of columns,
      2) Limiting the number of rows,
      3) Attempting to convert object columns to numeric (with clamping of extremely small values),
      4) Formatting numeric columns to fixed decimal notation,
      5) Finally displaying the DataFrame with a fallback to an HTML table if necessary.
    """
    display_df = df.copy()

    # Optionally select columns
    if columns is not None:
        valid_columns = [col for col in columns if col in display_df.columns]
        if valid_columns:
            display_df = display_df[valid_columns]

    # Limit to the desired number of rows
    display_df = display_df.head(num_rows)

    def clamp_float_str(value_str):
        """
        Attempt to convert a string to a float. If the value is extremely small,
        return 0.0 to avoid issues in conversion.
        """
        try:
            val = float(value_str)
            if abs(val) < 1e-308:
                return 0.0
            if abs(val) > 1e308:
                return float('inf') if val > 0 else float('-inf')
            return val
        except:
            return None

    # For object-type columns, apply clamping and conversion
    for col in display_df.columns:
        if pd.api.types.is_object_dtype(display_df[col]):
            display_df[col] = display_df[col].apply(clamp_float_str)
            # Attempt to convert any remaining strings to numeric
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

    # Format numeric columns to fixed decimal strings
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            try:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
            except Exception:
                display_df[col] = display_df[col].astype(str)
        else:
            display_df[col] = display_df[col].astype(str)

    # Display the DataFrame in Streamlit, fallback to HTML if needed
    try:
        st.dataframe(display_df)
    except Exception as e:
        st.error(f"Error displaying dataframe: {str(e)}")
        st.write("Falling back to HTML table display:")
        st.write(display_df.to_html(index=False), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper Function: Display OLS Regression Summary as Tables
# -----------------------------------------------------------------------------
def display_ols_summary_as_tables(model):
    """
    Displays an OLS regression summary in Streamlit as formatted tables,
    including model information, coefficients (with significant p-values highlighted),
    and residual diagnostics.
    """
    st.subheader("Model Information")
    model_info = {
        "Dependent Variable": "Price",
        "R-squared": f"{model.rsquared:.3f}",
        "Adjusted R-squared": f"{model.rsquared_adj:.3f}",
        "F-statistic": f"{model.fvalue:.2f}",
        "Prob (F-statistic)": f"{model.f_pvalue:.4g}",
        "Log-Likelihood": f"{model.llf:.1f}",
        "AIC": f"{model.aic:.1f}",
        "BIC": f"{model.bic:.1f}",
        "Observations": f"{int(model.nobs)}"
    }
    model_info_df = pd.DataFrame({
        "Metric": list(model_info.keys()),
        "Value": list(model_info.values())
    })
    st.table(model_info_df)

    st.subheader("Coefficients")
    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std. Error": model.bse,
        "t-value": model.tvalues,
        "P>|t|": model.pvalues,
        "95% CI Lower": model.conf_int()[0],
        "95% CI Upper": model.conf_int()[1]
    })
    numeric_cols = ["Coefficient", "Std. Error", "t-value", "P>|t|", "95% CI Lower", "95% CI Upper"]
    coef_df[numeric_cols] = coef_df[numeric_cols].round(4)
    st.table(coef_df.style.applymap(highlight_pval, subset=["P>|t|"]))

    st.subheader("Residual Diagnostics")
    try:
        diag_stats = {
            "Observations": f"{int(model.nobs)}",
            "Degrees of Freedom Residuals": f"{int(model.df_resid)}",
            "Degrees of Freedom Model": f"{int(model.df_model)}"
        }
        if hasattr(model, 'omnibus'):
            diag_stats["Omnibus"] = f"{model.omnibus:.3f}"
        if hasattr(model, 'omnibus_pval'):
            diag_stats["Prob(Omnibus)"] = f"{model.omnibus_pval:.4g}"
        if hasattr(model, 'durbin_watson'):
            diag_stats["Durbin-Watson"] = f"{model.durbin_watson:.3f}"
        if hasattr(model, 'jarque_bera'):
            diag_stats["Jarque-Bera"] = f"{model.jarque_bera:.3f}"
        if hasattr(model, 'jarque_bera_pval'):
            diag_stats["Prob(JB)"] = f"{model.jarque_bera_pval:.4g}"
        if hasattr(model, 'skew'):
            diag_stats["Skewness"] = f"{model.skew:.3f}"
        if hasattr(model, 'kurtosis'):
            diag_stats["Kurtosis"] = f"{model.kurtosis:.3f}"
        if hasattr(model, 'condition_number'):
            diag_stats["Condition Number"] = f"{model.condition_number:.2e}"

        diag_df = pd.DataFrame({
            "Metric": list(diag_stats.keys()),
            "Value": list(diag_stats.values())
        })
        st.table(diag_df)
    except Exception as e:
        st.error(f"Error displaying residual diagnostics: {str(e)}")
        st.write(f"Observations: {int(model.nobs)}")
        st.write(f"Degrees of Freedom Residuals: {int(model.df_resid)}")
        st.write(f"Degrees of Freedom Model: {int(model.df_model)}")

# -----------------------------------------------------------------------------
# Helper Function: Create Polynomial Features
# -----------------------------------------------------------------------------
def create_polynomial_features(df, feature_columns, degree=2):
    """
    Creates polynomial features from specified feature columns.
    Missing values are imputed with the mean.
    Returns a DataFrame with the polynomial (interaction and squared) terms.
    """
    df_temp = df[feature_columns].copy()
    missing_values = df_temp.isnull().sum()
    for col in feature_columns:
        if missing_values[col] > 0:
            df_temp[col] = df_temp[col].fillna(df_temp[col].mean())
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(df_temp)
    poly_feature_names = poly.get_feature_names_out(feature_columns)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=df.index)
    # Exclude the original features
    X_poly_df = X_poly_df.iloc[:, len(feature_columns):]
    return X_poly_df

# =============================================================================
# NOAA API Config & Data Fetching (Cached)
# =============================================================================
NOAA_API_TOKEN = os.getenv("NCEI_API_TOKEN", "CCSqekrmuGGNtWRrttiTKdVifdRuJpSi")
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
NOAA_DATASET_ID = "GHCND"
NOAA_LOCATION_ID = "FIPS:FI"
NOAA_DATATYPE_ID = ["TAVG", "AWND", "PRCP"]
NOAA_UNITS = "metric"
NOAA_LIMIT = 1000

@st.cache_data
def fetch_noaa_data(start_date, end_date, token=NOAA_API_TOKEN,
                    datasetid=NOAA_DATASET_ID, location_id=NOAA_LOCATION_ID,
                    datatypeid=NOAA_DATATYPE_ID, units=NOAA_UNITS, limit=NOAA_LIMIT,
                    show_messages=True):
    """
    Fetch NOAA weather data for a specified date range in chunks. The data is
    aggregated, pivoted to a wide format, and columns are renamed for clarity.
    This function is cached to speed up subsequent runs.
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
        if show_messages:
            status_code = requests.get(NOAA_BASE_URL, headers=headers, params=params).status_code
            st.write(f"Fetching NOAA data from {params['startdate']} to {params['enddate']}: Status {status_code}")
        response = requests.get(NOAA_BASE_URL, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                df_chunk = pd.DataFrame(data["results"])
                df_noaa = pd.concat([df_noaa, df_chunk], ignore_index=True)
            elif show_messages:
                st.write(f"No results for period {params['startdate']} to {params['enddate']}")
        elif show_messages:
            st.write(f"Error {response.status_code} for period {params['startdate']} to {params['enddate']} - {response.reason}")
        current_start = current_end + timedelta(days=1)

    if not df_noaa.empty:
        df_noaa["date"] = pd.to_datetime(df_noaa["date"])
        df_noaa = df_noaa.groupby(["date", "datatype"])["value"].mean().reset_index()
        df_noaa = df_noaa.pivot(index="date", columns="datatype", values="value").reset_index()
        df_noaa.columns.name = None
        df_noaa.rename(columns={
            "TAVG": "temperature_C",
            "AWND": "wind_speed",
            "PRCP": "precipitation"
        }, inplace=True)
    return df_noaa

# =============================================================================
# Sidebar Navigation
# =============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Data Overview", "Time Series Visualizations", "Regression Analysis", "Polynomial Regression", "Residual Diagnostics"],
    key="page_nav"
)

# Decide whether to show data loading messages based on the selected page
SHOW_DATA_LOADING_MESSAGES = (page == "Data Overview")

# =============================================================================
# Load Electricity Price Data
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
electricity_file = os.path.join(script_dir, "fmi_weather_and_price.csv")
if not os.path.exists(electricity_file):
    st.error(f"File '{electricity_file}' not found.")
    electricity_file = st.text_input("Enter the path to the electricity price CSV file:")
    if not electricity_file:
        st.stop()

try:
    df_electricity = pd.read_csv(electricity_file)
except FileNotFoundError:
    st.error(f"File '{electricity_file}' still not found.")
    st.stop()

df_electricity.rename(columns={"Time": "date"}, inplace=True)
df_electricity["date"] = pd.to_datetime(df_electricity["date"], errors="coerce").dt.date

elec_min_date = pd.to_datetime(df_electricity["date"]).min().date()
elec_max_date = pd.to_datetime(df_electricity["date"]).max().date()

finnish_min_date = f"{elec_min_date.day}.{elec_min_date.month}.{elec_min_date.year}"
finnish_max_date = f"{elec_max_date.day}.{elec_max_date.month}.{elec_max_date.year}"

if SHOW_DATA_LOADING_MESSAGES:
    st.write(f"Electricity dataset date range: {finnish_min_date} to {finnish_max_date}")

# =============================================================================
# Fetch NOAA Weather Data & Merge with Electricity Data
# =============================================================================
start_date = datetime(elec_min_date.year, elec_min_date.month, elec_min_date.day)
end_date = datetime(elec_max_date.year, elec_max_date.month, elec_max_date.day)
df_noaa = fetch_noaa_data(start_date, end_date, show_messages=SHOW_DATA_LOADING_MESSAGES)

if not df_noaa.empty:
    df_noaa["date"] = df_noaa["date"].dt.date
    noaa_min_date = df_noaa["date"].min()
    noaa_max_date = df_noaa["date"].max()
    finnish_noaa_min = f"{noaa_min_date.day}.{noaa_min_date.month}.{noaa_min_date.year}"
    finnish_noaa_max = f"{noaa_max_date.day}.{noaa_max_date.month}.{noaa_max_date.year}"
    if SHOW_DATA_LOADING_MESSAGES:
        st.write(f"NOAA dataset date range: {finnish_noaa_min} to {finnish_noaa_max}")
else:
    if SHOW_DATA_LOADING_MESSAGES:
        st.warning("No NOAA data fetched. The app might not work as intended.")

if not df_noaa.empty and not df_electricity.empty:
    df_merged = pd.merge(df_electricity, df_noaa, on="date", how="left")
    if SHOW_DATA_LOADING_MESSAGES:
        st.write("Merged dataset preview (first 5 rows):")
        safe_dataframe_display(df_merged, num_rows=5)
    merged_csv = os.path.join(script_dir, "merged_weather_electricity.csv")
    df_merged.to_csv(merged_csv, index=False)
    if SHOW_DATA_LOADING_MESSAGES:
        st.write(f"Merged dataset saved as '{merged_csv}'.")

    # =============================================================================
    # Feature Engineering & Finnish Date Formatting
    # =============================================================================
    df_merged["date_FI"] = pd.to_datetime(df_merged["date"]).dt.strftime("%-d.%-m.%Y")
    if df_merged["date_FI"].str.contains('%').any():
        df_merged["date_FI"] = pd.to_datetime(df_merged["date"]).dt.strftime("%#d.%#m.%Y")
    df = df_merged.copy()
    df["Price_RollingMean"] = df["Price"].rolling(window=30, min_periods=1).mean()

    # If both Temp and temperature_C exist, choose the one with fewer missing values
    if "Temp" in df.columns and "temperature_C" in df.columns:
        temp_nulls = df["temperature_C"].isnull().sum()
        Temp_nulls = df["Temp"].isnull().sum()
        if SHOW_DATA_LOADING_MESSAGES:
            st.write(f"Missing values in 'temperature_C': {temp_nulls}, in 'Temp': {Temp_nulls}")
        if temp_nulls > Temp_nulls:
            df["temperature_C"] = df["temperature_C"].fillna(df["Temp"])
        else:
            df["Temp"] = df["Temp"].fillna(df["temperature_C"])
    elif "Temp" in df.columns and "temperature_C" not in df.columns:
        df["temperature_C"] = df["Temp"]
    elif "temperature_C" in df.columns and "Temp" not in df.columns:
        df["Temp"] = df["temperature_C"]

    # Create lagged features and an interaction term (if columns exist)
    if "temperature_C" in df.columns and "Wind" in df.columns:
        temp_col_series = df["temperature_C"].fillna(df["temperature_C"].mean())
        wind_col_series = df["Wind"].fillna(df["Wind"].mean())
        df["temp_lag1"] = temp_col_series.shift(1)
        df["wind_lag1"] = wind_col_series.shift(1)
        df["temp_wind_interaction"] = temp_col_series * wind_col_series
else:
    st.error("Please check your data loading and NOAA API. The program cannot continue without proper data.")
    st.stop()

# =============================================================================
# Page: Data Overview
# =============================================================================
if page == "Data Overview":
    st.header("Data Overview")
    st.write("""
    **Overview:** This page provides a summary of the merged electricity and weather data.
    It includes a data preview, DataFrame structure, missing values, and summary statistics.
    The 'date_FI' column is formatted in the Finnish date format (dd.mm.yyyy) for readability.
    """)
    st.subheader("Data Preview (First 10 Rows)")
    preview_columns = ["date_FI", "Price", "temperature_C", "Wind", "precipitation"]
    if all(col in df.columns for col in preview_columns):
        safe_dataframe_display(df, preview_columns, 10)
    else:
        safe_dataframe_display(df, num_rows=10)

    st.subheader("DataFrame Information")
    column_info = []
    for i, (col_name, dtype) in enumerate(zip(df.columns, df.dtypes)):
        non_null_count = df[col_name].count()
        column_info.append({
            "Index": i,
            "Column": col_name,
            "Non-Null Count": f"{non_null_count} / {len(df)}",
            "Data Type": str(dtype)
        })
    col_info_df = pd.DataFrame(column_info)
    st.table(col_info_df)

    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.write(f"Total memory usage: {memory_usage:.2f} MB")

    st.subheader("Missing Values by Column")
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Count"]
    missing_values["Percentage"] = (missing_values["Missing Count"] / len(df) * 100).round(2)
    missing_values = missing_values.sort_values(by="Missing Count", ascending=False)
    st.table(missing_values)

    st.subheader("Summary Statistics")
    summary_stats = df.describe()
    st.dataframe(summary_stats)

# =============================================================================
# Page: Time Series Visualizations
# =============================================================================
elif page == "Time Series Visualizations":
    st.header("Time Series Visualizations")
    st.write("""
    **Overview:** This page shows how electricity prices and weather variables evolve over time.
    The visualizations include time series plots, rolling averages, distributions, and scatter plots
    to highlight relationships between price and weather variables.
    """)
    date_form = mdates.DateFormatter("%d.%m.%Y")
    st.subheader("Electricity Price and Temperature Over Time")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    temp_col = "temperature_C" if "temperature_C" in df.columns else "Temp"
    ax1.plot(pd.to_datetime(df["date"]), df["Price"], label="Electricity Price (€)", color="red")
    if temp_col in df.columns:
        ax1.plot(pd.to_datetime(df["date"]), df[temp_col], label="Temperature (°C)", color="blue", alpha=0.6)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.set_title("Electricity Price and Temperature Over Time")
    ax1.legend()
    ax1.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("30-Day Rolling Mean")
    st.write("""
    **Explanation:** The 30-day rolling mean smooths out daily fluctuations in electricity prices,
    revealing longer-term trends.
    """)
    fig13, ax13 = plt.subplots(figsize=(12, 6))
    ax13.plot(pd.to_datetime(df["date"]), df["Price"], label="Daily Prices", color="red", alpha=0.5)
    ax13.plot(pd.to_datetime(df["date"]), df["Price_RollingMean"], label="30-Day Rolling Mean", color="black")
    ax13.set_xlabel("Date")
    ax13.set_ylabel("Electricity Price (€)")
    ax13.set_title("Electricity Price Trend with 30-Day Rolling Mean")
    ax13.legend()
    ax13.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45)
    ax13.grid(True)
    st.pyplot(fig13)

    st.subheader("Distribution of Electricity Prices")
    st.write("""
    **Explanation:** This histogram shows the frequency distribution of daily electricity prices,
    helping identify the central tendency and spread of the data.
    """)
    fig14, ax14 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Price"], bins=50, kde=True, color="red", ax=ax14)
    ax14.set_xlabel("Electricity Price (€)")
    ax14.set_ylabel("Frequency")
    ax14.set_title("Distribution of Electricity Prices")
    ax14.grid(True)
    st.pyplot(fig14)

    st.subheader("Relationship Between Price and Weather Variables")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Price vs Temperature:** This scatter plot visualizes how electricity price relates to temperature.")
        fig_scatter1, ax_scatter1 = plt.subplots(figsize=(8, 6))
        ax_scatter1.scatter(df[temp_col], df["Price"], alpha=0.5, color="blue")
        ax_scatter1.set_xlabel("Temperature (°C)")
        ax_scatter1.set_ylabel("Price (€)")
        ax_scatter1.set_title("Price vs Temperature")
        ax_scatter1.grid(True, alpha=0.3)
        st.pyplot(fig_scatter1)
    with col2:
        st.write("**Price vs Wind Speed:** This scatter plot visualizes how electricity price relates to wind speed.")
        fig_scatter2, ax_scatter2 = plt.subplots(figsize=(8, 6))
        ax_scatter2.scatter(df["Wind"], df["Price"], alpha=0.5, color="green")
        ax_scatter2.set_xlabel("Wind Speed")
        ax_scatter2.set_ylabel("Price (€)")
        ax_scatter2.set_title("Price vs Wind Speed")
        ax_scatter2.grid(True, alpha=0.3)
        st.pyplot(fig_scatter2)

    st.subheader("Boxplots for Outlier Detection")
    st.write("""
    **Explanation:** Boxplots help to visually identify outliers in the data for temperature, wind, and price.
    Outliers may affect model performance and can be handled separately.
    """)
    fig21, (ax21, ax22_box, ax23_box) = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(y=df[temp_col], ax=ax21)
    ax21.set_title("Temperature")
    sns.boxplot(y=df["Wind"], ax=ax22_box)
    ax22_box.set_title("Wind")
    sns.boxplot(y=df["Price"], ax=ax23_box)
    ax23_box.set_title("Price")
    st.pyplot(fig21)

# =============================================================================
# Page: Regression Analysis
# =============================================================================
elif page == "Regression Analysis":
    st.header("Regression Analysis")
    st.write("""
    **Overview:** This section performs an Ordinary Least Squares (OLS) regression to analyze
    the linear relationship between temperature, wind, and electricity price. The results include
    model statistics, coefficients, and confidence intervals.
    """)
    temp_col = "temperature_C" if "temperature_C" in df.columns else "Temp"
    df_reg = df.copy()
    for col in [temp_col, "Wind", "Price"]:
        df_reg[col] = df_reg[col].fillna(df_reg[col].mean())
    df_reg = df_reg.dropna(subset=[temp_col, "Wind", "Price"])
    st.write(f"Number of observations used: {len(df_reg)}")
    X = df_reg[[temp_col, "Wind"]]
    X = sm.add_constant(X)
    y = df_reg["Price"]
    model = sm.OLS(y, X).fit()

    st.subheader("OLS Regression Results")
    st.write("**Model Statistics:**")
    model_stats = pd.DataFrame({
        "Statistic": ["Observations", "R-squared", "Adjusted R-squared", "F-statistic", "Prob (F-statistic)"],
        "Value": [int(model.nobs), round(model.rsquared, 3), round(model.rsquared_adj, 3),
                  round(model.fvalue, 3), f"{model.f_pvalue:.3g}"]
    })
    st.table(model_stats)

    st.write("**Coefficients:**")
    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std. Error": model.bse,
        "t-value": model.tvalues,
        "P>|t|": model.pvalues,
    }).round(4)
    st.table(coef_df.style.applymap(highlight_pval, subset=["P>|t|"]))

    st.write("**95% Confidence Intervals:**")
    conf_int = model.conf_int()
    conf_int.columns = ["Lower", "Upper"]
    conf_int_df = pd.DataFrame({
        "Variable": conf_int.index,
        "Lower 95% CI": conf_int["Lower"],
        "Upper 95% CI": conf_int["Upper"]
    }).round(4)
    st.table(conf_int_df)

    st.subheader("Actual vs. Predicted Prices")
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
    ax_pred.scatter(y, model.fittedvalues, alpha=0.5)
    ax_pred.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax_pred.set_xlabel("Actual Price (€)")
    ax_pred.set_ylabel("Predicted Price (€)")
    ax_pred.set_title("Actual vs. Predicted Prices")
    st.pyplot(fig_pred)

# =============================================================================
# Page: Polynomial Regression
# =============================================================================
elif page == "Polynomial Regression":
    st.header("Polynomial Regression Analysis")
    st.write("""
    **Overview:** This section expands the regression model by including polynomial terms
    (squared and interaction terms) to capture potential nonlinear relationships.
    """)
    temp_col = "temperature_C" if "temperature_C" in df.columns else "Temp"
    df_poly = df.copy()
    df_poly["Price"] = df_poly["Price"].fillna(df_poly["Price"].mean())

    st.subheader("Missing Values Before Imputation")
    missing_before = df_poly[[temp_col, "Wind"]].isnull().sum()
    st.write(missing_before)

    for col in [temp_col, "Wind"]:
        if col in df_poly.columns:
            df_poly[col] = df_poly[col].fillna(df_poly[col].mean())

    poly_features = create_polynomial_features(df_poly, [temp_col, "Wind"], degree=2)
    poly_features.columns = [f"{temp_col}^2", f"{temp_col}_Wind", "Wind^2"]
    df_poly = pd.concat([df_poly, poly_features], axis=1)

    st.subheader("Data with Polynomial Features (First 5 Rows)")
    display_cols = ["date_FI", "Price", temp_col, "Wind"] + list(poly_features.columns)
    safe_dataframe_display(df_poly, display_cols, 5)

    X_poly = df_poly[[temp_col, "Wind"] + list(poly_features.columns)]
    X_poly = sm.add_constant(X_poly)
    y_poly = df_poly["Price"]
    model_poly = sm.OLS(y_poly, X_poly).fit()

    st.subheader("Polynomial OLS Regression Summary")
    model_stats = pd.DataFrame({
        "Statistic": ["Observations", "R-squared", "Adjusted R-squared", "F-statistic", "Prob (F-statistic)"],
        "Value": [int(model_poly.nobs), round(model_poly.rsquared, 3), round(model_poly.rsquared_adj, 3),
                  round(model_poly.fvalue, 3), f"{model_poly.f_pvalue:.3g}"]
    })
    st.table(model_stats)

    st.write("**Coefficients:**")
    coef_df = pd.DataFrame({
        "Variable": model_poly.params.index,
        "Coefficient": model_poly.params.values,
        "Std. Error": model_poly.bse,
        "t-value": model_poly.tvalues,
        "P>|t|": model_poly.pvalues,
    }).round(4)
    st.table(coef_df.style.applymap(highlight_pval, subset=["P>|t|"]))

    st.write("**95% Confidence Intervals:**")
    conf_int = model_poly.conf_int()
    conf_int.columns = ["Lower", "Upper"]
    conf_int_df = pd.DataFrame({
        "Variable": conf_int.index,
        "Lower 95% CI": conf_int["Lower"],
        "Upper 95% CI": conf_int["Upper"]
    }).round(4)
    st.table(conf_int_df)

    st.subheader("Residual Diagnostics (Polynomial Model)")
    residuals_poly = model_poly.resid
    col1, col2 = st.columns(2)
    with col1:
        fig8, ax8 = plt.subplots(figsize=(8, 6))
        sm.qqplot(residuals_poly, line='s', ax=ax8)
        ax8.set_title("QQ Plot of Residuals")
        st.pyplot(fig8)
    with col2:
        fig9, ax9 = plt.subplots(figsize=(8, 6))
        ax9.hist(residuals_poly, bins=20, edgecolor='k', alpha=0.7)
        ax9.set_xlabel("Residuals")
        ax9.set_ylabel("Frequency")
        ax9.set_title("Histogram of Residuals")
        st.pyplot(fig9)

    fig10, ax10 = plt.subplots(figsize=(10, 6))
    ax10.scatter(model_poly.fittedvalues, residuals_poly, alpha=0.5)
    ax10.axhline(y=0, color='r', linestyle='-')
    ax10.set_xlabel("Fitted Values")
    ax10.set_ylabel("Residuals")
    ax10.set_title("Residuals vs Fitted Values")
    st.pyplot(fig10)

# =============================================================================
# Page: Residual Diagnostics (Interaction Model)
# =============================================================================
elif page == "Residual Diagnostics":
    st.header("Residual Diagnostics (Interaction Model)")
    st.write("""
    **Overview:** This section builds an interaction model that includes lagged features and interaction terms.
    It then displays regression results and various residual plots to check model assumptions.
    """)
    temp_col = "temperature_C" if "temperature_C" in df.columns else "Temp"
    df_int = df.copy()
    df_int["Price"] = df_int["Price"].fillna(df_int["Price"].mean())
    for col in [temp_col, "Wind"]:
        if col in df_int.columns:
            df_int[col] = df_int[col].fillna(df_int[col].mean())
    df_int["temp_lag1"] = df_int[temp_col].shift(1)
    df_int["wind_lag1"] = df_int["Wind"].shift(1)
    df_int["temp_wind_interaction"] = df_int[temp_col] * df_int["Wind"]
    df_int[f"{temp_col}^2"] = df_int[temp_col] ** 2
    df_int["Wind^2"] = df_int["Wind"] ** 2
    df_int = df_int.dropna()

    st.subheader("Data with Interaction Features (First 5 Rows)")
    interaction_cols = ["date_FI", "Price", temp_col, "Wind", "temp_lag1", "wind_lag1", "temp_wind_interaction", f"{temp_col}^2", "Wind^2"]
    safe_dataframe_display(df_int, interaction_cols, 5)

    st.subheader("Correlation Matrix")
    numeric_cols = ["Price", temp_col, "Wind", "temp_lag1", "wind_lag1", "temp_wind_interaction", f"{temp_col}^2", "Wind^2"]
    corr_matrix = df_int[numeric_cols].corr()
    fig12, ax12 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, annot_kws={"size": 10}, ax=ax12)
    ax12.set_title("Correlation Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig12)

    X_interaction = df_int[[temp_col, "Wind", "temp_lag1", "wind_lag1", "temp_wind_interaction", f"{temp_col}^2", "Wind^2"]]
    X_interaction = sm.add_constant(X_interaction)
    y_interaction = df_int["Price"]
    model_interaction = sm.OLS(y_interaction, X_interaction).fit()

    st.subheader("Interaction Model Regression Summary")
    model_stats = pd.DataFrame({
        "Statistic": ["Observations", "R-squared", "Adjusted R-squared", "F-statistic", "Prob (F-statistic)"],
        "Value": [int(model_interaction.nobs), round(model_interaction.rsquared, 3), round(model_interaction.rsquared_adj, 3),
                  round(model_interaction.fvalue, 3), f"{model_interaction.f_pvalue:.3g}"]
    })
    st.table(model_stats)

    st.write("**Coefficients:**")
    coef_df = pd.DataFrame({
        "Variable": model_interaction.params.index,
        "Coefficient": model_interaction.params.values,
        "Std. Error": model_interaction.bse,
        "t-value": model_interaction.tvalues,
        "P>|t|": model_interaction.pvalues,
    }).round(4)
    st.table(coef_df.style.applymap(highlight_pval, subset=["P>|t|"]))

    st.write("**95% Confidence Intervals:**")
    conf_int = model_interaction.conf_int()
    conf_int.columns = ["Lower", "Upper"]
    conf_int_df = pd.DataFrame({
        "Variable": conf_int.index,
        "Lower 95% CI": conf_int["Lower"],
        "Upper 95% CI": conf_int["Upper"]
    }).round(4)
    st.table(conf_int_df)

    st.subheader("Residual Plots for Interaction Model")
    residuals = model_interaction.resid
    col1, col2 = st.columns(2)
    with col1:
        fig22, ax22 = plt.subplots(figsize=(8, 6))
        ax22.scatter(model_interaction.fittedvalues, residuals, alpha=0.5)
        ax22.axhline(y=0, color='red', linestyle='--')
        ax22.set_xlabel("Fitted Values")
        ax22.set_ylabel("Residuals")
        ax22.set_title("Residuals vs Fitted Values")
        st.pyplot(fig22)
    with col2:
        fig23, ax23 = plt.subplots(figsize=(8, 6))
        sm.qqplot(residuals, line='45', fit=True, ax=ax23)
        ax23.set_title("Q-Q Plot of Residuals")
        st.pyplot(fig23)
    fig24, ax24 = plt.subplots(figsize=(10, 6))
    ax24.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax24.set_xlabel("Residuals")
    ax24.set_ylabel("Frequency")
    ax24.set_title("Histogram of Residuals")
    st.pyplot(fig24)
    shapiro_test = stats.shapiro(residuals)
    st.write("### Shapiro-Wilk Test for Normality of Residuals")
    st.write(f"Statistic: {shapiro_test.statistic:.4f}")
    st.write(f"p-value: {shapiro_test.pvalue:.4g}")
    if shapiro_test.pvalue < 0.05:
        st.write("Residuals are not normally distributed (p < 0.05).")
    else:
        st.write("Residuals appear normally distributed (p >= 0.05).")

# =============================================================================
# Page: Testing Price Shifting
# =============================================================================
st.header("Testing Price Shifting")
st.write("""
**Overview:** In some electricity markets (such as Nordpool), spot prices are estimated one day before.
This section shifts the price data by one day to explore how weather conditions on a given day might affect the next day’s prices.
""")
df_merged["Price_shifted"] = df_merged["Price"].shift(-1)
st.write("Preview of Original and Shifted Prices:")
safe_dataframe_display(df_merged[["date", "Price", "Price_shifted"]], num_rows=5)

corr_original = df_merged[['temperature_C', 'Price']].corr()
corr_shifted = df_merged[['temperature_C', 'Price_shifted']].corr()
st.write("Correlation Matrix (Original Price):")
st.write(corr_original)
st.write("Correlation Matrix (Shifted Price):")
st.write(corr_shifted)

df_reg_shifted = df_merged.dropna(subset=["temperature_C", "Wind", "Price_shifted"]).copy()
X_shifted = sm.add_constant(df_reg_shifted[["temperature_C", "Wind"]])
y_shifted = df_reg_shifted["Price_shifted"]
model_shifted = sm.OLS(y_shifted, X_shifted).fit()

st.subheader("OLS Regression Summary (Shifted Price)")
st.text(model_shifted.summary())