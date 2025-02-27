#!/usr/bin/env python3
import streamlit as st
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

# Global variable to control data loading messages
SHOW_DATA_LOADING_MESSAGES = True

# Remove default bullet style from sidebar radio buttons
st.markdown(
    """
    <style>
    .stRadio > label { list-style-type: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===============================================================================
# Helper functions for displaying OLS results and handling data
# ===============================================================================
def display_ols_summary_as_tables(model):
    """
    Displays a statsmodels OLS regression summary as nice Streamlit tables
    """
    # Model Information Table
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
        "Number of Observations": f"{int(model.nobs)}"
    }
    
    # Convert to two-column format for better display
    model_info_df = pd.DataFrame({
        "Metric": list(model_info.keys()),
        "Value": list(model_info.values())
    })
    
    # Display as a styled table
    st.table(model_info_df)
    
    # Coefficients Table
    st.subheader("Coefficients")
    
    # Extract coefficient data
    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std. Error": model.bse,
        "t-value": model.tvalues,
        "P>|t|": model.pvalues,
        "95% CI Lower": model.conf_int()[0],
        "95% CI Upper": model.conf_int()[1]
    })
    
    # Round numeric columns for better display
    numeric_cols = ["Coefficient", "Std. Error", "t-value", "P>|t|", "95% CI Lower", "95% CI Upper"]
    coef_df[numeric_cols] = coef_df[numeric_cols].round(4)
    
    # Highlight significant coefficients (p < 0.05)
    def highlight_significant(val):
        if isinstance(val, float) and val < 0.05:
            return 'background-color: rgba(0, 255, 0, 0.2)'
        return ''
    
    # Display as a styled table
    st.table(coef_df.style.applymap(highlight_significant, subset=["P>|t|"]))
    
    # Residual Statistics - use direct attributes instead of diagn dictionary
    st.subheader("Residual Diagnostics")
    
    try:
        # Try to access these attributes safely
        omnibus = getattr(model, 'omnibus', None)
        omnibus_pval = getattr(model, 'omnibus_pval', None)
        durbin_watson = getattr(model, 'durbin_watson', None)
        jarque_bera = getattr(model, 'jarque_bera', None)
        jarque_bera_pval = getattr(model, 'jarque_bera_pval', None)
        skew = getattr(model, 'skew', None)
        kurtosis = getattr(model, 'kurtosis', None)
        condition_number = getattr(model, 'condition_number', None)
        
        diag_stats = {
            "Observations": f"{int(model.nobs)}",
            "Degrees of Freedom Residuals": f"{int(model.df_resid)}",
            "Degrees of Freedom Model": f"{int(model.df_model)}"
        }
        
        # Only add statistics that exist
        if omnibus is not None:
            diag_stats["Omnibus"] = f"{omnibus:.3f}"
        if omnibus_pval is not None:
            diag_stats["Prob(Omnibus)"] = f"{omnibus_pval:.4g}"
        if durbin_watson is not None:
            diag_stats["Durbin-Watson"] = f"{durbin_watson:.3f}"
        if jarque_bera is not None and isinstance(jarque_bera, (int, float)):
            diag_stats["Jarque-Bera"] = f"{jarque_bera:.3f}"
        elif jarque_bera is not None and hasattr(jarque_bera, '__iter__') and len(jarque_bera) > 0:
            diag_stats["Jarque-Bera"] = f"{jarque_bera[0]:.3f}"
        if jarque_bera_pval is not None and isinstance(jarque_bera_pval, (int, float)):
            diag_stats["Prob(JB)"] = f"{jarque_bera_pval:.4g}"
        elif jarque_bera_pval is not None and hasattr(jarque_bera_pval, '__iter__') and len(jarque_bera_pval) > 0:
            diag_stats["Prob(JB)"] = f"{jarque_bera_pval[0]:.4g}"
        if skew is not None:
            diag_stats["Skewness"] = f"{skew:.3f}"
        if kurtosis is not None:
            diag_stats["Kurtosis"] = f"{kurtosis:.3f}"
        if condition_number is not None:
            diag_stats["Condition Number"] = f"{condition_number:.2e}"
            
        # Convert to two-column format
        diag_df = pd.DataFrame({
            "Metric": list(diag_stats.keys()),
            "Value": list(diag_stats.values())
        })
        
        # Display as a styled table
        st.table(diag_df)
    except Exception as e:
        st.error(f"Error displaying residual diagnostics: {str(e)}")
        # Fallback to just displaying basic stats
        st.write(f"Number of Observations: {int(model.nobs)}")
        st.write(f"Degrees of Freedom Residuals: {int(model.df_resid)}")
        st.write(f"Degrees of Freedom Model: {int(model.df_model)}")


def create_polynomial_features(df, feature_columns, degree=2):
    """
    Creates polynomial features without duplicates
    
    Args:
        df: DataFrame with the original features
        feature_columns: List of column names to use for polynomial features
        degree: Degree of polynomial features
        
    Returns:
        DataFrame with polynomial features
    """
    # Create a temporary dataframe for imputation
    df_temp = df[feature_columns].copy()
    
    # Check and report missing values
    missing_values = df_temp.isnull().sum()
    
    # Impute missing values with the mean
    for col in feature_columns:
        if missing_values[col] > 0:
            df_temp[col] = df_temp[col].fillna(df_temp[col].mean())
    
    # Apply PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(df_temp)
    poly_feature_names = poly.get_feature_names_out(feature_columns)
    
    # Create DataFrame with polynomial features
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=df.index)
    
    # Keep only interaction terms and squared terms (skip original terms)
    X_poly_df = X_poly_df.iloc[:, len(feature_columns):]
    
    return X_poly_df


def safe_dataframe_display(df, columns=None, num_rows=5):
    """
    Safely displays a dataframe with mixed types in Streamlit
    
    Args:
        df: DataFrame to display
        columns: List of column names to display (optional)
        num_rows: Number of rows to display
    """
    # Create a copy to avoid modifying the original dataframe
    display_df = df.copy()
    
    # If specific columns are requested
    if columns is not None:
        valid_columns = [col for col in columns if col in display_df.columns]
        if len(valid_columns) > 0:
            display_df = display_df[valid_columns]
    
    # Limit the number of rows
    display_df = display_df.head(num_rows)
    
    # Convert all columns to strings for safety
    for col in display_df.columns:
        # Try to format numeric columns with standard notation instead of scientific
        if pd.api.types.is_numeric_dtype(display_df[col]):
            try:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
            except:
                display_df[col] = display_df[col].astype(str)
        else:
            display_df[col] = display_df[col].astype(str)
            
    # Try to display the dataframe
    try:
        st.dataframe(display_df)
    except Exception as e:
        st.error(f"Error displaying dataframe: {str(e)}")
        # Fallback to HTML display
        st.write("Displaying as HTML table instead:")
        st.write(display_df.to_html(index=False), unsafe_allow_html=True)


st.title("A3 Analysis")
st.write("Electricity vs Weather Analysis")

# =============================================================================
# 1. NOAA API Configuration & Data Fetching (with caching)
# =============================================================================

NOAA_API_TOKEN = os.getenv("NCEI_API_TOKEN", "YOUR_API_KEY_HERE")  # Replace with your API key or use Streamlit secrets
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
    Fetch NOAA weather data for a given date range. This function is cached.
    
    Args:
        show_messages: Whether to display status messages (default True)
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
        
        # Only show messages if requested
        if show_messages:
            st.write(f"Fetching NOAA data from {params['startdate']} to {params['enddate']}: "
                    f"Status {requests.get(NOAA_BASE_URL, headers=headers, params=params).status_code}")
                    
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
        df_noaa.columns.name = None  # Remove any MultiIndex name
        df_noaa.rename(columns={
            "TAVG": "temperature_C", 
            "AWND": "wind_speed",
            "PRCP": "precipitation"
        }, inplace=True)
    return df_noaa


# =============================================================================
# 5. Sidebar Navigation (List View)
# =============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Data Overview", "Time Series Visualizations", "Regression Analysis", "Polynomial Regression", "Residual Diagnostics"]
)

# Set whether to show data loading messages based on page
SHOW_DATA_LOADING_MESSAGES = (page == "Data Overview")

# =============================================================================
# 2. Load Electricity Price Data
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
electricity_file = os.path.join(script_dir, "fmi_weather_and_price.csv")
if not os.path.exists(electricity_file):
    st.error(f"File '{electricity_file}' not found. Please specify the correct path.")
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

# Format dates in Finnish format
finnish_min_date = f"{elec_min_date.day}.{elec_min_date.month}.{elec_min_date.year}"
finnish_max_date = f"{elec_max_date.day}.{elec_max_date.month}.{elec_max_date.year}"

# Only show on Data Overview page
if SHOW_DATA_LOADING_MESSAGES:
    st.write(f"Electricity dataset date range: {finnish_min_date} to {finnish_max_date}")

# =============================================================================
# 3. Fetch NOAA Weather Data (Cached) & Merge Data
# =============================================================================

start_date = datetime(elec_min_date.year, elec_min_date.month, elec_min_date.day)
end_date = datetime(elec_max_date.year, elec_max_date.month, elec_max_date.day)
df_noaa = fetch_noaa_data(start_date, end_date, show_messages=SHOW_DATA_LOADING_MESSAGES)

if not df_noaa.empty:
    df_noaa["date"] = df_noaa["date"].dt.date
    noaa_min_date = df_noaa["date"].min()
    noaa_max_date = df_noaa["date"].max()
    
    # Format dates in Finnish format
    finnish_noaa_min = f"{noaa_min_date.day}.{noaa_min_date.month}.{noaa_min_date.year}"
    finnish_noaa_max = f"{noaa_max_date.day}.{noaa_max_date.month}.{noaa_max_date.year}"
    
    # Only show on Data Overview page
    if SHOW_DATA_LOADING_MESSAGES:
        st.write(f"NOAA dataset date range: {finnish_noaa_min} to {finnish_noaa_max}")
else:
    if SHOW_DATA_LOADING_MESSAGES:
        st.warning("No NOAA data fetched. The app might not work as intended.")

if not df_noaa.empty and not df_electricity.empty:
    df_merged = pd.merge(df_electricity, df_noaa, on="date", how="left")
    if SHOW_DATA_LOADING_MESSAGES:
        st.write("Merged dataset preview (first 5 rows):")
        safe_dataframe_display(df_merged)
    
    merged_csv = os.path.join(script_dir, "merged_weather_electricity.csv")
    df_merged.to_csv(merged_csv, index=False)
    if SHOW_DATA_LOADING_MESSAGES:
        st.write(f"Merged dataset saved as '{merged_csv}'.")

    # =============================================================================
    # 4. Feature Engineering + Finnish Date Format
    # =============================================================================
    # Convert dates to Finnish format "day.month.year" (e.g., "25.3.2024")
    df_merged["date_FI"] = pd.to_datetime(df_merged["date"]).dt.strftime("%-d.%-m.%Y")
    # (On Windows, you may need to use "%#d.%#m.%Y".)
    
    # Fix date formatting if needed
    if df_merged["date_FI"].str.contains('%').any():
        df_merged["date_FI"] = pd.to_datetime(df_merged["date"]).dt.strftime("%#d.%#m.%Y")

    # Common DataFrame for subsequent pages
    df = df_merged.copy()
    df["Price_RollingMean"] = df["Price"].rolling(window=30, min_periods=1).mean()

    # Fix inconsistent temperature columns
    if "Temp" in df.columns and "temperature_C" in df.columns:
        # Check which column has more data
        temp_nulls = df["temperature_C"].isnull().sum()
        Temp_nulls = df["Temp"].isnull().sum()
        
        if SHOW_DATA_LOADING_MESSAGES:
            st.write(f"Missing values in temperature columns - temperature_C: {temp_nulls}, Temp: {Temp_nulls}")
        
        if temp_nulls > Temp_nulls:
            # If Temp has more data, use it for temperature_C
            if SHOW_DATA_LOADING_MESSAGES:
                st.write("Using 'Temp' column to fill 'temperature_C' column")
            df["temperature_C"] = df["temperature_C"].fillna(df["Temp"])
        elif Temp_nulls > temp_nulls:
            # If temperature_C has more data, use it for Temp
            if SHOW_DATA_LOADING_MESSAGES:
                st.write("Using 'temperature_C' column to fill 'Temp' column")
            df["Temp"] = df["Temp"].fillna(df["temperature_C"])
    elif "Temp" in df.columns and "temperature_C" not in df.columns:
        # If only Temp exists, create temperature_C
        if SHOW_DATA_LOADING_MESSAGES:
            st.write("Creating 'temperature_C' column from 'Temp' column")
        df["temperature_C"] = df["Temp"]
    elif "temperature_C" in df.columns and "Temp" not in df.columns:
        # If only temperature_C exists, create Temp
        if SHOW_DATA_LOADING_MESSAGES:
            st.write("Creating 'Temp' column from 'temperature_C' column")
        df["Temp"] = df["temperature_C"]

    # Create lagged features and interaction term
    # First, handle missing values to avoid issues
    if "temperature_C" in df.columns and "Wind" in df.columns:
        # Temporary columns for calculations to avoid modifying original data
        temp_col = df["temperature_C"].fillna(df["temperature_C"].mean())
        wind_col = df["Wind"].fillna(df["Wind"].mean())
        
        df["temp_lag1"] = temp_col.astype(float).shift(1)
        df["wind_lag1"] = wind_col.astype(float).shift(1)
        df["temp_wind_interaction"] = temp_col * wind_col
else:
    st.error("Please check your data loading and NOAA API. The program cannot continue without the proper data.")
    st.stop()

# =============================================================================
# 6. Data Overview Page
# =============================================================================
if page == "Data Overview":
    st.header("Data Overview")
    st.write(
        "This table shows the merged electricity and weather data. "
        "The column 'date_FI' is formatted in the Finnish date format (dd.mm.yyyy) for better readability."
    )
    
    # Display a better preview of the data
    st.subheader("Data Preview (First 10 Rows)")
    preview_columns = ["date_FI", "Price", "temperature_C", "Wind", "precipitation"]
    if all(col in df.columns for col in preview_columns):
        safe_dataframe_display(df, preview_columns, 10)
    else:
        safe_dataframe_display(df, num_rows=10)
    
    # Display DataFrame information in a more readable format
    st.subheader("DataFrame Information")
    
    # Create a better formatted DataFrame info display
    column_info = []
    for i, (col_name, dtype) in enumerate(zip(df.columns, df.dtypes)):
        non_null_count = df[col_name].count()
        column_info.append({
            "Index": i,
            "Column": col_name,
            "Non-Null Count": f"{non_null_count} / {len(df)}",
            "Data Type": str(dtype)
        })
    
    # Display as a nice table
    col_info_df = pd.DataFrame(column_info)
    st.table(col_info_df)
    
    # Display memory usage
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.write(f"Total memory usage: {memory_usage:.2f} MB")
    
    # Display detailed missing value information
    st.subheader("Missing Values by Column")
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Count"]
    missing_values["Percentage"] = (missing_values["Missing Count"] / len(df) * 100).round(2)
    missing_values = missing_values.sort_values(by="Missing Count", ascending=False)
    st.table(missing_values)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    summary_stats = df.describe()
    st.dataframe(summary_stats)

# =============================================================================
# 7. Time Series Visualizations Page
# =============================================================================
elif page == "Time Series Visualizations":
    st.header("Time Series Visualizations")
    st.write(
        "The following plots show the evolution of electricity prices and weather data over time. "
        "The dates on the x-axis are formatted as dd.mm.yyyy."
    )
    date_form = mdates.DateFormatter("%d.%m.%Y")
    
    # Electricity Price & Temperature Over Time
    st.subheader("Electricity Price and Temperature Over Time")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Use either temperature_C or Temp based on data availability
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
    
    # 30-Day Rolling Mean Plot
    st.subheader("30-Day Rolling Mean")
    st.write(
        "This plot illustrates the 30-day rolling average of electricity prices, "
        "which helps to smooth short-term fluctuations and highlight longer-term trends."
    )
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
    
    # Distribution of Electricity Prices
    st.subheader("Distribution of Electricity Prices")
    st.write("The histogram below shows the frequency distribution of daily electricity prices.")
    fig14, ax14 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Price"], bins=50, kde=True, color="red", ax=ax14)
    ax14.set_xlabel("Electricity Price (€)")
    ax14.set_ylabel("Frequency")
    ax14.set_title("Distribution of Electricity Prices")
    ax14.grid(True)
    st.pyplot(fig14)
    
    # Scatter Plots for Relationships
    st.subheader("Relationship Between Price and Weather Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Temperature Scatter Plot
        fig_scatter1, ax_scatter1 = plt.subplots(figsize=(8, 6))
        ax_scatter1.scatter(df[temp_col], df["Price"], alpha=0.5, color="blue")
        ax_scatter1.set_xlabel("Temperature (°C)")
        ax_scatter1.set_ylabel("Price (€)")
        ax_scatter1.set_title("Price vs Temperature")
        ax_scatter1.grid(True, alpha=0.3)
        st.pyplot(fig_scatter1)
    
    with col2:
        # Price vs Wind Scatter Plot
        fig_scatter2, ax_scatter2 = plt.subplots(figsize=(8, 6))
        ax_scatter2.scatter(df["Wind"], df["Price"], alpha=0.5, color="green")
        ax_scatter2.set_xlabel("Wind Speed")
        ax_scatter2.set_ylabel("Price (€)")
        ax_scatter2.set_title("Price vs Wind Speed")
        ax_scatter2.grid(True, alpha=0.3)
        st.pyplot(fig_scatter2)
    
    # Boxplots for Outlier Detection
    st.subheader("Boxplots for Outlier Detection")
    st.write("Boxplots help identify outliers in temperature, wind, and price data.")
    fig21, (ax21, ax22_box, ax23_box) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use the appropriate temperature column
    sns.boxplot(y=df[temp_col], ax=ax21)
    ax21.set_title("Temperature")
    
    sns.boxplot(y=df["Wind"], ax=ax22_box)
    ax22_box.set_title("Wind")
    
    sns.boxplot(y=df["Price"], ax=ax23_box)
    ax23_box.set_title("Price")
    
    st.pyplot(fig21)

# =============================================================================
# 8. Regression Analysis Page (Custom OLS Table)
# =============================================================================
elif page == "Regression Analysis":
    st.header("Regression Analysis")
    st.write(
        "This section shows the results of an Ordinary Least Squares (OLS) regression model. "
        "The model examines the relationship between temperature, wind, and electricity price."
    )
    
    # Use the appropriate temperature column
    temp_col = "temperature_C" if "temperature_C" in df.columns else "Temp"
    
    # Impute missing values for regression
    df_reg = df.copy()
    for col in [temp_col, "Wind", "Price"]:
        df_reg[col] = df_reg[col].fillna(df_reg[col].mean())
    
    # Drop any remaining rows with NaNs
    df_reg = df_reg.dropna(subset=[temp_col, "Wind", "Price"])
    
    # Report the number of observations after handling missing values
    st.write(f"Number of observations used in regression: {len(df_reg)}")
    
    # Create the OLS model
    X = df_reg[[temp_col, "Wind"]]
    X = sm.add_constant(X)
    y = df_reg["Price"]
    model = sm.OLS(y, X).fit()

    # Display a simplified regression summary
    st.subheader("OLS Regression Results")
    
    # Display model statistics
    st.write("**Model Statistics:**")
    model_stats = pd.DataFrame({
        "Statistic": ["Observations", "R-squared", "Adjusted R-squared", "F-statistic", "Prob (F-statistic)"],
        "Value": [int(model.nobs), round(model.rsquared, 3), round(model.rsquared_adj, 3), 
                  round(model.fvalue, 3), f"{model.f_pvalue:.3g}"]
    })
    st.table(model_stats)
    
    # Display coefficients
    st.write("**Coefficients:**")
    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std. Error": model.bse,
        "t-value": model.tvalues,
        "P>|t|": model.pvalues,
    })
    
    # Round values for better display
    coef_df = coef_df.round(4)
    
    # Highlight significant p-values
    def highlight_pval(val):
        if isinstance(val, float) and val < 0.05:
            return 'background-color: rgba(0, 255, 0, 0.2)'
        return ''
    
    st.table(coef_df.style.applymap(highlight_pval, subset=["P>|t|"]))
    
    # Display confidence intervals
    st.write("**95% Confidence Intervals:**")
    conf_int = model.conf_int()
    conf_int.columns = ["Lower", "Upper"]
    conf_int_df = pd.DataFrame({
        "Variable": conf_int.index,
        "Lower 95% CI": conf_int["Lower"],
        "Upper 95% CI": conf_int["Upper"]
    }).round(4)
    
    st.table(conf_int_df)
    
    # Create a scatter plot with regression line
    st.subheader("Actual vs. Predicted Prices")
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
    ax_pred.scatter(y, model.fittedvalues, alpha=0.5)
    ax_pred.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax_pred.set_xlabel("Actual Price (€)")
    ax_pred.set_ylabel("Predicted Price (€)")
    ax_pred.set_title("Actual vs. Predicted Prices")
    st.pyplot(fig_pred)

# =============================================================================
# 9. Polynomial Regression Page
# =============================================================================
elif page == "Polynomial Regression":
    st.header("Polynomial Regression Analysis")
    st.write(
        "In this section, we expand the regression model by adding polynomial (squared and interaction) terms. "
        "This helps capture potential nonlinear relationships between the variables and electricity price."
    )
    
    # Use the appropriate temperature column
    temp_col = "temperature_C" if "temperature_C" in df.columns else "Temp"
    
    # Start with a clean copy of the data
    df_poly = df.copy()
    
    # Impute missing values for price (target variable)
    df_poly["Price"] = df_poly["Price"].fillna(df_poly["Price"].mean())
    
    # Display missing values in features before imputation
    st.subheader("Missing Values Before Imputation")
    missing_before = df_poly[[temp_col, "Wind"]].isnull().sum()
    st.write(missing_before)
    
    # Impute missing values in predictors
    for col in [temp_col, "Wind"]:
        if col in df_poly.columns:
            df_poly[col] = df_poly[col].fillna(df_poly[col].mean())
    
    # Create polynomial features
    poly_features = create_polynomial_features(df_poly, [temp_col, "Wind"], degree=2)
    
    # Rename columns to avoid confusion with original features
    poly_features.columns = [
        f"{temp_col}^2",
        f"{temp_col}_Wind", 
        "Wind^2"
    ]
    
    # Add the polynomial features to the DataFrame
    df_poly = pd.concat([df_poly, poly_features], axis=1)
    
    # Display the DataFrame with polynomial features
    st.subheader("Data with Polynomial Features (First 5 Rows)")
    display_cols = ["date_FI", "Price", temp_col, "Wind"] + list(poly_features.columns)
    safe_dataframe_display(df_poly, display_cols)
    
    # Prepare the regression input
    X_poly = df_poly[[temp_col, "Wind"] + list(poly_features.columns)]
    X_poly = sm.add_constant(X_poly)
    y_poly = df_poly["Price"]
    
    # Fit the polynomial regression model
    model_poly = sm.OLS(y_poly, X_poly).fit()
    
    # Display the regression results
    st.subheader("Polynomial OLS Regression Summary")
    
    # Display model statistics
    st.write("**Model Statistics:**")
    model_stats = pd.DataFrame({
        "Statistic": ["Observations", "R-squared", "Adjusted R-squared", "F-statistic", "Prob (F-statistic)"],
        "Value": [int(model_poly.nobs), round(model_poly.rsquared, 3), round(model_poly.rsquared_adj, 3), 
                  round(model_poly.fvalue, 3), f"{model_poly.f_pvalue:.3g}"]
    })
    st.table(model_stats)
    
    # Display coefficients
    st.write("**Coefficients:**")
    coef_df = pd.DataFrame({
        "Variable": model_poly.params.index,
        "Coefficient": model_poly.params.values,
        "Std. Error": model_poly.bse,
        "t-value": model_poly.tvalues,
        "P>|t|": model_poly.pvalues,
    })
    
    # Round values for better display
    coef_df = coef_df.round(4)
    
    # Highlight significant p-values
    def highlight_pval(val):
        if isinstance(val, float) and val < 0.05:
            return 'background-color: rgba(0, 255, 0, 0.2)'
        return ''
    
    st.table(coef_df.style.applymap(highlight_pval, subset=["P>|t|"]))
    
    # Display confidence intervals
    st.write("**95% Confidence Intervals:**")
    conf_int = model_poly.conf_int()
    conf_int.columns = ["Lower", "Upper"]
    conf_int_df = pd.DataFrame({
        "Variable": conf_int.index,
        "Lower 95% CI": conf_int["Lower"],
        "Upper 95% CI": conf_int["Upper"]
    }).round(4)
    
    st.table(conf_int_df)
    
    # Residual Diagnostics
    st.subheader("Residuals Diagnostics for Polynomial Model")
    residuals_poly = model_poly.resid
    
    col1, col2 = st.columns(2)
    
    with col1:
        # QQ Plot
        fig8, ax8 = plt.subplots(figsize=(8, 6))
        sm.qqplot(residuals_poly, line='s', ax=ax8)
        ax8.set_title("QQ Plot of Residuals")
        st.pyplot(fig8)
    
    with col2:
        # Histogram of Residuals
        fig9, ax9 = plt.subplots(figsize=(8, 6))
        ax9.hist(residuals_poly, bins=20, edgecolor='k', alpha=0.7)
        ax9.set_xlabel("Residuals")
        ax9.set_ylabel("Frequency")
        ax9.set_title("Histogram of Residuals")
        st.pyplot(fig9)
    
    # Residuals vs Fitted Values
    fig10, ax10 = plt.subplots(figsize=(10, 6))
    ax10.scatter(model_poly.fittedvalues, residuals_poly, alpha=0.5)
    ax10.axhline(y=0, color='r', linestyle='-')
    ax10.set_xlabel("Fitted Values")
    ax10.set_ylabel("Residuals")
    ax10.set_title("Residuals vs Fitted Values")
    st.pyplot(fig10)

# =============================================================================
# 10. Residual Diagnostics (Interaction Model) Page
# =============================================================================
elif page == "Residual Diagnostics":
    st.header("Residual Diagnostics (Interaction Model)")
    st.write(
        "This section builds an interaction model by including lagged features and interaction terms "
        "to capture potential time-lagged effects. The residual plots help evaluate model fit and assumptions."
    )
    
    # Use the appropriate temperature column
    temp_col = "temperature_C" if "temperature_C" in df.columns else "Temp"
    
    # Create a copy for the interaction model
    df_int = df.copy()
    
    # Impute missing values for price (target variable)
    df_int["Price"] = df_int["Price"].fillna(df_int["Price"].mean())
    
    # Impute missing values in predictors
    for col in [temp_col, "Wind"]:
        if col in df_int.columns:
            df_int[col] = df_int[col].fillna(df_int[col].mean())
    
    # Create lagged features
    df_int["temp_lag1"] = df_int[temp_col].shift(1)
    df_int["wind_lag1"] = df_int["Wind"].shift(1)
    
    # Create interaction term
    df_int["temp_wind_interaction"] = df_int[temp_col] * df_int["Wind"]
    
    # Create polynomial terms
    df_int[f"{temp_col}^2"] = df_int[temp_col] ** 2
    df_int["Wind^2"] = df_int["Wind"] ** 2
    
    # Drop rows with NaN values (due to shift operation)
    df_int = df_int.dropna()
    
    # Display DataFrame with interaction features
    st.subheader("Data with Interaction Features (First 5 Rows)")
    interaction_cols = ["date_FI", "Price", temp_col, "Wind", "temp_lag1", "wind_lag1", 
                        "temp_wind_interaction", f"{temp_col}^2", "Wind^2"]
    safe_dataframe_display(df_int, interaction_cols)
    
    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numeric_cols = ["Price", temp_col, "Wind", "temp_lag1", "wind_lag1", 
                   "temp_wind_interaction", f"{temp_col}^2", "Wind^2"]
    corr_matrix = df_int[numeric_cols].corr()
    
    fig12, ax12 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, annot_kws={"size": 10}, ax=ax12)
    ax12.set_title("Correlation Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig12)
    
    # Fit the interaction model
    X_interaction = df_int[[temp_col, "Wind", "temp_lag1", "wind_lag1", 
                          "temp_wind_interaction", f"{temp_col}^2", "Wind^2"]]
    X_interaction = sm.add_constant(X_interaction)
    y_interaction = df_int["Price"]
    
    model_interaction = sm.OLS(y_interaction, X_interaction).fit()
    
    # Display the regression results
    st.subheader("Interaction Model Regression Summary")
    
    # Display model statistics
    st.write("**Model Statistics:**")
    model_stats = pd.DataFrame({
        "Statistic": ["Observations", "R-squared", "Adjusted R-squared", "F-statistic", "Prob (F-statistic)"],
        "Value": [int(model_interaction.nobs), round(model_interaction.rsquared, 3), round(model_interaction.rsquared_adj, 3), 
                  round(model_interaction.fvalue, 3), f"{model_interaction.f_pvalue:.3g}"]
    })
    st.table(model_stats)
    
    # Display coefficients
    st.write("**Coefficients:**")
    coef_df = pd.DataFrame({
        "Variable": model_interaction.params.index,
        "Coefficient": model_interaction.params.values,
        "Std. Error": model_interaction.bse,
        "t-value": model_interaction.tvalues,
        "P>|t|": model_interaction.pvalues,
    })
    
    # Round values for better display
    coef_df = coef_df.round(4)
    
    # Highlight significant p-values
    def highlight_pval(val):
        if isinstance(val, float) and val < 0.05:
            return 'background-color: rgba(0, 255, 0, 0.2)'
        return ''
    
    st.table(coef_df.style.applymap(highlight_pval, subset=["P>|t|"]))
    
    # Display confidence intervals
    st.write("**95% Confidence Intervals:**")
    conf_int = model_interaction.conf_int()
    conf_int.columns = ["Lower", "Upper"]
    conf_int_df = pd.DataFrame({
        "Variable": conf_int.index,
        "Lower 95% CI": conf_int["Lower"],
        "Upper 95% CI": conf_int["Upper"]
    }).round(4)
    
    st.table(conf_int_df)
    
    # Residual Plots
    st.subheader("Residual Plots for Interaction Model")
    residuals = model_interaction.resid
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals vs Fitted Values
        fig22, ax22 = plt.subplots(figsize=(8, 6))
        ax22.scatter(model_interaction.fittedvalues, residuals, alpha=0.5)
        ax22.axhline(y=0, color='red', linestyle='--')
        ax22.set_xlabel("Fitted Values")
        ax22.set_ylabel("Residuals")
        ax22.set_title("Residuals vs Fitted Values")
        st.pyplot(fig22)
    
    with col2:
        # QQ Plot
        fig23, ax23 = plt.subplots(figsize=(8, 6))
        sm.qqplot(residuals, line='45', fit=True, ax=ax23)
        ax23.set_title("Q-Q Plot of Residuals")
        st.pyplot(fig23)
    
    # Histogram of Residuals
    fig24, ax24 = plt.subplots(figsize=(10, 6))
    ax24.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax24.set_xlabel("Residuals")
    ax24.set_ylabel("Frequency")
    ax24.set_title("Histogram of Residuals")
    st.pyplot(fig24)
    
    # Shapiro-Wilk Test for Normality
    shapiro_test = stats.shapiro(residuals)
    st.write("### Shapiro-Wilk Test for Normality of Residuals")
    st.write(f"Statistic: {shapiro_test.statistic:.4f}")
    st.write(f"p-value: {shapiro_test.pvalue:.4g}")
    
    if shapiro_test.pvalue < 0.05:
        st.write("The residuals are not normally distributed (p < 0.05)")
    else:
        st.write("The residuals appear to be normally distributed (p >= 0.05)")