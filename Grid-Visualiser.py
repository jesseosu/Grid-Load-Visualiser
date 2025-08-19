import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Functions

def simulate_load_data(start_time, hours, base_load, temp, zone_name):
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    load = []
    for i in range(hours):
        hour = timestamps[i].hour
        if 6 <= hour <= 9:
            fluctuation = np.random.normal(0, 15)
            current_load = base_load + 100 + fluctuation
        elif 10 <= hour <= 17:
            fluctuation = np.random.normal(0, 20)
            current_load = base_load + 150 + fluctuation + (temp - 25) * 5
        elif 18 <= hour <= 21:
            fluctuation = np.random.normal(0, 25)
            current_load = base_load + 200 + fluctuation
        else:
            fluctuation = np.random.normal(0, 10)
            current_load = base_load + fluctuation
        load.append(max(current_load, 0))

    return pd.DataFrame({
        "Time": timestamps,
        "Zone": zone_name,
        "Load (MW)": load,
        "Temp (°C)": temp
    })

def fetch_temperature(city):
    try:
        api_key = "your_openweathermap_api_key"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        return response.json()['main']['temp']
    except:
        return 30

def forecast_load(df_zone, model_type="Linear"):
    df_zone = df_zone.copy()
    df_zone['Hour'] = range(len(df_zone))
    X = df_zone[['Hour']]
    y = df_zone['Load (MW)']

    model = LinearRegression() if model_type == "Linear" else RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    X_future = pd.DataFrame({"Hour": range(len(df_zone), len(df_zone)+12)})
    y_future = model.predict(X_future)
    future_times = [df_zone['Time'].max() + timedelta(hours=i+1) for i in range(12)]

    return pd.DataFrame({
        "Time": future_times,
        "Zone": df_zone['Zone'].iloc[0],
        "Load (MW)": y_future
    })


# Streamlit UI

st.set_page_config("Powerlink Grid Load Forecaster", layout="wide")
st.title("⚡ Powerlink Grid Load Simulator & Forecasting Tool")

if st.button("Refresh Data"):
    st.rerun()

city = st.sidebar.selectbox("Select City for Weather", ["Brisbane", "Gold Coast", "Cairns"])
temperature = fetch_temperature(city)
st.sidebar.write(f"Current Temperature: {temperature:.1f} °C")

zones = st.sidebar.multiselect("Zones to Simulate", ["North Brisbane", "South Brisbane", "Gold Coast", "Cairns"], default=["North Brisbane"])
base_load = st.sidebar.slider("Base Load (MW)", 200, 500, 350)
hours = st.sidebar.slider("Simulation Duration (Hours)", 12, 72, 24)
threshold = st.sidebar.slider("Overload Threshold (MW)", 800, 1500, 950)
model_choice = st.sidebar.selectbox("Forecasting Model", ["Linear", "Random Forest"])
renewable_percent = st.sidebar.slider("Renewable Contribution (%)", 0, 100, 30)
use_battery = st.sidebar.checkbox("Enable Battery Storage")

adjusted_base_load = base_load * (1 - renewable_percent / 100)
start_time = datetime.now()

dfs = []
if zones:
    for zone in zones:
        df_zone = simulate_load_data(start_time, hours, adjusted_base_load, temperature, zone)

        if use_battery:
            df_zone['Battery Action'] = np.where(df_zone['Load (MW)'] > threshold, 'Charging', 'Discharging')
            df_zone['Adjusted Load (MW)'] = np.where(df_zone['Load (MW)'] > threshold,
                                                    df_zone['Load (MW)'] - 50,
                                                    df_zone['Load (MW)'] + 30)
        else:
            df_zone['Battery Action'] = 'N/A'
            df_zone['Adjusted Load (MW)'] = df_zone['Load (MW)']

        dfs.append(df_zone)

if not dfs:
    st.error("Please select at least one zone.")
    st.stop()

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = st.sidebar.file_uploader("Or Upload Historical Load CSV")

if st.session_state.uploaded_file:
    df_all = pd.read_csv(st.session_state.uploaded_file, parse_dates=["Time"])
else:
    df_all = pd.concat(dfs)

# Scenario tag
df_all["Scenario"] = f"BaseLoad={base_load}, Temp={temperature:.1f}°C, Renewable={renewable_percent}%"

# Smoothed Load
df_all["Smoothed Load"] = df_all["Adjusted Load (MW)"].rolling(window=3).mean()

# Emissions estimate
grid_intensity = 0.8  # kg CO2 per kWh
df_all["Estimated Emissions (t)"] = df_all["Adjusted Load (MW)"] * 1000 * grid_intensity / 1e6

# Main Chart
st.subheader("Load Over Time")
fig = px.line(df_all, x="Time", y="Adjusted Load (MW)", color="Zone")
fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Overload Threshold")
st.plotly_chart(fig, use_container_width=True)

# Forecasting
st.subheader("Forecasting")
forecast_zone = st.selectbox("Select Zone for Forecasting", zones)
df_zone = df_all[df_all["Zone"] == forecast_zone]
df_forecast = forecast_load(df_zone, model_choice)

if df_forecast['Load (MW)'].max() > threshold:
    st.warning("Forecast indicates demand may exceed capacity. Consider initiating demand response protocols.")

fig2 = px.line(pd.concat([df_zone[['Time', 'Adjusted Load (MW)']].rename(columns={'Adjusted Load (MW)': 'Load (MW)'}), df_forecast]), x="Time", y="Load (MW)", color_discrete_sequence=["blue"])
fig2.update_traces(line=dict(dash="dot"), selector=dict(mode="lines"))
fig2.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Overload Threshold")
fig2.update_layout(title=f"Forecast for {forecast_zone} ({model_choice})")
st.plotly_chart(fig2, use_container_width=True)

# Peak Load Summary
st.subheader("📊 Peak Load Summary")
summary = df_all.groupby("Zone").agg(Max_Load_MW=("Adjusted Load (MW)", "max"), Avg_Load_MW=("Adjusted Load (MW)", "mean"), Total_Emissions_t=("Estimated Emissions (t)", "sum"))
st.dataframe(summary.style.highlight_max(axis=0))

# Load Duration Curve
st.subheader("Load Duration Curve")
ldf = df_all.copy()
ldf.sort_values("Adjusted Load (MW)", ascending=False, inplace=True)
ldf["Duration %"] = 100 * np.arange(len(ldf)) / len(ldf)
fig_ldc = px.line(ldf, x="Duration %", y="Adjusted Load (MW)")
fig_ldc.update_layout(title="Load Duration Curve")
st.plotly_chart(fig_ldc, use_container_width=True)

# Temperature vs Load Correlation
st.subheader("Temperature vs Load")
scatter = px.scatter(df_all, x="Temp (°C)", y="Adjusted Load (MW)", color="Zone", trendline="ols")
st.plotly_chart(scatter, use_container_width=True)

# Download Button
st.download_button("Download CSV file", df_all.to_csv(index=False).encode(), file_name="powerlink_simulation.csv")

# Footer
st.markdown("""
---
### Project Notes
- Simulates demand across zones with weather impact, renewable offset, and optional battery storage
- Forecasts based on linear or machine learning models
- Adds emissions analysis, load duration curves, demand response warnings
- Designed in context of Australia's NEM grid
- Built by a passionate engineer inspired by Powerlink QLD’s mission to lead the energy transition
""")
