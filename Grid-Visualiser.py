import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Load Simulation Function

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


# Weather API

def fetch_temperature(city):
    try:
        api_key = "your_openweathermap_api_key"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        return response.json()['main']['temp']
    except:
        return 30


# Forecasting Function

def forecast_load(df_zone, model_type="Linear"):
    df_zone = df_zone.copy()
    df_zone['Hour'] = range(len(df_zone))
    X = df_zone[['Hour']]
    y = df_zone['Load (MW)']

    if model_type == "Linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, y)

    X_future = pd.DataFrame({"Hour": range(len(df_zone), len(df_zone)+12)})
    y_future = model.predict(X_future)
    future_times = [df_zone['Time'].max() + timedelta(hours=i+1) for i in range(12)]

    return pd.DataFrame({
        "Time": future_times,
        "Zone": df_zone['Zone'].iloc[0],
        "Load (MW)": y_future
    })


# Streamlit App Layout

st.set_page_config("Powerlink Grid Load Simulator", layout="wide")
st.title("⚡ Grid Load Simulator & Forecaster for Powerlink")

# --- Sidebar Settings ---
st.sidebar.header("Simulation Controls")
city = st.sidebar.selectbox("Weather City", ["Brisbane", "Gold Coast", "Cairns"])
temperature = fetch_temperature(city)
st.sidebar.markdown(f"**Current Temp:** {temperature:.1f} °C")

zones = st.sidebar.multiselect("Zones", ["North Brisbane", "South Brisbane", "Gold Coast", "Cairns"], default=["North Brisbane"])
base_load = st.sidebar.slider("Base Load (MW)", 200, 500, 350)
hours = st.sidebar.slider("Simulation Hours", 12, 72, 24)
threshold = st.sidebar.slider("Overload Threshold (MW)", 800, 1500, 950)
model_choice = st.sidebar.selectbox("Forecasting Model", ["Linear", "Random Forest"])


# Simulate and Display Load

start_time = datetime.now()
df_all = pd.concat([simulate_load_data(start_time, hours, base_load, temperature, zone) for zone in zones])

st.subheader("Load Over Time")
fig = px.line(df_all, x="Time", y="Load (MW)", color="Zone", markers=False)
fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Overload Threshold")
st.plotly_chart(fig, use_container_width=True)


# Forecasting

st.subheader("Load Forecasting")
forecast_zone = st.selectbox("Select Zone for Forecast", zones)
df_zone = df_all[df_all["Zone"] == forecast_zone]
df_forecast = forecast_load(df_zone, model_type=model_choice)

fig2 = px.line(pd.concat([df_zone[['Time', 'Load (MW)']], df_forecast]), x="Time", y="Load (MW)", color_discrete_sequence=["blue"])
fig2.update_traces(line=dict(dash="dot"), selector=dict(mode="lines"))
fig2.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Overload Threshold")
fig2.update_layout(title=f"Forecast for {forecast_zone} ({model_choice})")
st.plotly_chart(fig2, use_container_width=True)


# Peak Load Summary

st.subheader("Peak Load Summary")
summary = df_all.groupby("Zone").agg(Max_Load_MW=("Load (MW)", "max"))
st.dataframe(summary.style.highlight_max(axis=0))


# Weather vs Load Plot

st.subheader("Weather vs Load Correlation")
scatter = px.scatter(df_all, x="Temp (°C)", y="Load (MW)", color="Zone", trendline="ols")
st.plotly_chart(scatter, use_container_width=True)


# Download Button

st.download_button("Export CSV", df_all.to_csv(index=False).encode(), file_name="powerlink_load_simulation.csv")


# Info Footer

st.markdown("""
---
### Project Summary
This interactive simulation helps model and forecast electrical grid load across Queensland zones under varying temperature conditions.

- Demonstrates correlation between heat and load
- Alerts for overload risks
- Implements forecasting models
- Simulates real-time grid load dynamics
- Showcases user interface development with Streamlit and Plotly

Built as a showcase for Powerlink’s **Energy Insights & Digital Systems Internship**.
""")
