import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ----------------------------
# Simulate load data per zone
# ----------------------------
def simulate_load_data(start_time, hours, base_load, temp, zone_name):
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    load = []
    for hour in range(hours):
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
        "Load (MW)": load
    })

# ----------------------------
# Weather API Integration
# ----------------------------
def fetch_temperature(city):
    try:
        api_key = "your_openweathermap_api_key"  # Replace with your real API key
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        return data['main']['temp']
    except:
        return 30  # fallback temperature

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Grid Load Visualiser", layout="wide")
st.title("⚡ Grid Load Visualiser & Forecaster")

st.sidebar.header("Simulation Settings")
cities = ["Brisbane", "Gold Coast", "Cairns", "Toowoomba"]
selected_city = st.sidebar.selectbox("Select City (for Weather)", cities)
temperature = fetch_temperature(selected_city)
st.sidebar.write(f"Current temperature: {temperature:.1f} °C")

zones = st.sidebar.multiselect("Zones to Simulate", ["North Brisbane", "South Brisbane", "Gold Coast", "Cairns"], default=["North Brisbane", "Gold Coast"])
base_load = st.sidebar.slider("Base Load (MW)", 200, 500, 400)
hours = st.sidebar.slider("Simulation Duration (hours)", 12, 48, 24)
thresh = st.sidebar.slider("Overload Threshold (MW)", 800, 1500, 950)

# ----------------------------
# Generate Data
# ----------------------------
start_time = datetime.now()
df_all = pd.concat([simulate_load_data(start_time, hours, base_load, temperature, zone) for zone in zones])

# ----------------------------
# Plot Load Curves
# ----------------------------
st.subheader("Simulated Load Over Time")
fig = px.line(df_all, x="Time", y="Load (MW)", color="Zone", markers=True, line_shape="spline")
fig.update_layout(xaxis_title="Time", yaxis_title="Load (MW)", title_x=0.5)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Forecasting (Regression)
# ----------------------------
st.subheader("Simple Forecasting (Linear Regression)")
zone_to_forecast = st.selectbox("Zone to Forecast", zones)
df_zone = df_all[df_all['Zone'] == zone_to_forecast]
df_zone['Hour'] = range(len(df_zone))
X = df_zone[['Hour']]
y = df_zone['Load (MW)']
model = LinearRegression().fit(X, y)
X_future = pd.DataFrame({"Hour": range(len(df_zone), len(df_zone)+12)})
y_future = model.predict(X_future)
future_times = [df_zone['Time'].max() + timedelta(hours=i+1) for i in range(12)]

df_forecast = pd.DataFrame({
    "Time": future_times,
    "Zone": zone_to_forecast,
    "Load (MW)": y_future
})

fig2 = px.line(pd.concat([df_zone[['Time', 'Load (MW)']], df_forecast]), x="Time", y="Load (MW)", title=f"Forecast for {zone_to_forecast}")
fig2.update_traces(line=dict(dash="dot"), selector=dict(mode="lines"))
fig2.update_layout(xaxis_title="Time", yaxis_title="Load (MW)", title_x=0.5)
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Overload Alerts
# ----------------------------
if df_all['Load (MW)'].max() > thresh:
    st.error(f"Peak load exceeded threshold ({thresh} MW)! Max observed: {df_all['Load (MW)'].max():.2f} MW")

# ----------------------------
# Download CSV Option
# ----------------------------
st.download_button("Download Simulation Data as CSV", df_all.to_csv(index=False).encode(), file_name="grid_load_simulation.csv")

# ----------------------------
# Info
# ----------------------------
st.markdown("""
---
### Notes
- Load is based on simulated values per region adjusted by time of day and temperature
- Forecast uses simple linear regression for demo purposes
- Weather data via [OpenWeatherMap](https://openweathermap.org)
- Supports multi-zone input and downloadable reporting

Tip: Try setting temperature above 40°C and observe the impact.
""")