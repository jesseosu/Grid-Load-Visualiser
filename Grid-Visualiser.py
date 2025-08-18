import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

# ----------------------------
# Simulate load data
# ----------------------------
def simulate_load_data(start_time, hours=24, base_load=400, temp=30):
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    load = []
    for hour in range(hours):
        # Simulate load pattern: higher during day, lower at night
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
        "Load (MW)": load
    })

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Grid Load Visualiser & Forecaster")

st.sidebar.header("Simulation Settings")
temperature = st.sidebar.slider("Ambient Temperature (°C)", min_value=10, max_value=45, value=30)
base_load = st.sidebar.slider("Base Load (MW)", min_value=200, max_value=500, value=400)
hours = st.sidebar.slider("Duration (hours)", min_value=12, max_value=48, value=24)

start_time = datetime.now()
df = simulate_load_data(start_time, hours=hours, base_load=base_load, temp=temperature)

# Plot
fig = px.line(df, x="Time", y="Load (MW)", title="Simulated Grid Load Over Time",
              markers=True, line_shape="spline")
fig.update_layout(xaxis_title="Time", yaxis_title="Load (MW)", title_x=0.5)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
This simulation forecasts grid load using simplified assumptions:
- Higher temperature increases daytime load
- Peak demand typically occurs around late afternoon/evening
- Load includes simulated random fluctuations for realism

Modify temperature or base load to see impact on the system.

- Full support for Streamlit, Python virtual environments, and data science tools
- Excellent UI, autocomplete, debugging, and Plotly integration

To run this project in PyCharm:
1. Clone the repo or open the directory
2. Create a virtual environment
3. Install dependencies (e.g., `pip install streamlit pandas numpy plotly`)
4. Run: streamlit run grid-visualiser.py

Consider adding a `requirements.txt` to streamline setup.
""")
