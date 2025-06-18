import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Mix Optimizer", layout="centered")
st.title("⚡ Energy Mix Optimizer")

# Load dataset
data = pd.read_csv("data/energy_mix_dataset.csv", parse_dates=["datetime"], index_col="datetime")

# Sidebar inputs
st.sidebar.header("Input Forecast")
ssr = st.sidebar.slider("Solar radiation (W/m²)", 0, 1000, 250)
wind_u = st.sidebar.slider("Wind U (m/s)", -10, 10, 2)
wind_v = st.sidebar.slider("Wind V (m/s)", -10, 10, 2)
temp = st.sidebar.slider("Temperature (K)", 260, 310, 295)
price = st.sidebar.slider("OMIE price (€/MWh)", 0, 200, 75)
demand = st.sidebar.slider("Forecasted demand (MWh)", 100, 1000, 600)

# Build input row
row = pd.DataFrame([{
    "surface_solar_radiation_downwards": ssr,
    "10m_u_component_of_wind": wind_u,
    "10m_v_component_of_wind": wind_v,
    "2m_temperature": temp,
    "hour": 13,
    "month": 6,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_holiday": 0,
    "price": price
}])

# Train models
features = data.drop(columns=["demand", "solar_pv", "wind", "hydro"])
y_solar = data["solar_pv"]
y_wind = data["wind"]

X_train_s, _, y_train_s, _ = train_test_split(features, y_solar, test_size=0.2, shuffle=False)
X_train_w, _, y_train_w, _ = train_test_split(features, y_wind, test_size=0.2, shuffle=False)

model_s = XGBRegressor(n_estimators=100)
model_w = XGBRegressor(n_estimators=100)
model_s.fit(X_train_s, y_train_s)
model_w.fit(X_train_w, y_train_w)

solar_pred = model_s.predict(row)[0]
wind_pred = model_w.predict(row)[0]
hydro_pred = data["hydro"].mean()

# Optimisation
costs = [30, 25, 40, 80, 90]
emissions = [0, 0, 10, 450, 900]
bounds = [
    (0, solar_pred),
    (0, wind_pred),
    (0, hydro_pred),
    (0, 300),
    (0, 300)
]

A_eq = [[1, 1, 1, 1, 1]]
b_eq = [demand]

res = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

# Results
if res.success:
    mix = res.x
    sources = ["Solar", "Wind", "Hydro", "Gas", "Nuclear"]
    total_cost = np.dot(mix, costs)
    total_emissions = np.dot(mix, emissions) / 1000

    st.subheader("⚙️ Optimised Energy Mix")
    df_result = pd.DataFrame({"Technology": sources, "Dispatch (MWh)": mix})
    st.bar_chart(df_result.set_index("Technology"))

    st.markdown(f"**💰 Total Cost:** €{total_cost:.2f}")
    st.markdown(f"**🌱 Emissions:** {total_emissions:.2f} tCO₂")
else:
    st.error("❌ Optimisation failed.")
