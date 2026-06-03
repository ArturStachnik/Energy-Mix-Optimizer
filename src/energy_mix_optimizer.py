import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# -----------------------------------------
# Load dataset
# -----------------------------------------
data = pd.read_csv("data/energy_mix_dataset.csv", parse_dates=["datetime"], index_col="datetime")

# Target demand and features
features = data.drop(columns=["demand", "solar_pv", "wind", "hydro"])
target_solar = data["solar_pv"]
target_wind = data["wind"]
target_hydro = data["hydro"]  # Optional: use historical mean or simple model

def train_and_predict(X, y, climate_input):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"RMSE: {mean_squared_error(y_test, model.predict(X_test), squared=False):.2f}")
    return model.predict(climate_input)[0]

# -----------------------------------------
# Forecast with current climate data
# -----------------------------------------
tomorrow_input = pd.DataFrame([{
    "surface_solar_radiation_downwards": 210,
    "10m_u_component_of_wind": 2.5,
    "10m_v_component_of_wind": 1.8,
    "2m_temperature": 295,
    "hour": 13,
    "month": 6,
    "day_of_week": 3,
    "is_weekend": 0,
    "is_holiday": 0,
    "price": 75
}])

solar_pred = train_and_predict(features, target_solar, tomorrow_input)
wind_pred  = train_and_predict(features, target_wind, tomorrow_input)
hydro_pred = data["hydro"].mean()  # Static average as a placeholder

print(f"\nForecast:")
print(f"Solar: {solar_pred:.2f} MWh")
print(f"Wind: {wind_pred:.2f} MWh")
print(f"Hydro: {hydro_pred:.2f} MWh")

# -----------------------------------------
# Optimisation: minimise cost under constraints
# -----------------------------------------
demand_forecast = 580  # You can predict this with ML too
costs = [30, 25, 40, 80, 90]         # €/MWh for [solar, wind, hydro, gas, nuclear]
emissions = [0, 0, 10, 450, 900]     # gCO₂/kWh
bounds = [
    (0, solar_pred),
    (0, wind_pred),
    (0, hydro_pred),
    (0, 300),  # gas
    (0, 300)   # nuclear
]

A_eq = [[1, 1, 1, 1, 1]]
b_eq = [demand_forecast]

res = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

# -----------------------------------------
# Output
# -----------------------------------------
if res.success:
    mix = res.x
    sources = ["Solar", "Wind", "Hydro", "Gas", "Nuclear"]
    for s, v in zip(sources, mix):
        print(f"{s}: {v:.2f} MWh")

    total_cost = np.dot(mix, costs)
    total_emissions = np.dot(mix, emissions) / 1000
    print(f"\nTotal cost: €{total_cost:.2f}")
    print(f"Total CO₂: {total_emissions:.2f} t")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(sources, mix)
    plt.ylabel("MWh")
    plt.title("Optimised Energy Mix")
    plt.tight_layout()
    plt.show()
else:
    print("✖ Optimisation failed.")

