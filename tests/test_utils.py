import pytest
import numpy as np
import pandas as pd
from src.utils import predict_generation, optimise_mix

# -------------------------
# Dummy data for testing
# -------------------------
X = pd.DataFrame({
    "surface_solar_radiation_downwards": [200, 250, 300, 400],
    "10m_u_component_of_wind": [2, 3, 4, 5],
    "10m_v_component_of_wind": [1, 1.5, 2, 2.5],
    "2m_temperature": [290, 292, 294, 296],
    "hour": [12, 13, 14, 15],
    "month": [6, 6, 6, 6],
    "day_of_week": [1, 2, 3, 4],
    "is_weekend": [0, 0, 0, 0],
    "is_holiday": [0, 0, 0, 0],
    "price": [50, 55, 60, 65]
})

y = pd.Series([100, 120, 140, 160])

X_input = pd.DataFrame([{
    "surface_solar_radiation_downwards": 280,
    "10m_u_component_of_wind": 4,
    "10m_v_component_of_wind": 2,
    "2m_temperature": 295,
    "hour": 13,
    "month": 6,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_holiday": 0,
    "price": 60
}])

# -------------------------
# Test predict_generation
# -------------------------
def test_predict_generation():
    pred = predict_generation(X, y, X_input)
    assert isinstance(pred, float)
    assert pred > 0

# -------------------------
# Test optimise_mix
# -------------------------
def test_optimise_mix_success():
    solar = 200
    wind = 150
    hydro = 100
    demand = 300
    mix, cost, emissions, success = optimise_mix(solar, wind, hydro, demand)
    assert success
    assert abs(sum(mix) - demand) < 1e-3
    assert cost > 0
    assert emissions >= 0

def test_optimise_mix_failure():
    solar = 0
    wind = 0
    hydro = 0
    demand = 1000
    mix, cost, emissions, success = optimise_mix(solar, wind, hydro, demand)
    assert not success
