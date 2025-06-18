import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.optimize import linprog

# ---------------------------
# Train a time-series model
# ---------------------------
def train_model(X, y):
    model = XGBRegressor(n_estimators=100, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=10, 
                  verbose=False)
        break  # Use first split only
    return model

# ---------------------------
# Predict generation for new input
# ---------------------------
def predict_generation(X_hist, y_hist, X_input):
    model = train_model(X_hist, y_hist)
    prediction = model.predict(X_input)
    return prediction[0]

# ---------------------------
# Optimise generation mix
# ---------------------------
def optimise_mix(solar, wind, hydro, demand):
    # Cost and emissions vectors for [solar, wind, hydro, gas, nuclear]
    costs = np.array([30, 25, 40, 80, 90])
    emissions = np.array([0, 0, 10, 450, 900])
    availability = [solar, wind, hydro, 300, 300]

    bounds = [(0, a) for a in availability]
    A_eq = [np.ones(len(costs))]
    b_eq = [demand]

    res = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    
    if res.success:
        mix = res.x
        total_cost = np.dot(mix, costs)
        total_emissions = np.dot(mix, emissions) / 1000  # kg → t
        return mix, total_cost, total_emissions, True
    else:
        return [0]*5, 0, 0, False
