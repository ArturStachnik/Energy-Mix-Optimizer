import pandas as pd
import numpy as np
import os
import holidays

# -----------------------------------------
# Configuration
# -----------------------------------------
SAVE_PATH = "data/energy_mix_dataset.csv"

# -----------------------------------------
# Functions
# -----------------------------------------
def load_data():
    esios = pd.read_csv("data/esios_mix.csv", parse_dates=["datetime"], index_col="datetime")
    omie = pd.read_csv("data/omie_prices.csv", parse_dates=["datetime"], index_col="datetime")
    omie = omie.rename(columns={"price_spain": "price"})
    clima = pd.read_csv("data/era5_weather.csv", parse_dates=["datetime"], index_col="datetime")
    return esios, omie, clima

def add_time_features(df):
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = df.index.date.astype("datetime64").isin(holidays.Spain()).astype(int)
    return df

def merge_all():
    esios, omie, clima = load_data()
    df = esios.join([omie, clima], how="inner")
    df = add_time_features(df)
    df = df.dropna().sort_index()
    os.makedirs("data", exist_ok=True)
    df.to_csv(SAVE_PATH)
    print(f"✔ Dataset saved to {SAVE_PATH} with shape {df.shape}")

# -----------------------------------------
# CLI entry point
# -----------------------------------------
if __name__ == "__main__":
    merge_all()

