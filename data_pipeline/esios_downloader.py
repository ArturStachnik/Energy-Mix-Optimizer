import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------------------
# Configuration
# -----------------------------------------
BASE_URL = "https://api.esios.ree.es/indicators/{indicator_id}?start_date={start}&end_date={end}"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "x-api-key": os.getenv("ESIOS_TOKEN"),  # Set your API token as an environment variable
    "Host": "api.esios.ree.es"
}

# List of indicator IDs to fetch (REE official codes)
INDICATORS = {
    "demand": 460,
    "wind": 540,
    "solar_pv": 542,
    "solar_thermal": 543,
    "hydro": 72,     # You may combine 71 (ugh) and 72 if desired
    "nuclear": 74,
    "gas_cc": 79
}

# -----------------------------------------
# Functions
# -----------------------------------------
def fetch_indicator(indicator_id, start_date, end_date):
    url = BASE_URL.format(indicator_id=indicator_id, start=start_date.isoformat(), end=end_date.isoformat())
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch indicator {indicator_id}: {response.status_code}")
    data = response.json()["indicator"]["values"]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")["value"].astype(float)
    return df

def download_esios_data(start_date="2018-01-01", end_date=None, save_path="data/esios_mix.csv"):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    frames = {}
    for name, code in INDICATORS.items():
        print(f"Fetching {name}...")
        df = fetch_indicator(code, start, end)
        frames[name] = df

    combined = pd.concat(frames.values(), axis=1)
    combined.columns = list(frames.keys())
    combined.to_csv(save_path)
    print(f"✔ Data saved to {save_path}")

# -----------------------------------------
# CLI entry point
# -----------------------------------------
if __name__ == "__main__":
    download_esios_data(start_date="2018-01-01")

