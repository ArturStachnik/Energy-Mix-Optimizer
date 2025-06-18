import os
import zipfile
import io
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# -----------------------------------------
# Configuration
# -----------------------------------------
BASE_URL = "https://www.omie.es/sites/default/files/dados/AGNO_{year}/MES_{month:02d}/TXT/INT_PBC_PBP_{year}{month:02d}{day:02d}.zip"

# -----------------------------------------
# Functions
# -----------------------------------------
def fetch_price_for_day(day):
    url = BASE_URL.format(year=day.year, month=day.month, day=day.day)
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            filename = [f for f in z.namelist() if f.endswith(".csv")][0]
            with z.open(filename) as f:
                df = pd.read_csv(f, sep=";", decimal=",", header=None,
                                 names=["date", "hour", "price_spain", "price_portugal"])
                df["datetime"] = pd.to_datetime(df["date"] + " " + df["hour"].astype(str) + ":00")
                df = df.set_index("datetime")[["price_spain"]]
                return df
    except Exception as e:
        print(f"Failed to fetch {day}: {e}")
        return None

def download_omie_prices(start_date="2018-01-01", end_date=None, save_path="data/omie_prices.csv"):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    all_days = pd.date_range(start, end, freq="D")
    frames = []
    for day in tqdm(all_days, desc="Downloading OMIE daily prices"):
        df = fetch_price_for_day(day)
        if df is not None:
            frames.append(df)

    if frames:
        result = pd.concat(frames)
        result.to_csv(save_path)
        print(f"✔ OMIE price data saved to {save_path}")
    else:
        print("✖ No price data fetched.")

# -----------------------------------------
# CLI entry point
# -----------------------------------------
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_omie_prices(start_date="2018-01-01")
