import cdsapi
import os
import pathlib
from datetime import datetime

# -----------------------------------------
# Configuration
# -----------------------------------------
VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards"
]

AREA = [44, -10, 36, 4]  # [North, West, South, East] - Iberian Peninsula
FOLDER = "data/era5"

os.makedirs(FOLDER, exist_ok=True)

# -----------------------------------------
# Download function
# -----------------------------------------
def download_era5_variable(variable, year):
    file_path = f"{FOLDER}/{variable}_{year}.nc"
    if pathlib.Path(file_path).exists():
        print(f"✔ File already exists: {file_path}")
        return

    print(f"↓ Downloading {variable} for {year}...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": variable,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
        },
        file_path
    )
    print(f"✔ Saved: {file_path}")

# -----------------------------------------
# Main execution
# -----------------------------------------
def download_all(start_year=2018, end_year=None):
    if end_year is None:
        end_year = datetime.today().year

    for year in range(start_year, end_year + 1):
        for variable in VARIABLES:
            download_era5_variable(variable, year)

# -----------------------------------------
# CLI entry point
# -----------------------------------------
if __name__ == "__main__":
    download_all(start_year=2018)

