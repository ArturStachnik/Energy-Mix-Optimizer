#Energy-Mix Optimizer

> **Predict Solar & Wind output, forecast demand & automatically recommend the cheapest, lowest-carbon generation mix for the next 24 h.**  
> Built with Python | XGBoost | SciPy | pandas | Streamlit

---

## 1. Why does this project matter?

**Energy companies lose millions every year** by over- or under-estimating renewable output and by dispatching expensive thermal plants when cheaper green capacity is available.  
This repository shows how a data-driven approach can:

| Business Question | Current Pain | Energy-Mix Optimizer Benefit |
|-------------------|-------------|-------------------------------|
| *How much solar & wind will I really have tomorrow?* | Operators rely on rule-of-thumb or overnight forecasts, ignoring local micro-climate. | ML models trained on historical weather → **±4 % RMSE** prediction accuracy. |
| *Which units should I schedule to cover demand at least cost?* | Manual decision ≈ conservative (expensive gas fired in reserve). | Linear optimisation chooses cheapest mix given predicted availability. **Up to 8 % fuel-cost savings¹**. |
| *Can I cut CO₂ without breaching security of supply?* | No quantitative trade-off dashboard. | Emission-weighted cost function delivers **up to 10 kt CO₂ avoided per year** for a 5 GW portfolio. |

¹Back-cast on Spanish 2024 market data; see `notebooks/backtest_2024.ipynb`.

---

## 2. What’s inside?

```text
energy-mix-optimizer/
├─ data_pipeline/                 <- Scripts to download & clean raw data
│   ├─ esios_downloader.py        demand & generation (ESIOS API)
│   ├─ omie_prices_downloader.py  price curves (OMIE)
│   ├─ era5_downloader.py         weather re-analysis (Copernicus ERA5)
│   └─ make_dataset.py            merge & feature-engineer → CSV
├─ src/
│   ├─ energy_mix_optimizer.py    main ML + optimisation engine
│   └─ utils.py                   shared helpers
├─ app/
│   └─ dashboard.py               Streamlit front-end (mix explorer)
├─ notebooks/                     exploratory & benchmarking notebooks
├─ tests/                         unit tests (pytest)
├─ README.md                      ← you are here
└─ requirements.txt
