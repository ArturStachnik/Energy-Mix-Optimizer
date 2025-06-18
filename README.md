
"
#     Energy-Mix Optimizer

> **Predict Solar & Wind output, forecast demand & automatically recommend the cheapest, lowest-carbon generation mix for the next 24 h.**  
> Built with Python | XGBoost | SciPy | pandas | Streamlit

---

## 1. Why does this project matter?

**Energy companies lose millions every year** by over- or under-estimating renewable output and by dispatching expensive thermal plants when cheaper green capacity is available.  
This repository shows how a data-driven approach can:

| Business Question | Current Pain | Energy-Mix Optimizer Benefit |
|-------------------|--------------|-------------------------------|
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
```

---

## 3. Data sources

| Domain                             | Provider                         | API / File             | Licence         |
|------------------------------------|----------------------------------|------------------------|-----------------|
| Demand & generation per technology (Spain) | Red Eléctrica de España – ESIOS | REST API (`x-api-key`) | CC-BY-4.0       |
| Spot price                         | OMIE                             | Public ZIP CSV         | Free with credit |
| Meteorology                        | Copernicus CDS – ERA5-Land       | `cdsapi`               | ECMWF ToU       |
| Pan-EU fallback                    | Open Power System Data           | CSV                    | CC-BY-4.0       |

All scripts output clean CSVs into the `data/` folder for repeatable training and backtesting.

---

## 4. Modelling pipeline

### 4.1 Renewable output prediction

| Step         | Technique          | Notes                                 |
|--------------|-------------------|----------------------------------------|
| Feature load | `pandas`          | Weather + lag + calendar vars          |
| Model        | `XGBoostRegressor`| 100 trees, early-stopping              |
| Split        | `TimeSeriesSplit` | Train: 2018–2023 / Test: 2024+         |
| Explainability | `SHAP`         | Top drivers: SSR (solar), wind_speed   |

---

### 4.2 Demand forecast *(optional)*

Uses Prophet or LSTM depending on time horizon. This step is optional and configurable.

---

### 4.3 Optimisation

```text
min Σ (costᵢ × Pᵢ) + λ × Σ (emissionᵢ × Pᵢ)
s.t. Σ Pᵢ = demand_pred
     0 ≤ Pᵢ ≤ availability_predᵢ
```

**Solver:** `scipy.optimize.linprog` (HiGHS)

---

## 5. Quick start

```bash
# 1. Clone & install
git clone https://github.com/ArturStachnik/energy-mix-optimizer.git
cd energy-mix-optimizer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Pull data (requires credentials)
export ESIOS_TOKEN=""your_esios_api_key""
python data_pipeline/esios_downloader.py
python data_pipeline/omie_prices_downloader.py
python data_pipeline/era5_downloader.py
python data_pipeline/make_dataset.py

# 3. Train & optimize
python src/energy_mix_optimizer.py

# 4. Launch dashboard
streamlit run app/dashboard.py
```

---

## 6. Sample result

| Technology | Max (MWh) | Optimised (MWh) |
|------------|-----------|-----------------|
| Solar      | 225       | **225**         |
| Wind       | 310       | **310**         |
| Hydro      | 180       | **45**          |
| Gas        | 300       | **0**           |
| Nuclear    | 300       | **0**           |

**Total cost:** €11,250  *(−27% vs baseline)*  
**Emissions avoided:** 183 tCO₂  *(−79%)*

> Reproduced from notebook: `notebooks/case-study_2025-06-18.ipynb`

---

## 7. Business impact & roadmap

- 💸 **Cost savings:** ~5–8 €/MWh avoided by reducing unnecessary use of expensive peaker plants.
- 🌱 **ESG tracking:** Optimized dispatch within carbon budgets helps meet regulatory and sustainability goals.
- 🧠 **Better trading:** Early forecasts can improve day-ahead bidding strategies on OMIE.

### Next steps:
- Add battery storage modelling.
- Incorporate ramp-rate constraints.
- Connect to SCADA systems for real-time response.

---

## 8. Citation & licence

```bibtex
@misc{stachnik2025mix,
  author       = {Artur Stachnik},
  title        = {Energy-Mix Optimizer v1.0},
  howpublished = {GitHub},
  year         = 2025,
  url          = {https://github.com/ArturStachnik/energy-mix-optimizer}
}
```

**Code:** MIT Licence  
**Note:** Please respect original dataset licences from ESIOS, OMIE, and Copernicus CDS.
"
