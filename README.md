# Energy Mix Optimizer

Day-ahead renewable generation forecast and economic dispatch for the
Iberian power system. Built on public, no-auth APIs (Energy-Charts from
Fraunhofer ISE and Open-Meteo) so the whole training and inference loop
runs without manual file downloads, vendor credentials or background CDS
jobs.

The project is organised as a Python package with a FastAPI service for
inference and command-line entry points for training. Models are XGBoost
regressors with strict time-series cross-validation; the dispatch is a
linear program solved with HiGHS via SciPy.

## Scope

What the system does:

- Forecasts hourly solar PV, wind and demand for mainland Spain up to 72h
  ahead, using calendar features and capacity-weighted weather aggregates
  across seven representative Iberian locations.
- Returns the cost-optimal economic dispatch per hour under a configurable
  carbon price, broken down by technology, with marginal cost and
  curtailment.

What the system does **not** do (intentional, documented limitations):

- No inter-temporal constraints. The LP is solved independently per hour;
  ramp limits, minimum up and down times, start-up costs and storage state
  variables are out of scope. A production unit-commitment model would
  need a MILP and additional asset-level data.
- No network or nodal modeling. We treat the peninsula as a single bus.
- No reserve requirements, ancillary services, or interconnection flows
  with France and Portugal.
- Weather aggregation uses seven points and rough installed-capacity
  weights, not an asset-resolved siting register.
- Anything labeled as a metric in this README is taken from the model
  metadata file produced by the training pipeline. **Do not** rely on any
  fixed numeric claim until you have run the training pipeline yourself
  and inspected the saved metadata.

## Data sources

| Source | Endpoint | Auth | Coverage |
|---|---|---|---|
| Energy-Charts (Fraunhofer ISE) | `https://api.energy-charts.info` | none | generation by technology + demand (Load), hourly, all EU countries from 2011 |
| Open-Meteo forecast | `https://api.open-meteo.com/v1/forecast` | none | weather forecast, ~16d horizon |
| Open-Meteo archive | `https://archive-api.open-meteo.com/v1/archive` | none | historical weather reanalysis from 1940 |

References:
- Energy-Charts API: <https://api.energy-charts.info>
- Open-Meteo: <https://open-meteo.com/en/docs>

Electricity data is provided by Energy-Charts.info (Fraunhofer Institute
for Solar Energy Systems ISE) under the CC BY 4.0 license. Energy-Charts
re-publishes ENTSO-E Transparency Platform data under a permissive license
and without authentication. Any derived work must credit
**Energy-Charts.info** as the data source.

Both services impose fair-use rate limits. The default HTTP client sets a
30-second timeout and three exponential retries; tune via the `EMO_HTTP_*`
environment variables.

## Installation

```bash
git clone <repo-url>
cd energy-mix-optimizer
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11 or newer is required. No GPU is needed.

## Configuration

All settings are read from environment variables prefixed with `EMO_` or
from a local `.env` file. Defaults are functional; see `.env.example`.

## Training

Real data:

```bash
emo-train --start 2022-01-01 --end 2024-12-31 --artifacts-dir artifacts/models
```

Offline synthetic data, useful for CI and local smoke tests:

```bash
emo-train --synthetic --start 2023-01-01 --end 2023-06-30
```

The pipeline:

1. Fetches observed generation and demand from Energy-Charts for the
   requested window (single call returns all technologies and the demand
   series; resampled from 15-min to hourly).
2. Fetches historical weather from Open-Meteo Archive for the same window
   across the seven panel locations.
3. Aggregates weather with capacity weights, adds calendar and lag
   features (24h, 48h, 168h), drops rows with missing lags.
4. Fits an XGBoost regressor for each target (`solar`, `wind`, `demand`)
   using `TimeSeriesSplit` cross-validation followed by a final hold-out
   on the last 15% of data.
5. Writes each model to `<artifacts>/models/<target>.joblib` together
   with a JSON metadata sidecar that captures hyperparameters, feature
   list, training window, CV metrics and hold-out metrics.

## Inference

Command line:

```bash
emo-forecast --as-of 2025-07-01T00:00 --horizon-hours 24 --carbon-price 80
```

API:

```bash
uvicorn energy_mix_optimizer.api.main:app --host 0.0.0.0 --port 8000
```

If models have not been trained yet, the service still starts; the
`/forecast` and `/models/info` endpoints will return `503 Service
Unavailable` with a message pointing back to the training command.

## API surface

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check, including whether models are loaded. |
| `GET` | `/models/info` | Model metadata: feature list, CV and hold-out metrics, training window, XGBoost and package versions. |
| `POST` | `/forecast` | Full pipeline: fetch weather and recent actuals, run forecasters, run dispatch, return hourly result. |
| `POST` | `/optimize` | Forecast-free dispatch: caller supplies demand and renewable availability, server returns dispatch. |

Interactive OpenAPI docs are served at `/docs` and `/redoc` when the
service is running.

### Example: forecast

```bash
curl -s -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"horizon_hours": 24, "carbon_price_eur_per_t": 80.0}' | jq .summary
```

The full response shape (per hour, plus an aggregate summary):

```json
{
  "as_of_utc": "2025-07-01T00:00:00+00:00",
  "horizon_hours": 24,
  "carbon_price_eur_per_t": 80.0,
  "hourly": [
    {
      "timestamp": "2025-07-01T00:00:00+00:00",
      "demand_mw": 27450.3,
      "solar_pv_mw": 0.0,
      "wind_mw": 6840.1,
      "dispatch_mw": {
        "solar_pv": 0.0,
        "wind": 6840.1,
        "hydro": 15010.2,
        "nuclear": 5600.0,
        "combined_cycle": 0.0,
        "coal": 0.0
      },
      "marginal_cost_eur_per_mwh": 5.0,
      "total_energy_cost_eur": 131051.0,
      "total_emissions_t": 0.0,
      "feasible": true
    }
  ],
  "summary": {
    "total_demand_mwh": 658807.2,
    "total_energy_cost_eur": 3144320.0,
    "total_emissions_t": 0.0,
    "renewable_share": 0.237,
    "average_marginal_cost_eur_per_mwh": 6.67
  },
  "model_versions": {
    "solar": "2025-07-01T08:42:13+00:00",
    "wind":  "2025-07-01T08:42:15+00:00",
    "demand": "2025-07-01T08:42:17+00:00"
  }
}
```

Numbers above are illustrative of the schema; actuals depend on the
forecast window and the trained models.

### Example: forecast-free dispatch

```bash
curl -s -X POST http://localhost:8000/optimize \
  -H 'Content-Type: application/json' \
  -d '{
    "demand_mw":          [28000, 27500, 27000],
    "solar_forecast_mw":  [    0,     0,     0],
    "wind_forecast_mw":   [ 6000,  6500,  7000],
    "carbon_price_eur_per_t": 80.0
  }'
```

Response (real output of this exact call against the running service):

```json
{
  "hourly": [
    {
      "timestamp": "...",
      "demand_mw": 28000.0,
      "solar_pv_mw": 0.0,
      "wind_mw": 6000.0,
      "dispatch_mw": {
        "solar_pv": 0.0,
        "wind": 6000.0,
        "hydro": 16000.0,
        "nuclear": 6000.0,
        "combined_cycle": 0.0,
        "coal": 0.0
      },
      "marginal_cost_eur_per_mwh": 10.0,
      "total_energy_cost_eur": 140000.0,
      "total_emissions_t": 0.0,
      "feasible": true
    }
  ],
  "summary": {
    "total_demand_mwh": 82500.0,
    "total_energy_cost_eur": 399500.0,
    "total_emissions_t": 0.0,
    "renewable_share": 0.2376,
    "average_marginal_cost_eur_per_mwh": 6.6667
  }
}
```

Note how the LP correctly fills the residual demand with the cheapest
available dispatchable technologies (hydro at 5 EUR/MWh, then nuclear at
10 EUR/MWh) before touching combined-cycle gas or coal. Zero emissions
because no thermal generation was needed.

## Smoke verification

The training pipeline includes a `--synthetic` flag for offline
verification. This generates a deterministic toy dataset (seed=2024) and
runs the full training loop without hitting any external service. The
numbers below come from running:

```bash
emo-train --synthetic --start 2023-01-01 --end 2023-04-30 \
    --artifacts-dir artifacts/models --cv-splits 3
```

Output (logged at INFO level):

```
[solar]  CV MAE=643.28, hold-out MAE=429.17 RMSE=639.82 NMAE=9.75%
[wind]   CV MAE=1358.13, hold-out MAE=1691.10 RMSE=2060.82 NMAE=13.17%
[demand] CV MAE=905.99, hold-out MAE=752.31 RMSE=937.28 NMAE=2.74%
```

**These are not performance claims.** They quantify how well XGBoost
recovers a known synthetic signal where the generator function is fully
deterministic and the noise is Gaussian by construction. They exist
purely to confirm that the training, evaluation and persistence loop
runs end-to-end without errors. Real-world metrics will be saved into
`artifacts/models/<target>.metadata.json` after running `emo-train`
against actual Energy-Charts and Open-Meteo data.

## Project layout

```
src/energy_mix_optimizer/
  config.py                Pydantic settings
  logging_config.py        Stdlib logging with optional JSON formatter
  exceptions.py            Domain exceptions
  data/
    locations.py           Iberian weather sampling panel and weights
    energy_charts_client.py Async client for Energy-Charts (Fraunhofer ISE)
    open_meteo_client.py   Async client for Open-Meteo forecast and archive
    feature_engineering.py Calendar features, lag features, weather aggregation
  models/
    forecaster.py          XGBoost time-series forecaster
  optimization/
    dispatch.py            Linear-program dispatch with merit order
  pipelines/
    train.py               Training CLI
    forecast.py            Inference pipeline and CLI
  api/
    main.py                FastAPI application
    schemas.py             Pydantic request and response models
    dependencies.py        Dependency providers
tests/                     Pytest suite with respx-mocked HTTP
```

## Development

```bash
make install   # editable install with dev extras
make lint      # ruff and mypy
make test      # pytest
make api       # uvicorn with reload
make train-synthetic   # fast offline smoke train
```

## Docker

```bash
docker build -t energy-mix-optimizer .
docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" energy-mix-optimizer
```

The image runs `uvicorn` as a non-root user and mounts the `artifacts/`
directory so models persist across container restarts.

## License

MIT. See `LICENSE` if you intend to redistribute.
