# Demand Forecasting & Inventory Optimization Using Machine Learning
### Data Science & Machine Learning Internship — Four-Week Implementation & Production System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)

---

## 1. Project Overview & System Context

Retailers and supply-chain organizations face the perpetual challenge of balancing product availability against the cost of holding inventory. Inaccurate demand estimates cause stockouts, lost sales, emergency supplier replenishment fees, or excessive inventory holding costs and spoilage.

This production-grade system develops an end-to-end predictive analytics workflow that:
1. **Forecasts Future SKU Demand** from historical sales, weekly sell prices, and multi-layered calendar and event features using ensemble machine learning.
2. **Eliminates Target Leakage** using a strict chronological rolling window partition (`Train` $\rightarrow$ `Validation` $\rightarrow$ `Test`) and non-leaking lag/rolling statistics anchored strictly prior to the forecast origin.
3. **Benchmarks Against Credible Baselines** (Naive Lag 7, Seasonal Naive Lag 28, 7-Day & 28-Day Moving Averages) across MAE, RMSE, WAPE, and Forecast Bias.
4. **Translates Demand Forecasts into Inventory Decisions** by calculating dynamic **Safety Stock ($SS$)**, **Reorder Points ($ROP$)**, **Suggested Order Quantities ($Q$)**, and **Stockout Probability ($P(\text{Stockout})$)** under documented service-level assumptions ($\alpha = 95\%$).
5. **Provides Operational Visibility** through an interactive 4-page **Streamlit Dashboard** and a high-performance **FastAPI REST Service**.

---

## 2. Four-Week Repository Organization

The project is structured both as a standard data science package and organized into four self-contained weekly milestone deliverable directories:

```
P1/
├── Week 1/                                   # Week 1 Deliverables: Data Ingestion & EDA
│   ├── data_dictionary.md                   # Full schema definitions, types, and descriptions
│   ├── data_source_notes.md                 # Unconstrained demand vs stockout assumptions
│   ├── data_quality_report.md               # Quality audit report on nulls, dates, and keys
│   ├── 01_exploratory_data_analysis.ipynb   # Executed Jupyter notebook with rich EDA
│   └── figures/                             # Visualizations from exploratory analysis
│
├── Week 2/                                   # Week 2 Deliverables: Feature Engineering & Baselines
│   ├── feature_engineering.py               # Standalone feature generation pipeline
│   ├── chronological_split_plan.md          # Train/Val/Test cutoffs and leakage protocol
│   ├── baseline_forecasts.py                # Heuristic baselines execution script
│   ├── baseline_evaluation_table.md         # Baseline evaluation scorecard (markdown)
│   ├── baseline_evaluation_table.csv        # Baseline metrics dataset
│   ├── 02_feature_engineering_and_baselines.ipynb # Executed feature & baseline notebook
│   └── figures/                             # Correlation heatmaps and split diagrams
│
├── Week 3/                                   # Week 3 Deliverables: Model Training & Replenishment
│   ├── model_training.py                    # Random Forest, XGBoost, LightGBM training script
│   ├── model_evaluation_report.md           # Formal ML vs baseline comparative report
│   ├── forecast_output.csv                  # Test period actuals, baselines, and ML forecasts
│   ├── inventory_replenishment.py           # Statistical inventory optimization script
│   ├── replenishment_recommendations.csv    # Actionable purchase orders and ROP thresholds
│   ├── 03_model_training_and_replenishment.ipynb # Executed modeling & inventory notebook
│   └── figures/                             # Residual diagnostics, feature importances, ROP bars
│
├── Week 4/                                   # Week 4 Deliverables: Dashboard, API & Final Reporting
│   ├── dashboard.py                         # Interactive multi-page Streamlit application
│   ├── api.py                               # FastAPI REST service with Pydantic validation
│   ├── final_analytical_report.md           # Executive technical and business report
│   ├── final_presentation.md                # Executive slide presentation deck
│   ├── test_pipeline.py                     # Automated unit test suite
│   ├── README.md                            # Complete system setup and documentation
│   └── requirements.txt                     # Pinned project dependencies
│
├── data/
│   ├── raw/                                 # Immutable raw datasets (calendar, sales, prices)
│   ├── interim/                             # Cleaned longitudinal dataset
│   └── processed/                           # Engineered features and forecast outputs
├── src/                                     # Reusable production source code
│   ├── data_loader.py                       # Data ingestion and longitudinal transformation
│   ├── data_quality.py                      # Data validation and audit functions
│   ├── features.py                          # Leakage-free lag, rolling, calendar, price features
│   ├── splitting.py                         # Chronological partition generator
│   ├── baselines.py                         # Naive, seasonal-naive, and moving average baselines
│   ├── metrics.py                           # MAE, RMSE, WAPE, Non-Zero MAPE, Forecast Bias
│   ├── models.py                            # Random Forest, XGBoost, LightGBM models
│   └── inventory.py                         # Safety stock, ROP, suggested orders, stockout risk
├── app/                                     # Web application package
│   ├── dashboard.py                         # Streamlit dashboard
│   └── api.py                               # FastAPI inference service
├── notebooks/                               # Numbered reproducible notebooks (01, 02, 03)
├── reports/                                 # Generated figures and analytical markdown reports
├── tests/                                   # Automated test suite (pytest)
├── requirements.txt                         # Dependency specifications
└── README.md                                # Root documentation
```

---

## 3. Installation & Environment Setup

### 3.1 Prerequisites
- Python 3.9 or higher (tested on Python 3.14)
- Git

### 3.2 Setup Virtual Environment & Install Dependencies
```powershell
# Clone or navigate to the repository directory
cd c:\Users\prana\Desktop\P1

# Create and activate virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Quickstart Execution Guide

### 4.1 Running the End-to-End Pipeline
Each weekly module can be executed independently:

```powershell
# Step 1: Ingest raw data, audit quality, and create interim dataset (Week 1)
python -c "from src.data_loader import generate_interim_dataset; generate_interim_dataset()"
python -c "from src.data_quality import run_data_quality_audit; run_data_quality_audit()"

# Step 2: Feature engineering and baseline benchmarking (Week 2)
python "Week 2/feature_engineering.py"
python "Week 2/baseline_forecasts.py"

# Step 3: Model training, evaluation, and inventory replenishment (Week 3)
python "Week 3/model_training.py"

# Step 4: Run automated test suite (Week 4)
pytest tests/test_pipeline.py -v
```

---

## 5. Launching the Interactive Applications

### 5.1 Interactive Streamlit Dashboard
Launch the multi-page decision-support dashboard:
```powershell
streamlit run app/dashboard.py
```
Open your browser to `http://localhost:8501`.

#### Dashboard Pages:
1. **Demand Overview**: Executive KPIs (total actual demand, total forecast demand, aggregate WAPE, forecast bias), filterable actual vs forecast daily time series, net error distribution, and cumulative demand tracking.
2. **Product / Store Forecasts**: Searchable SKU selector, granular 28-day trajectories comparing ground truth against XGBoost, Moving Average (7d), and Naive (Lag 7), plus 120-day historical sales profiling.
3. **Inventory Risk & Replenishment**: Real-time slider controls for Supplier Lead Time (3 to 21 days), Target Cycle Service Level (85% to 99%), wholesale pack sizes (6, 12, 24). Renders Stock vs Reorder Point (ROP) comparisons, stockout risk tiers (HIGH / MEDIUM / SAFE), and actionable purchase order recommendations.
4. **Model Performance & Benchmarks**: Complete benchmarking table, WAPE bar chart comparison, and residual diagnostic histograms.

---

### 5.2 Production FastAPI REST Service
Launch the high-throughput inference service:
```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

#### Sample API Queries:

**1. Health Check (`GET /health`)**:
```powershell
curl http://localhost:8000/health
```

**2. Generate Multi-Step Demand Forecast (`POST /forecast`)**:
```powershell
curl -X POST "http://localhost:8000/forecast" `
  -H "Content-Type: application/json" `
  -d '{"item_id": "FOODS_3_090", "store_id": "CA_1", "horizon_days": 14}'
```

**3. Generate Inventory Replenishment Recommendation (`POST /replenishment`)**:
```powershell
curl -X POST "http://localhost:8000/replenishment" `
  -H "Content-Type: application/json" `
  -d '{"item_id": "FOODS_3_090", "store_id": "CA_1", "current_stock": 150, "lead_time_days": 7, "service_level": 0.95, "pack_size": 6}'
```

---

## 6. Model Evaluation Scorecard

All candidate models were tuned on the 28-day chronological validation set and evaluated on the strictly held-out 28-day test set (`2016-03-28` to `2016-04-24`):

| Model Architecture | Type | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) | Status |
|---|---|---|---|---|---|---|---|
| `Naive (Lag 7)` | Heuristic Baseline | 6.969 | 11.117 | 47.22% | 74.62% | +2.22% | Benchmark |
| `Seasonal Naive (Lag 28)` | Heuristic Baseline | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% | Benchmark |
| `Moving Average (7-Day)` | Heuristic Baseline | 6.658 | 10.171 | 45.11% | 72.24% | +3.16% | Benchmark |
| `Moving Average (28-Day)` | Heuristic Baseline | 6.219 | 9.634 | 42.14% | 66.48% | +0.09% | Best Baseline |
| `Random Forest` | Machine Learning | 5.717 | 8.489 | 38.74% | 63.40% | +3.29% | ML Candidate |
| `LightGBM` | Machine Learning | 5.632 | 8.447 | 38.16% | 64.33% | +3.86% | ML Candidate |
| **`XGBoost`** | **Machine Learning** | **5.612** | **8.410** | **38.03%** | **62.43%** | **+3.03%** | **Production Winner** |

---

## 7. Inventory Optimization & Replenishment Mathematics

Demand forecasts directly drive statistical inventory policies:

1. **Lead Time Demand ($D_L$)**:
   $$D_L = \mu_d \times L$$
2. **Safety Stock ($SS$)**:
   $$SS = Z_{\alpha} \times \sigma_d \times \sqrt{L}$$
   where $\alpha = 0.95 \implies Z = 1.645$, and $\sigma_d$ is the standard deviation of daily demand over active periods.
3. **Reorder Point ($ROP$)**:
   $$ROP = D_L + SS$$
4. **Suggested Replenishment Quantity ($Q$)**:
   $$Q = \max\left(0, \left\lceil \frac{ROP - I_{\text{current}} + \hat{D}_{\text{horizon}}}{\text{pack\_size}} \right\rceil \times \text{pack\_size}\right)$$
5. **Stockout Probability**:
   $$P(\text{Stockout}) = 1 - \Phi\left(\frac{I_{\text{current}} - D_L}{\sigma_L}\right)$$

---

## 8. Responsible-Use & Human Interpretation Guidelines

1. **Decision Support, Not Autopilot**: Recommendations assist planning managers. Final purchase approval should consider supplier minimum order values, warehouse cubic capacity, and market anomalies.
2. **Stockouts vs. True Zero Demand**: Periods of zero sales during active listing windows represent potential stockouts rather than absent consumer demand. Calculating variance over active periods prevents downward replenishment spirals.
3. **Lead Time Uncertainty**: The baseline assumes a deterministic 7-day supplier lead time. In facilities tracking lead time variance ($\sigma_L$), safety stock should be adjusted to $Z_{\alpha} \sqrt{L\sigma_d^2 + \mu_d^2\sigma_L^2}$.
