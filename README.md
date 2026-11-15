# Demand Forecasting & Inventory Optimization Using Machine Learning
### Data Science & Machine Learning Internship — Complete Four-Week Production System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-5%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Status](https://img.shields.io/badge/Project-Complete%20(Weeks%201--4)-success.svg)](#)

---

## 1. Executive Summary & Problem Context

In multi-echelon retail supply chains, operations teams must continuously balance on-shelf product availability against the substantial financial carrying costs of excess inventory. Inaccurate demand estimates trigger stockouts, forfeited sales revenue, emergency supplier replenishment fees, and customer defection. Conversely, over-forecasting demand leads to tied-up working capital, warehouse congestion, spoilage of perishable goods, and margin-eroding promotional markdowns.

This repository implements an enterprise-grade, end-to-end **Predictive Demand Forecasting and Inventory Decision-Support System** developed across four structured lifecycle milestones:
1. **Audits Multi-Source Retail Telemetry** (1,913 days of point-of-sale sales, weekly shelf prices, promotional calendars, and SNAP distribution schedules across Walmart retail stores).
2. **Differentiates Unconstrained Demand from Observed Sales** during stockouts to prevent persistent downward replenishment spirals.
3. **Engineers Non-Leaking Predictive Features** (lags strictly $\ge 7$ days, rolling window statistics anchored prior to forecast horizons, calendar harmonics, and promotional markdown indicators).
4. **Enforces Strict Chronological Splitting** (`Train` $\rightarrow$ `Validation` $\rightarrow$ `Test`) to guarantee zero forward-looking data leakage.
5. **Benchmarks Against Heuristic Baselines** (Naive Lag 7, Seasonal Naive Lag 28, 7-Day & 28-Day Moving Averages) evaluated across MAE, RMSE, WAPE, and Forecast Bias.
6. **Trains, Tunes & Evaluates Machine Learning Models** (Random Forest, LightGBM, and XGBoost), with the winning **XGBoost regressor** achieving **38.03% WAPE** and **5.612 unit MAE** on an out-of-sample 28-day test horizon.
7. **Translates Demand Forecasts into Inventory Replenishment Decisions**, calculating dynamic **Safety Stock ($SS$)**, **Reorder Points ($ROP$)**, **Stockout Probability ($P(\text{Stockout})$)**, and case-pack rounded order quantities under a documented 95% cycle service level.
8. **Operationalizes Decision Support** through an interactive 4-page **Streamlit Dashboard** and a high-throughput **FastAPI REST Service** backed by automated unit tests.

---

## 2. Complete Repository Organization

The repository is organized strictly into four self-contained milestone deliverables matching the internship lifecycle requirements:

```
P1/
├── Week 1/                                   # Week 1 Deliverables: Ingestion, Quality Audit & EDA
│   ├── data_dictionary.md                   # Schema definitions, types, sources, and nullability
│   ├── data_source_notes.md                 # Unconstrained demand vs observed sales during stockouts
│   ├── data_quality_report.md               # Quality audit report on nulls, duplicates, and continuity
│   ├── sales_interim.csv                    # Cleaned longitudinal interim dataset (14.2 MB)
│   ├── 01_exploratory_data_analysis.ipynb   # Executed Jupyter notebook with rich exploratory analysis
│   └── figures/                             # Visualizations from exploratory data analysis
│       ├── 01_aggregate_sales_trend.png     # Long-term sales trend with rolling smoothers
│       ├── 01_seasonality_patterns.png      # Day-of-week and monthly seasonality distributions
│       ├── 01_zero_sales_distribution.png   # Intermittency and zero-sales frequency
│       ├── 01_event_impact.png              # Impact of national events and SNAP distribution
│       └── 01_price_distribution.png        # Price distribution across product SKUs
│
├── Week 2/                                   # Week 2 Deliverables: Feature Engineering & Baselines
│   ├── feature_engineering.py               # Standalone non-leaking feature pipeline script
│   ├── baseline_forecasts.py                # Standalone heuristic baselines evaluation script
│   ├── chronological_split_plan.md          # Train/Val/Test cutoffs and leakage prevention protocol
│   ├── baseline_evaluation_table.md         # Baseline evaluation scorecard (Markdown format)
│   ├── baseline_evaluation_table.csv        # Baseline evaluation metrics dataset (CSV format)
│   ├── 02_feature_engineering_and_baselines.ipynb # Executed feature & baseline notebook
│   ├── features_engineered.csv              # Fully transformed dataset with lag/rolling features (33.8 MB)
│   └── figures/                             # Visualizations from feature and baseline analysis
│       ├── 02_feature_correlation.png       # Correlation heatmap of features vs sales demand
│       ├── 02_chronological_splits.png      # Visualization of chronological partitions
│       └── 02_baseline_comparison.png       # Sample product baseline forecasts vs actual demand
│
├── Week 3/                                   # Week 3 Deliverables: ML Training, Evaluation & Replenishment
│   ├── model_training.py                    # Standalone model training, tuning & evaluation script
│   ├── inventory_replenishment.py           # Safety stock, ROP, stockout probability & ordering logic
│   ├── demand_forecaster.joblib             # Serialized winning XGBoost demand forecasting model (662 KB)
│   ├── model_evaluation_report.md           # Comparative evaluation report and scorecard vs baselines
│   ├── forecast_output.csv                  # Test partition actuals, baselines, and ML forecast outputs
│   ├── replenishment_recommendations.csv    # Actionable SKU-level replenishment orders & stockout risks
│   ├── 03_model_training_and_replenishment.ipynb # Executed training, evaluation & inventory notebook
│   └── figures/                             # Model performance and replenishment visualizations
│       ├── 03_actual_vs_forecast.png        # Test set actual demand vs XGBoost predictions
│       ├── 03_feature_importance.png        # Top predictive features in the XGBoost model
│       ├── 03_item_trajectories.png         # Per-SKU demand forecast trajectories over 28-day horizon
│       ├── 03_model_wape_comparison.png     # WAPE metric comparison across baselines and ML models
│       ├── 03_residual_distribution.png     # Forecast error residual distribution and homoscedasticity
│       └── 03_stock_vs_rop.png              # Inventory on-hand vs Reorder Point safety buffers
│
├── Week 4/                                   # Week 4 Deliverables: Dashboard, API & Final Reporting
│   ├── dashboard.py                         # Interactive multi-page Streamlit decision support application
│   ├── api.py                               # Production FastAPI REST service with Pydantic validation
│   ├── final_analytical_report.md           # Formal 10+ page executive technical and business report
│   ├── final_presentation.pptx              # Native 12-slide executive presentation deck
│   ├── final_presentation.md                # Presentation speaker script and slide-by-slide notes
│   ├── test_pipeline.py                     # Automated unit test suite verifying metrics, inventory & API
│   ├── README.md                            # Week 4 milestone overview and runbook
│   └── requirements.txt                     # Pinned project dependencies
│
├── streamlit_app.py                         # 1-Click Streamlit Community Cloud root entrypoint
├── requirements.txt                         # Root dependency specifications
├── .gitignore                               # Standardized Git ignore rules (ignoring files > 100 MB)
└── README.md                                # Root comprehensive documentation & execution guide
```

---

## 3. Four-Week Technical Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 WEEK 1: INGESTION & AUDIT                              │
│  Raw POS Sales (1,913 Days) ──► Date Continuity & Key Audit ──► Stockout Identification │
│  Calendar & Pricing Feeds   ──► Zero Sales vs Intermittency  ──► Longitudinal Interim   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              WEEK 2: FEATURES & BASELINES                              │
│  Non-Leaking Lags (t-7 to t-42) ──► Anchored Rolling Stats ──► Chronological Splits     │
│  Calendar & Markdown Signals    ──► Heuristic Baselines   ──► WAPE & Bias Benchmarking │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       WEEK 3: ML TRAINING & REPLENISHMENT ENGINE                       │
│  Random Forest / LightGBM / XGBoost ──► Cross-Validation Tuning ──► Champion XGBoost   │
│  Lead Time Demand & Safety Stock   ──► Reorder Points (ROP)    ──► Pack-Size Orders   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         WEEK 4: DECISION SUPPORT APPLICATIONS                          │
│  Interactive Streamlit Dashboard ◄──► Enterprise FastAPI REST Service ◄──► Unit Tests  │
│  Formal 10-Page Analytical Report ◄──► 12-Slide Native PowerPoint Deck ◄──► Cloud App   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Milestone Breakdown

### 4.1 Week 1: Data Ingestion, Quality Audit & Exploratory Analysis
- **Longitudinal Span**: Verified **1,913 contiguous calendar days** (`2011-01-29` to `2016-04-24`) across retail merchandise departments (`HOBBIES`, `FOODS`, `HOUSEHOLD`).
- **Data Quality Audit**: Confirmed zero duplicate primary keys `(store_id, item_id, date)`, zero negative sales, and zero negative sell prices.
- **Stockout vs Unconstrained Demand Protocol**:
  - Point-of-sale data records observed sales ($S_t$), not latent demand ($D_t$). When stock runs out, $S_t = 0$ while $D_t > 0$.
  - Items prior to their initial store launch date are flagged as `is_active = False` to prevent negative demand bias.
  - Active items (`is_active = True`) with zero sales despite valid pricing and historical demand are flagged as `potential_stockout` (13.38% of active records).
  - Demand variance ($\sigma_d$) calculations in replenishment logic use active periods, preventing artificial safety stock deflation.
- **Exploratory Insights**:
  - Strong day-of-week seasonality (sales surge on Friday through Sunday).
  - Statistically significant sales lift on California SNAP benefit distribution days.
  - Store closure on Christmas Day produces genuine structural zero sales.

---

### 4.2 Week 2: Feature Engineering & Baseline Forecasts
- **Leakage Prevention Principle**: Every feature is strictly calculated using historical information available prior to the forecast horizon ($t \ge 7$).
- **Engineered Feature Registry (27 Attributes)**:
  - **Lags (6)**: `sales_lag_7`, `sales_lag_14`, `sales_lag_21`, `sales_lag_28`, `sales_lag_35`, `sales_lag_42`.
  - **Rolling Statistics (9)**: Rolling mean, standard deviation, and maximum over 7, 14, and 28-day windows, anchored strictly at `sales_lag_7` to eliminate contemporaneous leakage.
  - **Calendar Metadata (8)**: Day of week, day of month, month, year, weekend indicator, holiday event indicator, event category codes, SNAP benefit distribution flags.
  - **Pricing & Elasticity (4)**: Selling price, price deviation from SKU historical average, promotional markdown ratios, and discount flags.
- **Strict Chronological Partitions**:
  - **Training Partition**: `2011-01-29` to `2016-02-28` (~5 years, 90,850 rows).
  - **Validation Partition**: `2016-02-29` to `2016-03-27` (28 days, 1,400 rows).
  - **Testing Partition**: `2016-03-28` to `2016-04-24` (28 days, 1,400 rows).
- **Heuristic Baselines Scorecard**:
  - Naive Lag 7, Seasonal Naive Lag 28, 7-Day Moving Average, and 28-Day Moving Average evaluated on MAE, RMSE, WAPE, and Forecast Bias.
  - Best heuristic baseline: **Moving Average (28-Day)** at **42.14% WAPE** and **6.219 unit MAE**.

---

### 4.3 Week 3: Model Training, Evaluation & Replenishment Logic
- **Supervised Machine Learning Models**:
  - **Random Forest**: 100 trees, `max_depth=12`, `min_samples_leaf=5`.
  - **LightGBM**: Fast leaf-wise gradient boosting (`n_estimators=150`, `learning_rate=0.05`, `num_leaves=31`).
  - **XGBoost**: Level-wise gradient boosting with second-order gradient regularization (`n_estimators=150`, `learning_rate=0.05`, `max_depth=6`, `reg_alpha=0.1`, `reg_lambda=1.0`).
- **Comparative Evaluation Scorecard on 28-Day Test Set**:

| Model Architecture | Model Category | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) | Operational Status |
|---|---|---|---|---|---|---|---|
| `Naive (Lag 7)` | Heuristic Baseline | 6.969 | 11.117 | 47.22% | 74.62% | +2.22% | Baseline |
| `Seasonal Naive (Lag 28)` | Heuristic Baseline | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% | Baseline |
| `Moving Average (7-Day)` | Heuristic Baseline | 6.658 | 10.171 | 45.11% | 72.24% | +3.16% | Baseline |
| `Moving Average (28-Day)` | Heuristic Baseline | 6.219 | 9.634 | 42.14% | 66.48% | +0.09% | Best Baseline |
| `Random Forest` | Machine Learning | 5.717 | 8.489 | 38.74% | 63.40% | +3.29% | Evaluated Candidate |
| `LightGBM` | Machine Learning | 5.632 | 8.447 | 38.16% | 64.33% | +3.86% | Evaluated Candidate |
| **`XGBoost`** | **Machine Learning** | **5.612** | **8.410** | **38.03%** | **62.43%** | **+3.03%** | **Winning Production Model** |

- **XGBoost Outperformance Highlights**:
  - Outperforms the leading baseline by **4.11 percentage points of WAPE** (a **9.7% relative error reduction**).
  - Reduces RMSE by **12.7%** (from 9.634 to 8.410 units).
  - Maintains a balanced forecast bias of **+3.03%**, providing mild buffer protection against stockouts while avoiding excessive holding costs.
- **Top Demand Drivers**: `rolling_mean_28` (31.4%), `sales_lag_7` (18.6%), `rolling_mean_7` (12.8%), `sell_price` (9.5%), `day_of_week` (8.2%).

#### Statistical Replenishment Mathematics
1. **Expected Lead Time Demand ($D_L$)**:
   $$D_L = \bar{d} \times L$$
2. **Dynamic Safety Stock ($SS$)**:
   $$SS = Z_{\alpha} \times \sigma_d \times \sqrt{L}$$
   Where $Z_{\alpha} = 1.645$ for a **95% cycle service level** ($CSL$), $L = 7\text{ days}$ lead time, and $\sigma_d$ is the standard deviation of daily demand over active periods.
3. **Reorder Point ($ROP$)**:
   $$ROP = D_L + SS$$
4. **Stockout Probability ($P(\text{Stockout})$)**:
   $$P(\text{Stockout}) = P(D_L > I_{current}) = 1 - \Phi\left(\frac{I_{current} - D_L}{\sigma_L}\right)$$
   - **HIGH RISK**: $I_{current} < D_L$ (Stockout imminent during lead time; $P \ge 50\%$).
   - **MEDIUM RISK**: $D_L \le I_{current} < ROP$ (Inventory has breached reorder point; safety buffer eroding).
   - **SAFE**: $I_{current} \ge ROP$ (Inventory remains above reorder point).
5. **Suggested Order Quantity ($Q$)**:
   $$Q_{raw} = \max\left(0, ROP - I_{current} + \hat{D}_{horizon}\right)$$
   Adjusted to wholesale case pack constraints ($\text{pack\_size} = 6$, $\text{MOQ} = 12$):
   $$Q = \max\left(\text{MOQ}, \left\lceil \frac{Q_{raw}}{\text{pack\_size}} \right\rceil \times \text{pack\_size}\right)$$

#### Sample Actionable Replenishment Recommendations Table

| Item ID | Store | Current Stock | Lead Time Demand | Safety Stock | ROP | Stockout Risk | Suggested Order | Est. Capital ($) |
|---|---|---|---|---|---|---|---|---|
| `FOODS_3_744` | CA_1 | 44 | 83.4 | 15.6 | 99.0 | **HIGH** (100.0%) | **390 units** | \$1,365.00 |
| `FOODS_3_090` | CA_1 | 182 | 351.0 | 94.7 | 445.7 | **HIGH** (99.8%) | **1668 units** | \$2,502.00 |
| `FOODS_3_362` | CA_1 | 62 | 121.0 | 35.5 | 156.5 | **HIGH** (99.7%) | **582 units** | \$1,746.00 |
| `FOODS_3_694` | CA_1 | 46 | 80.1 | 20.5 | 100.7 | **HIGH** (99.7%) | **378 units** | \$1,134.00 |
| `FOODS_2_019` | CA_1 | 61 | 104.2 | 29.9 | 134.1 | **HIGH** (99.1%) | **492 units** | \$1,476.00 |
| `FOODS_1_085` | CA_1 | 48 | 76.7 | 22.4 | 99.1 | **HIGH** (98.2%) | **360 units** | \$1,080.00 |
| `FOODS_3_702` | CA_1 | 43 | 65.8 | 19.9 | 85.7 | **HIGH** (97.0%) | **306 units** | \$918.00 |
| `FOODS_3_099` | CA_1 | 75 | 103.9 | 27.9 | 131.8 | **HIGH** (95.6%) | **474 units** | \$1,422.00 |
| `FOODS_3_739` | CA_1 | 88 | 104.7 | 17.9 | 122.6 | **HIGH** (93.8%) | **456 units** | \$1,368.00 |
| `FOODS_2_276` | CA_1 | 57 | 80.1 | 25.3 | 105.4 | **HIGH** (93.4%) | **372 units** | \$1,116.00 |

---

### 4.4 Week 4: Dashboard, API & Executive Reporting

#### 1. Interactive Streamlit Dashboard (`Week 4/dashboard.py` and `streamlit_app.py`)
A 4-page enterprise decision-support interface:
- **Page 1: Demand Overview**: Executive KPIs (Total Actual Demand, Total Forecast Demand, Aggregate WAPE, Forecast Bias), interactive actual vs forecast daily time series with store/department filtering, residual error distributions, and cumulative demand tracking.
- **Page 2: Product / Store Forecasts**: Searchable SKU selector displaying 28-day trajectories comparing actual sales against XGBoost, Moving Average (7-Day), and Naive (Lag 7), alongside 120-day historical demand history.
- **Page 3: Inventory Risk & Replenishment**: Dynamic slider controls for supplier lead time (3 to 21 days), target cycle service level (80% to 99.9%), wholesale pack sizes (6, 12, 24, 48), and MOQ. Renders current inventory vs Reorder Point safety buffers, risk tier badges, and an **actionable purchase order table with 1-click CSV Export (`st.download_button`)**.
- **Page 4: Model Performance**: Comprehensive benchmark evaluation table, WAPE & RMSE bar comparisons, and XGBoost feature importance rankings.

#### 2. Enterprise FastAPI REST Service (`Week 4/api.py`)
A production REST service with full Pydantic request and response validation, error handling, and auto-generated Swagger documentation at `http://localhost:8000/docs`:
- `GET /health`: Service health status, version, and catalog metadata.
- `GET /products`: List of monitored SKU product IDs and store keys.
- `GET /metrics`: Out-of-sample benchmark scorecard comparing ML models against baselines.
- `POST /forecast`: Generates multi-step demand forecasts for a given SKU and horizon (1 to 28 days).
- `POST /replenishment`: Computes Safety Stock, Reorder Point, stockout probability, and suggested replenishment quantity.

#### 3. Automated Test Suite (`Week 4/test_pipeline.py`)
Automated pytest suite validating:
- Core KPI calculation accuracy (MAE, RMSE, WAPE, Forecast Bias).
- Analytical Safety Stock and Reorder Point formulas.
- Order quantity replenishment logic and pack-size rounding rules.
- Stockout risk probability categorization under normal distributions.
- FastAPI REST endpoint request/response contracts and status codes.

#### 4. Final Executive Deliverables
- **Formal 10+ Page Analytical Report** ([Week 4/final_analytical_report.md](file:///c:/Users/prana/Desktop/P1/Week%204/final_analytical_report.md)): Exhaustive 4,600+ word technical report covering business problem context, data architecture, feature engineering, modeling results, replenishment formulation, and operational limitations.
- **Native Executive PowerPoint Presentation** ([Week 4/final_presentation.pptx](file:///c:/Users/prana/Desktop/P1/Week%204/final_presentation.pptx)): Professional 12-slide executive deck with custom corporate visual hierarchy, KPI callouts, and structured cards.
- **Presentation Deck Notes** ([Week 4/final_presentation.md](file:///c:/Users/prana/Desktop/P1/Week%204/final_presentation.md)): Slide-by-slide speaker script, visual structure, and talking points.

---

## 5. How to Deploy the Streamlit Dashboard

### Option A: Free 1-Click Deployment to Streamlit Community Cloud (Recommended)
You can deploy this dashboard to the cloud for free with a public shareable URL:

1. **Push Repository to GitHub**:
   Ensure all changes are pushed to your GitHub repository:
   `https://github.com/Pranav-2910/Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning`
2. **Sign in to Streamlit Community Cloud**:
   Go to [share.streamlit.io](https://share.streamlit.io/) and click **"Continue with GitHub"**.
3. **Deploy New Application**:
   - Click the blue **"New app"** button.
   - **Repository**: `Pranav-2910/Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py` (or `Week 4/dashboard.py`)
4. **Click "Deploy!"**:
   Streamlit Cloud automatically installs packages from `requirements.txt` and launches your live interactive dashboard in approximately 2 minutes!

### Option B: Local Execution
Run the dashboard locally in your terminal:
```powershell
# Using the root entrypoint:
streamlit run streamlit_app.py

# Or running directly from Week 4:
streamlit run "Week 4/dashboard.py"
```
Navigate your browser to `http://localhost:8501`.

---

## 6. Quickstart Execution Guide for All Pipelines

### 6.1 Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Pranav-2910/Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning.git
cd Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning

# Install pinned dependencies
pip install -r requirements.txt
```

### 6.2 Reproducing All Pipelines (Weeks 1 to 4)

**1. Run Feature Engineering Pipeline (Week 2)**:
```powershell
python "Week 2/feature_engineering.py"
```
*Outputs: `Week 2/features_engineered.csv` and `Week 2/chronological_split_plan.md`.*

**2. Run Heuristic Baseline Forecasts (Week 2)**:
```powershell
python "Week 2/baseline_forecasts.py"
```
*Outputs: `Week 2/baseline_evaluation_table.csv` and `Week 2/baseline_evaluation_table.md`.*

**3. Run Machine Learning Model Training & Evaluation (Week 3)**:
```powershell
python "Week 3/model_training.py"
```
*Outputs: `Week 3/demand_forecaster.joblib`, `Week 3/forecast_output.csv`, and `Week 3/model_evaluation_report.md`.*

**4. Run Inventory Replenishment & Ordering Logic (Week 3)**:
```powershell
python "Week 3/inventory_replenishment.py"
```
*Outputs: `Week 3/replenishment_recommendations.csv`.*

**5. Run Automated Unit Test Suite (Week 4)**:
```powershell
pytest "Week 4/test_pipeline.py" -v
```
*Verifies all 5 test suites (metrics, safety stock, replenishment, stockout risk, and API routes).*

---

## 7. Launching the Production FastAPI REST Service

```powershell
uvicorn "Week 4.api:app" --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Sample API Queries:

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

## 8. Interactive Jupyter Notebooks

- [Week 1/01_exploratory_data_analysis.ipynb](file:///c:/Users/prana/Desktop/P1/Week%201/01_exploratory_data_analysis.ipynb): Exploratory analysis of sales trends, seasonality cycles, intermittency, and promotional event lift.
- [Week 2/02_feature_engineering_and_baselines.ipynb](file:///c:/Users/prana/Desktop/P1/Week%202/02_feature_engineering_and_baselines.ipynb): Non-leaking lag generation, correlation matrices, and heuristic baseline evaluations.
- [Week 3/03_model_training_and_replenishment.ipynb](file:///c:/Users/prana/Desktop/P1/Week%203/03_model_training_and_replenishment.ipynb): Model training, cross-validation tuning, residual diagnostic plots, and inventory replenishment simulation.

---

## 9. Responsible-Use & Human Interpretation Guidelines

1. **Decision Support, Not Autonomous Ordering**: Recommended order quantities assist planning managers. Final purchase approvals should evaluate supplier minimum order values (MOV), warehouse cubic storage limits, and non-modeled macroeconomic disruptions.
2. **Stockouts vs Latent Demand**: Zero sales during active listing periods reflect stockout constraints rather than absent customer demand. Calculating safety stock buffers over active periods prevents downward replenishment spirals.
3. **Supplier Lead Time Uncertainty**: The baseline system models deterministic 7-day supplier lead times ($L = 7, \sigma_L = 0$). When supplier delivery variance is documented, the standard deviation expands to $\sigma_{LT} = \sqrt{L\sigma_d^2 + \bar{d}^2\sigma_L^2}$.

---

## 10. Dataset Provenance & Full Raw Data Access

The cleaned interim dataset (`Week 1/sales_interim.csv`, 14.2 MB) and the engineered features dataset (`Week 2/features_engineered.csv`, 33.8 MB) are included directly in the repository for immediate execution.

To access or rebuild the full multi-gigabyte raw dataset:
1. Download the source files (`calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv`) from the official [Kaggle M5 Forecasting Accuracy Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data).
2. The raw files exceed GitHub's 100 MB per-file limit and are managed locally via `.gitignore`.
