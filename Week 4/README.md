# Demand Forecasting & Inventory Optimization Using Machine Learning
### Data Science & Machine Learning Internship — Week 4: Dashboard, API & Final Reporting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-5%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## 1. Week 4 Deliverables Overview

Week 4 packages the predictive demand forecasting models and statistical inventory logic into enterprise decision-support applications:

1. **Interactive Multi-Page Streamlit Dashboard (`dashboard.py`)**:
   - **Page 1: Demand Overview**: High-level KPIs, actual vs forecast time series, residual error distributions, cumulative demand tracking.
   - **Page 2: Product / Store Forecasts**: Searchable SKU selector, granular 28-day trajectories comparing ground truth against XGBoost, Moving Average (7d), and Naive (Lag 7), plus 120-day historical demand history.
   - **Page 3: Inventory Risk & Replenishment**: Dynamic slider controls for supplier lead time (3 to 21 days), target cycle service level (80% to 99.9%), wholesale pack sizes (6, 12, 24, 48), and MOQ. Renders current stock vs Reorder Point (ROP) buffers, risk tier badges (HIGH / MEDIUM / SAFE), and **1-click CSV order export**.
   - **Page 4: Model Performance**: Comprehensive benchmark scorecard table, WAPE & RMSE bar comparisons, and XGBoost feature importance rankings.

2. **Enterprise FastAPI REST Service (`api.py`)**:
   - `GET /health`: Service health check, version, and monitored SKU count.
   - `GET /products`: List of monitored SKU items and store identifiers.
   - `GET /metrics`: Out-of-sample benchmark scorecard comparing ML models against baselines.
   - `POST /forecast`: Generates multi-step demand forecasts for a given SKU and horizon (1 to 28 days).
   - `POST /replenishment`: Computes Safety Stock, Reorder Point, stockout probability, and suggested replenishment quantity adhering to wholesale pack-size constraints.

3. **Automated Unit Test Suite (`test_pipeline.py`)**:
   - Verifies KPI metrics calculations (MAE, RMSE, WAPE, Forecast Bias).
   - Validates analytical Safety Stock and Reorder Point formulas.
   - Verifies replenishment ordering logic and pack-size rounding rules.
   - Validates normal-distribution stockout risk probability categorization.
   - Verifies FastAPI route handlers, Pydantic request models, and response structures.

4. **Executive Documentation & Reporting**:
   - **Formal 10+ Page Analytical Report (`final_analytical_report.md`)**: Comprehensive technical and business methodology report (4,600+ words).
   - **Native Executive PowerPoint Presentation (`final_presentation.pptx`)**: Professional 12-slide executive presentation deck.
   - **Presentation Script & Notes (`final_presentation.md`)**: Slide-by-slide speaker script, visual structure, and talking points.
   - **Requirements Specification (`requirements.txt`)**: Pinned production dependencies.

---

## 2. Deploying the Streamlit Dashboard

### Free 1-Click Deployment to Streamlit Community Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `Pranav-2910/Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning`
4. Set **Branch**: `main`
5. Set **Main file path**: `streamlit_app.py` (or `Week 4/dashboard.py`)
6. Click **"Deploy!"** — your dashboard goes live with a public URL in ~2 minutes.

### Local Execution
```powershell
streamlit run streamlit_app.py
# or
streamlit run "Week 4/dashboard.py"
```

---

## 3. Running the Automated Test Suite

```powershell
pytest "Week 4/test_pipeline.py" -v
```

---

## 4. Launching the Production FastAPI Service

```powershell
uvicorn "Week 4.api:app" --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## 5. Model Evaluation Scorecard

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
