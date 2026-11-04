# Demand Forecasting & Inventory Optimization Using Machine Learning
### Data Science & Machine Learning Internship — Week 4: Dashboard, API & Final Reporting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-5%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## 1. Week 4 Deliverables Overview

Week 4 packages the demand forecasting models and statistical inventory logic into enterprise-grade decision-support applications:

1. **Interactive Multi-Page Streamlit Dashboard (`dashboard.py`)**:
   - **Page 1: Demand Overview**: High-level KPIs, actual vs forecast time series, residual error distributions, cumulative demand tracking.
   - **Page 2: Product / Store Forecasts**: Searchable SKU selector, granular 28-day trajectories comparing ground truth against XGBoost, Moving Average (7d), and Naive (Lag 7), plus 120-day historical demand history.
   - **Page 3: Inventory Risk & Replenishment**: Dynamic slider controls for supplier lead time (3 to 21 days), target cycle service level (85% to 99%), and wholesale pack sizes (6, 12, 24). Renders current stock vs Reorder Point (ROP) buffers, risk tier badges (HIGH / MEDIUM / SAFE), and actionable purchase order tables.
   - **Page 4: Model Performance**: Comprehensive benchmark scorecard table, WAPE bar comparisons, and residual diagnostic histograms.

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
   - **Final Analytical Report (`final_analytical_report.md`)**: Comprehensive technical and business methodology report.
   - **Final Presentation Deck (`final_presentation.md`)**: 8-slide executive presentation deck for business stakeholders.
   - **Requirements Specification (`requirements.txt`)**: Pinned production dependencies.

---

## 2. Quickstart Execution Guide

### 2.1 Running the Automated Test Suite
```powershell
pytest "Week 4/test_pipeline.py" -v
```

### 2.2 Launching the Streamlit Dashboard
```powershell
streamlit run "Week 4/dashboard.py"
```
Navigate your browser to `http://localhost:8501`.

### 2.3 Launching the FastAPI REST Service
```powershell
uvicorn "Week 4.api:app" --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

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

## 3. Evaluated Model Performance Scorecard

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

## 4. Statistical Inventory Replenishment Mathematics

Demand forecasts directly drive statistical inventory policies:

1. **Lead Time Demand ($D_L$)**:
   $$D_L = \bar{d} \times L$$
2. **Safety Stock ($SS$)**:
   $$SS = Z_{\alpha} \times \sigma_d \times \sqrt{L}$$
   where $\alpha = 0.95 \implies Z = 1.645$, and $\sigma_d$ is the standard deviation of daily demand over active periods.
3. **Reorder Point ($ROP$)**:
   $$ROP = D_L + SS$$
4. **Suggested Replenishment Quantity ($Q$)**:
   $$Q = \max\left(\text{MOQ}, \left\lceil \frac{ROP - I_{\text{current}} + \hat{D}_{\text{horizon}}}{\text{pack\_size}} \right\rceil \times \text{pack\_size}\right)$$
5. **Stockout Probability**:
   $$P(\text{Stockout}) = 1 - \Phi\left(\frac{I_{\text{current}} - D_L}{\sigma_L}\right)$$

---

## 5. Human-in-the-Loop Interpretation Guidelines

1. **Decision Support, Not Autopilot**: Recommendations assist planning managers. Final purchase approvals should evaluate supplier minimum order values, warehouse cubic capacity, and market anomalies.
2. **Stockouts vs True Zero Demand**: Periods of zero sales during active listing windows represent potential stockouts rather than absent consumer demand. Calculating variance over active periods prevents downward replenishment spirals.
3. **Lead Time Uncertainty**: The baseline assumes a deterministic 7-day supplier lead time. In facilities tracking lead time variance ($\sigma_L$), safety stock should be adjusted to $Z_{\alpha} \sqrt{L\sigma_d^2 + \mu_d^2\sigma_L^2}$.
