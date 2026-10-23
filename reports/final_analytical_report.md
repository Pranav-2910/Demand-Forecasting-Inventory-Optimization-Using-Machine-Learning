# Demand Forecasting & Inventory Optimization Using Machine Learning
## Final Analytical & Technical Report

**Project**: Data Science & Machine Learning Internship — Final Submission  
**Author**: Data Science & Machine Learning Engineering Team  
**Evaluation Period**: 28-Day Chronological Test Window (`2016-03-28` to `2016-04-24`)  
**Data Scope**: Walmart US Retail Sales & Inventory Operations (M5 Case Study)

---

## 1. Executive Summary

In modern retail and supply-chain operations, profitability hinges on maintaining optimal stock availability while minimizing capital tied up in holding costs, depreciation, and warehouse overhead. Underestimating customer demand results in stockouts, forfeited revenue, and customer churn; overestimating demand leads to costly excess inventory, discounting, and obsolescence.

This project delivers an end-to-end, production-grade **Predictive Demand Forecasting and Inventory Decision-Support System** that:
1. **Accurately Forecasts Demand**: Employs non-leaking feature engineering and ensemble gradient boosting (XGBoost) to achieve a **38.03% WAPE** and **5.612 unit MAE**, significantly outperforming traditional retail heuristics (Naive Lag 7 WAPE: 47.22%; Moving Average 28-Day WAPE: 42.14%).
2. **Prevents Data Leakage**: Enforces strict chronological partitioning (Train $\rightarrow$ Validation $\rightarrow$ Test) where all lag features ($t \ge 7$) and rolling statistics are anchored prior to forecast horizons.
3. **Translates Predictions into Optimal Inventory Decisions**: Implements a statistical replenishment engine calculating dynamic **Safety Stock ($SS$)**, **Reorder Points ($ROP$)**, and **Suggested Order Quantities ($Q$)** under documented service-level constraints ($\alpha = 95\%$).
4. **Delivers Operational Usability**: Packages the models into an interactive, multi-page **Streamlit Dashboard** and a high-throughput **FastAPI REST Service** with full input validation and automated error handling.

---

## 2. Business Context & Problem Statement

### 2.1 The Retail Inventory Balancing Dilemma
Supply chain planners face two competing cost drivers:
- **Stockout Cost ($C_s$)**: Lost revenue, emergency supplier expedited shipping fees, and diminished customer goodwill.
- **Inventory Holding Cost ($C_h$)**: Capital financing costs, warehouse storage fees, shrinkage, spoilage, and markdown losses.

### 2.2 Unconstrained Demand vs. Observed Sales
A critical insight established during exploratory data analysis is that **observed point-of-sale transactions do not necessarily represent true consumer demand**. When an active product runs out of stock:
- Point-of-sale systems record $S_t = 0$.
- Actual latent consumer demand $D_t > 0$.
- If treated naively, machine learning models learn that demand dropped to zero, creating a destructive feedback loop of declining replenishment orders and persistent stockouts.

Our workflow resolves this by:
- Explicitly separating pre-launch periods (`is_active = False`) from active selling windows.
- Identifying potential stockout intervals where active items with valid pricing suddenly register zero sales despite strong historical velocity.
- Calculating demand variance ($\sigma_d$) using active windows to prevent the artificial deflation of safety stock buffers.

---

## 3. Data Collection, Quality Checks & Exploratory Insights (Week 1)

The system ingests three relational tables:
- **Sales Data (`sales_train_validation.csv`)**: 1,913 daily observations across 30,490 item-store series.
- **Calendar Data (`calendar.csv`)**: Date attributes, national/cultural holidays, and state-level SNAP distribution dates.
- **Sell Prices Data (`sell_prices.csv`)**: Weekly shelf prices per item-store combination.

### 3.1 Data Quality Audit Findings
- **Date Continuity**: Verified 1,913 contiguous days from `2011-01-29` to `2016-04-24` with zero missing calendar dates.
- **Key Uniqueness**: Zero duplicate records on `(store_id, item_id, date)`.
- **Value Validity**: Zero negative unit sales; price anomalies and zeros were verified and cleaned.
- **Intermittency**: 17.14% of overall records contained zero sales, of which 13.38% of active records were flagged as potential stockout periods.

### 3.2 Exploratory Demand Patterns
1. **Weekend Surges**: Sales exhibit massive peaks on Saturdays and Sundays, with Friday through Sunday driving approximately 58% of weekly volume.
2. **Holiday Elasticity**: Major sporting events (e.g. Super Bowl) and holidays stimulate sharp demand spikes, whereas Christmas Day registers total store closure.
3. **Price Sensitivity**: Promotional discounts exceeding 5% produce an average unit sales lift of 18.4%.

---

## 4. Feature Engineering & Chronological Split Strategy (Week 2)

### 4.1 Chronological Split Plan
To guarantee zero lookahead bias, we established a strict three-tier temporal split:
- **Training Set**: Inception to `2016-02-28` (`d_1` to `d_1857`) — 93,550 records for model parameter fitting.
- **Validation Set**: `2016-02-29` to `2016-03-27` (`d_1858` to `d_1885`, 28 days) — 1,400 records for model tuning.
- **Test Set**: `2016-03-28` to `2016-04-24` (`d_1886` to `d_1913`, 28 days) — 1,400 records held out exclusively for final operational evaluation.

### 4.2 Non-Leaking Feature Registry
1. **Past-Only Lag Features**: `sales_lag_7`, `sales_lag_14`, `sales_lag_21`, `sales_lag_28`, `sales_lag_35`, `sales_lag_42`. All lags are strictly $\ge 7$ days prior to forecast origin.
2. **Rolling Statistics**: Rolling mean, standard deviation, and maximums over 7, 14, and 28-day windows, anchored strictly at `lag_7` or earlier.
3. **Calendar Indicators**: Day of week, day of month, month, year, weekend indicator, holiday event presence, event category encoding, and SNAP food stamp distribution flags.
4. **Pricing & Promotions**: Shelf price, price deviation from historical SKU mean, discount ratio relative to historical maximum, and binary promotional discount flags.

---

## 5. Machine Learning Modeling & Baseline Evaluation (Week 3)

### 5.1 Evaluated Model Architectures
- **Baseline 1: Naive (Lag 7)**: Predicts sales from the exact same day of the prior week.
- **Baseline 2: Seasonal Naive (Lag 28)**: Predicts sales from exactly 4 weeks prior.
- **Baseline 3: Moving Average (7-Day & 28-Day)**: Historical trailing rolling averages.
- **Model 1: Random Forest Regressor**: 100 trees, max depth 12, bagging ensemble.
- **Model 2: LightGBM Regressor**: Gradient boosting with histogram-based binning.
- **Model 3: XGBoost Regressor**: Gradient boosted decision trees with column subsampling and shrinkage.

### 5.2 Out-of-Sample Performance Scorecard (28-Day Test Window)

| Model Name | Type | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) |
|---|---|---|---|---|---|---|
| `Naive (Lag 7)` | Heuristic Baseline | 6.969 | 11.117 | 47.22% | 74.62% | +2.22% |
| `Seasonal Naive (Lag 28)` | Heuristic Baseline | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% |
| `Moving Average (7-Day)` | Heuristic Baseline | 6.658 | 10.171 | 45.11% | 72.24% | +3.16% |
| `Moving Average (28-Day)` | Heuristic Baseline | 6.219 | 9.634 | 42.14% | 66.48% | +0.09% |
| `Random Forest` | Machine Learning | 5.717 | 8.489 | 38.74% | 63.40% | +3.29% |
| `LightGBM` | Machine Learning | 5.632 | 8.447 | 38.16% | 64.33% | +3.86% |
| **`XGBoost`** | **Machine Learning (Winner)** | **5.612** | **8.410** | **38.03%** | **62.43%** | **+3.03%** |

### 5.3 Analytical Conclusions
- **XGBoost achieved the lowest error across all metrics**, outperforming the best heuristic baseline (28-day moving average) by **4.11 percentage points of WAPE** and **0.607 units of MAE**.
- **WAPE Superiority**: Because individual SKU daily sales contain zeros, MAPE produces erratic distortions (62%–85%). WAPE ($\frac{\sum |y - \hat{y}|}{\sum y}$) weights errors by sales volume, providing an operationally interpretable percentage.
- **Balanced Forecast Bias**: The winning XGBoost model produced a controlled bias of **+3.03%**, providing a slight conservative buffer without creating chronic overstocking.

---

## 6. Inventory Replenishment & Decision Support Logic (Week 3)

Point forecasts were transformed into concrete replenishment policies using statistical inventory control:

### 6.1 Mathematical Formulations
1. **Lead Time Demand ($D_L$)**:
   $$D_L = \mu_d \times L$$
   where $\mu_d$ is the mean daily forecast demand and $L = 7$ days is supplier lead time.
2. **Lead Time Demand Variability ($\sigma_L$)**:
   $$\sigma_L = \sigma_d \times \sqrt{L}$$
   where $\sigma_d$ is the daily demand standard deviation over active selling periods.
3. **Safety Stock ($SS$)**:
   $$SS = Z_{\alpha} \times \sigma_L$$
   For target cycle service level $\alpha = 0.95$, $Z_{0.95} = 1.645$.
4. **Reorder Point ($ROP$)**:
   $$ROP = D_L + SS$$
5. **Suggested Replenishment Order Quantity ($Q$)**:
   $$Q_{\text{net}} = \max\left(0, ROP - I_{\text{current}} + \hat{D}_{\text{horizon}}\right)$$
   $$Q_{\text{order}} = \left\lceil \frac{Q_{\text{net}}}{\text{pack\_size}} \right\rceil \times \text{pack\_size}$$
   subject to supplier wholesale case pack sizes (e.g. multiples of 6) and Minimum Order Quantities ($MOQ = 12$).

### 6.2 Stockout Risk Categorization
- **HIGH RISK** ($I_{\text{current}} < D_L$): Inventory on hand cannot cover expected lead time demand; stockout is imminent prior to supplier arrival. Immediate purchase order required.
- **MEDIUM RISK** ($D_L \le I_{\text{current}} < ROP$): Inventory is below reorder point; order must be scheduled within standard replenishment cycle.
- **SAFE** ($I_{\text{current}} \ge ROP$): Sufficient safety stock on hand; no purchase order required.

---

## 7. Interactive Dashboard & Production API (Week 4)

### 7.1 Multi-Page Streamlit Dashboard (`app/dashboard.py`)
- **Page 1: Demand Overview**: Executive KPIs (Units, Revenue, WAPE, Bias), filterable daily actual vs forecast trends, error distribution, and cumulative demand tracking.
- **Page 2: Product / Store Forecasts**: Searchable product selector, granular 28-day trajectories vs baselines, and 120-day historical demand profiling.
- **Page 3: Inventory Risk & Replenishment**: Dynamic lead-time and service-level sliders ($85\% - 99\%$), real-time ROP recomputation, stockout risk alerts, and order recommendations.
- **Page 4: Model Performance & Benchmarks**: Complete benchmarking table, WAPE bar chart comparison, and residual diagnostic histograms.

### 7.2 FastAPI Production REST API (`app/api.py`)
- `GET /health`: System health and model registry status.
- `GET /products`: Catalog lookup of monitored items and stores.
- `GET /metrics`: Out-of-sample accuracy scorecard across baselines and ML models.
- `POST /forecast`: Returns multi-day daily demand forecasts and aggregate totals for specified SKU and horizon.
- `POST /replenishment`: Returns dynamic lead-time demand, safety stock, ROP, stockout probability, and suggested order quantity.

---

## 8. Responsible Use, Limitations & Human-in-the-Loop Guidelines

1. **Model Outputs as Decision Support**: Machine learning predictions represent statistical likelihoods, not infallible ground truth. Human inventory planners should cross-reference suggestions against real-world supplier allocations, shelf capacity constraints, and warehouse bottlenecks.
2. **Supplier Lead Time Variance**: While the current formulation assumes a deterministic 7-day lead time, real-world logistics experience stochastic delivery delays. Where supplier lead time variance ($\sigma_L^2$) is tracked, the formula should be expanded to:
   $$\sigma_{\text{lead}} = \sqrt{L \sigma_d^2 + \mu_d^2 \sigma_L^2}$$
3. **Inventory Record Accuracy (IRA)**: Physical inventory counts can drift from telemetry due to shrinkage, cashier scanning errors, or damaged goods. Replenishment logic requires periodic physical cycle counts to maintain data integrity.
