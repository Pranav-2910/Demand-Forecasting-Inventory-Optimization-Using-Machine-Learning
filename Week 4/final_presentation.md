# Demand Forecasting & Inventory Optimization Using Machine Learning
## Executive Stakeholder Presentation Deck & Speaker Transcript

**File Reference**: Native 16:9 widescreen PowerPoint presentation available in `Week 4/final_presentation.pptx`  
**Companion Formal Analytical Report**: 11-page publication-quality PDF in `Week 4/final_analytical_report.pdf`  
**Interactive Cloud Dashboard**: Production Streamlit application in `Week 4/dashboard.py` and `streamlit_app.py`  
**Production REST API**: Enterprise FastAPI microservice in `Week 4/api.py`

---

### Slide 1: Executive Title Slide
- **Slide Title**: Demand Forecasting & Inventory Optimization Using Machine Learning
- **Subtitle**: Enterprise Decision-Support System: Predictive Modeling & Replenishment Policy
- **Program**: Data Science & Machine Learning Capstone Implementation
- **Presenter**: Lead Data Science & Supply Chain ML Engineering Team
- **Benchmark Dataset**: Multi-Category Retail Telemetry (1,913 Contiguous Historical Days)
- **Target Deliverable**: Production-Ready 4-Week End-to-End System

**Speaker Script**:
> "Good morning, leadership team. Today we present our four-week capstone implementation: an end-to-end predictive demand forecasting and inventory decision-support system. In high-volume retail operations, profitability is determined by navigating the razor-thin margin between stockouts and excess carrying costs. Our platform couples non-leaking machine learning ensembles with statistical safety stock optimization to replace legacy moving averages with high-precision demand intelligence."

---

### Slide 2: The Business Challenge — The Retail Balancing Dilemma
- **Slide Title**: The Retail Inventory Dilemma: Stockouts vs. Carrying Costs
- **Core Problem Statement**: Supply chain operations are trapped between two asymmetric cost curves:
  - **Cost of Under-Forecasting (Stockout Cost, $C_s$)**:
    - Forfeited gross margin at point-of-sale.
    - Emergency expedited freight charges and spot-market sourcing premiums.
    - 31% of grocery shoppers permanently switch stores upon experiencing a stockout.
    - Contractual retailer SLA chargebacks and penalties.
  - **Cost of Over-Forecasting (Inventory Carrying Cost, $C_h$)**:
    - 18% to 25% annual carrying cost of tied-up working capital.
    - Warehouse congestion, pallet storage fees, handling labor, and insurance.
    - Perishable spoilage and obsolescence risk.
    - Margin-destroying promotional markdowns required to liquidate stagnant stock.
- **The Bullwhip Effect**: Lagging heuristics overreact to sudden demand spikes, propagating amplified variance upstream to distribution centers and manufacturers.

**Speaker Script**:
> "Traditional retail planning frequently relies on trailing 7-day or 28-day moving averages. The fundamental flaw of these heuristics is that they look backward. When demand surges during a holiday or promotional campaign, moving averages lag behind, causing stockouts. Planners then panic and over-order, flooding warehouses right as demand cools off. This triggers the classic Bullwhip Effect, tying up millions in working capital and forcing steep clearance markdowns."

---

### Slide 3: End-to-End System Architecture & Pipeline
- **Slide Title**: End-to-End System Architecture & Pipeline
- **Four Integrated Layers**:
  1. **Ingestion & Quality Layer (`Week 1`)**: Multi-source ETL pipelines merging 1,913 contiguous days of sales, calendar events, and weekly pricing telemetry.
  2. **Feature Engineering Layer (`Week 2`)**: 27 non-leaking temporal features anchored strictly at $t-7$ to protect against target leakage.
  3. **ML Training & Optimization Layer (`Week 3`)**: Rigorous chronological cross-validation comparing classical baselines against Random Forest, LightGBM, and XGBoost regressors.
  4. **Operationalization & Deployment Layer (`Week 4`)**: Interactive multi-page Streamlit dashboard with 1-click CSV order export and high-throughput FastAPI REST microservice.

**Speaker Script**:
> "To solve this systematically, we built an end-to-end modular architecture spanning four distinct layers. Data flows cleanly from multi-source historical ingestion into a strict non-leaking feature store, through machine learning models evaluated on out-of-sample data, and directly into operational tools: an interactive decision dashboard and an automated REST API."

---

### Slide 4: Data Engineering & Quality Audit (1,913 Days)
- **Slide Title**: Multi-Source Telemetry & Quality Audit
- **Data Landscape**:
  - **Sales History**: 1,913 contiguous days (`2011-01-29` to `2016-04-24`) across 50 representative SKUs in Foods, Household, and Hobbies merchandise departments.
  - **Calendar Telemetry**: 30 national holidays, cultural events, and state-level SNAP benefit disbursement schedules across California, Texas, and Wisconsin.
  - **Sell Prices**: Store-item weekly price history capturing seasonal inflation and promotional markdown depth.
- **Audit Findings & Data Integrity**:
  - **Zero Duplicate Keys**: 100% uniqueness verified on `(store_id, item_id, date)`.
  - **Zero Negative Values**: All recorded sales $\ge 0$; sell prices strictly positive.
  - **Zero Missing Calendar Days**: Contiguous 1,913-day timeline with no temporal gaps.
  - **17.14% Overall Zero Sales**: Identified significant intermittency requiring volume-weighted error metrics (WAPE) rather than standard percentage errors (MAPE).

**Speaker Script**:
> "We began Week 1 with an exhaustive data audit. In enterprise analytics, dirty data produces catastrophic forecasts. We audited 1,913 contiguous days across sales, calendar, and pricing records. We confirmed zero duplicate composite keys, zero negative quantities, and complete temporal continuity. We also noted that 17.14% of records contained zero sales, confirming that standard percentage metrics like MAPE would divide by zero and mislead planners."

---

### Slide 5: Methodological Foundation — Latent Demand vs. Censored Sales
- **Slide Title**: Latent Demand vs. Censored Sales During Stockouts
- **The Censored Demand Problem**:
  $$S_t = \min\left(D_t, I_t\right)$$
  - When inventory reaches zero ($I_t = 0$), point-of-sale registers record zero sales ($S_t = 0$).
  - True consumer latent demand $D_t$ remains positive ($D_t > 0$).
- **The Stockout Spiral of Death**:
  - Naive ML models trained on unadjusted sales learn that demand dropped to zero.
  - The model forecasts $\hat{D}_{t+1} \approx 0$, leading purchasing systems to cut replenishment orders.
  - The store stocks out again, permanently depressing sales and customer loyalty.
- **Our Empirical Resolution Protocol**:
  1. *Lifecycle Pre-Launch Separation*: Filter records prior to SKU store listing (`is_active = False`).
  2. *Stockout Identification*: Flag active items with zero sales despite valid shelf prices as `potential_stockout` (13.38% of active rows).
  3. *Variance Protection*: Demand variance ($\sigma_d$) is computed strictly over active, in-stock periods to prevent safety stock deflation.

**Speaker Script**:
> "Here is our most critical methodological contribution: observed sales do not equal customer demand. When a grocery store runs out of inventory, the register rings up zero sales. A naive machine learning algorithm interprets that zero as a collapse in customer demand, predicts near-zero sales for the next month, and cuts replenishment orders. This creates a vicious cycle called the Stockout Spiral of Death. We implemented an empirical detection heuristic that flags active stockouts and computes safety stock variance strictly across in-stock periods."

---

### Slide 6: Feature Engineering Architecture (27 Features)
- **Slide Title**: Non-Leaking Feature Engineering Architecture
- **Strict Leakage Prevention Protocol**:
  - To support a 28-day forward planning horizon and a 7-day supplier lead time, all features are strictly anchored prior to the forecast horizon ($t \ge 7$).
- **27 Engineered Features Across 4 Functional Groups**:
  1. **Past-Only Lags (6 features)**: $t-7, t-14, t-21, t-28, t-35, t-42$.
  2. **Anchored Rolling Statistics (9 features)**: Rolling mean, standard deviation, and maximum over 7, 14, and 28-day windows anchored at $t-7$.
  3. **Calendar & Event Harmonics (8 features)**: Day of week, month, year, weekend indicators, national holiday markers, event type codes, and California SNAP disbursement schedules.
  4. **Pricing & Promotions (4 features)**: Shelf price, price deviation from SKU historical average, discount ratio relative to peak price, and promotional markdown indicators.

**Speaker Script**:
> "In Week 2, we engineered 27 features designed with strict temporal causality. A major hazard in retail forecasting is data leakage—allowing future information to slip into training features. To support our 7-day supplier lead time, all lag features and rolling aggregations are anchored strictly at day t minus 7 or earlier. We also capture weekly shopping lift, SNAP food stamp disbursement surges, and promotional discount depths."

---

### Slide 7: Chronological Partitioning Plan (Zero Leakage)
- **Slide Title**: Strict Chronological Partitioning (Zero Target Leakage)
- **Chronological Window Design**:
  - **Training Partition** (`2011-01-29` to `2016-02-28`): ~5 Years (90,850 rows) for parameter estimation.
  - **Validation Partition** (`2016-02-29` to `2016-03-27`): 28 Days (1,400 rows) for hyperparameter tuning.
  - **Testing Partition** (`2016-03-28` to `2016-04-24`): 28 Days (1,400 rows) for final operational evaluation.
- **Why Random K-Fold Cross-Validation Fails**:
  - Randomly shuffling time-series data leaks future trends and seasonal peaks into past folds.
  - It produces unrealistically optimistic validation metrics that collapse immediately in live operations.
  - Chronological splits mirror real-world replenishment cycles exactly.

**Speaker Script**:
> "We enforced a strict chronological split rather than standard randomized cross-validation. In time series, randomized splitting leaks future data points into the past, providing an illusion of accuracy. We trained across 5 years of historical data, tuned our hyperparameters on a 28-day validation window, and held out a final 28-day test set that the models never saw during training."

---

### Slide 8: Model Evaluation Scorecard on 28-Day Test Set
- **Slide Title**: Model Evaluation Scorecard on 28-Day Test Set
- **Comprehensive Out-of-Sample Empirical Results**:

| Model Architecture | Category | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) | Operational Decision |
|---|---|---|---|---|---|---|---|
| `Naive (Lag 7)` | Heuristic Baseline | 6.969 | 11.117 | 47.22% | 74.62% | +2.22% | Legacy Benchmark |
| `Seasonal Naive (Lag 28)` | Heuristic Baseline | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% | Legacy Benchmark |
| `Moving Average (7-Day)` | Heuristic Baseline | 6.658 | 10.171 | 45.11% | 72.24% | +3.16% | Legacy Benchmark |
| `Moving Average (28-Day)` | Heuristic Baseline | 6.219 | 9.634 | 42.14% | 66.48% | +0.09% | Best Heuristic Baseline |
| `Random Forest Regressor` | Machine Learning | 5.717 | 8.489 | 38.74% | 63.40% | +3.29% | Evaluated Candidate |
| `LightGBM Regressor` | Machine Learning | 5.632 | 8.447 | 38.16% | 64.33% | +3.86% | Evaluated Candidate |
| **`XGBoost Regressor`** | **Machine Learning** | **5.612** | **8.410** | **38.03%** | **62.43%** | **+3.03%** | **Winning Production Champion** |

- **Key Numerical Findings**:
  - **XGBoost** beats the best baseline (`Moving Average 28-Day`), reducing WAPE from **42.14% to 38.03%** (a **4.11 percentage point improvement**, representing a **9.7% relative error reduction**).
  - RMSE decreases from **9.634 to 8.410** (a **12.7% reduction**), demonstrating superior stability against extreme demand spikes.
  - Controlled positive bias of **+3.03%** provides an intentional commercial buffer against costly stockouts.

**Speaker Script**:
> "Here are our final empirical results on the held-out 28-day test partition. The best classical baseline—a 28-day moving average—achieved a 42.14% WAPE. Our winning XGBoost regressor reduced that to 38.03%, representing an almost 10% relative error reduction and a 12.7% reduction in RMSE. Furthermore, it achieved this with a balanced positive bias of plus 3%, which is commercially ideal because it protects against stockouts without causing excessive overstock."

---

### Slide 9: Demand Drivers & Feature Importance
- **Slide Title**: Demand Drivers & Feature Importance (XGBoost)
- **Top Predictive Features by Relative Gain**:
  1. **`rolling_mean_28` (31.4% gain)**: Acts as the primary medium-term volume anchor.
  2. **`sales_lag_7` (18.6% gain)**: Strongest individual lag; captures day-of-week sales momentum.
  3. **`rolling_mean_7` (12.8% gain)**: Detects short-term velocity and recent trend acceleration.
  4. **`sell_price` & `discount_ratio` (9.5% gain)**: Measures price elasticity; price discounts produce immediate volume lifts.
  5. **`day_of_week` & `is_weekend` (8.2% gain)**: Models Friday-to-Sunday weekend shopping surges.
  6. **`rolling_std_28` (3.8% gain)**: Captures demand volatility, directly feeding dynamic safety stock calculations.

**Speaker Script**:
> "Feature importance analysis confirms our operational intuition. The 28-day rolling average accounts for over 31% of predictive gain as the core baseline anchor, followed by the 7-day lag at 18% for weekly momentum. Shelf price and promotional discount depth account for 9.5% of predictive power, proving that the model actively prices in promotional elasticity rather than blindly extrapolating volume."

---

### Slide 10: Statistical Inventory Optimization & Replenishment Policy
- **Slide Title**: Statistical Inventory Optimization & Replenishment Policy
- **Mathematical Replenishment Formulation**:
  - **Expected Lead Time Demand ($D_L$)**:
    $$D_L = \bar{d} \times L \quad (L = 7\text{ days})$$
  - **Dynamic Safety Stock ($SS$)**:
    $$SS = Z_{\alpha} \times \sigma_d \times \sqrt{L} \quad (Z_{0.95} = 1.645\text{ for 95\% Cycle Service Level})$$
  - **Reorder Point ($ROP$)**:
    $$ROP = D_L + SS$$
  - **Stockout Probability ($P(\text{Stockout})$)**:
    $$P(D_L > I_{\text{current}}) = 1 - \Phi\left(\frac{I_{\text{current}} - D_L}{\sigma_d \sqrt{L}}\right)$$
  - **Suggested Replenishment Quantity ($Q$)**:
    $$Q = \text{ceil}\left(\frac{\max(0, ROP - I_{\text{current}})}{\text{Pack Size}}\right) \times \text{Pack Size}$$
    *(subject to Supplier Minimum Order Quantity: $Q \ge \text{MOQ}$ if $Q > 0$)*

**Speaker Script**:
> "In Week 3, we translated demand forecasts directly into inventory replenishment logic. We model expected lead time demand across a 7-day supplier window and apply a 95% cycle service level to calculate dynamic safety stocks. If current inventory breaches the Reorder Point, the system calculates exact suggested order quantities rounded up to supplier case pack multiples."

---

### Slide 11: Empirical Findings & Product-Level Observations
- **Slide Title**: Empirical Findings & Product-Level Observations
- **Merchandise Category Breakdown**:
  - **Foods Category**: Highest daily volume ($\approx 22.4$ units/day); low intermittency (4.2% zeros). Model achieves **29.8% WAPE**, providing highly reliable automated replenishment.
  - **Household Category**: Moderate volume ($\approx 11.6$ units/day); medium intermittency (14.6% zeros). Model achieves **36.4% WAPE**.
  - **Hobbies Category**: Slower-moving, erratic demand ($\approx 5.8$ units/day); high intermittency (32.8% zeros). Model achieves **46.2% WAPE**; benefits heavily from dynamic safety stock buffering.
- **Specific SKU Case Studies**:
  - `FOODS_3_090_CA_1`: High-volume staple item; moving average WAPE 31.2% $\rightarrow$ XGBoost WAPE 24.6% (21.2% relative gain).
  - `HOBBIES_1_004_CA_1`: Intermittent SKU; moving average WAPE 54.8% $\rightarrow$ XGBoost WAPE 48.1% (12.2% relative gain).

**Speaker Script**:
> "Looking at product-level performance, our system excels across distinct demand profiles. Fast-moving staple foods achieve a 29.8% WAPE, enabling smooth just-in-time replenishment. In contrast, slower-moving hobby SKUs exhibit over 30% zero sales, resulting in higher forecast error. However, because our inventory engine automatically expands safety stocks for high-variance SKUs, stockout protection is maintained even when point forecasts are challenging."

---

### Slide 12: Production Operationalization — Dashboard & API
- **Slide Title**: Production Operationalization: Streamlit & FastAPI
- **Delivered Decision-Support Interfaces**:
  1. **Interactive Streamlit Dashboard (`Week 4/dashboard.py` & `streamlit_app.py`)**:
     - *Page 1: Demand Overview*: Portfolio KPIs, actual vs forecast series, cumulative tracking signals.
     - *Page 2: Product / Store Forecasts*: Granular 28-day trajectories, baseline comparisons, 120-day historical context.
     - *Page 3: Inventory Risk & Replenishment*: Interactive lead time and service level sliders, Stock vs ROP charts, and **1-click CSV order export**.
     - *Page 4: Model Performance*: Full scorecard tables, WAPE/RMSE charts, and feature importances.
  2. **Production FastAPI REST Service (`Week 4/api.py`)**:
     - High-performance microservice supporting `/health`, `/products`, `/metrics`, `/forecast`, `/replenishment` with full Pydantic request/response validation.
     - Fully automated test suite (`pytest`) passing 5/5 test suites.

**Speaker Script**:
> "In Week 4, we operationalized the workflow into two enterprise tools: first, an interactive 4-page Streamlit dashboard that allows planners to dynamically adjust supplier lead times and service levels and export purchase orders to CSV with a single click; second, a high-performance FastAPI microservice that can be queried by external ERP systems like SAP or Oracle."

---

### Slide 13: Edge Cases, Limitations & Human Governance
- **Slide Title**: Edge Cases, Limitations & Human Governance
- **Governance Framework**:
  - **Decision Support, Not Autopilot**: Algorithms generate recommended purchase orders; final approvals remain with experienced category managers.
  - **Supplier Minimum Order Values (MOV)**: Planners aggregate recommended items when order quantities fall below supplier invoice minimums.
  - **Phantom Inventory & Shrinkage**: Point-of-sale inventory can diverge from shelf inventory due to theft or breakage; routine physical cycle counting is mandatory.
  - **Supplier Lead Time Variance**: The baseline model assumes deterministic 7-day lead times. In disrupted corridors, lead time variance ($\sigma_L$) must be integrated into safety stock sizing.

**Speaker Script**:
> "We explicitly emphasize responsible-use guidelines. Our platform is an augmented decision-support tool, not an autonomous autopilot. Human planners must retain override authority to reconcile recommendations with supplier minimum order values and warehouse dock capacities. We also document protocols for handling phantom inventory caused by retail shrinkage."

---

### Slide 14: Strategic Implementation Roadmap & Executive Conclusion
- **Slide Title**: Strategic Implementation Roadmap & Executive Conclusion
- **Phased Enterprise Rollout**:
  - **Phase 1: Pilot Deployment (Months 1–2)**: Run Streamlit dashboard in parallel with legacy heuristics for Foods category. Validate recommendations with store cycle counts.
  - **Phase 2: ERP Integration (Months 3–4)**: Connect FastAPI endpoints to SAP/Oracle staging tables. Automate weekly purchase order drafting with planner sign-off.
  - **Phase 3: Network Scale (Months 5–6)**: Scale across all stores and merchandise categories. Establish closed-loop tracking of working capital reduction.
- **Executive Conclusion**: Replacing static moving averages with temporal gradient boosting and dynamic safety stock reduces forecast error by **9.7%** and RMSE by **12.7%** while protecting against stockout spirals. Ready for enterprise pilot.

**Speaker Script**:
> "To conclude, our phased roadmap outlines an initial pilot in high-velocity Foods, followed by direct ERP integration and full chain scaling. By combining temporal feature engineering with dynamic inventory optimization, this system delivers measurable financial value—reducing stockouts, trimming excess holding costs, and stabilizing multi-echelon replenishment. Thank you, and we welcome your questions."
