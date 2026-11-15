# Demand Forecasting & Inventory Optimization Using Machine Learning
## Executive Stakeholder Presentation Deck & Speaker Transcript

---

### Slide 1: Executive Title Slide
**Slide Title**: Demand Forecasting & Inventory Optimization Using Machine Learning  
**Subtitle**: Data Science & Machine Learning Internship Final Capstone Presentation  
**Program**: Advanced Machine Learning & Supply Chain Analytics  
**Presenter**: Data Science & ML Engineering Team  
**Benchmark Case Study**: Walmart US M5 Retail Dataset (1,913 Historical Days)  
**File Reference**: Native PowerPoint slides available in `Week 4/final_presentation.pptx`

**Speaker Script**:
> "Good morning, leadership team. Today we present our four-week capstone implementation: an end-to-end predictive demand forecasting and inventory decision-support system. In high-volume retail environments, the difference between profitability and loss frequently comes down to balancing shelf availability against the carrying costs of excess inventory. Our system couples non-leaking gradient boosted ensembles with stochastic inventory replenishment policies to solve this problem empirically."

---

### Slide 2: The Business Challenge — The Retail Balancing Dilemma
**Slide Title**: The Retail Inventory Dilemma: Stockouts vs. Carrying Costs  
**Core Problem**: Supply chain operations are trapped between two competing cost curves:
- **Cost of Under-Forecasting (Stockout Cost, $C_s$)**:
  - Forfeited gross margin at point-of-sale.
  - Emergency expedited supplier rush and freight charges.
  - 31% of grocery shoppers permanently switch stores upon encountering a stockout.
  - SLA chargebacks and penalties from retail partners.
- **Cost of Over-Forecasting (Inventory Carrying Cost, $C_h$)**:
  - 18% to 25% annual carrying cost of tied-up working capital.
  - Warehouse congestion, pallet storage fees, and handling labor.
  - Spoilage and expiration of perishable goods.
  - Margin-eroding promotional markdowns required to clear excess stock.

**Speaker Script**:
> "Traditional retail planning often relies on trailing moving averages or simple rules of thumb. The issue is that simple averages lag behind sudden demand surges during holidays and promotional markdowns. When planners overreact, they create the Bullwhip Effect, amplifying order swings upstream and either incurring expensive emergency freight or tying up working capital in stagnant inventory."

---

### Slide 3: Multi-Source Retail Telemetry & Quality Audit
**Slide Title**: Multi-Source Retail Telemetry & Quality Audit  
**Data Landscape (1,913 Contiguous Days)**:
- **Sales History**: Daily unit sales across Foods, Household, and Hobbies merchandise departments.
- **Calendar Metadata**: 30 national holidays, cultural events, and state-level SNAP benefit distribution schedules across California, Texas, and Wisconsin.
- **Weekly Shelf Prices**: Store-item weekly prices capturing inflation and promotional markdown depth.
- **Quality Audit Findings**:
  - 1,913 contiguous calendar days (`2011-01-29` to `2016-04-24`) verified with zero missing days.
  - Zero duplicate records on composite key `(store_id, item_id, date)`.
  - Zero negative sales quantities; 100% valid positive selling prices.
  - 17.14% zero-sales observations identified, requiring volume-weighted metrics (WAPE) over MAPE.

**Speaker Script**:
> "We began Week 1 with an exhaustive data audit. A critical principle in retail analytics is data cleanliness. We audited 1,913 continuous days of sales, calendar, and pricing records. We verified that our composite keys had zero duplicates, no negative sales, and complete date continuity. We also identified that 17% of observations contained zero sales, which guided our selection of error metrics."

---

### Slide 4: Core Methodological Insight — Latent Demand vs. Censored Sales
**Slide Title**: Latent Demand vs. Observed Sales During Stockouts  
**The Censored Demand Problem**:
$$S_t = \min\left(D_t, I_t\right)$$
- When stock runs out ($I_t = 0$), point-of-sale registers record $S_t = 0$.
- Actual consumer demand $D_t$ remains positive ($D_t > 0$).
- **The Stockout Spiral of Death**: Naive machine learning models trained on raw sales learn that demand dropped to zero, falsely depressing future forecasts ($\hat{D}_{t+1} \approx 0$). Planners place smaller purchase orders, triggering persistent, recurring stockouts.
- **Our Resolution Protocol**:
  1. *Pre-Launch Separation*: Filter out records prior to initial SKU store listing (`is_active = False`).
  2. *Stockout Identification*: Flag active items with zero sales despite valid pricing as `potential_stockout` (13.38% of active rows).
  3. *Variance Protection*: Demand variance ($\sigma_d$) is calculated strictly over active, in-stock periods to prevent safety stock deflation.

**Speaker Script**:
> "Here is our most important methodological distinction: observed sales do not equal customer demand. When a grocery store runs out of milk, the cash register records zero sales. If an algorithm is trained naively on that zero, it concludes customers no longer want milk and recommends ordering even less next week. We developed an empirical filtering heuristic that separates pre-launch records from active stockouts, protecting our safety stock calculations from downward death spirals."

---

### Slide 5: Feature Engineering Architecture
**Slide Title**: Non-Leaking Feature Engineering Architecture  
**Strict Leakage Prevention Protocol**:
- To support a 28-day forward planning horizon and a 7-day supplier lead time, all features are strictly anchored prior to the forecast horizon ($t \ge 7$).
- **27 Non-Leaking Features Created**:
  1. *Past-Only Lags (6 features)*: $t-7, t-14, t-21, t-28, t-35, t-42$.
  2. *Anchored Rolling Statistics (9 features)*: Rolling mean, standard deviation, and maximum over 7, 14, and 28-day windows anchored at $t-7$.
  3. *Calendar & Events (8 features)*: Day of week, month, year, weekend indicators, holiday markers, event type codes, and California SNAP distribution schedules.
  4. *Price Elasticity & Promotions (4 features)*: Shelf price, deviation from SKU average price, percentage discount relative to historical peak, and promotional markdown indicators.

**Speaker Script**:
> "In Week 2, we engineered 27 features designed with strict temporal causality. A major trap in applied forecasting is target leakage—using information from tomorrow to predict today. To support our 7-day lead time and 28-day replenishment horizon, all lag features and rolling aggregations are anchored strictly at day t minus 7 or earlier. We also capture calendar harmonics, weekend shopping lifts, California SNAP food stamp cycles, and promotional discount depths."

---

### Slide 6: Chronological Partitioning Plan
**Slide Title**: Strict Chronological Partitioning (Zero Target Leakage)  
**Partition Structure**:
- **Training Partition** (`2011-01-29` to `2016-02-28`): ~5 Years (90,850 rows) for model parameter estimation.
- **Validation Partition** (`2016-02-29` to `2016-03-27`): 28 Days (1,400 rows) for hyperparameter tuning.
- **Testing Partition** (`2016-03-28` to `2016-04-24`): 28 Days (1,400 rows) for unbiased out-of-sample operational evaluation.
- **Why Random K-Fold Fails**: Shuffling time-series observations leaks future information into training sets, creating artificially optimistic accuracy that collapses in production.

**Speaker Script**:
> "We enforce a strict chronological split rather than standard randomized k-fold cross-validation. In time series, randomized splitting leaks future data points into the past, providing an illusion of accuracy. We trained across 5 years of historical data, tuned our hyperparameters on a 28-day validation window, and held out a final 28-day test set that the models never saw during training."

---

### Slide 7: Model Evaluation Scorecard
**Slide Title**: Model Evaluation Scorecard on 28-Day Test Set  
**Out-of-Sample Empirical Results**:

| Model Architecture | Category | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) | Operational Decision |
|---|---|---|---|---|---|---|---|
| `Naive (Lag 7)` | Heuristic Baseline | 6.969 | 11.117 | 47.22% | 74.62% | +2.22% | Benchmark |
| `Seasonal Naive (Lag 28)` | Heuristic Baseline | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% | Benchmark |
| `Moving Average (7-Day)` | Heuristic Baseline | 6.658 | 10.171 | 45.11% | 72.24% | +3.16% | Benchmark |
| `Moving Average (28-Day)` | Heuristic Baseline | 6.219 | 9.634 | 42.14% | 66.48% | +0.09% | Best Heuristic Baseline |
| `Random Forest Regressor` | Machine Learning | 5.717 | 8.489 | 38.74% | 63.40% | +3.29% | Evaluated Candidate |
| `LightGBM Regressor` | Machine Learning | 5.632 | 8.447 | 38.16% | 64.33% | +3.86% | Evaluated Candidate |
| **`XGBoost Regressor`** | **Machine Learning** | **5.612** | **8.410** | **38.03%** | **62.43%** | **+3.03%** | **Winning Production Champion** |

**Key Takeaways**:
- **XGBoost** beats the best heuristic baseline (`Moving Average 28-Day`), reducing WAPE from **42.14% to 38.03%** (a **4.11 percentage point improvement**, representing a **9.7% relative error reduction**).
- RMSE drops from **9.634 to 8.410** (a **12.7% reduction**), proving resilience against outlier demand spikes.
- Controlled bias of **+3.03%** provides mild commercial buffer protection against stockouts.

**Speaker Script**:
> "Here are our final empirical results on the held-out test partition. The best classical baseline—a 28-day moving average—achieved a 42.14% WAPE. Our winning XGBoost regressor reduced that to 38.03%, representing an almost 10% relative error reduction and a 12.7% reduction in RMSE. Furthermore, it achieved this with a balanced positive bias of plus 3%, which is commercially ideal because it protects against stockouts without causing excessive overstock."

---

### Slide 8: Demand Drivers & Feature Importance
**Slide Title**: Demand Drivers & Feature Importance (XGBoost)  
**Primary Drivers of Retail Demand**:
1. **`rolling_mean_28` (31.4%)**: Serves as the primary medium-term volume anchor.
2. **`sales_lag_7` (18.6%)**: Strongest single lag; captures day-of-week sales momentum.
3. **`rolling_mean_7` (12.8%)**: Detects short-term demand velocity and recent acceleration.
4. **`sell_price` & `discount_ratio` (9.5%)**: Measures price elasticity; price cuts produce immediate volume expansion.
5. **`day_of_week` & `is_weekend` (8.2%)**: Models Friday-to-Sunday weekend shopping surges.
6. **`rolling_std_28` (3.8%)**: Measures demand volatility, directly informing safety stock sizing.

**Speaker Script**:
> "Feature importance analysis from our winning model confirms our operational hypotheses. The 28-day rolling average accounts for over 31% of predictive gain as the core baseline anchor, followed by the 7-day lag at 18% for weekly momentum. Shelf price and promotional discount depth account for 9.5% of predictive power, allowing the model to capture price elasticity."

---

### Slide 9: Statistical Inventory Optimization & Replenishment
**Slide Title**: Statistical Inventory Optimization & Replenishment Policy  
**Mathematical Formulation**:
- **Expected Lead Time Demand ($D_L$)**: $D_L = \bar{d} \times L$ ($L = 7\text{ days}$).
- **Dynamic Safety Stock ($SS$)**: $SS = Z_{\alpha} \times \sigma_d \times \sqrt{L}$ ($Z_{0.95} = 1.645$ for 95% Cycle Service Level).
- **Reorder Point ($ROP$)**: $ROP = D_L + SS$.
- **Stockout Probability ($P(\text{Stockout})$)**: $P(D_L > I_{\text{current}}) = 1 - \Phi\left(\frac{I_{\text{current}} - D_L}{\sigma_d \sqrt{L}}\right)$.
- **Suggested Order Quantity ($Q$)**: Net requirement rounded up to wholesale pack multiples (6, 12, 24) and MOQ constraints.

**Speaker Script**:
> "In Week 3, we translated demand forecasts directly into inventory replenishment logic. We model expected lead time demand across a 7-day supplier window and apply a 95% cycle service level to calculate dynamic safety stocks. If current inventory breaches the Reorder Point, the system calculates exact suggested order quantities rounded up to supplier case pack multiples."

---

### Slide 10: Production Operationalization — Dashboard & API
**Slide Title**: Production Operationalization: Streamlit & FastAPI  
**Delivered Decision-Support Interfaces**:
1. **Interactive Streamlit Dashboard (`Week 4/dashboard.py`)**:
   - *Page 1: Demand Overview*: Portfolio KPIs, actual vs forecast series, cumulative tracking signals.
   - *Page 2: Product / Store Forecasts*: Granular 28-day trajectories, baseline comparisons, 120-day history.
   - *Page 3: Inventory Risk & Replenishment*: Interactive lead time and service level sliders, Stock vs ROP charts, and **1-click CSV order export**.
   - *Page 4: Model Performance*: Full scorecard tables, WAPE/RMSE charts, and feature importances.
2. **Production FastAPI REST Service (`Week 4/api.py`)**:
   - `/health`, `/products`, `/metrics`, `/forecast`, `/replenishment` with full Pydantic validation.
   - Fully automated test suite (`pytest`) passing 5/5 test suites.

**Speaker Script**:
> "In Week 4, we operationalized the workflow into two enterprise tools: first, an interactive 4-page Streamlit dashboard that allows planners to dynamically adjust supplier lead times and service levels and export purchase orders to CSV with a single click; second, a high-performance FastAPI microservice that can be queried by external ERP systems like SAP or Oracle."

---

### Slide 11: Responsible Use & Human Governance
**Slide Title**: Limitations, Edge Cases & Human-in-the-Loop Governance  
**Governance Framework**:
- **Decision Support, Not Autopilot**: Algorithms generate recommended purchase orders; final approvals remain with experienced category managers.
- **Supplier Minimum Order Values (MOV)**: Planners aggregate recommended items when order quantities fall below supplier invoice minimums.
- **Phantom Inventory & Shrinkage**: Point-of-sale inventory can diverge from shelf inventory due to theft or breakage; routine physical cycle counting is mandatory.
- **Supplier Lead Time Variance**: The baseline model assumes deterministic 7-day lead times. In disrupted corridors, lead time variance ($\sigma_L$) must be integrated into safety stock sizing.

**Speaker Script**:
> "We explicitly emphasize responsible-use guidelines. Our platform is an augmented decision-support tool, not an autonomous autopilot. Human planners must retain override authority to reconcile recommendations with supplier minimum order values and warehouse dock capacities. We also document protocols for handling phantom inventory caused by retail shrinkage."

---

### Slide 12: Strategic Roadmap & Executive Conclusion
**Slide Title**: Strategic Implementation Roadmap & Executive Conclusion  
**Deployment Phases**:
- **Phase 1: Pilot Deployment (Months 1–2)**: Run Streamlit dashboard in parallel with legacy heuristics for Foods category. Validate recommendations with store cycle counts.
- **Phase 2: ERP Integration (Months 3–4)**: Connect FastAPI endpoints to SAP/Oracle staging tables. Automate weekly purchase order drafting with planner sign-off.
- **Phase 3: Network Scale (Months 5–6)**: Scale across all stores and merchandise categories. Establish closed-loop tracking of working capital reduction.
- **Executive Conclusion**: Replacing static moving averages with temporal gradient boosting and dynamic safety stock reduces forecast error by 9.7% while protecting against stockout spirals. Ready for enterprise pilot.

**Speaker Script**:
> "To conclude, our phased roadmap outlines an initial pilot in high-velocity Foods, followed by direct ERP integration and full chain scaling. By combining temporal feature engineering with dynamic inventory optimization, this system delivers measurable financial value—reducing stockouts, trimming excess holding costs, and stabilizing multi-echelon replenishment. Thank you, and we welcome your questions."
