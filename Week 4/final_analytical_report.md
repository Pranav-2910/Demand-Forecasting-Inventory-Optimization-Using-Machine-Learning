# Predictive Demand Forecasting & Inventory Optimization Using Machine Learning
## An End-to-End Decision Support System for Multi-Echelon Retail Supply Chains

---

### Document Control & Metadata
- **Project Title**: Predictive Demand Forecasting & Inventory Optimization Using Machine Learning
- **Program**: Data Science & Machine Learning Internship — Final Capstone Deliverable
- **Document Type**: Formal Technical & Analytical Report (Comprehensive 4-Week Implementation)
- **Target Audience**: Supply Chain Executives, Retail Category Planners, Machine Learning Engineers, and Operations Directors
- **Evaluation Benchmark**: Walmart US Retail Operations (M5 Case Study — 1,913 Historical Days)
- **Primary Methodologies**: Temporal Feature Engineering, Chronological Splitting, Heuristic Baselines, Gradient Boosted Ensembles (XGBoost), and Dynamic Safety Stock Stochastic Modeling
- **Production Status**: Production-Ready Prototypes Delivered (Interactive Streamlit Dashboard & FastAPI REST Service)

---

## 1. Executive Summary

In high-velocity retail supply chains, sustained profitability depends upon maintaining high on-shelf product availability while simultaneously minimizing the substantial financial carrying costs of excess inventory. Underestimating consumer demand creates product stockouts, forfeited revenue, brand defection, and emergency freight costs. Conversely, overestimating demand results in bloated working capital commitments, warehouse storage bottlenecks, shrinkage, and margin-eroding promotional markdowns.

This report presents the complete research, technical architecture, empirical findings, and deployment artifacts of an enterprise-grade **Predictive Demand Forecasting and Inventory Decision-Support System**. Implemented across a four-week lifecycle, the project addresses the core challenge of transforming noisy, intermittent point-of-sale telemetry into statistically sound, non-leaking demand forecasts and translating those forecasts directly into actionable purchase orders.

### Key Technical Achievements
1. **Zero-Leakage Temporal Feature Engineering**: Formulated 27 non-leaking predictive features utilizing historical sales lags strictly anchored prior to the forecast horizon ($t \ge 7$), anchored rolling window statistics (7, 14, 28 days), calendar harmonics, and promotional markdown indicators.
2. **Empirical Stockout & Latent Demand Heuristics**: Established analytical criteria distinguishing true zero customer demand from stockout-induced zero sales, preventing the negative feedback loops that degrade standard retail machine learning models.
3. **Rigorous Out-of-Sample Chronological Validation**: Established non-overlapping temporal partitions (5-year training window, 28-day validation window, and 28-day test window) eliminating lookahead bias and forward-looking data contamination.
4. **Demonstrated Machine Learning Superiority**: Tuned and evaluated candidate models against four classical retail baselines. The winning model—an optimized **XGBoost Regressor**—achieved a **38.03% Weighted Absolute Percentage Error (WAPE)** and a **5.612 unit Mean Absolute Error (MAE)** on the out-of-sample 28-day test horizon, outperforming the best heuristic baseline (28-Day Moving Average at 42.14% WAPE) by **4.11 percentage points**, representing a **9.7% relative error reduction** and a **12.7% reduction in RMSE**.
5. **Stochastic Inventory Replenishment Engine**: Developed an automated inventory optimization module that computes dynamic **Safety Stock ($SS$)**, **Reorder Points ($ROP$)**, **Stockout Risk Probabilities ($P(\text{Stockout})$)**, and suggested purchase orders rounded to wholesale case pack constraints under a target 95% cycle service level.
6. **Operational Decision-Support Interfaces**: Deployed an interactive 4-page **Streamlit Analytics Dashboard** and an enterprise-ready **FastAPI REST Service** with strict Pydantic contract validation, supported by a 100% passing automated test suite (`pytest`).

---

## 2. Business Context & Problem Formulation

### 2.1 The Retail Inventory Dilemma
Retail organizations operate on tight operating margins, typically ranging between 2% and 5% in the grocery and fast-moving consumer goods (FMCG) sectors. Within this economic environment, inventory investment constitutes the largest single commitment of working capital. Supply chain managers are forced to navigate two opposing economic cost curves:

```
Total Cost
    │            Total Cost Curve (Holding + Stockout)
    │                  \             /
    │                   \   *Min*   /
    │                    \    │    /
    │  Stockout Cost      \   │   /       Holding Cost
    │  (Lost Margin, Churn)\  │  /        (Capital, Spoilage, Space)
    │                       \ │ /
    └─────────────────────────┴──────────────────────── Inventory Level
```

1. **The Cost of Under-Forecasting (Stockout Cost, $C_s$)**:
   When consumer demand exceeds available on-hand stock ($D_t > I_t$), the immediate cost includes the forfeited gross margin of the lost sale. However, the indirect costs are often far more severe: expedited supplier rush fees, premium logistics freight, penalty fees from service level agreements (SLAs), and permanent customer defection to competing retailers. Studies in retail operations indicate that up to 31% of consumers navigate to another store when encountering a stockout in essential FMCG categories.

2. **The Cost of Over-Forecasting (Inventory Carrying Cost, $C_h$)**:
   Holding excess stock incurs continuous carrying costs, traditionally estimated between 18% and 25% of the inventory's cost of goods sold (COGS) annually. These carrying costs comprise:
   - *Cost of Capital*: The opportunity cost of working capital tied up in slow-moving pallets rather than revenue-generating investments.
   - *Storage & Warehousing Fees*: Pallet racking expenses, climate-controlled energy costs, and material handling labor.
   - *Shrinkage, Damage & Spoilage*: Perishable food expiration, packaging deterioration, and warehouse handling breakage.
   - *Markdown & Liquidation Losses*: End-of-cycle price slashes required to clear unsold merchandise from prime shelf space.

### 2.2 The Bullwhip Effect in Multi-Echelon Supply Chains
When retail stores base their replenishment decisions on inaccurate, highly volatile demand estimates, that variance is amplified upstream through regional distribution centers, central warehouses, and tiered suppliers—a phenomenon known as the *Bullwhip Effect*. By delivering accurate, low-bias demand forecasts and stabilizing safety stock buffers, machine learning reduces order variance amplification across the entire supply chain network.

---

## 3. Data Provenance, Scope & Multi-Source Schema Audit

### 3.1 Dataset Provenance & Scope
The analytical foundation for this system is built upon longitudinal point-of-sale and supply telemetry from the **Walmart US M5 Forecasting Benchmark**, encompassing 1,913 consecutive days of transactions from January 29, 2011 to April 24, 2016. The dataset mirrors real-world retail complexity across merchandise departments, temporal calendars, and pricing tiers.

The telemetry comprises three primary data feeds:
1. **Daily Unit Sales Telemetry (`sales_train_validation.csv`)**: Daily point-of-sale unit sales aggregated at the SKU-store level across merchandise categories:
   - `FOODS`: Fast-moving perishable and non-perishable consumables exhibiting weekly replenishment velocity.
   - `HOUSEHOLD`: Cleaning, paper goods, and home maintenance items with steady baseline consumption.
   - `HOBBIES`: Discretionary merchandise characterized by high demand intermittency and long tail patterns.
2. **Calendar & Event Metadata (`calendar.csv`)**: Multi-layered temporal attributes spanning 1,969 calendar days:
   - Date keys, day of the week, week numbering (`wm_yr_wk`), month, and year.
   - 30 distinct national, cultural, religious, and sporting event markers (e.g., Super Bowl, Thanksgiving, Easter, Labor Day, Eid al-Fitr).
   - State-level SNAP (Supplemental Nutrition Assistance Program) benefit distribution schedules across California (`snap_CA`), Texas (`snap_TX`), and Wisconsin (`snap_WI`).
3. **Weekly Shelf Pricing Records (`sell_prices.csv`)**: Shelf prices recorded weekly per store-item combination, enabling the tracking of price inflation, permanent price resets, and temporary promotional markdown events.

### 3.2 Data Quality Audit Findings
A comprehensive data quality audit script was executed across the source datasets, generating the following validation results:

| Audit Parameter | Verification Rule | Observed Metric | Status |
|---|---|---|---|
| **Date Continuity** | Contiguous calendar dates without temporal gaps | 1,913 consecutive days (`2011-01-29` to `2016-04-24`) | **PASSED** (0 missing days) |
| **Composite Key Uniqueness** | Zero duplicate records on `(store_id, item_id, date)` | 0 duplicate records identified | **PASSED** (100% unique keys) |
| **Sales Validity** | All observed sales quantities must satisfy $S_t \ge 0$ | Minimum sales = 0; 0 negative records | **PASSED** (No negative values) |
| **Price Validity** | All shelf prices must satisfy $P_t > 0$ | Min price = \$0.05, Max price = \$20.00; 0 nulls | **PASSED** (Valid positive pricing) |
| **Intermittency Ratio** | Proportion of zero-sales records across all rows | 17.14% of overall daily observations | **VERIFIED** (Requires zero-handling) |
| **Pre-Launch Records** | Zero sales prior to initial SKU store listing | 3.76% of raw initial records | **IDENTIFIED** (Requires filtering) |

---

## 4. Unconstrained Demand vs. Observed Sales During Stockouts

### 4.1 The Censored Demand Problem
A critical insight established during Week 1 exploratory analysis is that **point-of-sale cash register transactions do not equal true consumer demand**. In retail supply chain economics, observed sales ($S_t$) represent *censored demand*:

$$S_t = \min\left(D_t, I_t\right)$$

Where:
- $D_t$ is true unconstrained consumer demand.
- $I_t$ is available on-hand inventory at the start of period $t$.

When an item stockout occurs ($I_t = 0$):
- The point-of-sale register records $S_t = 0$.
- However, latent customer demand $D_t$ remains positive ($D_t > 0$).

### 4.2 The Destructive Feedback Loop of Naive Forecasting
If a machine learning algorithm is trained directly on unadjusted sales data without accounting for stockouts, the model interprets $S_t = 0$ as a genuine absence of consumer interest. In subsequent planning cycles:
1. The model forecasts reduced demand ($\hat{D}_{t+1} \approx 0$).
2. The inventory planner orders fewer replenishment units.
3. The store stocks out even faster in period $t+1$.
4. The item is eventually dropped from store shelves due to "poor sales," even though customer demand was consistently high.

```
┌─────────────────────────────────────────────────────────────┐
│              THE STOCKOUT SPIRAL OF DEATH                   │
│                                                             │
│   Stockout Occurs (Inventory = 0)                           │
│        │                                                    │
│        ▼                                                    │
│   Observed POS Sales = 0                                    │
│        │                                                    │
│        ▼                                                    │
│   Naive Model Interprets as "Zero Consumer Demand"          │
│        │                                                    │
│        ▼                                                    │
│   Forecast Artificially Depressed                           │
│        │                                                    │
│        ▼                                                    │
│   Replenishment Order Reduced                               │
│        │                                                    │
│        ▼                                                    │
│   Next Period Stockout Occurs Faster ──► (Cycle Repeats)    │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Analytical Identification Heuristic
To prevent this negative bias without fabricating ungrounded telemetry, our pipeline implements a two-stage identification heuristic:
1. **Pre-Launch Period Separation**: Many items were introduced into stores months or years after the January 2011 start date. Zero sales recorded prior to an item's first observed sale or valid pricing record are flagged as `is_active = False` and excluded from training sets.
2. **Active Listing Potential Stockout Flagging**: For actively listed items (`is_active = True`), instances where `sales == 0` while a valid `sell_price` exists and historical demand velocity was positive are flagged as `potential_stockout` (accounting for 13.38% of active records).
3. **Variance Protection**: In the downstream inventory engine, demand variance ($\sigma_d$) is calculated strictly over active, in-stock windows. This prevents artificial deflation of safety stock buffers.

---

## 5. Non-Leaking Feature Engineering Architecture

### 5.1 Temporal Causality & Target Leakage Prevention
In time-series forecasting, target leakage occurs when a feature inadvertently incorporates information that would not be available in production at the forecast origin time. To support a **28-day retail replenishment horizon** and a **7-day supplier lead time**, the forecasting pipeline enforces strict temporal causality:

$$\text{Information Cutoff: } t_0 \le T - \text{Horizon}$$

**Core Architectural Rule**: Every lag feature and rolling aggregation window is anchored strictly at $t-7$ or earlier ($t \ge 7$). Contemporaneous features ($t-1$ through $t-6$) are strictly excluded because in real-world retail ordering, purchase orders must be submitted 7 days prior to delivery.

```
       T - 42      T - 28       T - 14        T - 7        T (Forecast Origin)     T + 28 (Horizon End)
         │           │            │            │                   │                         │
Historical Information Available (Lags >= 7d)  │◄── Lead Time ────►│◄── Forecast Window ────►│
═══════════════════════════════════════════════╪═══════════════════╪═════════════════════════╡
[================ ALLOWED ====================]│[==== BLOCKED ====]│[====== TARGET Y =======]
```

### 5.2 Feature Registry Breakdown (27 Engineered Attributes)

The transformed feature matrix (`Week 2/features_engineered.csv`, 33.8 MB) comprises 27 features categorized into four functional groups:

#### Group A: Past-Only Lag Features (6 Attributes)
- `sales_lag_7`: Unit sales exactly 7 days prior; captures day-of-week persistence.
- `sales_lag_14`: Unit sales 14 days prior; captures bi-weekly shopping momentum.
- `sales_lag_21`: Unit sales 21 days prior; three-week cyclical indicator.
- `sales_lag_28`: Unit sales 28 days prior; monthly calendar cycle anchor.
- `sales_lag_35`: Unit sales 35 days prior; seasonal persistence marker.
- `sales_lag_42`: Unit sales 42 days prior; medium-term historical baseline.

#### Group B: Anchored Rolling Statistics (9 Attributes)
All rolling aggregations are anchored at $t-7$ to eliminate lookahead contamination:
- `rolling_mean_7`, `rolling_std_7`, `rolling_max_7`: Short-term 7-day demand velocity and volatility.
- `rolling_mean_14`, `rolling_std_14`, `rolling_max_14`: Bi-weekly demand trend and peak volume.
- `rolling_mean_28`, `rolling_std_28`, `rolling_max_28`: Monthly volume anchor and baseline demand variance.

#### Group C: Calendar, Seasonality & Promotional Attributes (8 Attributes)
- `day_of_week` (0=Monday, 6=Sunday): Captures substantial weekend shopping surges.
- `day_of_month` (1 to 31): Captures pay-cycle demand spikes (1st and 15th of the month).
- `month` (1 to 12) & `year` (2011 to 2016): Captures macro trend and annual seasonality.
- `is_weekend`: Binary flag indicating Friday through Sunday shopping days.
- `has_event`: Binary flag indicating national holidays or cultural events.
- `event_type_code`: Categorical integer encoding event type (Sporting, Cultural, National, Religious).
- `snap_CA`: Binary flag indicating active SNAP food stamp distribution days in California.

#### Group D: Price Elasticity & Promotional Markdown Metrics (4 Attributes)
- `sell_price`: Current shelf price in US dollars.
- `price_diff_to_mean`: Dollar deviation of current price from the SKU's historical average:
  $$\Delta P_{it} = P_{it} - \bar{P}_i$$
- `discount_ratio`: Percentage discount relative to historical peak shelf price:
  $$\text{Discount}_{it} = \frac{\max_{\tau \le t}(P_{i\tau}) - P_{it}}{\max_{\tau \le t}(P_{i\tau})}$$
- `is_discounted`: Binary indicator flagging whether an item is on promotional markdown ($\text{Discount}_{it} > 0.05$).

---

## 6. Chronological Partitioning & Model Selection Strategy

### 6.1 The Fallacy of Random K-Fold Cross-Validation
A frequent flaw in applied time-series machine learning is the use of randomized k-fold cross-validation. In temporal datasets, randomly shuffling observations causes future observations ($t+k$) to leak into the training partition, allowing models to predict past sales using future knowledge. This inflates evaluation scores during development but causes severe catastrophic failure when deployed to production.

### 6.2 Strict Chronological Split Design
To ensure that models are evaluated under authentic out-of-sample operational conditions, a **strict non-overlapping chronological split** was established:

| Partition | Start Date | End Date | Duration | Record Count | Operational Purpose |
|---|---|---|---|---|---|
| **Training Set** | `2011-01-29` | `2016-02-28` | ~5 Years (1,857 Days) | 90,850 rows | Learning feature representations and fitting model weights |
| **Validation Set** | `2016-02-29` | `2016-03-27` | 28 Days (4 Weeks) | 1,400 rows | Hyperparameter tuning and model selection |
| **Testing Set** | `2016-03-28` | `2016-04-24` | 28 Days (4 Weeks) | 1,400 rows | Final unbiased evaluation of operational performance |

```
2011-01-29                                      2016-02-28     2016-03-27     2016-04-24
├──────────────────────────────────────────────────────┼──────────────┼──────────────┤
│                TRAINING SET (~5 Years)               │  VALIDATION  │   TEST SET   │
│             Fitting Model Parameters                 │ Hyper-tuning │ Final Eval   │
└──────────────────────────────────────────────────────┴──────────────┴──────────────┘
```

The 28-day duration exactly mirrors the standard four-week retail planning horizon, ensuring that performance metrics directly reflect commercial reality.

---

## 7. Candidate Forecasting Architectures & Heuristic Baselines

### 7.1 Heuristic Benchmark Formulations
To validate whether machine learning provides genuine commercial value, candidates must be benchmarked against standard retail heuristics:

1. **Naive Forecast (Lag 7)**:
   Assumes sales on day $t$ will exactly equal sales on the same day of the prior week:
   $$\hat{y}_{t} = y_{t-7}$$
   *Role*: Establishes baseline performance of day-of-week persistence.

2. **Seasonal Naive Forecast (Lag 28)**:
   Assumes sales on day $t$ will equal sales exactly four weeks prior:
   $$\hat{y}_{t} = y_{t-28}$$
   *Role*: Evaluates whether four-week monthly cycles alone can forecast demand.

3. **7-Day Moving Average Baseline**:
   Takes the rolling arithmetic mean of the preceding 7 days (anchored at $t-7$):
   $$\hat{y}_t = \frac{1}{7}\sum_{k=7}^{13} y_{t-k}$$
   *Role*: Evaluates whether short-term momentum smoothing captures retail velocity.

4. **28-Day Moving Average Baseline**:
   Takes the rolling arithmetic mean of the preceding 28 days (anchored at $t-7$):
   $$\hat{y}_t = \frac{1}{28}\sum_{k=7}^{34} y_{t-k}$$
   *Role*: Evaluates medium-term volume stability. In retail practice, this is the most common heuristic baseline.

### 7.2 Supervised Machine Learning Architectures
Three competitive machine learning algorithms were formulated and tuned:

1. **Random Forest Regressor**:
   An ensemble of 100 de-correlated decision trees utilizing bootstrap aggregating (bagging):
   - Hyperparameters: `n_estimators=100`, `max_depth=12`, `min_samples_leaf=5`, `n_jobs=-1`.
   - Strengths: Robust to feature collinearity and outliers; provides low variance estimates.

2. **LightGBM Regressor**:
   Fast, leaf-wise gradient boosting trees using histogram-based binning:
   - Hyperparameters: `n_estimators=150`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`.
   - Strengths: Highly efficient leaf-wise growth capturing deep feature interactions.

3. **XGBoost Regressor (Winning Champion)**:
   Extreme Gradient Boosting using second-order Taylor series approximations of the loss function:
   - Objective: Minimize regularized mean squared error:
     $$\mathcal{L} = \sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2 + \sum_{k=1}^K \left(\gamma T_k + \frac{1}{2}\lambda \|w_k\|^2 + \alpha \|w_k\|_1\right)$$
   - Hyperparameters: `n_estimators=150`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`.
   - Strengths: Depth-wise level growth prevents severe overfitting; $L_1/L_2$ regularization handles correlated lag features effectively.

---

## 8. Empirical Evaluation, Metric Selection & Benchmark Scorecard

### 8.1 Metric Selection Rationale in Retail Forecasting
In standard regression, Mean Squared Error (MSE) and Mean Absolute Error (MAE) are common. However, retail demand forecasting requires specific consideration:

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
   Interpretable directly in retail units sold.

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   Penalizes large forecast errors quadratically; critical for identifying catastrophic under-forecasts on high-volume products.

3. **Weighted Absolute Percentage Error (WAPE) — Primary Metric**:
   $$\text{WAPE} = \frac{\sum_{i=1}^n |y_i - \hat{y}_i|}{\sum_{i=1}^n y_i} \times 100\%$$
   *Why WAPE is essential*: Standard Mean Absolute Percentage Error (MAPE) divides by actual demand ($y_i$). When sales are zero ($y_i = 0$), MAPE divides by zero and becomes undefined or produces extreme values (e.g. 500%+). WAPE sums all errors across the entire portfolio and divides by total volume, providing a stable, volume-weighted percentage metric.

4. **Forecast Bias (%)**:
   $$\text{Bias} = \frac{\sum_{i=1}^n (\hat{y}_i - y_i)}{\sum_{i=1}^n y_i} \times 100\%$$
   Indicates whether a model systematically over-predicts (positive bias) or under-predicts (negative bias).

### 8.2 Final Evaluation Scorecard on 28-Day Test Set

All models and heuristic baselines were evaluated on the strictly held-out 28-day test set (`2016-03-28` to `2016-04-24`):

| Model Architecture | Category | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) | Operational Decision |
|---|---|---|---|---|---|---|---|
| `Naive (Lag 7)` | Heuristic Baseline | 6.969 | 11.117 | 47.22% | 74.62% | +2.22% | Benchmark |
| `Seasonal Naive (Lag 28)` | Heuristic Baseline | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% | Benchmark |
| `Moving Average (7-Day)` | Heuristic Baseline | 6.658 | 10.171 | 45.11% | 72.24% | +3.16% | Benchmark |
| `Moving Average (28-Day)` | Heuristic Baseline | 6.219 | 9.634 | 42.14% | 66.48% | +0.09% | Best Heuristic Baseline |
| `Random Forest` | Machine Learning | 5.717 | 8.489 | 38.74% | 63.40% | +3.29% | Strong ML Candidate |
| `LightGBM` | Machine Learning | 5.632 | 8.447 | 38.16% | 64.33% | +3.86% | Competitive Candidate |
| **`XGBoost`** | **Machine Learning** | **5.612** | **8.410** | **38.03%** | **62.43%** | **+3.03%** | **Winning Production Champion** |

```
WAPE (%) Across Tested Architectures (Lower is Better)
┌─────────────────────────────────────────────────────────────┐
│ Naive (Lag 7)              47.22% ██████████████████████    │
│ Seasonal Naive (Lag 28)    47.58% ██████████████████████    │
│ Moving Avg (7-Day)         45.11% ████████████████████      │
│ Moving Avg (28-Day)        42.14% ██████████████████        │
│ Random Forest              38.74% ███████████████           │
│ LightGBM                   38.16% ███████████████           │
│ XGBoost (Winner)           38.03% ███████████████           │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 In-Depth Analysis of Results
1. **Statistically Significant Baseline Outperformance**:
   **XGBoost** beats the leading heuristic benchmark (`Moving Average 28-Day`) across every metric, reducing WAPE from **42.14% to 38.03%** (a **4.11 percentage point improvement**, representing a **9.7% relative error reduction**). Furthermore, XGBoost reduces RMSE from **9.634 to 8.410** (a **12.7% reduction**), indicating that gradient boosting suppresses large catastrophic errors on promotional surge days.
2. **Controlled, Operationally Desirable Bias (+3.03%)**:
   The winning XGBoost model exhibits a mild positive bias of **+3.03%**. In retail replenishment, a slight positive bias is commercially preferred over negative bias because the marginal revenue loss of an out-of-stock event ($C_s$) is generally higher than the carrying cost of holding a slight excess buffer ($C_h$).
3. **Feature Importance Rankings**:
   Analysis of Gini impurity gain from the XGBoost model identifies the primary drivers of demand:
   - `rolling_mean_28` (31.4%): Acts as the primary medium-term volume anchor.
   - `sales_lag_7` (18.6%): Captures weekly day-of-week consumption momentum.
   - `rolling_mean_7` (12.8%): Captures short-term velocity.
   - `sell_price` & `discount_ratio` (9.5%): Captures consumer price sensitivity during promotional discount events.
   - `day_of_week` & `is_weekend` (8.2%): Accounts for major Friday–Sunday volume expansion.

---

## 9. Statistical Inventory Replenishment & Decision Policy

### 9.1 Mathematical Formulation of Replenishment Parameters
Machine learning forecasts alone do not solve supply chain problems; they must be integrated into inventory decision policies. Our system translates daily forecasts into automated replenishment orders using classical inventory control theory:

#### 1. Expected Lead Time Demand ($D_L$)
The expected customer consumption during the supplier replenishment lead time ($L = 7\text{ days}$):
$$D_L = \sum_{t=1}^{L} \hat{d}_t = \bar{d} \times L$$

#### 2. Dynamic Safety Stock ($SS$)
To absorb demand volatility during lead time and prevent stockouts, safety stock is calculated based on active listing demand standard deviation ($\sigma_d$):
$$SS = Z_{\alpha} \times \sigma_d \times \sqrt{L}$$
Where:
- $Z_{\alpha} = 1.645$ corresponds to a **95% Cycle Service Level ($CSL$)** under standard normal distribution assumptions ($\alpha = 0.05$).
- $\sigma_d$: Standard deviation of daily demand computed over active listing windows.
- $L$: Supplier lead time (7 days).

#### 3. Reorder Point ($ROP$)
The inventory level that triggers a replenishment order:
$$ROP = D_L + SS$$
When available stock drops to or below $ROP$, an immediate order must be placed.

#### 4. Stockout Risk Probability ($P(\text{Stockout})$)
The cumulative probability that lead time demand exceeds current on-hand stock ($I_{\text{current}}$):
$$P(\text{Stockout}) = P(D_L > I_{\text{current}}) = 1 - \Phi\left(\frac{I_{\text{current}} - D_L}{\sigma_d \sqrt{L}}\right)$$
- **HIGH RISK**: $I_{\text{current}} < D_L \implies P(\text{Stockout}) \ge 50\%$. Immediate expedited ordering required.
- **MEDIUM RISK**: $D_L \le I_{\text{current}} < ROP$. Regular replenishment window active; safety buffer is eroding.
- **SAFE**: $I_{\text{current}} \ge ROP$. Stock levels are sufficient.

#### 5. Suggested Order Quantity ($Q$)
Net order requirement over the 28-day forecast horizon ($\hat{D}_{\text{horizon}}$):
$$Q_{\text{raw}} = \max\left(0, ROP - I_{\text{current}} + \hat{D}_{\text{horizon}}\right)$$

To respect commercial supplier constraints, $Q_{\text{raw}}$ is adjusted for case pack size ($\text{Pack Size} = 6$) and Minimum Order Quantity ($\text{MOQ} = 12$):
$$Q = \max\left(\text{MOQ}, \left\lceil \frac{Q_{\text{raw}}}{\text{Pack Size}} \right\rceil \times \text{Pack Size}\right)$$

### 9.2 Actionable Replenishment Recommendations Sample
The table below illustrates sample output from `inventory_replenishment.py` (available in `Week 3/replenishment_recommendations.csv`):

| Item ID | Store | On-Hand Stock | Lead-Time Demand | Safety Stock | Reorder Point (ROP) | Stockout Risk Tier | Stockout Prob (%) | Suggested Order | Est. Capital ($) |
|---|---|---|---|---|---|---|---|---|---|
| `FOODS_3_744` | CA_1 | 44 | 83.4 | 15.6 | 99.0 | **HIGH** | 100.0% | **390 units** | \$1,365.00 |
| `FOODS_3_090` | CA_1 | 182 | 351.0 | 94.7 | 445.7 | **HIGH** | 99.8% | **1,668 units** | \$2,502.00 |
| `FOODS_3_362` | CA_1 | 62 | 121.0 | 35.5 | 156.5 | **HIGH** | 99.7% | **582 units** | \$1,746.00 |
| `FOODS_3_694` | CA_1 | 46 | 80.1 | 20.5 | 100.7 | **HIGH** | 99.7% | **378 units** | \$1,134.00 |
| `FOODS_2_019` | CA_1 | 61 | 104.2 | 29.9 | 134.1 | **HIGH** | 99.1% | **492 units** | \$1,476.00 |
| `FOODS_1_085` | CA_1 | 48 | 76.7 | 22.4 | 99.1 | **HIGH** | 98.2% | **360 units** | \$1,080.00 |
| `FOODS_3_702` | CA_1 | 43 | 65.8 | 19.9 | 85.7 | **HIGH** | 97.0% | **306 units** | \$918.00 |
| `FOODS_3_099` | CA_1 | 75 | 103.9 | 27.9 | 131.8 | **HIGH** | 95.6% | **474 units** | \$1,422.00 |
| `FOODS_3_739` | CA_1 | 88 | 104.7 | 17.9 | 122.6 | **HIGH** | 93.8% | **456 units** | \$1,368.00 |
| `FOODS_2_276` | CA_1 | 57 | 80.1 | 25.3 | 105.4 | **HIGH** | 93.4% | **372 units** | \$1,116.00 |

---

## 10. Operationalization: Interactive Dashboard & FastAPI Microservice

### 10.1 Multi-Page Streamlit Analytics Application (`Week 4/dashboard.py`)
To make predictive analytics accessible to inventory planners, a 4-page Streamlit dashboard was developed:
1. **Demand Overview**: Displays aggregate actual vs forecast demand, cumulative tracking signals, portfolio WAPE, and residual error distributions, with interactive category/department filters.
2. **Product / Store Forecasts**: Enables SKU search, displaying 28-day forward trajectory curves comparing Ground Truth against XGBoost, Moving Average (7d), and Naive (Lag 7), with 120 days of historical demand context.
3. **Inventory Risk & Replenishment**: Provides real-time sliders for Supplier Lead Time (3–21 days) and Service Level (80%–99.9%). Renders stock vs ROP comparative bar charts and an actionable purchase orders table with a **1-click CSV Export button**.
4. **Model Performance & Benchmarks**: Displays the complete scorecard table, WAPE bar charts, and feature importance rankings.

### 10.2 Production FastAPI REST Microservice (`Week 4/api.py`)
To integrate forecasts into enterprise ERP systems (e.g. SAP, Oracle SCM), a high-throughput REST API was implemented:
- `GET /health`: Microservice health check, version, and catalog metadata.
- `GET /products`: Enumeration of all monitored SKU identifiers and store keys.
- `GET /metrics`: Comprehensive model performance scorecard comparing ML models against baselines.
- `POST /forecast`: Generates validated multi-step demand predictions for a specified SKU, store, and horizon (1 to 28 days).
- `POST /replenishment`: Computes dynamic Safety Stock, Reorder Point, stockout probability, and suggested replenishment order quantities with strict Pydantic contract validation.

### 10.3 Automated Test Pipeline (`Week 4/test_pipeline.py`)
The pipeline includes 5 automated test suites verified under `pytest`:
- `test_metrics_calculation`: Validates MAE, RMSE, WAPE, and Bias formulas against known inputs.
- `test_safety_stock_and_reorder_point`: Validates analytical safety stock and ROP formulas.
- `test_suggested_replenishment_orders`: Validates replenishment order generation and pack-size rounding.
- `test_stockout_risk_assessment`: Tests risk categorization (`HIGH`, `MEDIUM`, `SAFE`) and normal distribution probabilities.
- `test_api_endpoints`: Verifies all FastAPI route handlers, Pydantic schemas, and status codes.

---

## 11. Limitations, Edge Cases & Responsible-Use Guidelines

### 11.1 Methodological Limitations
1. **Deterministic Lead Time Assumption**:
   The primary model assumes a constant 7-day supplier lead time ($L = 7, \sigma_L = 0$). In global supply chains facing port congestion or carrier delays, lead time variance ($\sigma_L$) can be substantial. When supplier delivery tracking is available, the safety stock formula should be extended to:
   $$\sigma_{LT} = \sqrt{L\sigma_d^2 + \bar{d}^2\sigma_L^2}$$
2. **Inventory Telemetry Accuracy (Phantom Inventory)**:
   The replenishment logic assumes that recorded on-hand inventory matches physical shelf stock. In retail practice, shoplifting, cashier scan errors, and undamaged returns create *phantom inventory* (system records stock, but shelves are empty). Routine physical cycle counting remains essential.
3. **Unmodeled Macroeconomic Disruptions**:
   Extreme black swan disruptions (e.g., pandemic supply shocks, natural disasters, sudden competitor store closures) fall outside historical training distributions.

### 11.2 Human-in-the-Loop Governance
This system is designed as an **augmented decision-support platform**, not an autonomous purchasing autopilot:
- **Planner Override Authority**: Recommended order quantities ($Q$) should be reviewed by category managers prior to ERP purchase order release.
- **Supplier Minimum Order Values (MOV)**: In cases where suggested order quantities do not satisfy supplier minimum invoice dollar thresholds, human planners must aggregate multiple SKUs into consolidated supplier purchase orders.
- **Warehouse Cubic Capacity Limits**: Category planners must ensure total incoming replenishment volume does not exceed local distribution center receiving dock and pallet racking capacities.

---

## 12. Strategic Recommendations, Implementation Roadmap & Conclusion

### 12.1 Strategic Implementation Roadmap

```
PHASE 1: Pilot Deployment (Months 1–2)
├── Deploy Streamlit Dashboard in parallel with legacy planning heuristics.
├── Monitor WAPE and Forecast Bias across High-Velocity Foods category.
└── Validate inventory recommendations with weekly store cycle counts.

PHASE 2: Automated ERP Integration (Months 3–4)
├── Connect FastAPI endpoints directly to SAP/Oracle ERP staging tables.
├── Implement automated lead time variance tracking (supplier scorecarding).
└── Automate weekly case pack purchase order generation with planner sign-off.

PHASE 3: Full Multi-Store Scaling (Months 5–6)
├── Scale pipeline across all regional retail stores and product categories.
├── Implement automated monthly model retraining and concept drift detection.
└── Establish closed-loop KPI tracking: measure working capital reduction and stockout reduction.
```

### 12.2 Conclusion
This capstone project delivers an end-to-end, production-ready solution to one of retail's most critical financial and operational dilemmas. By enforcing strict temporal causality, identifying latent demand during stockouts, establishing non-leaking feature representations, and tuning gradient-boosted ensembles, the system outperforms classical retail heuristics by **4.11 percentage points of WAPE** and **12.7% of RMSE**. When linked directly to statistical inventory control policies, the platform reduces stockout risk while preventing excess capital lockup.

Backed by reproducible modular code, high-resolution diagnostic visualizations, automated unit test coverage, an interactive Streamlit analytics dashboard, and a high-performance FastAPI microservice, this repository establishes a solid foundation for enterprise deployment in modern retail supply chains.
