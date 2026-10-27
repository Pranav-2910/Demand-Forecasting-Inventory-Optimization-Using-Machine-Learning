# Demand Forecasting & Inventory Optimization Using Machine Learning
### Data Science & Machine Learning Internship — Phase 2: Weeks 1, 2 & 3 Implementation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Weeks%201--3-Completed-success.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Walmart%20M5-orange.svg)](https://www.kaggle.com/c/m5-forecasting-accuracy)
[![Winning Model](https://img.shields.io/badge/Best%20Model-XGBoost%20(38.0%25%20WAPE)-brightgreen.svg)](#)

---

## 1. Project Overview & Business Context

Retailers and supply-chain organizations face the perpetual challenge of balancing high on-shelf product availability against the substantial financial carrying costs of holding excess inventory. Inaccurate demand forecasting leads directly to stockouts, lost sales revenue, expedited replenishment freight, or costly inventory markdowns and spoilage.

This repository covers **Weeks 1, 2, and 3** of an end-to-end predictive analytics and inventory decision-support system:
1. **Audits and Ingests Multi-Source Retail Telemetry** (daily unit sales, weekly shelf prices, and calendar/promotional events) spanning 1,913 days across retail departments.
2. **Distinguishes True Zero Demand from Stockout Periods** to prevent negative demand bias in downstream models.
3. **Engineers Non-Leaking Predictive Features** (lags strictly $\ge 7$ days, rolling window statistics anchored prior to forecast horizons, calendar attributes, and promotional markdown indicators).
4. **Enforces Strict Chronological Splitting** (`Train` $\rightarrow$ `Validation` $\rightarrow$ `Test`) to guarantee zero forward-looking data leakage.
5. **Establishes Rigorous Baseline Forecast Benchmarks** (Naive Lag 7, Seasonal Naive Lag 28, 7-Day & 28-Day Moving Averages) evaluated on MAE, RMSE, WAPE, and Forecast Bias.
6. **Trains, Tunes & Evaluates Supervised Machine Learning Models** (Random Forest, LightGBM, and XGBoost) on a strictly out-of-sample 28-day horizon, demonstrating statistically significant gains over baselines.
7. **Translates Demand Forecasts into Inventory Replenishment Decisions**, calculating dynamic Safety Stock ($SS$), Reorder Points ($ROP$), Stockout Risk probabilities, and actionable order quantities adhering to case-pack constraints under a documented 95% service level.

---

## 2. Repository Structure

```
P1/
├── Week 1/                                   # Week 1 Deliverables: Data Ingestion & Quality Audit
│   ├── data_dictionary.md                   # Full schema definitions, types, sources, and nullability
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
│   ├── model_training.py                    # Standalone model training, cross-validation & evaluation script
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
├── requirements.txt                         # Pinned project dependencies
├── .gitignore                               # Standardized Git ignore rules
└── README.md                                # Project overview, methodology, and execution guide
```

---

## 3. Week 1: Data Collection, Quality Checks & Exploratory Analysis

### 3.1 Data Provenance & Scope
The data is derived from point-of-sale transactions across Walmart retail stores (Kaggle M5 Forecasting benchmark):
- **Sales History**: 1,913 days of daily unit sales across merchandise departments (`HOBBIES`, `FOODS`, `HOUSEHOLD`).
- **Calendar Metadata**: Date attributes, national/cultural/sporting events, and state-level SNAP (Supplemental Nutrition Assistance Program) distribution schedules.
- **Selling Prices**: Weekly store-level shelf prices per SKU.

### 3.2 Data Quality Audit Findings
A comprehensive automated audit was conducted on the longitudinal sales dataset:
- **Date Continuity**: Verified **1,913 contiguous calendar days** (`2011-01-29` to `2016-04-24`) with zero missing days.
- **Key Uniqueness**: **0 duplicate records** on composite key `(store_id, item_id, date)`.
- **Value Validity**: **0 negative sales quantities**; zero negative or invalid prices.
- **Intermittency & Zero Sales**: 17.14% of overall records contained zero sales.

### 3.3 Stockout Identification vs. Unconstrained Demand
> [!IMPORTANT]
> Observed sales during stockouts register as $S_t = 0$, but true unconstrained consumer demand is $D_t > 0$. Naive models trained on raw sales will interpret out-of-stock periods as lack of consumer interest, creating a self-reinforcing downward spiral of inventory starvation.

**Methodological Treatment**:
- **Pre-Launch Period Separation**: Rows before an item's first observed sale or pricing record are flagged as `is_active = False` to prevent negative demand bias.
- **Potential Stockout Flagging**: Active records (`is_active = True`) with zero sales despite valid pricing and positive historical demand velocity are flagged as `potential_stockout` (13.38% of active records).
- **Variance Protection**: Downstream demand variance ($\sigma_d$) calculations use active windows, preventing artificial deflation of safety stock buffers.

### 3.4 Key Exploratory Findings
1. **Strong Weekend Seasonality**: Friday through Sunday account for the majority of weekly sales volume. Day-of-week lag features ($t-7, t-14$) provide strong predictive signals.
2. **Promotional Event Lifts**: National holidays and sporting events (e.g., Super Bowl) stimulate substantial demand spikes, while Christmas Day exhibits absolute store closures (true zeros).
3. **SNAP Food Stamp Impact**: Sales on SNAP distribution days in California exhibit a statistically significant lift over non-SNAP days.

---

## 4. Week 2: Feature Engineering & Baseline Forecasts

### 4.1 Chronological Partitioning Plan (Zero Leakage)
To ensure that models are evaluated under realistic forward-looking operational conditions, a **strict chronological rolling window partition** was established:

| Partition | Start Date | End Date | Duration | Record Count | Operational Purpose |
|---|---|---|---|---|---|
| **Training** | `2011-01-29` | `2016-02-28` | ~5 years | 90,850 rows | Fitting model parameters and feature representations |
| **Validation** | `2016-02-29` | `2016-03-27` | 28 days | 1,400 rows | Hyperparameter tuning and model selection |
| **Testing** | `2016-03-28` | `2016-04-24` | 28 days | 1,400 rows | Final unbiased out-of-sample operational evaluation |

*Forecast Horizon: 28 days (4 retail planning weeks).*

### 4.2 Non-Leaking Feature Registry
All features use strictly past information available at the forecast origin:
- **Past-Only Lag Features**: `sales_lag_7`, `sales_lag_14`, `sales_lag_21`, `sales_lag_28`, `sales_lag_35`, `sales_lag_42`. All lags are strictly $\ge 7$ days prior to the target date.
- **Anchored Rolling Statistics**: Rolling mean, standard deviation, and maximum over 7, 14, and 28 days, anchored strictly at `sales_lag_7` to eliminate contemporaneous leakage.
- **Calendar & Event Features**: Day of week, day of month, month, year, weekend indicator, holiday event indicator, event category codes, and SNAP benefit distribution flags.
- **Pricing & Promotions**: Selling price, price deviation from SKU historical mean, discount ratio relative to historical maximum, and promotional markdown flags.

### 4.3 Baseline Forecasting Benchmarks

Four standard heuristic baselines were evaluated on both validation and testing partitions using five key metrics:
- **MAE** (Mean Absolute Error in units)
- **RMSE** (Root Mean Squared Error)
- **WAPE** (Weighted Absolute Percentage Error = $\frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$, robust to intermittent zero sales)
- **Non-Zero MAPE** (Mean Absolute Percentage Error calculated exclusively where actual demand $y > 0$)
- **Forecast Bias** ($\frac{\sum (\hat{y} - y)}{\sum y} \times 100\%$, indicating over/under-prediction)

#### Baseline Evaluation Scorecard

| Model Name & Partition | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) | Benchmark Role |
|---|---|---|---|---|---|---|
| `Naive (Lag 7) [Validation]` | 6.664 | 10.696 | 46.58% | 73.55% | -2.97% | Day-of-week persistence |
| `Seasonal Naive (Lag 28) [Validation]` | 7.564 | 12.312 | 52.88% | 85.17% | +2.02% | Monthly cycle persistence |
| `Moving Average (7-Day) [Validation]` | 6.400 | 10.401 | 44.73% | 68.45% | -3.54% | Short-term rolling trend |
| `Moving Average (28-Day) [Validation]` | 6.402 | 10.247 | 44.75% | 67.65% | -0.60% | Medium-term rolling trend |
| `Naive (Lag 7) [Test]` | 6.969 | 11.117 | 47.22% | 74.62% | +2.22% | Out-of-sample benchmark |
| `Seasonal Naive (Lag 28) [Test]` | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% | Out-of-sample benchmark |
| `Moving Average (7-Day) [Test]` | 6.658 | 10.171 | 45.11% | 72.24% | +3.16% | Out-of-sample benchmark |
| **`Moving Average (28-Day) [Test]`** | **6.219** | **9.634** | **42.14%** | **66.48%** | **+0.09%** | **Leading Benchmark** |

---

## 5. Week 3: Model Training, Evaluation & Replenishment Logic

### 5.1 Machine Learning Modeling Methodology
During Week 3, three competitive supervised machine learning algorithms were formulated and tuned to capture complex non-linear demand interactions:
1. **Random Forest Regressor**: Non-parametric ensemble of 100 decision trees (`n_estimators=100`, `max_depth=12`, `min_samples_leaf=5`, `n_jobs=-1`).
2. **LightGBM Regressor**: Fast, leaf-wise gradient boosting trees (`n_estimators=150`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8`).
3. **XGBoost Regressor**: Level-wise gradient boosting with second-order gradient optimization (`n_estimators=150`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`).

All candidate models were trained strictly on historical data up to `2016-02-28`, tuned on the chronological validation window (`2016-02-29` to `2016-03-27`), and evaluated once on the strictly withheld 28-day test partition (`2016-03-28` to `2016-04-24`).

### 5.2 Comparative Evaluation Scorecard (ML Models vs Baselines)

The complete benchmark scorecard on the out-of-sample test window is presented below:

| Model / Baseline Name | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) | Evaluation Status |
|---|---|---|---|---|---|---|
| `Naive (Lag 7) [Baseline]` | 6.969 | 11.117 | 47.22% | 74.62% | +2.22% | Heuristic Benchmark |
| `Seasonal Naive (Lag 28) [Baseline]` | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% | Heuristic Benchmark |
| `Moving Average (7-Day) [Baseline]` | 6.658 | 10.171 | 45.11% | 72.24% | +3.16% | Heuristic Benchmark |
| `Moving Average (28-Day) [Baseline]` | 6.219 | 9.634 | 42.14% | 66.48% | +0.09% | Best Heuristic Benchmark |
| `Random Forest [ML]` | 5.717 | 8.489 | 38.74% | 63.40% | +3.29% | Evaluated Candidate |
| `LightGBM [ML]` | 5.632 | 8.447 | 38.16% | 64.33% | +3.86% | Evaluated Candidate |
| **`XGBoost [ML]`** | **5.612** | **8.410** | **38.03%** | **62.43%** | **+3.03%** | **Winning Champion Model** |

#### Key Analytical Insights
1. **Significant Baseline Outperformance**: **XGBoost** surpasses the best heuristic baseline (`Moving Average 28-Day`), reducing WAPE from **42.14% to 38.03%** (a **4.11 percentage point improvement**, representing a **9.7% relative error reduction**) and decreasing RMSE from **9.634 to 8.410** (a **12.7% reduction**).
2. **Balanced Forecast Bias**: The winning XGBoost model maintains a controlled bias of **+3.03%**. This modest positive bias is commercially desirable in retail, offering mild buffer protection against stockouts while preventing excessive capital tie-up.
3. **Intermittent Robustness**: WAPE successfully overcomes MAPE distortion caused by intermittent zero-sales records, providing an actionable volume-weighted KPI for inventory planners.

### 5.3 Feature Importance & Demand Drivers
Feature importance analysis from the winning XGBoost model reveals the primary structural drivers of consumer demand:
- **`rolling_mean_28` (31.4% importance)**: Serves as the primary medium-term volume anchor.
- **`sales_lag_7` (18.6% importance)**: Captures weekly day-of-week sales momentum.
- **`rolling_mean_7` (12.8% importance)**: Detects immediate short-term demand velocity.
- **`sell_price` & `discount_ratio` (9.5% importance)**: Captures consumer elasticity during price promotions.
- **`day_of_week` & `is_weekend` (8.2% importance)**: Accounts for substantial weekend shopping surges.

### 5.4 Inventory Replenishment Logic & Mathematical Formulation
Forecast demand outputs are directly coupled to an inventory replenishment optimization engine:

#### 1. Safety Stock ($SS$) Formulation
To protect against demand variability during replenishment lead time, Safety Stock is computed under a target Cycle Service Level ($CSL$):
$$SS = Z_{\alpha} \times \sigma_d \times \sqrt{L}$$
Where:
- $Z_{\alpha} = 1.645$ for a **95% cycle service level** ($CSL$).
- $L = 7\text{ days}$ (deterministic supplier lead time).
- $\sigma_d$: Standard deviation of daily demand calculated over active listing windows.

#### 2. Reorder Point ($ROP$) Formulation
The Reorder Point establishes the inventory trigger threshold:
$$ROP = D_L + SS = (\bar{d} \times L) + SS$$
Where $D_L = \bar{d} \times L$ represents expected lead-time demand based on daily forecast velocity.

#### 3. Stockout Risk Probability & Categorization
The probability of a stockout occurring prior to replenishment arrival is calculated using the standard normal cumulative distribution function $\Phi$:
$$P(\text{Stockout}) = P(D_L > I_{current}) = 1 - \Phi\left(\frac{I_{current} - D_L}{\sigma_L}\right)$$
- **HIGH RISK**: $I_{current} < D_L$ (Current inventory cannot cover expected lead time demand; stockout imminent).
- **MEDIUM RISK**: $D_L \le I_{current} < ROP$ (Inventory has breached reorder threshold; safety stock is eroding).
- **SAFE**: $I_{current} \ge ROP$ (Inventory remains above reorder point).

#### 4. Suggested Order Quantity ($Q$)
When inventory breaches the Reorder Point ($I_{current} \le ROP$), an replenishment order is generated:
$$Q_{raw} = \max\left(0, ROP - I_{current} + \hat{D}_{horizon}\right)$$
To respect real-world supplier packaging and logistics constraints, $Q_{raw}$ is adjusted to case pack multiples:
$$Q = \max\left(MOQ, \left\lceil \frac{Q_{raw}}{\text{Pack Size}} \right\rceil \times \text{Pack Size}\right)$$
*Parameters applied: $\text{Pack Size} = 6\text{ units}$, $\text{Minimum Order Quantity (MOQ)} = 12\text{ units}$.*

### 5.5 Actionable Replenishment Recommendations Sample

The table below illustrates sample actionable recommendations generated by `inventory_replenishment.py` (available in `Week 3/replenishment_recommendations.csv`):

| Item ID | Store | Current Stock | Lead Time Demand | Safety Stock | ROP | Stockout Risk | Suggested Order |
|---|---|---|---|---|---|---|---|
| `FOODS_3_744` | CA_1 | 44 | 83.4 | 15.6 | 99.0 | **HIGH** (100.0%) | **390 units** |
| `FOODS_3_090` | CA_1 | 182 | 351.0 | 94.7 | 445.7 | **HIGH** (99.8%) | **1668 units** |
| `FOODS_3_362` | CA_1 | 62 | 121.0 | 35.5 | 156.5 | **HIGH** (99.7%) | **582 units** |
| `FOODS_3_694` | CA_1 | 46 | 80.1 | 20.5 | 100.7 | **HIGH** (99.7%) | **378 units** |
| `FOODS_2_019` | CA_1 | 61 | 104.2 | 29.9 | 134.1 | **HIGH** (99.1%) | **492 units** |
| `FOODS_1_085` | CA_1 | 48 | 76.7 | 22.4 | 99.1 | **HIGH** (98.2%) | **360 units** |
| `FOODS_3_702` | CA_1 | 43 | 65.8 | 19.9 | 85.7 | **HIGH** (97.0%) | **306 units** |
| `FOODS_3_099` | CA_1 | 75 | 103.9 | 27.9 | 131.8 | **HIGH** (95.6%) | **474 units** |
| `FOODS_3_739` | CA_1 | 88 | 104.7 | 17.9 | 122.6 | **HIGH** (93.8%) | **456 units** |
| `FOODS_2_276` | CA_1 | 57 | 80.1 | 25.3 | 105.4 | **HIGH** (93.4%) | **372 units** |

### 5.6 Methodological Assumptions & Operational Limitations
- **Supplier Lead Time**: Modeled assuming a deterministic 7-day supplier lead time ($L = 7$, $\sigma_L = 0$). When supplier telemetry is available, the standard deviation formula expands to $\sigma_{LT} = \sqrt{L\sigma_d^2 + \bar{d}^2\sigma_L^2}$.
- **Unconstrained Demand Telemetry**: In retail POS systems, true lost sales during stockouts are unobserved. Demand variance ($\sigma_d$) is calculated strictly over active item records to minimize artificial deflation of safety stock buffers.
- **Decision Support Boundaries**: These replenishment calculations serve as decision-support recommendations. Human inventory controllers must review suggested orders against warehouse pallet space, minimum order costs, and supplier batch availability.

---

## 6. Installation & Execution Guide

### 6.1 Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Pranav-2910/Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning.git
cd Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Reproducing All Pipelines (Weeks 1, 2 & 3)

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

### 6.3 Interactive Jupyter Notebooks
- [Week 1/01_exploratory_data_analysis.ipynb](file:///c:/Users/prana/Desktop/P1/Week%201/01_exploratory_data_analysis.ipynb): Full exploratory data analysis, seasonality, intermittency, and promotional event lift.
- [Week 2/02_feature_engineering_and_baselines.ipynb](file:///c:/Users/prana/Desktop/P1/Week%202/02_feature_engineering_and_baselines.ipynb): Lag generation, correlation matrices, and heuristic baseline evaluations.
- [Week 3/03_model_training_and_replenishment.ipynb](file:///c:/Users/prana/Desktop/P1/Week%203/03_model_training_and_replenishment.ipynb): Model training, cross-validation tuning, residual diagnostic plots, and replenishment policy simulation.

---

## 7. Dataset Provenance & Full Raw Data Access

The cleaned interim dataset (`Week 1/sales_interim.csv`, 14.2 MB) and the engineered features dataset (`Week 2/features_engineered.csv`, 33.8 MB) are included directly in the repository for immediate execution.

To access or rebuild the full multi-gigabyte raw dataset:
1. Download the source files (`calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv`) from the official [Kaggle M5 Forecasting Accuracy Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data).
2. The raw files exceed GitHub's 100 MB per-file limit and are managed locally via `.gitignore`.
