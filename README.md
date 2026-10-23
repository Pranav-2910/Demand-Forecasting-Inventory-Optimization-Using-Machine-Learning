# Demand Forecasting & Inventory Optimization Using Machine Learning
### Data Science & Machine Learning Internship — Phase 1: Weeks 1 & 2 Implementation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Phase%201-Completed-success.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Walmart%20M5-orange.svg)](https://www.kaggle.com/c/m5-forecasting-accuracy)

---

## 1. Project Overview & Business Context

Retail and supply-chain organizations face the perpetual challenge of balancing product availability against the financial burden of holding excess stock. Inaccurate demand estimates can trigger stockouts, missed sales, expensive emergency replenishment, or costly excess inventory and product markdowns.

This repository represents **Phase 1 (Weeks 1 & 2)** of an end-to-end predictive analytics workflow that:
1. **Audits and Ingests Multi-Source Retail Data** (daily unit sales, weekly shelf prices, and calendar/promotional events) spanning 1,913 days.
2. **Distinguishes True Zero Demand from Stockout Periods** to prevent negative forecast bias.
3. **Engineers Non-Leaking Predictive Features** (lags strictly $\ge 7$ days, rolling window statistics anchored prior to forecast horizons, calendar attributes, and promotional markdown indicators).
4. **Enforces Strict Chronological Splitting** (`Train` $\rightarrow$ `Validation` $\rightarrow$ `Test`) to guarantee zero forward-looking data leakage.
5. **Establishes Rigorous Baseline Forecast Benchmarks** (Naive Lag 7, Seasonal Naive Lag 28, 7-Day & 28-Day Moving Averages) evaluated on MAE, RMSE, WAPE, and Forecast Bias.

---

## 2. Phase 1 Repository Structure

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
├── requirements.txt                         # Pinned project dependencies
├── .gitignore                               # Standardized Git ignore rules
└── README.md                                # Phase 1 project overview and execution guide
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

### 4.4 Baseline Takeaways for Modeling Phase
1. **Benchmark Target**: The 28-day moving average achieves the lowest baseline test error at **42.14% WAPE** and **6.219 unit MAE**. Machine learning models (e.g., XGBoost, LightGBM) must outperform this benchmark.
2. **Metric Stability**: Due to intermittent zero sales, MAPE exhibits high variance (66%–85%). **WAPE** serves as the primary evaluation metric because it weights errors by sales volume.
3. **Feature Correlation**: Rolling averages and 7-day lags have the highest linear correlation with sales demand ($r \approx 0.55 - 0.65$).

---

## 5. Installation & Execution Guide

### 5.1 Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Pranav-2910/Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning.git
cd Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Reproducing Phase 1 Pipelines

**1. Run Feature Engineering Pipeline**:
```powershell
python "Week 2/feature_engineering.py"
```
*Outputs: `Week 2/features_engineered.csv` and `Week 2/chronological_split_plan.md`.*

**2. Run Baseline Forecasts & Benchmarking**:
```powershell
python "Week 2/baseline_forecasts.py"
```
*Outputs: `Week 2/baseline_evaluation_table.csv` and `Week 2/baseline_evaluation_table.md`.*

**3. Inspect Interactive Notebooks**:
* Open [Week 1/01_exploratory_data_analysis.ipynb](file:///c:/Users/prana/Desktop/P1/Week%201/01_exploratory_data_analysis.ipynb) to view full exploratory charts.
* Open [Week 2/02_feature_engineering_and_baselines.ipynb](file:///c:/Users/prana/Desktop/P1/Week%202/02_feature_engineering_and_baselines.ipynb) to inspect feature distributions and baseline forecast plots.

---

## 6. Dataset Provenance & Full Raw Data Access

The cleaned interim dataset (`Week 1/sales_interim.csv`, 14.2 MB) and the engineered features dataset (`Week 2/features_engineered.csv`, 33.8 MB) are included directly in the repository for immediate execution.

To access or rebuild the full multi-gigabyte raw dataset:
1. Download the source files (`calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv`) from the official [Kaggle M5 Forecasting Accuracy Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data).
2. The raw files exceed GitHub's 100 MB per-file limit and are managed locally via `.gitignore`.
