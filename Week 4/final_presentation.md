# Demand Forecasting & Inventory Optimization Using Machine Learning
## Final Project Presentation Deck

---

### Slide 1: Executive Title Slide
**Title**: Demand Forecasting & Inventory Optimization Using Machine Learning  
**Subtitle**: Data Science & Machine Learning Internship Final Project Presentation  
**Domain**: Retail Supply Chain & Inventory Operations  
**Presenter**: Data Science & Machine Learning Engineering Intern  

---

### Slide 2: The Business Challenge — The Inventory Balancing Act
- **The Core Problem**: Retailers must constantly balance two opposing costs:
  - **Stockout Cost**: Lost immediate revenue, emergency supplier freight charges, and permanent customer defection.
  - **Holding Cost**: Tied-up working capital, warehouse carrying costs (15%–25% annually), spoilage, and forced price markdowns.
- **Why Simple Heuristics Fail**:
  - Trailing moving averages lag behind demand surges during holidays and promotional events.
  - Naive forecasts do not account for price elasticity or cross-calendar event dynamics.
- **The Core Objective**: Build an automated end-to-end forecasting and replenishment pipeline that predicts daily SKU demand, respects temporal causality, and produces actionable purchase orders.

---

### Slide 3: Technical Architecture & Leakage Prevention
```
[Raw POS & Calendar Data] 
       │
       ▼
[Data Quality Audit & Stockout Heuristics]
       │
       ▼
[Chronological Splitting: Train (Historic) | Val (28d) | Test (28d)]
       │
       ▼
[Leakage-Free Feature Engineering (Lags ≥ 7d, Anchored Rolling Stats)]
       │
       ▼
[Ensemble ML Models (Random Forest, XGBoost, LightGBM) vs Baselines]
       │
       ▼
[Statistical Inventory Replenishment Engine (SS, ROP, Order Quantities)]
       │
       ├──► [Interactive Streamlit Decision-Support Dashboard]
       └──► [Production-Ready FastAPI REST Service]
```
- **Golden Rule of Temporal Splitting**: Shuffling time series data introduces catastrophic forward-looking leakage. We enforce strict chronological boundaries:
  - **Train**: Day 1 to Day 1857
  - **Validation**: Day 1858 to Day 1885 (Model selection & hyperparameter tuning)
  - **Test**: Day 1886 to Day 1913 (Unseen operational benchmark)

---

### Slide 4: Empirical Forecasting Results — ML vs Baselines
*Evaluated on the 28-day out-of-sample chronological test partition:*

| Model Architecture | MAE (Units) | RMSE | WAPE (%) | Forecast Bias (%) | Verdict |
|---|---|---|---|---|---|
| Naive (Lag 7) | 6.969 | 11.117 | 47.22% | +2.22% | Heuristic Baseline |
| Seasonal Naive (Lag 28) | 7.023 | 11.642 | 47.58% | -3.07% | Heuristic Baseline |
| Moving Average (7-Day) | 6.658 | 10.171 | 45.11% | +3.16% | Heuristic Baseline |
| Moving Average (28-Day)| 6.219 | 9.634 | 42.14% | +0.09% | Best Baseline |
| Random Forest | 5.717 | 8.489 | 38.74% | +3.29% | ML Ensemble |
| LightGBM | 5.632 | 8.447 | 38.16% | +3.86% | ML Ensemble |
| **XGBoost (Winner)** | **5.612** | **8.410** | **38.03%** | **+3.03%** | **Production Model** |

**Key Metric Insight**: In intermittent retail sales with zeros, MAPE blows up to 60%–85%. **WAPE** ($\frac{\sum |y - \hat{y}|}{\sum y}$) provides volume-weighted stability and clear managerial interpretability.

---

### Slide 5: Statistical Inventory Replenishment Engine
Converting point forecasts into risk-mitigated supply chain policies:

1. **Lead Time Demand ($D_L$)**:
   $$D_L = \mu_d \times L \quad (\text{Default } L = 7 \text{ days})$$
2. **Safety Stock ($SS$)**:
   $$SS = Z_{\alpha} \times \sigma_d \times \sqrt{L} \quad (Z_{0.95} = 1.645 \text{ for } 95\% \text{ service level})$$
3. **Reorder Point ($ROP$)**:
   $$ROP = D_L + SS$$
4. **Suggested Order Quantity ($Q$)**:
   $$Q = \max(0, ROP - I_{\text{current}} + \hat{D}_{\text{horizon}})$$
   *Subject to wholesale case pack multiples (e.g. 6 or 12 units) and supplier Minimum Order Quantities (MOQ).*

---

### Slide 6: Operational Prototypes — Dashboard & API
- **Interactive Multi-Page Streamlit Dashboard**:
  - **Demand Overview**: Aggregate sales vs forecast, WAPE, and cumulative volume.
  - **Product / Store Forecasts**: Granular SKU trajectories vs baselines, historical patterns.
  - **Inventory Risk**: Real-time stock vs ROP, dynamic lead-time & service-level sliders ($85\% - 99\%$).
  - **Model Performance**: Full scorecard, residual error histograms, and baseline comparisons.
- **Production FastAPI REST Service**:
  - `/health`: Service health and model status.
  - `/products`: Catalog of monitored products and stores.
  - `/forecast`: Generates multi-day daily demand predictions for requested SKU and horizon.
  - `/replenishment`: Returns dynamic lead time demand, safety stock, ROP, stockout probability, and suggested order quantity.

---

### Slide 7: Responsible Use & Human Interpretation Guidelines
- **Decision Support, Not Autopilot**: Model outputs represent statistical likelihoods, not certainties. Supply chain managers must evaluate recommendations against supplier capacity, warehouse limits, and unexpected external shocks.
- **Distinguishing Stockouts from True Zero Demand**: Historical zero sales during active product windows reflect stockouts rather than absent consumer appetite. Our pipeline accounts for this distinction to prevent downward replenishment spirals.
- **Supplier Lead Time Uncertainty**: In production, lead time variance should be tracked so safety stock accommodates both demand volatility ($\sigma_d$) and transit delays ($\sigma_L$).

---

### Slide 8: Summary of Achievements
- [x] **Reproducible 4-Week Lifecycle**: Fully organized deliverables across Week 1, Week 2, Week 3, and Week 4.
- [x] **Leakage-Free Modeling**: Chronological validation with non-leaking features.
- [x] **State-of-the-Art Accuracy**: XGBoost outperforming all heuristics (38.03% WAPE).
- [x] **Mathematical Replenishment**: Dynamic safety stock and reorder point optimization.
- [x] **Operational Software**: Complete Streamlit dashboard, FastAPI service, and passing pytest test suite.
