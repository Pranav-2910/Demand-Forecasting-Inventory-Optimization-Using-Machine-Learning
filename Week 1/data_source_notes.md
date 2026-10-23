# Data Source Notes & Methodological Assumptions

## 1. Provenance and Scope
The underlying data reflects daily unit sales, temporal calendars, and weekly shelf prices across Walmart retail stores in the United States, spanning January 29, 2011 to April 24, 2016 (1,913 days).
- **Sales Data**: Point-of-sale transactions aggregated to daily unit sales per product-store pair.
- **Calendar Data**: Contains calendar date attributes, national, cultural, religious, and sporting events, and state-level SNAP (Supplemental Nutrition Assistance Program) distribution schedules for California, Texas, and Wisconsin.
- **Price Data**: Weekly selling prices per store-item combination.

---

## 2. Unconstrained Demand vs. Observed Sales During Stockouts

> [!IMPORTANT]
> A foundational principle in retail forecasting is that **observed point-of-sale transactions do not necessarily equal true consumer demand**.

When an item goes out of stock:
1. Observed sales register as exactly zero ($S_t = 0$).
2. Actual unconstrained demand ($D_t$) may be significantly greater than zero ($D_t > 0$).
3. Naive machine learning models trained directly on raw sales will learn that demand dropped to zero, falsely underforecasting future demand and perpetuating persistent stockout cycles.

### Our Analytical Protocol:
- **Pre-Launch Distinction**: Many items were introduced to stores after January 2011. Zero sales recorded prior to an item's first observed sale or pricing record represent pre-launch periods rather than zero demand or stockouts. These rows are explicitly flagged with `is_active = False` to prevent negative demand bias.
- **Stockout Identification**: For active products (`is_active = True`), instances where `sales == 0` while a valid `sell_price` exists and historical demand was consistently positive are flagged as `potential_stockout`.
- **Decision Guidance**: In the inventory decision module, demand variance ($\sigma_d$) is calculated using active periods, preventing artificial deflation of safety stock buffers caused by unrecorded stockout periods.

---

## 3. Assumptions and Edge Case Handling

1. **Returns and Negative Sales**:
   - The dataset records net positive units sold. Returns that exceed daily sales are capped at zero; no negative values exist in the source dataset.
2. **Missing Dates**:
   - The calendar table provides complete, contiguous date coverage from `2011-01-29` to `2016-06-19`. Christmas Day (`12-25`) stores are closed every year, registering genuine zero sales.
3. **Price Availability & Discounts**:
   - Prices change on weekly boundaries (`wm_yr_wk`). Price discounts relative to historical rolling maximums reflect promotional markdowns that stimulate demand elasticity.
4. **Lead Time & Stock Level Assumptions**:
   - In the absence of supplier telemetry, lead times are modeled under documented baseline assumptions ($L = 7$ days by default, configurable between 3 and 14 days). Current stock is simulated dynamically to test replenishment thresholds without fabricating ungrounded telemetry.
