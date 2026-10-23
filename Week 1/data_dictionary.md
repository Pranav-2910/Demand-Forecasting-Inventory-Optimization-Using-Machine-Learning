# Data Dictionary: Retail Sales, Calendar & Pricing

This document defines the schema, data types, business definitions, value ranges, and sources for all datasets utilized in the Demand Forecasting and Inventory Optimization system.

---

## 1. Daily Sales Table (`sales_train_validation.csv` / Interim Dataset)

| Field Name | Data Type | Key Type | Nullable | Description / Business Meaning | Allowed / Sample Values | Source Table |
|---|---|---|---|---|---|---|
| `id` | String | Surrogate Key | No | Unique series identifier combining product ID, store ID, and validation split | `HOBBIES_1_001_CA_1_validation` | `sales_train_validation` |
| `item_id` | String | Dimension Key | No | Unique product SKU identifier | `HOBBIES_1_001`, `FOODS_3_090` | `sales_train_validation` |
| `dept_id` | String | Attribute | No | Department department grouping | `HOBBIES_1`, `FOODS_1`, `HOUSEHOLD_2` | `sales_train_validation` |
| `cat_id` | String | Attribute | No | High-level product category | `HOBBIES`, `FOODS`, `HOUSEHOLD` | `sales_train_validation` |
| `store_id` | String | Dimension Key | No | Retail store location identifier | `CA_1`, `CA_2`, `TX_1`, `WI_1` | `sales_train_validation` |
| `state_id` | String | Attribute | No | US State location of the store | `CA`, `TX`, `WI` | `sales_train_validation` |
| `d` | String | Attribute | No | Relative day sequence label from day 1 to day 1913 | `d_1`, `d_2`, ..., `d_1913` | `sales_train_validation` |
| `sales` | Integer | Metric | No | Units sold on the specified date | $\ge 0$ (e.g. 0, 1, 15, 42) | `sales_train_validation` |
| `date` | Date | Dimension Key | No | Calendar date of observed sales (YYYY-MM-DD) | `2011-01-29` to `2016-04-24` | Joined via `calendar` |
| `is_active` | Boolean | Derived Flag | No | Flag indicating if product was actively listed (post-launch) | `True`, `False` | Derived |
| `is_zero_sales` | Boolean | Derived Flag | No | Flag indicating zero units sold on date | `True`, `False` | Derived |
| `potential_stockout`| Boolean | Derived Flag | No | Flag indicating zero sales during active period with valid price | `True`, `False` | Derived |

---

## 2. Calendar and Temporal Events Table (`calendar.csv`)

| Field Name | Data Type | Key Type | Nullable | Description / Business Meaning | Sample Values | Source Table |
|---|---|---|---|---|---|---|
| `date` | Date | Primary Key | No | Calendar date in format YYYY-MM-DD | `2011-01-29` | `calendar` |
| `wm_yr_wk` | Integer | Dimension Key | No | Walmart internal year-week identifier | `11101` (Year 2011, Week 1) | `calendar` |
| `weekday` | String | Attribute | No | Name of the day of the week | `Monday`, `Tuesday`, ..., `Sunday` | `calendar` |
| `wday` | Integer | Attribute | No | Integer encoding of weekday (Saturday=1 ... Friday=7) | `1` to `7` | `calendar` |
| `month` | Integer | Attribute | No | Calendar month index | `1` to `12` | `calendar` |
| `year` | Integer | Attribute | No | Calendar year | `2011` to `2016` | `calendar` |
| `d` | String | Dimension Key | No | Relative day sequence identifier | `d_1` to `d_1969` | `calendar` |
| `event_name_1` | String | Attribute | Yes | Name of holiday or promotional event occurring on date | `SuperBowl`, `Thanksgiving`, `LaborDay` | `calendar` |
| `event_type_1` | String | Attribute | Yes | Category classification of event 1 | `Sporting`, `Cultural`, `National`, `Religious` | `calendar` |
| `event_name_2` | String | Attribute | Yes | Name of secondary event on date (if overlapping) | `Easter`, `Cinco De Mayo` | `calendar` |
| `event_type_2` | String | Attribute | Yes | Category classification of event 2 | `Cultural`, `Religious` | `calendar` |
| `snap_CA` | Integer | Flag | No | Supplemental Nutrition Assistance Program benefit day in CA | `0` or `1` | `calendar` |
| `snap_TX` | Integer | Flag | No | Supplemental Nutrition Assistance Program benefit day in TX | `0` or `1` | `calendar` |
| `snap_WI` | Integer | Flag | No | Supplemental Nutrition Assistance Program benefit day in WI | `0` or `1` | `calendar` |

---

## 3. Selling Prices Table (`sell_prices.csv`)

| Field Name | Data Type | Key Type | Nullable | Description / Business Meaning | Sample Values | Source Table |
|---|---|---|---|---|---|---|
| `store_id` | String | Composite Key | No | Store location identifier | `CA_1` | `sell_prices` |
| `item_id` | String | Composite Key | No | Product SKU identifier | `HOBBIES_1_001` | `sell_prices` |
| `wm_yr_wk` | Integer | Composite Key | No | Walmart internal year-week identifier | `11325` | `sell_prices` |
| `sell_price` | Float | Metric | No | Selling price of the item at the store in US Dollars | `$0.42` to `$29.98` | `sell_prices` |

---

## 4. Derived Inventory Decision Support Fields

| Field Name | Data Type | Unit | Description | Mathematical Formulation |
|---|---|---|---|---|
| `lead_time_days` | Integer | Days | Replenishment lead time from supplier order to shelf availability | Assumed $L \in [3, 14]$ days (documented default: 7 days) |
| `service_level` | Float | % | Target cycle service level (probability of no stockout) | $\alpha \in [0.90, 0.99]$ (default: 0.95, $Z_{0.95} = 1.645$) |
| `safety_stock` | Float | Units | Buffer stock to absorb demand volatility over replenishment lead time | $SS = Z_{\alpha} \cdot \sigma_d \cdot \sqrt{L}$ |
| `lead_time_demand` | Float | Units | Expected total sales occurring during lead time | $D_L = \mu_d \cdot L$ |
| `reorder_point` | Float | Units | Inventory threshold triggering replenishment | $ROP = D_L + SS$ |
| `current_stock` | Integer | Units | On-hand physical inventory available for sale | Tracked inventory level |
| `suggested_order` | Integer | Units | Recommended replenishment order quantity | $Q = \max(0, \lceil ROP - I_{\text{current}} + \hat{D}_{\text{horizon}} \rceil)$ |
