# Chronological Validation and Evaluation Split Plan

## 1. Split Strategy and Rationale
Demand forecasting models must strictly respect temporal causality. Random cross-validation or randomly shuffling rows induces severe forward-looking leakage, producing artificially optimistic performance that degrades in production.

This project implements a **strict chronological rolling window partition**:
- **Forecast Horizon**: 28 days (4 full retail weeks, matching standard supply-chain replenishment cycles).
- **Leakage Prevention**: All lag and rolling features are anchored at or before forecast origin $t_0$.
- **Model Tuning**: Hyperparameter tuning and model architecture selection occur exclusively on the **Validation** window.
- **Unbiased Assessment**: The **Test** window remains completely held out until final benchmarking.

---

## 2. Partition Dates and Record Counts

| Partition | Start Date | End Date | Duration | Record Count | Purpose |
|---|---|---|---|---|---|
| **Training** | `2011-03-12` | `2016-02-28` | Historical Baseline | 90,750 | Fitting model parameters and feature representations |
| **Validation** | `2016-02-29` | `2016-03-27` | 28 days | 1,400 | Model selection, hyperparameter tuning, baseline comparison |
| **Testing** | `2016-03-28` | `2016-04-24` | 28 days | 1,400 | Final unbiased out-of-sample operational evaluation |

---

## 3. Data Leakage Audit Checklist
- [x] **Temporal Ordering**: Training dates precede validation dates; validation dates precede test dates.
- [x] **No Target Leakage in Features**: Lags are strictly $\ge 7$ days (or $\ge 28$ days for direct multi-step).
- [x] **Independent Pricing Statistics**: Price statistics are computed on past observations only.
- [x] **No Future Information**: Calendar and event features use only pre-scheduled date metadata.
