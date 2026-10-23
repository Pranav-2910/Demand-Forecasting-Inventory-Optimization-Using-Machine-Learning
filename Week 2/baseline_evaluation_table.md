# Baseline Forecasting Models Evaluation Scorecard

## Summary of Heuristic Baselines
To establish credible benchmarks for machine learning models, four standard heuristic baselines were evaluated on identical chronological validation and test partitions.

| Model Partition | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) |
|---|---|---|---|---|---|
| `Naive (Lag 7) [Validation]` | 6.664 | 10.696 | 46.58% | 73.55% | -2.97% |
| `Seasonal Naive (Lag 28) [Validation]` | 7.564 | 12.312 | 52.88% | 85.17% | 2.02% |
| `Moving Average (7-Day) [Validation]` | 6.4 | 10.401 | 44.73% | 68.45% | -3.54% |
| `Moving Average (28-Day) [Validation]` | 6.402 | 10.247 | 44.75% | 67.65% | -0.6% |
| `Naive (Lag 7) [Test]` | 6.969 | 11.117 | 47.22% | 74.62% | 2.22% |
| `Seasonal Naive (Lag 28) [Test]` | 7.023 | 11.642 | 47.58% | 67.47% | -3.07% |
| `Moving Average (7-Day) [Test]` | 6.658 | 10.171 | 45.11% | 72.24% | 3.16% |
| `Moving Average (28-Day) [Test]` | 6.219 | 9.634 | 42.14% | 66.48% | 0.09% |

## Key Baseline Insights
- **Seasonal Naive (Lag 28)** captures monthly cyclic demand but has higher latency.
- **Naive (Lag 7)** effectively captures strong weekly day-of-week seasonality.
- **Moving Averages** smooth noise and reduce variance, achieving lowest baseline error.