# Vendor Price Mark Regime and Vendor Selection Analysis

This folder contains reusable scaffolding for comparing three CMO price vendors
without an external truth price.

## Expected Inputs

Create two CSV files, or edit `config/regime_config.json` to point to your files.

### Price marks, long format

Required columns:

```text
date,cusip,vendor,price
```

Recommended optional columns:

```text
bucket,weight
```

Example:

```text
2020-03-16,123456AB1,VendorA,91.25,CM1Q,1000000
2020-03-16,123456AB1,VendorB,89.75,CM1Q,1000000
2020-03-16,123456AB1,VendorC,91.00,CM1Q,1000000
```

### Market data

Required columns:

```text
date,current_coupon,ten_year_rate
```

Optional but recommended:

```text
vix
```

### Vendor anomaly flags

Optional input for anomaly-based internal regimes:

```text
date,cusip,vendor,anomaly_flag
```

Recommended optional columns:

```text
bucket,anomaly_score
```

## Scripts

Run in this order:

```bash
python3 src/01_define_regimes.py --config config/regime_config.json
python3 src/04_define_anomaly_regimes.py --config config/regime_config.json
python3 src/02_nonparametric_tests.py --config config/regime_config.json
Rscript src/03_state_space_astsa.R config/regime_config.json
```

## Outputs

Regime/EDA outputs:

```text
output/tables/regimes_by_date.csv
output/tables/regime_feature_summary.csv
output/tables/anomaly_regime_by_date.csv
output/tables/anomaly_regime_by_bucket_date.csv
output/figures/regime_dashboard.html
output/figures/anomaly_regime_dashboard.html
```

Nonparametric vendor comparison outputs:

```text
output/tables/nonparam_vendor_scorecard.csv
output/tables/nonparam_pairwise_tests.csv
output/tables/nonparam_friedman_tests.csv
output/figures/nonparam_abs_deviation_boxplot.html
```

State-space outputs:

```text
output/tables/state_space_parameter_estimates.csv
output/tables/state_space_vendor_residual_scorecard.csv
output/tables/state_space_filtered_latent_price.csv
output/tables/state_space_innovations.csv
```

## Modeling Notes

The regime script creates three regime families:

1. Manual event windows, such as COVID and 2022 rate shock.
2. Market regimes from VIX and rate/current-coupon moves.
3. Internal pricing regimes from consensus price volatility and cross-vendor dispersion.
4. Optional anomaly-based internal regimes from vendor IQR/EWA flags.

For vendor selection, the most defensible evidence is a consistent ranking across
these regime families:

```text
low median absolute deviation from peer consensus
low outlier frequency
low stale frequency
low state-space residual variance
small regime bias
low residual autocorrelation
```

Because there is no external truth price, conclusions should be framed as:

```text
Vendor X is most consistent with the latent consensus price process and market
drivers, especially during stressed regimes.
```

Avoid saying:

```text
Vendor X is the true price.
```
