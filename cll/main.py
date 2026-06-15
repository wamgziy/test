from vendor_comparison import build_scorecard, label_regimes_from_index

# Optional: build a regime label from MOVE or VIX
regimes = label_regimes_from_index(move_series, quantile=0.90, min_run=5)

scorecard = build_scorecard(vendor_px, trades=trace_prints, regimes=regimes)
scorecard.to_csv("vendor_scorecard.csv")


# anomaly_flags: your existing DataFrame[date, cusip, vendor, anomaly_flag]
pre = label_regimes_predefined(vendor_px["date"].unique())
dd  = label_regimes_from_anomalies(anomaly_flags,
                                   vendor_quantile=0.95,
                                   min_vendors=2,
                                   min_run=3)
regimes = combine_regimes(pre, dd, mode="predefined_plus_unnamed")

# Diagnostic: are the two sources confirming each other?
print(regimes.groupby(["regime", "source"]).size().unstack(fill_value=0))

# Feed into scorecard
sc = build_scorecard(vendor_px,
                     trades=trace_prints,
                     regimes=regimes[["date", "regime"]])
