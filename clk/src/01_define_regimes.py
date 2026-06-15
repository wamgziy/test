#!/usr/bin/env python3
"""Define manual and data-driven regimes for CMO vendor mark analysis.

Inputs:
  - price marks in long format: date, cusip, vendor, price, optional bucket/weight
  - market data: date, current_coupon, ten_year_rate, optional vix

Outputs:
  - regimes_by_date.csv
  - regime_feature_summary.csv
  - interactive Plotly HTML dashboard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = Path(path).resolve().parents[1]
    cfg["_base_dir"] = str(base)
    return cfg


def resolve_path(cfg: Dict, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(cfg["_base_dir"]) / p
    return p


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float(values.mean())
    return float(np.average(values[mask], weights=weights[mask]))


def safe_nanmean(values: pd.Series) -> float:
    valid = values.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.mean())


def classify_by_quantile(x: pd.Series, elevated_q: float, stress_q: float) -> pd.Series:
    out = pd.Series("normal", index=x.index, dtype="object")
    valid = x.dropna()
    if len(valid) < 10:
        out[x.notna()] = "insufficient_history"
        return out
    elevated = valid.quantile(elevated_q)
    stress = valid.quantile(stress_q)
    out[(x > elevated) & (x <= stress)] = "elevated"
    out[x > stress] = "stress"
    out[x.isna()] = "missing"
    return out


def add_manual_regimes(dates: pd.Series, windows: List[Dict]) -> pd.DataFrame:
    reg = pd.DataFrame({"date": pd.to_datetime(dates).sort_values().unique()})
    reg["manual_regime"] = "normal"
    reg["manual_regime_detail"] = "normal"
    for w in windows:
        start = pd.to_datetime(w["start"])
        end = pd.to_datetime(w["end"])
        name = w["name"]
        mask = (reg["date"] >= start) & (reg["date"] <= end)
        reg.loc[mask, "manual_regime"] = "event_stress"
        reg.loc[mask, "manual_regime_detail"] = name
    return reg


def build_internal_features(prices: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    c = cfg["columns"]
    date_col, cusip_col, vendor_col, price_col = c["date"], c["cusip"], c["vendor"], c["price"]
    bucket_col = c.get("bucket")
    weight_col = c.get("weight")
    window = cfg["regime"]["rolling_window_days"]

    prices = prices.copy()
    prices[date_col] = pd.to_datetime(prices[date_col])
    if weight_col not in prices.columns:
        prices[weight_col] = 1.0
    if bucket_col not in prices.columns:
        prices[bucket_col] = "ALL"

    wide = (
        prices.pivot_table(
            index=[date_col, cusip_col, bucket_col, weight_col],
            columns=vendor_col,
            values=price_col,
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    vendor_cols = [v for v in cfg["vendors"] if v in wide.columns]
    if len(vendor_cols) < 2:
        raise ValueError("Need at least two vendor price columns after pivoting.")

    wide["consensus_price"] = wide[vendor_cols].median(axis=1, skipna=True)
    wide["vendor_dispersion"] = wide[vendor_cols].sub(wide["consensus_price"], axis=0).abs().median(axis=1)

    daily_rows = []
    for (dt, bucket), g in wide.groupby([date_col, bucket_col]):
        w = g[weight_col].astype(float)
        daily_rows.append(
            {
                "date": dt,
                "bucket": bucket,
                "consensus_index": weighted_mean(g["consensus_price"], w),
                "vendor_dispersion": weighted_mean(g["vendor_dispersion"], w),
                "n_cusips": g[cusip_col].nunique(),
            }
        )
    by_bucket = pd.DataFrame(daily_rows).sort_values(["bucket", "date"])

    by_bucket["consensus_index_change"] = by_bucket.groupby("bucket")["consensus_index"].diff()
    by_bucket["consensus_vol"] = by_bucket.groupby("bucket")["consensus_index_change"].transform(
        lambda s: s.rolling(window, min_periods=max(5, window // 4)).std()
    )
    by_bucket["dispersion_ma"] = by_bucket.groupby("bucket")["vendor_dispersion"].transform(
        lambda s: s.rolling(window, min_periods=max(5, window // 4)).mean()
    )

    overall = (
        by_bucket.groupby("date")
        .apply(
            lambda g: pd.Series(
                {
                    "consensus_index": np.average(g["consensus_index"], weights=np.maximum(g["n_cusips"], 1)),
                    "vendor_dispersion": np.average(g["vendor_dispersion"], weights=np.maximum(g["n_cusips"], 1)),
                    "consensus_vol": safe_nanmean(g["consensus_vol"]),
                    "dispersion_ma": safe_nanmean(g["dispersion_ma"]),
                    "n_cusips": g["n_cusips"].sum(),
                }
            )
        )
        .reset_index()
    )
    return overall


def build_market_features(market: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    c = cfg["columns"]
    date_col = c["date"]
    cc_col = c["current_coupon"]
    ten_col = c["ten_year_rate"]
    vix_col = c.get("vix")
    window = cfg["regime"]["rolling_window_days"]

    require_columns(market, [date_col, cc_col, ten_col], "market data")
    m = market.copy()
    m[date_col] = pd.to_datetime(m[date_col])
    m = m.sort_values(date_col).rename(columns={date_col: "date"})
    m["d_current_coupon"] = m[cc_col].diff()
    m["d_ten_year_rate"] = m[ten_col].diff()
    m["abs_d_current_coupon"] = m["d_current_coupon"].abs()
    m["abs_d_ten_year_rate"] = m["d_ten_year_rate"].abs()
    m["cc_10y_spread"] = m[cc_col] - m[ten_col]
    m["vol_current_coupon"] = m["d_current_coupon"].rolling(window, min_periods=max(5, window // 4)).std()
    m["vol_ten_year_rate"] = m["d_ten_year_rate"].rolling(window, min_periods=max(5, window // 4)).std()

    if vix_col in m.columns:
        m["log_vix"] = np.log(m[vix_col].where(m[vix_col] > 0))
        m["d_log_vix"] = m["log_vix"].diff()
    return m


def add_gmm_regime(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    k = int(cfg["regime"]["gmm_components"])
    min_rows = int(cfg["regime"]["min_gmm_rows"])
    feature_candidates = [
        "log_vix",
        "d_log_vix",
        "abs_d_ten_year_rate",
        "abs_d_current_coupon",
        "vol_ten_year_rate",
        "vol_current_coupon",
        "consensus_vol",
        "dispersion_ma",
    ]
    features = [f for f in feature_candidates if f in df.columns and df[f].notna().sum() >= min_rows]
    out = df.copy()
    out["gmm_market_regime"] = "insufficient_history"
    out["gmm_cluster"] = np.nan

    clean = out[["date"] + features].replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) < 2 or len(clean) < min_rows:
        return out

    scaler = StandardScaler()
    x = scaler.fit_transform(clean[features])
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=20)
    cluster = gmm.fit_predict(x)
    clean = clean.assign(gmm_cluster=cluster)

    z = pd.DataFrame(x, columns=features, index=clean.index)
    stress_features = [
        f
        for f in ["log_vix", "abs_d_ten_year_rate", "abs_d_current_coupon", "vol_ten_year_rate", "vol_current_coupon", "consensus_vol", "dispersion_ma"]
        if f in z.columns
    ]
    clean["stress_score"] = z[stress_features].mean(axis=1)
    cluster_score = clean.groupby("gmm_cluster")["stress_score"].mean().sort_values()
    labels = ["normal", "elevated", "stress"][:k]
    cluster_to_label = {cluster_id: labels[i] for i, cluster_id in enumerate(cluster_score.index)}

    probs = pd.DataFrame(gmm.predict_proba(x), columns=[f"gmm_prob_cluster_{i}" for i in range(k)])
    probs["date"] = clean["date"].values

    mapped = clean[["date", "gmm_cluster"]].copy()
    mapped["gmm_market_regime"] = mapped["gmm_cluster"].map(cluster_to_label)
    out = out.drop(columns=[c for c in out.columns if c.startswith("gmm_prob_cluster_")], errors="ignore")
    out = out.merge(mapped, on="date", how="left", suffixes=("", "_new"))
    out["gmm_market_regime"] = out["gmm_market_regime_new"].combine_first(out["gmm_market_regime"])
    out["gmm_cluster"] = out["gmm_cluster_new"].combine_first(out["gmm_cluster"])
    out = out.drop(columns=["gmm_market_regime_new", "gmm_cluster_new"])
    out = out.merge(probs, on="date", how="left")
    return out


def create_dashboard(regimes: pd.DataFrame, cfg: Dict) -> None:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Current Coupon and 10Y Rate",
            "VIX",
            "Consensus Price Index",
            "Consensus Price Volatility",
            "Cross-Vendor Dispersion",
        ],
    )
    c = cfg["columns"]
    date = regimes["date"]
    fig.add_trace(go.Scatter(x=date, y=regimes[c["current_coupon"]], name="Current coupon"), row=1, col=1)
    fig.add_trace(go.Scatter(x=date, y=regimes[c["ten_year_rate"]], name="10Y rate"), row=1, col=1)
    if c.get("vix") in regimes.columns:
        fig.add_trace(go.Scatter(x=date, y=regimes[c["vix"]], name="VIX"), row=2, col=1)
    fig.add_trace(go.Scatter(x=date, y=regimes["consensus_index"], name="Consensus index"), row=3, col=1)
    fig.add_trace(go.Scatter(x=date, y=regimes["consensus_vol"], name="Consensus vol"), row=4, col=1)
    fig.add_trace(go.Scatter(x=date, y=regimes["dispersion_ma"], name="Dispersion MA"), row=5, col=1)

    stress_colors = {
        "normal": "rgba(120,120,120,0.00)",
        "elevated": "rgba(255,190,80,0.18)",
        "stress": "rgba(220,60,60,0.18)",
        "event_stress": "rgba(100,90,220,0.12)",
    }
    for col in ["gmm_market_regime", "manual_regime"]:
        last_start = None
        last_val = None
        for _, row in regimes[["date", col]].iterrows():
            val = row[col]
            if val != last_val:
                if last_start is not None and last_val != "normal":
                    fig.add_vrect(
                        x0=last_start,
                        x1=row["date"],
                        fillcolor=stress_colors.get(last_val, "rgba(200,200,200,0.10)"),
                        line_width=0,
                        layer="below",
                    )
                last_start = row["date"]
                last_val = val
        if last_start is not None and last_val != "normal":
            fig.add_vrect(
                x0=last_start,
                x1=regimes["date"].max(),
                fillcolor=stress_colors.get(last_val, "rgba(200,200,200,0.10)"),
                line_width=0,
                layer="below",
            )

    fig.update_layout(height=1200, title="Vendor Price Regime EDA", hovermode="x unified")
    out = resolve_path(cfg, cfg["output"]["figures_dir"]) / "regime_dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)

    if "gmm_market_regime" in regimes.columns:
        scatter_features = [x for x in ["vol_ten_year_rate", "consensus_vol", "dispersion_ma"] if x in regimes.columns]
        if len(scatter_features) >= 2:
            scat = px.scatter(
                regimes,
                x=scatter_features[0],
                y=scatter_features[1],
                color="gmm_market_regime",
                hover_data=["date", "manual_regime_detail"],
                title="Data-Driven Regime Separation",
            )
            scat.write_html(resolve_path(cfg, cfg["output"]["figures_dir"]) / "gmm_regime_scatter.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    c = cfg["columns"]

    prices = pd.read_csv(resolve_path(cfg, cfg["input"]["price_file"]))
    market = pd.read_csv(resolve_path(cfg, cfg["input"]["market_file"]))
    require_columns(prices, [c["date"], c["cusip"], c["vendor"], c["price"]], "price data")

    internal = build_internal_features(prices, cfg)
    market_features = build_market_features(market, cfg)
    dates = pd.concat([internal["date"], market_features["date"]]).drop_duplicates()

    regimes = add_manual_regimes(dates, cfg["manual_regimes"])
    regimes = regimes.merge(market_features, on="date", how="left")
    regimes = regimes.merge(internal, on="date", how="left")

    elevated_q = float(cfg["regime"]["elevated_quantile"])
    stress_q = float(cfg["regime"]["stress_quantile"])
    vix_col = c.get("vix")
    if vix_col in regimes.columns:
        regimes["vix_percentile_regime"] = classify_by_quantile(regimes[vix_col], elevated_q, stress_q)
    regimes["consensus_vol_regime"] = classify_by_quantile(regimes["consensus_vol"], elevated_q, stress_q)
    regimes["vendor_dispersion_regime"] = classify_by_quantile(regimes["dispersion_ma"], elevated_q, stress_q)
    regimes = add_gmm_regime(regimes, cfg)

    tables_dir = resolve_path(cfg, cfg["output"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    regimes.sort_values("date").to_csv(tables_dir / "regimes_by_date.csv", index=False)

    regime_cols = [
        "manual_regime_detail",
        "vix_percentile_regime",
        "consensus_vol_regime",
        "vendor_dispersion_regime",
        "gmm_market_regime",
    ]
    summary_rows = []
    feature_cols = [
        c["current_coupon"],
        c["ten_year_rate"],
        "d_current_coupon",
        "d_ten_year_rate",
        vix_col if vix_col in regimes.columns else None,
        "consensus_vol",
        "dispersion_ma",
    ]
    feature_cols = [x for x in feature_cols if x and x in regimes.columns]
    for reg_col in [x for x in regime_cols if x in regimes.columns]:
        for name, g in regimes.groupby(reg_col, dropna=False):
            row = {"regime_type": reg_col, "regime": name, "n_days": len(g)}
            for f in feature_cols:
                row[f"{f}_mean"] = g[f].mean()
                row[f"{f}_p90"] = g[f].quantile(0.90)
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(tables_dir / "regime_feature_summary.csv", index=False)

    create_dashboard(regimes, cfg)
    print(f"Wrote {tables_dir / 'regimes_by_date.csv'}")
    print(f"Wrote {resolve_path(cfg, cfg['output']['figures_dir']) / 'regime_dashboard.html'}")


if __name__ == "__main__":
    main()
