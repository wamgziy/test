#!/usr/bin/env python3
"""Define internal regimes from vendor-level anomaly flags.

Expected anomaly input is long format:

    date,cusip,vendor,anomaly_flag

Optional columns:

    bucket,anomaly_score

The script aggregates vendor/CUSIP anomaly flags into date-level and
bucket-date-level features:

    total anomaly rate
    any-vendor anomaly rate
    shared anomaly rate: fraction of CUSIPs with >=2 anomalous vendors
    all-vendor anomaly rate: fraction of CUSIPs with all vendors anomalous
    vendor anomaly rates
    isolated anomaly rates by vendor

The resulting date-level label is written as `anomaly_internal_regime`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_base_dir"] = str(Path(path).resolve().parents[1])
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


def clean_vendor_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def add_bucket_from_prices(anom: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    c = cfg["columns"]
    bucket_col = c.get("bucket")
    if bucket_col in anom.columns:
        return anom

    price_path = resolve_path(cfg, cfg["input"]["price_file"])
    if not price_path.exists():
        anom[bucket_col] = "ALL"
        return anom

    prices = pd.read_csv(price_path, usecols=lambda x: x in {c["cusip"], bucket_col})
    if bucket_col not in prices.columns:
        anom[bucket_col] = "ALL"
        return anom

    mapping = prices[[c["cusip"], bucket_col]].drop_duplicates(subset=[c["cusip"]])
    out = anom.merge(mapping, on=c["cusip"], how="left")
    out[bucket_col] = out[bucket_col].fillna("ALL")
    return out


def aggregate_anomaly_features(anom: pd.DataFrame, cfg: Dict, group_cols: List[str]) -> pd.DataFrame:
    c = cfg["columns"]
    date_col, cusip_col, vendor_col, flag_col = c["date"], c["cusip"], c["vendor"], c["anomaly_flag"]
    vendors = [v for v in cfg["vendors"] if v in set(anom[vendor_col])]
    if len(vendors) < 2:
        raise ValueError("Need at least two configured vendors in anomaly data.")

    wide = (
        anom.pivot_table(
            index=group_cols + [cusip_col],
            columns=vendor_col,
            values=flag_col,
            aggfunc="max",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for v in vendors:
        if v not in wide.columns:
            wide[v] = 0
    wide[vendors] = wide[vendors].fillna(0).astype(int)
    wide["n_vendor_flags"] = wide[vendors].sum(axis=1)
    wide["any_vendor_anomaly"] = (wide["n_vendor_flags"] >= 1).astype(int)
    wide["shared_anomaly"] = (wide["n_vendor_flags"] >= 2).astype(int)
    wide["all_vendor_anomaly"] = (wide["n_vendor_flags"] == len(vendors)).astype(int)
    wide["one_vendor_anomaly"] = (wide["n_vendor_flags"] == 1).astype(int)

    for v in vendors:
        safe_v = clean_vendor_name(v)
        wide[f"vendor_anomaly_{safe_v}"] = wide[v]
        wide[f"isolated_anomaly_{safe_v}"] = ((wide[v] == 1) & (wide["n_vendor_flags"] == 1)).astype(int)

    agg_spec = {
        cusip_col: "nunique",
        "n_vendor_flags": "mean",
        "any_vendor_anomaly": "mean",
        "shared_anomaly": "mean",
        "all_vendor_anomaly": "mean",
        "one_vendor_anomaly": "mean",
    }
    for v in vendors:
        safe_v = clean_vendor_name(v)
        agg_spec[f"vendor_anomaly_{safe_v}"] = "mean"
        agg_spec[f"isolated_anomaly_{safe_v}"] = "mean"

    out = wide.groupby(group_cols).agg(agg_spec).reset_index()
    out = out.rename(
        columns={
            cusip_col: "n_cusips",
            "n_vendor_flags": "mean_anomalous_vendors_per_cusip",
            "any_vendor_anomaly": "any_anomaly_rate",
            "shared_anomaly": "shared_anomaly_rate",
            "all_vendor_anomaly": "all_vendor_anomaly_rate",
            "one_vendor_anomaly": "one_vendor_anomaly_rate",
        }
    )

    vendor_rate_cols = []
    isolated_cols = []
    for v in vendors:
        safe_v = clean_vendor_name(v)
        out = out.rename(
            columns={
                f"vendor_anomaly_{safe_v}": f"vendor_anomaly_rate_{safe_v}",
                f"isolated_anomaly_{safe_v}": f"isolated_anomaly_rate_{safe_v}",
            }
        )
        vendor_rate_cols.append(f"vendor_anomaly_rate_{safe_v}")
        isolated_cols.append(f"isolated_anomaly_rate_{safe_v}")

    out["total_anomaly_rate"] = out[vendor_rate_cols].mean(axis=1)
    out["max_vendor_anomaly_rate"] = out[vendor_rate_cols].max(axis=1)
    out["max_isolated_anomaly_rate"] = out[isolated_cols].max(axis=1)
    out["dominant_isolated_vendor"] = out[isolated_cols].idxmax(axis=1).str.replace("isolated_anomaly_rate_", "", regex=False)
    return out


def smooth_features(df: pd.DataFrame, cfg: Dict, group_cols: List[str]) -> pd.DataFrame:
    window = int(cfg["anomaly_regime"]["smooth_window_days"])
    if window <= 1:
        return df

    out = df.copy()
    out = out.sort_values(group_cols + ["date"])
    numeric_cols = [
        c
        for c in out.columns
        if c.endswith("_rate") or c in {"mean_anomalous_vendors_per_cusip", "total_anomaly_rate"}
    ]
    for col in numeric_cols:
        if group_cols:
            out[f"{col}_smoothed"] = out.groupby(group_cols, dropna=False)[col].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
        else:
            out[f"{col}_smoothed"] = out[col].rolling(window, min_periods=1).mean()
    return out


def quantile_thresholds(features: pd.DataFrame, cfg: Dict) -> Dict[str, float]:
    aq = cfg["anomaly_regime"]
    elevated_q = float(aq["elevated_quantile"])
    stress_q = float(aq["stress_quantile"])
    fields = [
        "total_anomaly_rate_smoothed",
        "shared_anomaly_rate_smoothed",
        "max_isolated_anomaly_rate_smoothed",
        "any_anomaly_rate_smoothed",
    ]
    thresholds = {}
    for field in fields:
        base = field.removesuffix("_smoothed")
        col = field if field in features.columns else base
        valid = features[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) == 0:
            thresholds[f"{base}_elevated"] = np.nan
            thresholds[f"{base}_stress"] = np.nan
        else:
            thresholds[f"{base}_elevated"] = float(valid.quantile(elevated_q))
            thresholds[f"{base}_stress"] = float(valid.quantile(stress_q))
    return thresholds


def label_anomaly_regime(features: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    out = features.copy()
    thresholds = quantile_thresholds(out, cfg)
    min_cusips = int(cfg["anomaly_regime"]["min_cusips_per_day"])

    def col(name: str) -> str:
        smoothed = f"{name}_smoothed"
        return smoothed if smoothed in out.columns else name

    total_col = col("total_anomaly_rate")
    shared_col = col("shared_anomaly_rate")
    any_col = col("any_anomaly_rate")
    iso_col = col("max_isolated_anomaly_rate")

    labels = []
    reasons = []
    for _, row in out.iterrows():
        if row.get("n_cusips", 0) < min_cusips:
            labels.append("insufficient_cusips")
            reasons.append("too_few_cusips")
            continue

        total = row[total_col]
        shared = row[shared_col]
        any_rate = row[any_col]
        isolated = row[iso_col]
        dom_vendor = row.get("dominant_isolated_vendor", "unknown")

        shared_stress = thresholds["shared_anomaly_rate_stress"]
        total_stress = thresholds["total_anomaly_rate_stress"]
        any_stress = thresholds["any_anomaly_rate_stress"]
        iso_stress = thresholds["max_isolated_anomaly_rate_stress"]

        shared_elev = thresholds["shared_anomaly_rate_elevated"]
        total_elev = thresholds["total_anomaly_rate_elevated"]
        any_elev = thresholds["any_anomaly_rate_elevated"]
        iso_elev = thresholds["max_isolated_anomaly_rate_elevated"]

        if np.isfinite(shared_stress) and shared >= shared_stress:
            labels.append("broad_internal_stress")
            reasons.append("high_shared_anomaly_rate")
        elif np.isfinite(iso_stress) and isolated >= iso_stress and (not np.isfinite(shared_elev) or shared < shared_elev):
            labels.append("vendor_divergence")
            reasons.append(f"high_isolated_anomaly_rate_{dom_vendor}")
        elif (
            (np.isfinite(total_stress) and total >= total_stress)
            or (np.isfinite(any_stress) and any_rate >= any_stress)
        ):
            labels.append("mixed_pricing_uncertainty")
            reasons.append("high_total_or_any_anomaly_rate")
        elif (
            (np.isfinite(total_elev) and total >= total_elev)
            or (np.isfinite(any_elev) and any_rate >= any_elev)
            or (np.isfinite(shared_elev) and shared >= shared_elev)
            or (np.isfinite(iso_elev) and isolated >= iso_elev)
        ):
            labels.append("elevated_internal_stress")
            reasons.append("elevated_anomaly_features")
        else:
            labels.append("normal")
            reasons.append("low_anomaly_features")

    out["anomaly_internal_regime"] = labels
    out["anomaly_regime_reason"] = reasons
    for k, v in thresholds.items():
        out[f"threshold_{k}"] = v
    return out


def write_dashboard(date_features: pd.DataFrame, cfg: Dict) -> None:
    figures_dir = resolve_path(cfg, cfg["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Anomaly Rates",
            "Shared vs Isolated Anomalies",
            "Vendor Anomaly Rates",
            "Anomaly Internal Regime",
        ],
    )
    date = date_features["date"]
    for col, name in [
        ("total_anomaly_rate_smoothed", "Total anomaly rate"),
        ("any_anomaly_rate_smoothed", "Any-vendor anomaly rate"),
    ]:
        if col in date_features.columns:
            fig.add_trace(go.Scatter(x=date, y=date_features[col], name=name), row=1, col=1)

    for col, name in [
        ("shared_anomaly_rate_smoothed", "Shared anomaly rate"),
        ("max_isolated_anomaly_rate_smoothed", "Max isolated anomaly rate"),
    ]:
        if col in date_features.columns:
            fig.add_trace(go.Scatter(x=date, y=date_features[col], name=name), row=2, col=1)

    for v in cfg["vendors"]:
        col = f"vendor_anomaly_rate_{clean_vendor_name(v)}_smoothed"
        if col in date_features.columns:
            fig.add_trace(go.Scatter(x=date, y=date_features[col], name=f"{v} anomaly rate"), row=3, col=1)

    regime_codes = {
        "normal": 0,
        "elevated_internal_stress": 1,
        "mixed_pricing_uncertainty": 2,
        "vendor_divergence": 3,
        "broad_internal_stress": 4,
        "insufficient_cusips": np.nan,
    }
    fig.add_trace(
        go.Scatter(
            x=date,
            y=date_features["anomaly_internal_regime"].map(regime_codes),
            mode="markers",
            name="Regime",
            text=date_features["anomaly_regime_reason"],
            hovertemplate="%{x}<br>regime code=%{y}<br>%{text}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.update_layout(height=1000, title="Anomaly-Based Internal Regime EDA", hovermode="x unified")
    fig.write_html(figures_dir / "anomaly_regime_dashboard.html")


def merge_into_regimes(date_features: pd.DataFrame, cfg: Dict) -> None:
    if not bool(cfg["anomaly_regime"].get("merge_into_regimes_by_date", True)):
        return
    regimes_path = resolve_path(cfg, cfg["output"]["tables_dir"]) / "regimes_by_date.csv"
    if not regimes_path.exists():
        return
    regimes = pd.read_csv(regimes_path)
    regimes["date"] = pd.to_datetime(regimes["date"])
    keep_cols = [
        "date",
        "anomaly_internal_regime",
        "anomaly_regime_reason",
        "total_anomaly_rate",
        "total_anomaly_rate_smoothed",
        "shared_anomaly_rate",
        "shared_anomaly_rate_smoothed",
        "max_isolated_anomaly_rate",
        "max_isolated_anomaly_rate_smoothed",
        "dominant_isolated_vendor",
    ]
    add = date_features[[c for c in keep_cols if c in date_features.columns]].copy()
    add["date"] = pd.to_datetime(add["date"])
    regimes = regimes.drop(columns=[c for c in add.columns if c != "date" and c in regimes.columns], errors="ignore")
    regimes = regimes.merge(add, on="date", how="left")
    regimes.to_csv(regimes_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    c = cfg["columns"]

    anomaly_path = resolve_path(cfg, cfg["input"]["anomaly_file"])
    if not anomaly_path.exists():
        raise FileNotFoundError(f"Missing anomaly file: {anomaly_path}")

    anom = pd.read_csv(anomaly_path)
    require_columns(anom, [c["date"], c["cusip"], c["vendor"], c["anomaly_flag"]], "anomaly data")
    anom[c["date"]] = pd.to_datetime(anom[c["date"]])
    anom[c["anomaly_flag"]] = anom[c["anomaly_flag"]].fillna(0).astype(int).clip(0, 1)
    anom = add_bucket_from_prices(anom, cfg)

    date_features = aggregate_anomaly_features(anom, cfg, [c["date"]]).rename(columns={c["date"]: "date"})
    date_features = smooth_features(date_features, cfg, [])
    date_features = label_anomaly_regime(date_features, cfg)

    bucket_col = c.get("bucket")
    bucket_features = aggregate_anomaly_features(anom, cfg, [c["date"], bucket_col]).rename(columns={c["date"]: "date"})
    bucket_features = smooth_features(bucket_features, cfg, [bucket_col])
    bucket_features = label_anomaly_regime(bucket_features, cfg)

    tables_dir = resolve_path(cfg, cfg["output"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    date_features.to_csv(tables_dir / "anomaly_regime_by_date.csv", index=False)
    bucket_features.to_csv(tables_dir / "anomaly_regime_by_bucket_date.csv", index=False)

    summary = (
        date_features.groupby("anomaly_internal_regime")
        .agg(
            n_days=("date", "count"),
            avg_total_anomaly_rate=("total_anomaly_rate", "mean"),
            avg_shared_anomaly_rate=("shared_anomaly_rate", "mean"),
            avg_max_isolated_anomaly_rate=("max_isolated_anomaly_rate", "mean"),
            avg_n_cusips=("n_cusips", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(tables_dir / "anomaly_regime_summary.csv", index=False)

    write_dashboard(date_features, cfg)
    merge_into_regimes(date_features, cfg)

    print(f"Wrote {tables_dir / 'anomaly_regime_by_date.csv'}")
    print(f"Wrote {tables_dir / 'anomaly_regime_by_bucket_date.csv'}")
    print(f"Wrote {resolve_path(cfg, cfg['output']['figures_dir']) / 'anomaly_regime_dashboard.html'}")


if __name__ == "__main__":
    main()
