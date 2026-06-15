#!/usr/bin/env python3
"""Nonparametric vendor comparison by regime.

The tests are based on each vendor's absolute deviation from the median of the
other vendors. This keeps the calculation symmetric and avoids choosing a truth
vendor.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats


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
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def robust_mad(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    med = x.median()
    return float((x - med).abs().median())


def build_vendor_deviation_panel(prices: pd.DataFrame, regimes: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    c = cfg["columns"]
    date_col, cusip_col, vendor_col, price_col = c["date"], c["cusip"], c["vendor"], c["price"]
    prices = prices.copy()
    prices[date_col] = pd.to_datetime(prices[date_col])
    regimes = regimes.copy()
    regimes["date"] = pd.to_datetime(regimes["date"])

    wide = (
        prices.pivot_table(index=[date_col, cusip_col], columns=vendor_col, values=price_col, aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    vendors = [v for v in cfg["vendors"] if v in wide.columns]
    if len(vendors) < 3:
        raise ValueError("This script expects at least three vendors in the data.")

    rows = []
    for v in vendors:
        other = [x for x in vendors if x != v]
        tmp = wide[[date_col, cusip_col, v] + other].copy()
        tmp = tmp.rename(columns={v: "price"})
        tmp["vendor"] = v
        tmp["peer_consensus"] = tmp[other].median(axis=1, skipna=True)
        tmp["deviation"] = tmp["price"] - tmp["peer_consensus"]
        tmp["abs_deviation"] = tmp["deviation"].abs()
        rows.append(tmp[[date_col, cusip_col, "vendor", "price", "peer_consensus", "deviation", "abs_deviation"]])
    panel = pd.concat(rows, ignore_index=True).rename(columns={date_col: "date", cusip_col: "cusip"})
    panel = panel.merge(regimes, on="date", how="left")

    panel = panel.sort_values(["vendor", "cusip", "date"])
    tol = float(cfg["nonparametric"]["stale_price_tolerance"])
    panel["price_change"] = panel.groupby(["vendor", "cusip"])["price"].diff()
    panel["is_stale"] = panel["price_change"].abs() <= tol

    mad_by_cusip_vendor = (
        panel.groupby(["cusip", "vendor"])["abs_deviation"]
        .apply(robust_mad)
        .reset_index(name="cusip_vendor_mad")
    )
    panel = panel.merge(mad_by_cusip_vendor, on=["cusip", "vendor"], how="left")
    abs_threshold = float(cfg["nonparametric"]["absolute_outlier_points"])
    z_threshold = float(cfg["nonparametric"]["robust_z_outlier_threshold"])
    robust_scale = 1.4826 * panel["cusip_vendor_mad"]
    panel["robust_z_abs_deviation"] = panel["abs_deviation"] / robust_scale.replace(0, np.nan)
    panel["is_outlier"] = (panel["abs_deviation"] >= abs_threshold) | (panel["robust_z_abs_deviation"] >= z_threshold)
    return panel


def scorecard(panel: pd.DataFrame) -> pd.DataFrame:
    regime_cols = [
        "manual_regime_detail",
        "vix_percentile_regime",
        "consensus_vol_regime",
        "vendor_dispersion_regime",
        "gmm_market_regime",
        "anomaly_internal_regime",
    ]
    rows = []
    for reg_col in [x for x in regime_cols if x in panel.columns]:
        for (regime, vendor), g in panel.groupby([reg_col, "vendor"], dropna=False):
            rows.append(
                {
                    "regime_type": reg_col,
                    "regime": regime,
                    "vendor": vendor,
                    "n": len(g),
                    "n_cusips": g["cusip"].nunique(),
                    "median_abs_deviation": g["abs_deviation"].median(),
                    "mean_abs_deviation": g["abs_deviation"].mean(),
                    "p95_abs_deviation": g["abs_deviation"].quantile(0.95),
                    "mean_signed_deviation": g["deviation"].mean(),
                    "outlier_rate": g["is_outlier"].mean(),
                    "stale_rate": g["is_stale"].mean(),
                    "missing_price_rate": g["price"].isna().mean(),
                }
            )
    return pd.DataFrame(rows)


def pairwise_tests(panel: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    regime_cols = [
        "manual_regime_detail",
        "vix_percentile_regime",
        "consensus_vol_regime",
        "vendor_dispersion_regime",
        "gmm_market_regime",
        "anomaly_internal_regime",
    ]
    vendors = cfg["vendors"]
    rows = []
    for reg_col in [x for x in regime_cols if x in panel.columns]:
        for regime, g in panel.groupby(reg_col, dropna=False):
            wide = g.pivot_table(index=["date", "cusip"], columns="vendor", values="abs_deviation", aggfunc="mean")
            for a, b in itertools.combinations([v for v in vendors if v in wide.columns], 2):
                paired = wide[[a, b]].dropna()
                if len(paired) < 20:
                    continue
                diff = paired[a] - paired[b]
                n_a_better = int((diff < 0).sum())
                n_b_better = int((diff > 0).sum())
                sign_res = stats.binomtest(n_a_better, n_a_better + n_b_better, p=0.5) if (n_a_better + n_b_better) > 0 else None
                try:
                    wilcox = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                    wilcox_stat = float(wilcox.statistic)
                    wilcox_p = float(wilcox.pvalue)
                except ValueError:
                    wilcox_stat = np.nan
                    wilcox_p = np.nan
                rows.append(
                    {
                        "regime_type": reg_col,
                        "regime": regime,
                        "vendor_a": a,
                        "vendor_b": b,
                        "n_pairs": len(paired),
                        "median_abs_dev_a": paired[a].median(),
                        "median_abs_dev_b": paired[b].median(),
                        "median_diff_a_minus_b": diff.median(),
                        "n_a_closer": n_a_better,
                        "n_b_closer": n_b_better,
                        "sign_test_pvalue": sign_res.pvalue if sign_res else np.nan,
                        "wilcoxon_stat": wilcox_stat,
                        "wilcoxon_pvalue": wilcox_p,
                    }
                )
    return pd.DataFrame(rows)


def friedman_tests(panel: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    regime_cols = [
        "manual_regime_detail",
        "vix_percentile_regime",
        "consensus_vol_regime",
        "vendor_dispersion_regime",
        "gmm_market_regime",
        "anomaly_internal_regime",
    ]
    vendors = cfg["vendors"]
    rows = []
    for reg_col in [x for x in regime_cols if x in panel.columns]:
        for regime, g in panel.groupby(reg_col, dropna=False):
            wide = g.pivot_table(index=["date", "cusip"], columns="vendor", values="abs_deviation", aggfunc="mean")
            have = [v for v in vendors if v in wide.columns]
            paired = wide[have].dropna()
            if len(paired) < 20 or len(have) < 3:
                continue
            stat, pvalue = stats.friedmanchisquare(*[paired[v].to_numpy() for v in have])
            ranks = paired.rank(axis=1, method="average", ascending=True)
            row = {
                "regime_type": reg_col,
                "regime": regime,
                "n_complete_blocks": len(paired),
                "friedman_stat": stat,
                "friedman_pvalue": pvalue,
            }
            for v in have:
                row[f"{v}_mean_rank"] = ranks[v].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    c = cfg["columns"]

    prices = pd.read_csv(resolve_path(cfg, cfg["input"]["price_file"]))
    regimes = pd.read_csv(resolve_path(cfg, cfg["output"]["tables_dir"]) / "regimes_by_date.csv")
    require_columns(prices, [c["date"], c["cusip"], c["vendor"], c["price"]], "price data")

    panel = build_vendor_deviation_panel(prices, regimes, cfg)
    tables_dir = resolve_path(cfg, cfg["output"]["tables_dir"])
    figures_dir = resolve_path(cfg, cfg["output"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(tables_dir / "nonparam_vendor_deviation_panel.csv", index=False)
    scorecard(panel).to_csv(tables_dir / "nonparam_vendor_scorecard.csv", index=False)
    pairwise_tests(panel, cfg).to_csv(tables_dir / "nonparam_pairwise_tests.csv", index=False)
    friedman_tests(panel, cfg).to_csv(tables_dir / "nonparam_friedman_tests.csv", index=False)

    plot_col = "gmm_market_regime" if "gmm_market_regime" in panel.columns else "manual_regime_detail"
    fig = px.box(
        panel.dropna(subset=["abs_deviation"]),
        x="vendor",
        y="abs_deviation",
        color="vendor",
        facet_col=plot_col,
        facet_col_wrap=3,
        points=False,
        title="Absolute Deviation from Peer Consensus by Regime",
    )
    fig.update_yaxes(matches=None)
    fig.write_html(figures_dir / "nonparam_abs_deviation_boxplot.html")

    print(f"Wrote {tables_dir / 'nonparam_vendor_scorecard.csv'}")
    print(f"Wrote {tables_dir / 'nonparam_pairwise_tests.csv'}")


if __name__ == "__main__":
    main()
