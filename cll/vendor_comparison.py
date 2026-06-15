"""
Vendor comparison framework for CMO daily price marks.

Three vendors, daily prices per CUSIP, 2012-2026. No ground truth — we build a
defensible "best vendor" argument from three angles:

  Step 1: TRACE-anchored accuracy   (which vendor's prior mark best predicts the trade?)
  Step 2: Staleness / responsiveness (which vendor is freshest, least matrix-priced?)
  Step 3: Cross-vendor consensus    (sanity layer, with regime split)

Outputs a single scorecard DataFrame keyed by vendor.

Expected inputs (long format):

  vendor_px : DataFrame[date, cusip, vendor, price]
  trades    : DataFrame[date, cusip, trade_price, trade_size]   (TRACE; optional)
  regimes   : DataFrame[date, regime]                           ('calm' or 'stress')

Dependencies: pandas, numpy, scipy>=1.7 (for friedmanchisquare, posthoc_nemenyi
via scikit-posthocs is optional; we ship a simple fallback).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Step 1: TRACE-anchored accuracy
# ----------------------------------------------------------------------------

def trace_anchored_errors(
    vendor_px: pd.DataFrame,
    trades: pd.DataFrame,
    regimes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For each trade print, attach each vendor's mark from the prior business day
    and compute signed pricing error. Returns one row per (trade, vendor).

    Signed error = trade_price - vendor_mark(t-1). Positive => vendor under-marked.
    """
    px = vendor_px.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["cusip", "vendor", "date"])

    # Forward-fill each vendor's price within cusip so a trade on day t can
    # use the last available mark even if the vendor skipped a day.
    px["price_ff"] = px.groupby(["cusip", "vendor"])["price"].ffill()

    tr = trades.copy()
    tr["date"] = pd.to_datetime(tr["date"])

    # As-of join: for each trade, pull each vendor's most recent mark strictly
    # before the trade date. merge_asof needs sorted frames.
    out = []
    for vendor, g in px.groupby("vendor"):
        g = g[["date", "cusip", "price_ff"]].rename(columns={"price_ff": "vendor_px"})
        g = g.sort_values(["cusip", "date"])
        merged = pd.merge_asof(
            tr.sort_values("date"),
            g.sort_values("date"),
            on="date",
            by="cusip",
            direction="backward",
            allow_exact_matches=False,  # strictly prior-day mark
        )
        merged["vendor"] = vendor
        out.append(merged)

    err = pd.concat(out, ignore_index=True)
    err = err.dropna(subset=["vendor_px"])
    err["signed_err"] = err["trade_price"] - err["vendor_px"]
    err["abs_err"] = err["signed_err"].abs()

    if regimes is not None:
        regimes = regimes.copy()
        regimes["date"] = pd.to_datetime(regimes["date"])
        err = err.merge(regimes, on="date", how="left")
        err["regime"] = err["regime"].fillna("calm")
    else:
        err["regime"] = "calm"

    return err


def summarize_trace_errors(err: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trade-level errors into per-vendor metrics, overall and by regime.
    Size-weighted MAE uses trade_size as weight (face value).
    """
    def _agg(g: pd.DataFrame) -> pd.Series:
        w = g["trade_size"].clip(lower=0).fillna(0)
        wsum = w.sum()
        size_w_mae = (g["abs_err"] * w).sum() / wsum if wsum > 0 else np.nan
        return pd.Series({
            "n_trades": len(g),
            "mae": g["abs_err"].mean(),
            "rmse": np.sqrt((g["signed_err"] ** 2).mean()),
            "bias": g["signed_err"].mean(),
            "size_w_mae": size_w_mae,
            "p95_abs_err": g["abs_err"].quantile(0.95),
        })

    overall = err.groupby("vendor", group_keys=False).apply(_agg).add_prefix("all_")
    by_regime = (
        err.groupby(["vendor", "regime"], group_keys=False)
           .apply(_agg)
           .unstack("regime")
    )
    by_regime.columns = [f"{reg}_{metric}" for metric, reg in by_regime.columns]
    return overall.join(by_regime)


# ----------------------------------------------------------------------------
# Step 2: Staleness / responsiveness
# ----------------------------------------------------------------------------

def staleness_metrics(vendor_px: pd.DataFrame) -> pd.DataFrame:
    """
    Per-vendor staleness diagnostics, averaged across cusips.

      zero_change_rate : fraction of (cusip, day) with price unchanged from prior day
      ret_autocorr1    : lag-1 autocorrelation of daily returns (high => smoothed)
      median_run_len   : median consecutive-flat-day run length
    """
    px = vendor_px.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["vendor", "cusip", "date"])
    px["ret"] = px.groupby(["vendor", "cusip"])["price"].pct_change()
    px["unchanged"] = (px["ret"].abs() < 1e-12).astype(int)

    def _per_cusip(g: pd.DataFrame) -> pd.Series:
        # consecutive-flat run lengths
        runs = (g["unchanged"] != g["unchanged"].shift()).cumsum()
        run_lens = g[g["unchanged"] == 1].groupby(runs).size()
        return pd.Series({
            "zero_change_rate": g["unchanged"].mean(),
            "ret_autocorr1": g["ret"].autocorr(lag=1),
            "median_run_len": run_lens.median() if len(run_lens) else 0.0,
        })

    per_cu = px.groupby(["vendor", "cusip"], group_keys=False).apply(_per_cusip)
    return per_cu.groupby("vendor").mean()


def lead_lag_matrix(
    vendor_px: pd.DataFrame,
    max_lag: int = 5,
    min_obs: int = 250,
) -> pd.DataFrame:
    """
    For each ordered pair (A -> B), median over cusips of argmax cross-correlation
    of returns at lags 0..max_lag. Positive => B lags A (A leads).
    """
    px = vendor_px.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["cusip", "vendor", "date"])
    px["ret"] = px.groupby(["vendor", "cusip"])["price"].pct_change()

    wide = px.pivot_table(index=["cusip", "date"], columns="vendor", values="ret")
    vendors = list(wide.columns)
    rows = []
    for cu, sub in wide.groupby(level="cusip"):
        if len(sub) < min_obs:
            continue
        for a in vendors:
            for b in vendors:
                if a == b:
                    continue
                ra, rb = sub[a].dropna(), sub[b].dropna()
                idx = ra.index.intersection(rb.index)
                if len(idx) < min_obs:
                    continue
                ra, rb = ra.loc[idx].values, rb.loc[idx].values
                # corr at lag k: corr(A_t, B_{t+k})  -> k>0 means B lags A
                best_k, best_c = 0, -np.inf
                for k in range(0, max_lag + 1):
                    if k == 0:
                        c = np.corrcoef(ra, rb)[0, 1]
                    else:
                        c = np.corrcoef(ra[:-k], rb[k:])[0, 1]
                    if c > best_c:
                        best_c, best_k = c, k
                rows.append({"cusip": cu, "leader": a, "follower": b, "best_lag": best_k})

    ll = pd.DataFrame(rows)
    if ll.empty:
        return ll
    return ll.groupby(["leader", "follower"])["best_lag"].median().unstack("follower")


# ----------------------------------------------------------------------------
# Step 3: Cross-vendor consensus + regime split
# ----------------------------------------------------------------------------

def consensus_deviation(
    vendor_px: pd.DataFrame,
    regimes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Per (date, cusip): median of the 3 vendor prices. Deviation per vendor =
    |P_v - median| / median (scale-free). Returns long DataFrame.
    """
    px = vendor_px.copy()
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index=["date", "cusip"], columns="vendor", values="price")
    med = wide.median(axis=1)
    dev = wide.sub(med, axis=0).abs().div(med, axis=0)
    dev = dev.stack().rename("rel_dev").reset_index().rename(columns={"vendor": "vendor"})

    if regimes is not None:
        regimes = regimes.copy()
        regimes["date"] = pd.to_datetime(regimes["date"])
        dev = dev.merge(regimes, on="date", how="left")
        dev["regime"] = dev["regime"].fillna("calm")
    else:
        dev["regime"] = "calm"
    return dev


def summarize_consensus(dev: pd.DataFrame) -> pd.DataFrame:
    overall = dev.groupby("vendor")["rel_dev"].agg(["mean", "median",
                                                    lambda s: s.quantile(0.95)])
    overall.columns = ["all_mean_dev", "all_median_dev", "all_p95_dev"]
    by_reg = dev.groupby(["vendor", "regime"])["rel_dev"].mean().unstack("regime")
    by_reg.columns = [f"{c}_mean_dev" for c in by_reg.columns]
    return overall.join(by_reg)


def friedman_on_panel(vendor_px: pd.DataFrame) -> dict:
    """
    Friedman test: are vendors systematically different in level?
    Operates on the wide panel of (date, cusip) x vendor. Drops rows with NaN
    in any vendor.
    """
    px = vendor_px.copy()
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index=["date", "cusip"], columns="vendor", values="price").dropna()
    cols = list(wide.columns)
    if wide.shape[0] < 50 or len(cols) < 3:
        return {"stat": np.nan, "pvalue": np.nan, "n": wide.shape[0]}
    stat, p = stats.friedmanchisquare(*[wide[c].values for c in cols])
    return {"stat": float(stat), "pvalue": float(p), "n": int(wide.shape[0]),
            "vendors": cols}


def nemenyi_posthoc(vendor_px: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Nemenyi post-hoc on the (date, cusip)-paired vendor levels.
    Uses scikit-posthocs if available; otherwise a manual Studentized-range
    approximation on average ranks.
    """
    px = vendor_px.copy()
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index=["date", "cusip"], columns="vendor", values="price").dropna()
    try:
        import scikit_posthocs as sp
        return sp.posthoc_nemenyi_friedman(wide.values, melted=False).set_axis(
            wide.columns, axis=0).set_axis(wide.columns, axis=1)
    except ImportError:
        # Fallback: average ranks + qsturng approximation
        ranks = wide.rank(axis=1).mean(axis=0)
        k, n = wide.shape[1], wide.shape[0]
        se = np.sqrt(k * (k + 1) / (6.0 * n))
        cols = list(wide.columns)
        out = pd.DataFrame(np.eye(k), index=cols, columns=cols)
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if i >= j:
                    continue
                q = abs(ranks[a] - ranks[b]) / se
                # 2-sided p via normal approx of studentized range, conservative
                p = 2 * (1 - stats.norm.cdf(q / np.sqrt(2)))
                out.loc[a, b] = out.loc[b, a] = p
        return out


# ----------------------------------------------------------------------------
# Scorecard
# ----------------------------------------------------------------------------

def build_scorecard(
    vendor_px: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    regimes: pd.DataFrame | None = None,
    anomaly_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble per-vendor metrics from all steps into one table.
    Adds a rank column per metric (1 = best). Lower is better for every metric
    here (bias is converted to |bias| for ranking).

    If `anomaly_flags` is supplied, also includes lone_dissent_rate and
    bad_day_count -- vendor-specific issue metrics from `vendor_issue_metrics`.
    """
    parts = []

    if trades is not None and len(trades):
        err = trace_anchored_errors(vendor_px, trades, regimes)
        parts.append(summarize_trace_errors(err))

    parts.append(staleness_metrics(vendor_px))

    dev = consensus_deviation(vendor_px, regimes)
    parts.append(summarize_consensus(dev))

    if anomaly_flags is not None and len(anomaly_flags):
        vi = vendor_issue_metrics(anomaly_flags)
        issue_part = vi["cell_summary"][["lone_dissent_rate", "concurrent_share"]].copy()
        issue_part["bad_day_count"] = vi["bad_day_count"]
        # concurrent_share: HIGHER is better, invert for ranking convention
        issue_part["solo_share"] = 1 - issue_part["concurrent_share"]
        issue_part = issue_part.drop(columns=["concurrent_share"])
        parts.append(issue_part)

    scorecard = pd.concat(parts, axis=1)

    # Bias: convert to |bias| for ranking
    for col in [c for c in scorecard.columns if c.endswith("bias")]:
        scorecard[f"{col}_abs"] = scorecard[col].abs()

    rank_cols = [c for c in scorecard.columns
                 if any(c.endswith(s) for s in
                        ("mae", "rmse", "p95_abs_err", "size_w_mae", "bias_abs",
                         "zero_change_rate", "ret_autocorr1", "median_run_len",
                         "mean_dev", "median_dev", "p95_dev",
                         "lone_dissent_rate", "bad_day_count", "solo_share"))]
    for c in rank_cols:
        scorecard[f"rank_{c}"] = scorecard[c].rank(method="min")

    scorecard["overall_rank_sum"] = scorecard[[f"rank_{c}" for c in rank_cols]].sum(axis=1)
    return scorecard.sort_values("overall_rank_sum")


# ----------------------------------------------------------------------------
# Regime labeling
#
# A regime label is just date -> 'calm'/'stress'. We support three sources:
#   (1) predefined named stress windows (taper tantrum, COVID, SVB, ...)
#   (2) data-driven from per-(date, cusip, vendor) anomaly flags you already
#       compute (IQR / EWA based)
#   (3) optional: a market stress series (MOVE/VIX), if you ever add one
# `combine_regimes` merges (1) and (2). Use `predefined_plus_unnamed` mode by
# default: keeps auditable named windows AND surfaces unnamed events.
# ----------------------------------------------------------------------------

# CMO-relevant stress windows. Edit/extend as you see fit.
PREDEFINED_STRESS_WINDOWS = [
    ("2013-05-01", "2013-09-30", "taper_tantrum"),
    ("2015-08-01", "2015-08-31", "oil_hy_aug2015"),
    ("2016-01-01", "2016-02-29", "oil_hy_early2016"),
    ("2020-03-01", "2020-04-30", "covid"),
    ("2022-03-01", "2022-10-31", "rate_shock_2022"),
    ("2023-03-01", "2023-03-31", "svb"),
    ("2023-04-01", "2023-05-31", "regional_bank_2023"),
]


def label_regimes_predefined(
    dates,
    windows: list[tuple[str, str, str]] = PREDEFINED_STRESS_WINDOWS,
) -> pd.DataFrame:
    """
    Hard-coded named stress windows. Returns DataFrame[date, regime, event].
    `event` carries the window name on stress days, empty string elsewhere.
    """
    dates = pd.to_datetime(pd.Index(dates).unique()).sort_values()
    regime = pd.Series("calm", index=dates)
    event = pd.Series("", index=dates)
    for start, end, name in windows:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        regime[mask] = "stress"
        event[mask] = name
    return pd.DataFrame({"date": dates, "regime": regime.values, "event": event.values})


def label_regimes_from_anomalies(
    anomaly_flags: pd.DataFrame,
    flag_col: str = "anomaly_flag",
    vendor_quantile: float = 0.95,
    min_vendors: int = 2,
    min_run: int = 3,
    min_history: int = 252,
) -> pd.DataFrame:
    """
    Build a regime label from your per-(date, cusip, vendor) anomaly flags.

    Logic:
      1. Per (date, vendor): anomaly rate = mean(flag) across cusips.
      2. Per vendor: expanding-quantile threshold (no look-ahead).
      3. A vendor is 'hot' on a day if its rate exceeds its own threshold.
      4. Day is stress-candidate if >= `min_vendors` vendors are hot
         (concurrence -- filters out single-vendor data quirks).
      5. Keep only stress-candidate runs of length >= `min_run`.

    The concurrence step (4) is why this works: a real market stress event
    shows up across vendors; a vendor-specific data glitch does not.

    Args:
      anomaly_flags : DataFrame with at least [date, cusip, vendor, <flag_col>].
                      Flag values 0/1 or boolean.
      vendor_quantile : per-vendor expanding quantile for "elevated" threshold.
      min_vendors     : concurrence requirement.
      min_run         : minimum consecutive days for a regime to count.
      min_history     : days of history before threshold kicks in (252 = 1y).

    Returns: DataFrame[date, regime, anomaly_rate_max] -- the last column is
             max anomaly rate across vendors that day (diagnostic).
    """
    df = anomaly_flags.copy()
    df["date"] = pd.to_datetime(df["date"])
    df[flag_col] = df[flag_col].astype(float)

    daily_rate = (
        df.groupby(["date", "vendor"])[flag_col].mean().unstack("vendor").sort_index()
    )

    thr = daily_rate.expanding(min_periods=min_history).quantile(vendor_quantile)
    hot = (daily_rate > thr).astype(int).fillna(0)

    candidate = (hot.sum(axis=1) >= min_vendors).astype(int)

    runs = (candidate != candidate.shift()).cumsum()
    run_len = candidate.groupby(runs).transform("size")
    stress = ((candidate == 1) & (run_len >= min_run)).astype(int)

    return pd.DataFrame({
        "date": daily_rate.index,
        "regime": np.where(stress.values == 1, "stress", "calm"),
        "anomaly_rate_max": daily_rate.max(axis=1).values,
    })


def vendor_issue_metrics(
    anomaly_flags: pd.DataFrame,
    flag_col: str = "anomaly_flag",
    vendor_quantile: float = 0.95,
    others_quantile: float = 0.75,
    min_history: int = 252,
) -> dict:
    """
    Detect vendor-specific problems by inverting the regime logic.

    Returns a dict with:

      'cell_summary' : DataFrame indexed by vendor with
            lone_dissent_count : # of (date, cusip) cells where only this vendor flagged
            lone_dissent_rate  : count / total cells observed for this vendor
            total_flagged      : # of cells this vendor flagged at all
            concurrent_share   : share of this vendor's flags that were shared
                                 with >= 1 other vendor (low => loner)

      'bad_days' : DataFrame[date, vendor, own_rate, other_max_rate]
            Days where this vendor's daily anomaly rate exceeds its own
            expanding `vendor_quantile` while every OTHER vendor's rate is
            below its own expanding `others_quantile`. These are the
            "vendor X had a bad day" candidates.

      'bad_day_count' : Series indexed by vendor, count of bad days.

    A vendor with low `lone_dissent_rate` AND low `bad_day_count` is clean.
    A vendor that scores high on either is a problem -- different failure
    modes (per-cusip mispricing vs. system-wide event).
    """
    df = anomaly_flags.copy()
    df["date"] = pd.to_datetime(df["date"])
    df[flag_col] = df[flag_col].astype(int)

    # --- Cell-level: lone-dissenter tally ---------------------------------
    wide = df.pivot_table(
        index=["date", "cusip"], columns="vendor", values=flag_col, fill_value=0
    ).astype(int)
    n_flag = wide.sum(axis=1)
    vendors = list(wide.columns)

    cell_rows = []
    for v in vendors:
        flagged = wide[v] == 1
        lone = flagged & (n_flag == 1)
        cell_rows.append({
            "vendor": v,
            "lone_dissent_count": int(lone.sum()),
            "total_flagged": int(flagged.sum()),
            "n_cells_observed": int(wide.shape[0]),
        })
    cell_summary = pd.DataFrame(cell_rows).set_index("vendor")
    cell_summary["lone_dissent_rate"] = (
        cell_summary["lone_dissent_count"] / cell_summary["n_cells_observed"]
    )
    cell_summary["concurrent_share"] = np.where(
        cell_summary["total_flagged"] > 0,
        1 - cell_summary["lone_dissent_count"] / cell_summary["total_flagged"],
        np.nan,
    )

    # --- Day-level: vendor-specific bad days ------------------------------
    daily_rate = (
        df.groupby(["date", "vendor"])[flag_col].mean().unstack("vendor").sort_index()
    )
    own_thr = daily_rate.expanding(min_periods=min_history).quantile(vendor_quantile)
    other_thr = daily_rate.expanding(min_periods=min_history).quantile(others_quantile)

    bad_rows = []
    for v in vendors:
        own_hot = daily_rate[v] > own_thr[v]
        others = [u for u in vendors if u != v]
        # Every other vendor must be BELOW its own others_quantile threshold
        others_cool = pd.concat(
            [daily_rate[u] <= other_thr[u] for u in others], axis=1
        ).all(axis=1)
        flagged_days = own_hot & others_cool
        idx = daily_rate.index[flagged_days.fillna(False)]
        for d in idx:
            bad_rows.append({
                "date": d,
                "vendor": v,
                "own_rate": float(daily_rate.loc[d, v]),
                "other_max_rate": float(daily_rate.loc[d, others].max()),
            })
    bad_days = pd.DataFrame(bad_rows)
    bad_day_count = (
        bad_days.groupby("vendor").size().reindex(vendors).fillna(0).astype(int)
        if len(bad_days) else pd.Series(0, index=vendors, name="bad_day_count")
    )
    bad_day_count.name = "bad_day_count"

    return {
        "cell_summary": cell_summary,
        "bad_days": bad_days,
        "bad_day_count": bad_day_count,
    }


def combine_regimes(
    predefined: pd.DataFrame,
    data_driven: pd.DataFrame,
    mode: str = "predefined_plus_unnamed",
) -> pd.DataFrame:
    """
    Combine predefined and data-driven regime sources.

      'union'                   : stress if either flags it.
      'intersection'            : stress only if both agree (conservative).
      'predefined_plus_unnamed' : default. Predefined wins on its days
                                  (keeps event names); data-driven hits
                                  outside any predefined window become
                                  'stress_unnamed'.

    Returns DataFrame[date, regime, event, source].
      regime : 'calm' | 'stress'
      event  : predefined event name, or 'unnamed', or ''
      source : 'predefined' | 'data_driven' | 'both' | 'none'
    """
    pre = predefined.copy()
    dd = data_driven[["date", "regime"]].rename(columns={"regime": "regime_dd"})
    pre["date"] = pd.to_datetime(pre["date"])
    dd["date"] = pd.to_datetime(dd["date"])

    df = pre.merge(dd, on="date", how="outer").sort_values("date")
    df["regime"] = df["regime"].fillna("calm")
    df["regime_dd"] = df["regime_dd"].fillna("calm")
    df["event"] = df["event"].fillna("")

    pre_hot = (df["regime"] == "stress").values
    dd_hot = (df["regime_dd"] == "stress").values

    if mode == "union" or mode == "predefined_plus_unnamed":
        out_regime = np.where(pre_hot | dd_hot, "stress", "calm")
        out_event = np.where(pre_hot, df["event"].values,
                             np.where(dd_hot, "unnamed", ""))
    elif mode == "intersection":
        out_regime = np.where(pre_hot & dd_hot, "stress", "calm")
        out_event = np.where(pre_hot & dd_hot, df["event"].values, "")
    else:
        raise ValueError(f"unknown mode: {mode}")

    source = np.where(pre_hot & dd_hot, "both",
              np.where(pre_hot, "predefined",
              np.where(dd_hot, "data_driven", "none")))

    return pd.DataFrame({
        "date": df["date"].values,
        "regime": out_regime,
        "event": out_event,
        "source": source,
    })


def label_regimes_from_index(
    idx: pd.Series,
    quantile: float = 0.90,
    min_run: int = 5,
) -> pd.DataFrame:
    """
    Optional: regime label from a market stress series (MOVE, VIX, ...).
    Kept for completeness; not needed if you use predefined + anomaly-driven.
    """
    s = idx.sort_index()
    thr = s.expanding(min_periods=252).quantile(quantile)
    hot = (s > thr).astype(int)
    runs = (hot != hot.shift()).cumsum()
    run_len = hot.groupby(runs).transform("size")
    stress = (hot == 1) & (run_len >= min_run)
    return pd.DataFrame({
        "date": s.index,
        "regime": np.where(stress, "stress", "calm"),
    })


if __name__ == "__main__":
    # Minimal smoke test on synthetic data
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    cusips = [f"CMO{i:04d}" for i in range(20)]
    rows = []
    for cu in cusips:
        true_px = 100 + np.cumsum(rng.normal(0, 0.1, len(dates)))
        for v, noise, lag, stale in [("A", 0.05, 0, 0.02),
                                     ("B", 0.07, 1, 0.05),
                                     ("C", 0.20, 0, 0.20)]:
            p = np.roll(true_px, lag) + rng.normal(0, noise, len(dates))
            mask = rng.random(len(dates)) < stale
            p = pd.Series(p).where(~mask).ffill().values
            for d, pr in zip(dates, p):
                rows.append({"date": d, "cusip": cu, "vendor": v, "price": pr})
    px = pd.DataFrame(rows)

    # Fake TRACE: 5% of (date, cusip) trade at true_px + small noise
    trades = (
        px[px["vendor"] == "A"]
        .sample(frac=0.05, random_state=1)
        .assign(trade_price=lambda d: d["price"] + rng.normal(0, 0.03, len(d)),
                trade_size=lambda d: rng.integers(1e5, 1e7, len(d)))
        [["date", "cusip", "trade_price", "trade_size"]]
    )

    # Synthesize per-(date, cusip, vendor) anomaly flags: deviation from
    # cross-vendor median exceeding 3x its rolling 251d MAD.
    wide = px.pivot_table(index=["date", "cusip"], columns="vendor", values="price")
    med = wide.median(axis=1)
    dev = wide.sub(med, axis=0).abs()
    flags = []
    for v in wide.columns:
        roll_mad = dev[v].groupby(level="cusip").transform(
            lambda s: s.rolling(251, min_periods=60).median()
        )
        flag = (dev[v] > 3 * roll_mad).astype(int)
        flags.append(flag.rename("anomaly_flag").reset_index().assign(vendor=v))
    anomaly_flags = pd.concat(flags, ignore_index=True)

    # Regime building
    pre = label_regimes_predefined(px["date"].unique())
    dd = label_regimes_from_anomalies(anomaly_flags)
    regimes = combine_regimes(pre, dd, mode="predefined_plus_unnamed")

    print("Regime composition:")
    print(regimes.groupby(["regime", "source"]).size().unstack(fill_value=0))
    print("\nStress days by event (top 10):")
    print(regimes[regimes["regime"] == "stress"]["event"].value_counts().head(10))

    # Vendor-issue diagnostics
    vi = vendor_issue_metrics(anomaly_flags)
    print("\nVendor-issue cell summary:")
    print(vi["cell_summary"].round(4))
    print("\nVendor bad-day count:")
    print(vi["bad_day_count"])

    sc = build_scorecard(px, trades=trades,
                         regimes=regimes[["date", "regime"]],
                         anomaly_flags=anomaly_flags)
    print("\nScorecard:")
    print(sc.filter(regex="^(all_|stress_|calm_|zero_|ret_|lone_|bad_|solo_|overall_)").round(4))
    print("\nFriedman:", friedman_on_panel(px))
