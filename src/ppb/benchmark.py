"""Peer benchmarks, savings opportunity and commodity pass-through."""
import numpy as np
import pandas as pd

from .simulate import CATEGORIES


def peer_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """P25 / median / P75 unit price per category and month across the community."""
    q = df.groupby(["category", "month"])["unit_price"].quantile([0.25, 0.5, 0.75]).unstack()
    q.columns = ["p25", "median", "p75"]
    return q.reset_index()


def price_position(df: pd.DataFrame, period: slice) -> pd.DataFrame:
    """Each company's average price vs the peer median over a period (100 = median)."""
    sub = df[df["month"].between(period.start, period.stop)]
    med = sub.groupby(["category", "month"])["unit_price"].transform("median")
    sub = sub.assign(rel=sub["unit_price"] / med * 100)
    out = sub.groupby(["company", "category"]).agg(rel_price=("rel", "mean"), spend=("spend", "sum")).reset_index()
    return out


def savings_to_p25(df: pd.DataFrame, period: slice) -> pd.DataFrame:
    """Spend saved per category if every buyer above the peer P25 had paid the P25 price."""
    sub = df[df["month"].between(period.start, period.stop)].copy()
    p25 = sub.groupby(["category", "month"])["unit_price"].transform(lambda s: s.quantile(0.25))
    sub["saving"] = np.clip(sub["unit_price"] - p25, 0, None) * sub["volume"]
    g = sub.groupby("category").agg(spend=("spend", "sum"), saving=("saving", "sum"))
    g["saving_pct"] = g["saving"] / g["spend"] * 100
    return g.sort_values("saving_pct", ascending=False)


def brent_elasticity(df: pd.DataFrame, market_idx: pd.DataFrame, max_lag: int = 9) -> pd.DataFrame:
    """For each category, find the lag at which median price best tracks Brent and the elasticity.

    Elasticity is the slope of log(median price) on log(Brent index) at the best lag:
    a 0.4 means a 10% move in oil shows up as roughly a 4% move in the purchase price.
    """
    med = df.groupby(["category", "month"])["unit_price"].median().unstack(0)
    rows = []
    for cat in med.columns:
        best = None
        for lag in range(max_lag + 1):
            x = np.log(market_idx["brent"].shift(lag)).dropna()
            y = np.log(med[cat]).reindex(x.index)
            slope, _ = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            if best is None or r > best[2]:
                best = (lag, slope, r)
        rows.append(dict(category=cat, best_lag_months=best[0], elasticity=round(best[1], 2), correlation=round(best[2], 2)))
    return pd.DataFrame(rows).sort_values("elasticity", ascending=False)


def shock_impact(market_idx: pd.DataFrame, before: str, after: str, spend_mix: dict, total_spend_m: float = 100) -> pd.DataFrame:
    """Should-cost impact of a market move on a company with a given category spend mix (EUR m).

    ``market_idx`` must be rebased (base month = 100). The cost index of each category is
    rebuilt from its weights at both dates, so the result is the full should-cost change.
    """
    rows = []
    for cat, share in spend_mix.items():
        w = CATEGORIES[cat]

        def cost(month):
            m = market_idx.loc[month]
            return w["brent"] * m["brent"] + w["gas"] * m["gas"] + w["cpi"] * m["cpi"] + w["fixed"] * 100

        pct = cost(after) / cost(before) - 1
        spend = total_spend_m * share
        rows.append(dict(category=cat, spend_m=spend, cost_change_pct=pct * 100, cost_change_m=spend * pct))
    return pd.DataFrame(rows).sort_values("cost_change_m", ascending=False)
