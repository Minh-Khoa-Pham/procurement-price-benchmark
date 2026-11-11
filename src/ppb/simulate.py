"""Generate an anonymised, synthetic 'community' dataset of purchase prices.

Each category's unit price follows a simple should-cost model: a weighted mix of
energy (Brent), gas, general inflation (CPI) and a fixed component. Companies
differ in three ways a procurement team controls: their negotiated price level,
how quickly their contracts pass index changes through (lag in months), and
whether their contracts are index-linked at all.
"""
import numpy as np
import pandas as pd

# cost structure per category: weights on (brent, gas, cpi, fixed)
CATEGORIES = {
    "Ocean freight": dict(brent=0.55, gas=0.00, cpi=0.15, fixed=0.30, base_price=1800.0, unit="EUR/TEU"),
    "Road freight": dict(brent=0.35, gas=0.00, cpi=0.35, fixed=0.30, base_price=1.60, unit="EUR/km"),
    "Packaging": dict(brent=0.20, gas=0.25, cpi=0.20, fixed=0.35, base_price=0.90, unit="EUR/kg"),
    "Industrial chemicals": dict(brent=0.40, gas=0.20, cpi=0.10, fixed=0.30, base_price=1.20, unit="EUR/kg"),
    "Facility services": dict(brent=0.00, gas=0.05, cpi=0.70, fixed=0.25, base_price=45.0, unit="EUR/hour"),
}
LAGS = [0, 3, 6]  # months of contractual pass-through lag


def simulate_community(market_idx: pd.DataFrame, n_companies: int = 60, seed: int = 7) -> pd.DataFrame:
    """Return long-format monthly purchase prices and volumes per company and category."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_companies + 1):
        company = f"C{c:03d}"
        annual_spend_m = rng.lognormal(mean=np.log(80), sigma=0.6)  # EUR millions
        for cat, p in CATEGORIES.items():
            premium = rng.normal(1.0, 0.07)  # negotiated price level vs market
            lag = int(rng.choice(LAGS, p=[0.3, 0.45, 0.25]))
            indexed = rng.random() > 0.2  # 20% buy at fixed prices, renegotiated yearly
            cost_idx = (
                p["brent"] * market_idx["brent"]
                + p["gas"] * market_idx["gas"]
                + p["cpi"] * market_idx["cpi"]
                + p["fixed"] * 100
            ) / 100
            if indexed:
                driver = cost_idx.shift(lag).bfill()
            else:  # fixed annual price set from the previous year's average
                yearly = cost_idx.groupby(cost_idx.index.year).mean().shift(1).bfill()
                driver = pd.Series(yearly.loc[cost_idx.index.year].values, index=cost_idx.index)
            noise = rng.normal(0, 0.015, len(driver))
            price = p["base_price"] * premium * driver.values * (1 + noise)
            share = rng.dirichlet(np.ones(len(CATEGORIES)))[list(CATEGORIES).index(cat)]
            monthly_spend = annual_spend_m * 1e6 * max(share, 0.05) / 12
            volume = monthly_spend / price * rng.normal(1, 0.05, len(price))
            for month, pr, vol in zip(driver.index, price, volume):
                rows.append((company, cat, month, pr, vol, lag, indexed))
    df = pd.DataFrame(rows, columns=["company", "category", "month", "unit_price", "volume", "lag_months", "index_linked"])
    df["spend"] = df["unit_price"] * df["volume"]
    return df
