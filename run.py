"""Run the full analysis: build the dataset, benchmark it, save figures and the insight report."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ppb.benchmark import brent_elasticity, peer_quartiles, price_position, savings_to_p25, shock_impact  # noqa: E402
from ppb.market import load_market, rebase  # noqa: E402
from ppb.simulate import simulate_community  # noqa: E402

ROOT = Path(__file__).parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
BLUE, GREY, RED = "#1F4E79", "#9AA5B1", "#C0392B"

market = load_market()
idx = rebase(market)
community = simulate_community(idx)
community.to_csv(ROOT / "data" / "community_prices_synthetic.csv", index=False)

# 1. market context
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(idx.index.to_timestamp(), idx["brent"], color=RED, label="Brent crude")
ax.plot(idx.index.to_timestamp(), idx["gas"], color=GREY, label="Henry Hub gas")
ax.plot(idx.index.to_timestamp(), idx["cpi"], color=BLUE, label="US CPI")
ax.axhline(100, color="black", lw=0.5)
ax.set_title("Market drivers, index (Jan 2022 = 100)", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=3)
fig.tight_layout(); fig.savefig(FIG / "01_market_drivers.png"); plt.close(fig)

# 2. pass-through by category
elast = brent_elasticity(community, idx)

# 3. peer benchmark band for ocean freight + two example buyers
q = peer_quartiles(community)
cat = "Ocean freight"
qc = q[q.category == cat]
t = qc["month"].dt.to_timestamp()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.fill_between(t, qc.p25, qc.p75, color=BLUE, alpha=0.15, label="Peer P25 to P75")
ax.plot(t, qc["median"], color=BLUE, label="Peer median")
pos = price_position(community, slice(pd.Period("2026-03"), pd.Period("2026-10")))
oc = pos[pos.category == cat].sort_values("rel_price")
for comp, col in [(oc.iloc[-1].company, RED), (oc.iloc[0].company, "#27AE60")]:
    s = community[(community.company == comp) & (community.category == cat)]
    ax.plot(s.month.dt.to_timestamp(), s.unit_price, color=col, lw=1.2, label=f"Buyer {comp}")
ax.set_title(f"{cat}: price paid vs peers (EUR/TEU)", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "02_ocean_freight_benchmark.png"); plt.close(fig)

# 4. savings opportunity during the shock period
shock = slice(pd.Period("2026-03"), pd.Period("2026-10"))
sav = savings_to_p25(community, shock)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(sav.index[::-1], sav["saving_pct"][::-1], color=BLUE)
for i, v in enumerate(sav["saving_pct"][::-1]):
    ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_title("Spend saved if buyers above P25 had paid the P25 price (Mar to Oct 2026)", loc="left", fontweight="bold", fontsize=10)
ax.set_xlabel("% of category spend")
fig.tight_layout(); fig.savefig(FIG / "03_savings_to_p25.png"); plt.close(fig)

# 5. contract design: lag vs price paid during the shock
lagged = community[community.month.between(shock.start, shock.stop)].copy()
lagged["rel"] = lagged.unit_price / lagged.groupby(["category", "month"]).unit_price.transform("median") * 100
by_terms = (
    lagged[lagged.category.isin(["Ocean freight", "Industrial chemicals", "Road freight"])]
    .assign(terms=lambda d: d.apply(lambda r: f"Indexed, {r.lag_months}-month lag" if r.index_linked else "Fixed annual price", axis=1))
    .groupby("terms").rel.mean().sort_values()
)
fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(by_terms.index, by_terms.values - 100, color=[("#27AE60" if v < 100 else RED) for v in by_terms.values])
ax.axvline(0, color="black", lw=0.6)
ax.set_title("Oil-exposed categories: price vs peer median by contract terms (Mar to Oct 2026)", loc="left", fontweight="bold", fontsize=10)
ax.set_xlabel("% above (+) or below (-) peer median")
fig.tight_layout(); fig.savefig(FIG / "04_contract_terms.png"); plt.close(fig)

# 6. impact of the oil shock on a typical EUR 100m spend
mix = {"Ocean freight": 0.25, "Road freight": 0.20, "Packaging": 0.20, "Industrial chemicals": 0.20, "Facility services": 0.15}
impact = shock_impact(idx, "2025-10", "2026-10", mix)

brent_move = market.loc["2026-10", "brent"] / market.loc["2025-10", "brent"] - 1
summary = {
    "brent_2025_10": market.loc["2025-10", "brent"],
    "brent_2026_10": market.loc["2026-10", "brent"],
    "brent_move_pct": brent_move * 100,
    "impact_total_m": impact.cost_change_m.sum(),
}
pd.Series(summary).to_csv(ROOT / "reports" / "summary.csv")
elast.to_csv(ROOT / "reports" / "pass_through.csv", index=False)
sav.to_csv(ROOT / "reports" / "savings_to_p25.csv")
impact.to_csv(ROOT / "reports" / "shock_impact.csv", index=False)
by_terms.to_csv(ROOT / "reports" / "contract_terms.csv")
print(pd.Series(summary).round(1), elast, sav.round(2), impact.round(2), by_terms.round(1), sep="\n\n")
