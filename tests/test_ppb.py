import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppb.benchmark import brent_elasticity, peer_quartiles, savings_to_p25, shock_impact  # noqa: E402
from ppb.market import load_market, rebase  # noqa: E402
from ppb.simulate import CATEGORIES, simulate_community  # noqa: E402


@pytest.fixture(scope="module")
def idx():
    return rebase(load_market())


@pytest.fixture(scope="module")
def community(idx):
    return simulate_community(idx, n_companies=20, seed=1)


def test_rebase_starts_at_100(idx):
    assert (idx.iloc[0].round(6) == 100).all()


def test_category_weights_sum_to_one():
    for w in CATEGORIES.values():
        assert w["brent"] + w["gas"] + w["cpi"] + w["fixed"] == pytest.approx(1.0)


def test_quartiles_are_ordered(community):
    q = peer_quartiles(community)
    assert (q.p25 <= q["median"]).all() and (q["median"] <= q.p75).all()


def test_savings_are_non_negative_and_below_spend(community):
    s = savings_to_p25(community, slice(pd.Period("2026-01"), pd.Period("2026-10")))
    assert (s.saving >= 0).all() and (s.saving < s.spend).all()


def test_oil_heavy_category_tracks_brent_more_than_services(community, idx):
    e = brent_elasticity(community, idx).set_index("category")
    assert e.loc["Ocean freight", "elasticity"] > e.loc["Facility services", "elasticity"]


def test_no_market_move_means_no_impact(idx):
    out = shock_impact(idx, "2026-10", "2026-10", {"Ocean freight": 1.0})
    assert out.cost_change_m.iloc[0] == pytest.approx(0)
