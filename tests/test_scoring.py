from datetime import date

import pandas as pd
import pytest

from src.config import load_settings
from src.score import confidence, coverage, gate, recency, req_score, role_fit, score_requirement

CFG = load_settings().scoring
REF = date(2026, 10, 1)


@pytest.mark.parametrize("ev,req,expected", [(0, 3, 0.0), (2, 4, 0.5), (3, 3, 1.0), (4, 3, 1.0), (4, 2, 1.0)])
def test_coverage_is_ratio_capped_at_one(ev, req, expected):
    assert coverage(ev, req) == pytest.approx(expected)


def test_recency_decays_geometrically():
    assert recency(0, 0.85) == 1.0
    assert recency(1, 0.85) == pytest.approx(0.85)
    assert recency(3, 0.85) == pytest.approx(0.85 ** 3)
    assert recency(-1, 0.85) == 1.0  # future-dated evidence is not rewarded


@pytest.mark.parametrize("n,expected", [(0, 0.0), (1, 0.5), (2, 0.75), (3, 1.0), (7, 1.0)])
def test_confidence_by_independent_sources(n, expected):
    assert confidence(n, 0.5, 0.25) == pytest.approx(expected)


def test_req_score_is_product():
    assert req_score(0.5, 0.85, 0.75) == pytest.approx(0.5 * 0.85 * 0.75)


def test_role_fit_is_weighted_mean():
    assert role_fit([5, 1], [1.0, 0.0]) == pytest.approx(5 / 6)
    assert role_fit([2, 2, 4], [0.5, 1.0, 0.25]) == pytest.approx((1 + 2 + 1) / 8)
    assert role_fit([], []) == 0.0


def test_gate_only_checks_must_haves():
    assert gate([True, False], [1.0, 0.2])
    assert not gate([True, True], [1.0, 0.99])
    assert gate([False, False], [0.0, 0.0])


def _ev(rows):
    return pd.DataFrame(rows, columns=["evidence_id", "level", "date", "source"])


def test_score_requirement_uses_evidence_meeting_required_level():
    ev = _ev([("E1", 3, "2026-04-01", "D1"), ("E2", 4, "2025-10-01", "D2"),
              ("E3", 4, "2024-10-01", "D3"), ("E4", 2, "2026-09-30", "D4")])
    res = score_requirement(ev, required_level=3, ref=REF, cfg=CFG)
    assert res["evidence_level"] == 4
    assert res["coverage"] == 1.0
    assert res["independent_sources"] == 3  # D1, D2, D3 meet level 3; D4 does not
    assert res["years_since"] == pytest.approx(0.5, abs=0.01)  # most recent supporting item: 2026-04-01
    assert set(res["evidence_ids"].split(";")) == {"E1", "E2", "E3"}
    expected = 1.0 * CFG.recency_decay ** ((REF - date(2026, 4, 1)).days / 365.25) * 1.0
    assert res["req_score"] == pytest.approx(expected)


def test_score_requirement_below_required_uses_best_level():
    ev = _ev([("E1", 2, "2025-10-01", "D1"), ("E2", 2, "2025-10-01", "D1"), ("E3", 1, "2026-09-01", "D2")])
    res = score_requirement(ev, required_level=4, ref=REF, cfg=CFG)
    assert res["coverage"] == 0.5
    assert res["independent_sources"] == 1  # two quotes, same document -> one source
    assert res["confidence"] == 0.5


def test_score_requirement_without_evidence_is_zero():
    res = score_requirement(_ev([]), required_level=3, ref=REF, cfg=CFG)
    assert res["req_score"] == 0.0 and res["coverage"] == 0.0 and res["independent_sources"] == 0
