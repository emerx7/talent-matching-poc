from src.config import Scoring
from src.score import LONGER, READY_NOW, READY_SOON, Gap, is_closable, readiness

CFG = Scoring(recency_decay=0.85, confidence_base=0.5, confidence_step=0.25, ready_now_min_fit=0.8,
              max_gaps_for_developing=2, max_closable_level_gap=1, thin_bench_max_ready=1)

SMALL = Gap("C01", 1, True)


def test_ready_now_at_exact_threshold():
    assert readiness(True, 0.8, [], CFG) == READY_NOW


def test_just_below_threshold_is_not_ready_now():
    assert readiness(True, 0.7999, [], CFG) == READY_SOON  # no gaps, evidence just needs refreshing


def test_gate_failure_blocks_ready_now_even_with_high_fit():
    assert readiness(False, 0.95, [SMALL], CFG) == READY_SOON


def test_two_closable_gaps_is_ready_soon():
    assert readiness(False, 0.5, [SMALL, Gap("C02", 1, True)], CFG) == READY_SOON


def test_three_gaps_is_longer_term():
    assert readiness(False, 0.5, [SMALL, Gap("C02", 1, True), Gap("C03", 1, True)], CFG) == LONGER


def test_gap_too_large_is_longer_term():
    assert readiness(False, 0.6, [Gap("C02", 2, True)], CFG) == LONGER


def test_gap_without_development_opportunity_is_longer_term():
    assert readiness(False, 0.6, [Gap("C02", 1, False)], CFG) == LONGER


def test_is_closable():
    assert is_closable(SMALL, 1)
    assert not is_closable(Gap("C01", 2, True), 1)
    assert is_closable(Gap("C01", 2, True), 2)
    assert not is_closable(Gap("C01", 0, True), 1)
