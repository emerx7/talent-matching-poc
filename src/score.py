"""Transparent, rule-based scoring of officer x role.

Per requirement (all constants from config.toml [scoring]):
    coverage   = min(evidence_level / required_level, 1.0)
    recency    = decay ** years_since_evidence
    confidence = min(1.0, base + step * (independent_sources - 1))
    req_score  = coverage * recency * confidence
role_fit = sum(weight * req_score) / sum(weight)
gate     = every must-have requirement has coverage >= 1.0

Which evidence backs a requirement: if any evidence meets the required level, the
supporting set is all evidence at >= required level; otherwise it is the evidence at the
highest level shown. Recency uses the most recent item in the supporting set, and
independent_sources counts the distinct documents in it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.config import Scoring, load_settings

READY_NOW = "Ready now"
READY_SOON = "Ready in 1–2 years"
LONGER = "Longer-term"
BANDS = [READY_NOW, READY_SOON, LONGER]


# ---------------------------------------------------------------------------
# Pure functions (unit-tested)
# ---------------------------------------------------------------------------
def coverage(evidence_level: int, required_level: int) -> float:
    if required_level <= 0:
        return 1.0
    return min(evidence_level / required_level, 1.0)


def recency(years_since: float, decay: float) -> float:
    return decay ** max(years_since, 0.0)


def confidence(independent_sources: int, base: float, step: float) -> float:
    if independent_sources <= 0:
        return 0.0
    return min(1.0, base + step * (independent_sources - 1))


def req_score(cov: float, rec: float, conf: float) -> float:
    return cov * rec * conf


def role_fit(weights: list[float], scores: list[float]) -> float:
    total = sum(weights)
    return sum(w * s for w, s in zip(weights, scores)) / total if total else 0.0


def gate(must_have: list[bool], coverages: list[float]) -> bool:
    return all(c >= 1.0 for mh, c in zip(must_have, coverages) if mh)


@dataclass
class Gap:
    competency_id: str
    level_gap: int
    has_opportunity: bool


def is_closable(g: Gap, max_level_gap: int) -> bool:
    return g.has_opportunity and 0 < g.level_gap <= max_level_gap


def readiness(gate_passed: bool, fit: float, gaps: list[Gap], cfg: Scoring) -> str:
    if gate_passed and fit >= cfg.ready_now_min_fit:
        return READY_NOW
    if len(gaps) <= cfg.max_gaps_for_developing and all(is_closable(g, cfg.max_closable_level_gap) for g in gaps):
        return READY_SOON
    return LONGER


def years_between(d: date, ref: date) -> float:
    return max((ref - d).days / 365.25, 0.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def score_requirement(ev: pd.DataFrame, required_level: int, ref: date, cfg: Scoring) -> dict:
    """ev: evidence rows for one officer x competency (may be empty)."""
    if ev.empty:
        return {"evidence_level": 0, "coverage": 0.0, "years_since": None, "recency": 0.0,
                "independent_sources": 0, "confidence": 0.0, "req_score": 0.0, "evidence_ids": ""}
    max_lvl = int(ev["level"].max())
    support = ev[ev["level"] >= required_level] if max_lvl >= required_level else ev[ev["level"] == max_lvl]
    latest = max(date.fromisoformat(d) for d in support["date"])
    yrs = years_between(latest, ref)
    n_src = support["source"].nunique()
    cov = coverage(max_lvl, required_level)
    rec = recency(yrs, cfg.recency_decay)
    conf = confidence(n_src, cfg.confidence_base, cfg.confidence_step)
    return {"evidence_level": max_lvl, "coverage": cov, "years_since": round(yrs, 2), "recency": rec,
            "independent_sources": n_src, "confidence": conf, "req_score": req_score(cov, rec, conf),
            "evidence_ids": ";".join(support["evidence_id"])}


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = load_settings()
    cfg, ref = s.scoring, s.general.reference_date
    evidence = pd.read_csv(s.outputs_dir / "evidence.csv")
    officers = pd.read_csv(s.data_dir / "officers.csv")
    reqs = pd.read_csv(s.data_dir / "role_requirements.csv")
    opps = pd.read_csv(s.data_dir / "opportunities.csv")

    # Highest level reachable through development, per competency.
    opp_max: dict[str, int] = {}
    for o in opps.itertuples():
        for c in o.competencies.split(";"):
            opp_max[c] = max(opp_max.get(c, 0), int(o.max_level))

    ev_groups = {k: g for k, g in evidence.groupby(["officer_id", "competency_id"])}
    empty = evidence.iloc[0:0]
    req_rows, fit_rows = [], []
    for oid in officers["officer_id"]:
        for rid, rr in reqs.groupby("role_id"):
            rows = []
            for r in rr.itertuples():
                res = score_requirement(ev_groups.get((oid, r.competency_id), empty), r.required_level, ref, cfg)
                level_gap = max(r.required_level - res["evidence_level"], 0)
                has_opp = opp_max.get(r.competency_id, 0) >= r.required_level
                rows.append({"officer_id": oid, "role_id": rid, "competency_id": r.competency_id,
                             "required_level": r.required_level, "weight": r.weight,
                             "must_have": bool(r.must_have), **res,
                             "is_gap": res["coverage"] < 1.0, "level_gap": level_gap,
                             "closable": is_closable(Gap(r.competency_id, level_gap, has_opp),
                                                     cfg.max_closable_level_gap)})
            req_rows.extend(rows)
            fit = role_fit([x["weight"] for x in rows], [x["req_score"] for x in rows])
            passed = gate([x["must_have"] for x in rows], [x["coverage"] for x in rows])
            gaps = [Gap(x["competency_id"], x["level_gap"],
                        opp_max.get(x["competency_id"], 0) >= x["required_level"]) for x in rows if x["is_gap"]]
            fit_rows.append({"officer_id": oid, "role_id": rid, "role_fit": round(fit, 4), "gate_passed": passed,
                             "n_gaps": len(gaps),
                             "n_closable_gaps": sum(is_closable(g, cfg.max_closable_level_gap) for g in gaps),
                             "readiness": readiness(passed, fit, gaps, cfg)})

    req_df, fit_df = pd.DataFrame(req_rows), pd.DataFrame(fit_rows)
    req_df.to_csv(s.outputs_dir / "requirement_scores.csv", index=False)
    fit_df.to_csv(s.outputs_dir / "role_fit.csv", index=False)
    bench = fit_df.pivot_table(index="role_id", columns="readiness", values="officer_id", aggfunc="count",
                               fill_value=0).reindex(columns=BANDS, fill_value=0)
    print("Bench strength (officers per readiness band):")
    print(bench.to_string())
    return req_df, fit_df


if __name__ == "__main__":
    run()
