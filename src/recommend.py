"""Suggest development opportunities for each gap.

Rank = tag_weight * tag_match + embedding_weight * cosine(gap need, opportunity description),
halved when the opportunity cannot take someone to the required level.
Without embeddings, ranking falls back to tag match + level fit.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_settings
from src.embeddings import cosine, embed


def rank_opportunities(competencies: pd.DataFrame, opps: pd.DataFrame, top_k: int, tag_w: float,
                       emb_w: float, use_embeddings: bool = True) -> pd.DataFrame:
    """Top-k opportunities for every (competency, required_level) pair."""
    comp = competencies.set_index("competency_id")
    keys = [(cid, lvl) for cid in comp.index for lvl in range(1, 5)]
    needs = [f"{comp.at[cid, 'description']} Target: {comp.at[cid, f'level_{lvl}']}" for cid, lvl in keys]
    need_emb = embed(needs) if use_embeddings else None
    opp_emb = embed(opps["description"].tolist()) if use_embeddings else None
    sims = cosine(need_emb, opp_emb) if need_emb is not None and opp_emb is not None else None

    tags = [set(t.split(";")) for t in opps["competencies"]]
    rows = []
    for i, (cid, lvl) in enumerate(keys):
        scored = []
        for j, o in enumerate(opps.itertuples()):
            tag = 1.0 if cid in tags[j] else 0.0
            sim = float(sims[i, j]) if sims is not None else 0.0
            score = tag_w * tag + (emb_w * sim if sims is not None else 0.0)
            if o.max_level < lvl:
                score *= 0.5
            scored.append((score, o.opportunity_id, tag, sim))
        scored.sort(key=lambda x: -x[0])
        for rank, (score, oid, tag, sim) in enumerate(scored[:top_k], 1):
            rows.append({"competency_id": cid, "required_level": lvl, "rank": rank, "opportunity_id": oid,
                         "score": round(score, 3), "tag_match": bool(tag), "similarity": round(sim, 3)})
    return pd.DataFrame(rows)


def run(use_embeddings: bool = True) -> pd.DataFrame:
    s = load_settings()
    comps = pd.read_csv(s.data_dir / "competencies.csv")
    opps = pd.read_csv(s.data_dir / "opportunities.csv")
    reqs = pd.read_csv(s.outputs_dir / "requirement_scores.csv")
    ranking = rank_opportunities(comps, opps, s.recommend.top_k, s.recommend.tag_weight,
                                 s.recommend.embedding_weight, use_embeddings)
    gaps = reqs.loc[reqs["is_gap"], ["officer_id", "role_id", "competency_id", "required_level"]]
    recs = gaps.merge(ranking, on=["competency_id", "required_level"])
    recs = recs.merge(opps[["opportunity_id", "type", "title"]], on="opportunity_id")
    recs = recs.sort_values(["officer_id", "role_id", "competency_id", "rank"])
    recs.to_csv(s.outputs_dir / "recommendations.csv", index=False)
    print(f"Recommendations: {len(recs)} rows for {len(gaps)} gaps")
    return recs


if __name__ == "__main__":
    run()
