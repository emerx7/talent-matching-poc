"""Fairness audit — the ONLY module that creates or reads the hidden group attribute.

1. A synthetic A/B attribute is written to data/_audit/group_attributes.csv with its own
   seed. Scoring and extraction never see it; the app shows only the aggregated report.
2. Adverse impact ratio (AIR) of "Ready now" shortlists per role: lower selection rate /
   higher selection rate, flagged when below the four-fifths threshold.
3. Counterfactual test: swap each document's officer name and gendered pronouns, re-run
   extraction, and check the extracted (competency, level) evidence is unchanged.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter

import pandas as pd

from src.config import load_settings
from src.extract import choose_mode, extract_validated, make_extractor
from src.generate_data import make_name_pool
from src.score import READY_NOW

# ---------------------------------------------------------------------------
# Hidden attribute
# ---------------------------------------------------------------------------


def generate_attributes(officer_ids: list[str], seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    return pd.DataFrame({"officer_id": officer_ids, "group": [rng.choice("AB") for _ in officer_ids]})


# ---------------------------------------------------------------------------
# Adverse impact ratio
# ---------------------------------------------------------------------------
def adverse_impact_ratio(rate_a: float, rate_b: float) -> float | None:
    """min(rate)/max(rate); None when nobody in either group was selected (ratio undefined)."""
    hi = max(rate_a, rate_b)
    if hi == 0:
        return None
    return min(rate_a, rate_b) / hi


def air_table(fit: pd.DataFrame, groups: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = fit.merge(groups, on="officer_id")
    df["selected"] = df["readiness"] == READY_NOW
    rows = []
    for rid, g in list(df.groupby("role_id")) + [("ALL", df)]:
        stats = {}
        for grp in "AB":
            sub = g[g["group"] == grp]
            stats[grp] = (len(sub), int(sub["selected"].sum()))
        rate_a = stats["A"][1] / stats["A"][0] if stats["A"][0] else 0.0
        rate_b = stats["B"][1] / stats["B"][0] if stats["B"][0] else 0.0
        air = adverse_impact_ratio(rate_a, rate_b)
        total_sel = stats["A"][1] + stats["B"][1]
        if air is None:
            note = "No 'Ready now' selections - ratio undefined"
        elif total_sel < 5:
            note = "Fewer than 5 selections - ratio is highly unstable"
        else:
            note = ""
        rows.append({"role_id": rid, "n_A": stats["A"][0], "selected_A": stats["A"][1], "rate_A": round(rate_a, 4),
                     "n_B": stats["B"][0], "selected_B": stats["B"][1], "rate_B": round(rate_b, 4),
                     "air": None if air is None else round(air, 3),
                     "flag": air is not None and air < threshold, "note": note})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Counterfactual name / pronoun swap
# ---------------------------------------------------------------------------
_PRONOUN_RE = re.compile(r"\b(he|she|they|his|her|hers|their|theirs|him|them)\b", re.I)
_SIMPLE_SWAP = {"he": "she", "she": "he", "they": "he", "his": "her", "their": "his", "theirs": "his",
                "hers": "his", "him": "her", "them": "him"}
# Words after "her" that mean it is an object pronoun ("thanked her for ..."), not possessive.
_NOT_A_NOUN = {"for", "to", "on", "in", "at", "with", "and", "as", "from", "by", "during", "after", "before",
               "through", "that", "about", "over", "when", "because", "but", "or", "so", "again"}


def _match_case(src: str, word: str) -> str:
    return word.capitalize() if src[0].isupper() else word


def swap_pronouns(text: str) -> str:
    def repl(m: re.Match) -> str:
        w = m.group(0)
        low = w.lower()
        if low == "her":
            nxt = re.match(r"\s*([A-Za-z'-]+)", text[m.end():])
            possessive = bool(nxt) and nxt.group(1).lower() not in _NOT_A_NOUN \
                and not re.match(r"\s*[.,;:!?]", text[m.end():])
            return _match_case(w, "his" if possessive else "him")
        return _match_case(w, _SIMPLE_SWAP[low])

    return _PRONOUN_RE.sub(repl, text)


def swap_name(text: str, old_full: str, new_full: str) -> str:
    old_first, new_first = old_full.split()[0], new_full.split()[0]
    text = text.replace(old_full, new_full)
    return re.sub(rf"\b{re.escape(old_first)}\b", new_first, text)


def counterfactual(text: str, old_name: str, new_name: str) -> str:
    return swap_pronouns(swap_name(text, old_name, new_name))


def evidence_signature(rows: list[dict]) -> Counter:
    return Counter((r["competency_id"], int(r["level"])) for r in rows)


def counterfactual_test(extractor, docs: list[tuple[str, str, str]], new_names: list[str]) -> dict:
    """docs: (doc_id, text, officer_name). Returns summary + per-doc differences."""
    diffs, unchanged = [], 0
    for (doc_id, text, name), new_name in zip(docs, new_names):
        orig, _ = extract_validated(extractor, text)
        swapped_text = counterfactual(text, name, new_name)
        swapped, _ = extract_validated(extractor, swapped_text)
        a, b = evidence_signature(orig), evidence_signature(swapped)
        if a == b:
            unchanged += 1
        else:
            diffs.append({"doc_id": doc_id, "only_original": sorted(map(list, (a - b).elements())),
                          "only_swapped": sorted(map(list, (b - a).elements()))})
    n = len(docs)
    return {"docs_tested": n, "unchanged": unchanged, "pct_unchanged": round(100 * unchanged / n, 1) if n else None,
            "differences": diffs[:25]}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(mode: str | None = None, limit: int | None = None) -> dict:
    s = load_settings()
    officers = pd.read_csv(s.data_dir / "officers.csv")
    s.audit_dir.mkdir(parents=True, exist_ok=True)
    groups = generate_attributes(officers["officer_id"].tolist(), s.fairness.attribute_seed)
    groups.to_csv(s.audit_dir / "group_attributes.csv", index=False)

    fit = pd.read_csv(s.outputs_dir / "role_fit.csv")
    air = air_table(fit, groups, s.fairness.air_threshold)
    air.to_csv(s.outputs_dir / "fairness_air.csv", index=False)

    mode = choose_mode(mode)
    extractor = make_extractor(mode)
    n = s.fairness.counterfactual_sample_llm if mode == "llm" else s.fairness.counterfactual_sample
    if limit:
        n = min(n, limit)
    docs_idx = pd.read_csv(s.data_dir / "documents_index.csv")
    sample = docs_idx.sample(n=min(n, len(docs_idx)), random_state=s.fairness.attribute_seed)
    names = dict(zip(officers["officer_id"], officers["name"]))
    docs = [(d.doc_id, (s.data_dir / d.path).read_text(encoding="utf-8"), names[d.officer_id])
            for d in sample.itertuples()]
    new_names = make_name_pool(random.Random(s.fairness.attribute_seed), len(docs), exclude=set(officers["name"]))
    cf = counterfactual_test(extractor, docs, new_names)
    cf["extractor"] = mode

    report = {"air_threshold": s.fairness.air_threshold, "air": air.to_dict(orient="records"),
              "roles_flagged": air.loc[air["flag"], "role_id"].tolist(), "counterfactual": cf}
    (s.outputs_dir / "fairness_report.json").write_text(json.dumps(report, indent=2))
    print(f"Fairness: AIR flagged roles = {report['roles_flagged'] or 'none'}; counterfactual "
          f"{cf['unchanged']}/{cf['docs_tested']} docs unchanged ({cf['pct_unchanged']}%)")
    return report


if __name__ == "__main__":
    run()
