"""Extraction quality against the 30 labelled documents (data/ground_truth.csv).

A predicted row matches a labelled sentence when the competency is the same and the quote
overlaps the sentence (one contains the other). Two views are reported:
  * competency  — right competency, any level
  * strict      — right competency AND right level
"""
from __future__ import annotations

import pandas as pd

from src.config import load_settings
from src.extract import choose_mode, extract_validated, make_extractor


def _overlaps(quote: str, sentence: str) -> bool:
    return quote in sentence or sentence in quote


def match_counts(pred: list[dict], gold: list[dict], strict: bool) -> dict[str, list[int]]:
    """Greedy one-to-one matching. Returns {competency_id: [tp, fp, fn]}."""
    counts: dict[str, list[int]] = {}
    used = set()
    for p in pred:
        hit = None
        for i, g in enumerate(gold):
            if i in used or g["competency_id"] != p["competency_id"]:
                continue
            if _overlaps(p["quote"], g["sentence"]) and (not strict or int(g["level"]) == int(p["level"])):
                hit = i
                break
        c = counts.setdefault(p["competency_id"], [0, 0, 0])
        if hit is None:
            c[1] += 1
        else:
            used.add(hit)
            c[0] += 1
    for i, g in enumerate(gold):
        if i not in used:
            counts.setdefault(g["competency_id"], [0, 0, 0])[2] += 1
    return counts


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def run(mode: str | None = None) -> pd.DataFrame:
    s = load_settings()
    gt = pd.read_csv(s.data_dir / "ground_truth.csv", keep_default_na=False)
    docs = pd.read_csv(s.data_dir / "documents_index.csv").set_index("doc_id")
    comps = pd.read_csv(s.data_dir / "competencies.csv").set_index("competency_id")["name"]
    mode = choose_mode(mode)
    extractor = make_extractor(mode)

    totals = {"competency": {}, "strict": {}}
    for doc_id, g in gt.groupby("doc_id"):
        text = (s.data_dir / docs.at[doc_id, "path"]).read_text(encoding="utf-8")
        pred, _ = extract_validated(extractor, text)
        gold = g[g["competency_id"] != ""].to_dict("records")
        for view in totals:
            for cid, (tp, fp, fn) in match_counts(pred, gold, strict=view == "strict").items():
                acc = totals[view].setdefault(cid, [0, 0, 0])
                acc[0] += tp
                acc[1] += fp
                acc[2] += fn

    rows = []
    for cid in comps.index:
        row = {"competency_id": cid, "name": comps[cid]}
        for view in totals:
            tp, fp, fn = totals[view].get(cid, [0, 0, 0])
            p, r, f = prf(tp, fp, fn)
            row.update({f"{view}_tp": tp, f"{view}_fp": fp, f"{view}_fn": fn, f"{view}_precision": round(p, 3),
                        f"{view}_recall": round(r, 3), f"{view}_f1": round(f, 3)})
        row["support"] = row["competency_tp"] + row["competency_fn"]
        rows.append(row)
    df = pd.DataFrame(rows)
    for label in ("MICRO", "MACRO"):
        row = {"competency_id": label, "name": f"{label.lower()} average", "support": int(df["support"].sum())}
        for view in totals:
            if label == "MICRO":
                tp, fp, fn = (int(df[f"{view}_{k}"].sum()) for k in ("tp", "fp", "fn"))
                p, r, f = prf(tp, fp, fn)
                row.update({f"{view}_tp": tp, f"{view}_fp": fp, f"{view}_fn": fn})
            else:
                has = df[df["support"] > 0]
                p, r, f = (has[f"{view}_{k}"].mean() for k in ("precision", "recall", "f1"))
            row.update({f"{view}_precision": round(p, 3), f"{view}_recall": round(r, 3), f"{view}_f1": round(f, 3)})
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.insert(2, "extractor", mode)
    df.to_csv(s.outputs_dir / "eval_metrics.csv", index=False)
    micro = df[df["competency_id"] == "MICRO"].iloc[0]
    print(f"Evaluation ({mode}, {gt['doc_id'].nunique()} docs): competency P={micro['competency_precision']:.2f} "
          f"R={micro['competency_recall']:.2f} | strict (with level) P={micro['strict_precision']:.2f} "
          f"R={micro['strict_recall']:.2f}")
    return df


if __name__ == "__main__":
    run()
