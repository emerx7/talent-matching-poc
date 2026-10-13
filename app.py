"""Succession Planning Assistant — Streamlit front end.

Display-only: reads the precomputed files in data/ and outputs/ and never calls an LLM API,
the extractors or the embedding model. Keep imports here to the lightweight set in
requirements.txt (streamlit, pandas, pydantic) so it runs on Streamlit Community Cloud.
It never sees the hidden fairness attribute.

Locally: `python run_all.py` (needs requirements-full.txt) to regenerate outputs, then `streamlit run app.py`.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import load_settings
from src.score import BANDS, READY_NOW

st.set_page_config(page_title="Succession Planning Assistant", layout="wide")
S = load_settings()

st.title("Succession Planning Assistant")
st.warning("**Decision support only — the panel makes all decisions. Synthetic data.**", icon="⚖️")

REQUIRED = ["evidence.csv", "requirement_scores.csv", "role_fit.csv", "recommendations.csv"]
if not all((S.outputs_dir / f).exists() for f in REQUIRED):
    st.error("Outputs not found. Run `python run_all.py` first, then reload this page.")
    st.stop()


@st.cache_data
def load():
    d, o = S.data_dir, S.outputs_dir
    out = {
        "roles": pd.read_csv(d / "roles.csv"),
        "comps": pd.read_csv(d / "competencies.csv"),
        "officers": pd.read_csv(d / "officers.csv"),
        "opps": pd.read_csv(d / "opportunities.csv"),
        "docs": pd.read_csv(d / "documents_index.csv"),
        "evidence": pd.read_csv(o / "evidence.csv"),
        "reqs": pd.read_csv(o / "requirement_scores.csv", keep_default_na=False, na_values=[""]),
        "fit": pd.read_csv(o / "role_fit.csv"),
        "recs": pd.read_csv(o / "recommendations.csv"),
        "suggestions": pd.read_csv(o / "suggestions.csv") if (o / "suggestions.csv").exists() else pd.DataFrame(),
    }
    for f in ("fairness_report.json", "extraction_log.json"):
        out[f] = json.loads((o / f).read_text()) if (o / f).exists() else None
    out["eval"] = pd.read_csv(o / "eval_metrics.csv") if (o / "eval_metrics.csv").exists() else None
    return out


D = load()
comp_name = dict(zip(D["comps"]["competency_id"], D["comps"]["name"]))
role_title = dict(zip(D["roles"]["role_id"], D["roles"]["title"]))
off_name = dict(zip(D["officers"]["officer_id"], D["officers"]["name"]))
off_label = {oid: f"{off_name[oid]} ({oid})" for oid in off_name}

if D["extraction_log.json"]:
    lg = D["extraction_log.json"]
    st.caption(f"Extractor: **{lg['mode']}** · {lg['evidence_rows']} evidence quotes from {lg['documents']} "
               f"documents · {lg['rejected_quotes']} quotes rejected (not verbatim in source)")

tab_role, tab_cand, tab_cmp, tab_audit = st.tabs(["Role view", "Candidate view", "Compare", "Audit"])


def fit_table(role_id: str) -> pd.DataFrame:
    f = D["fit"][D["fit"]["role_id"] == role_id].copy()
    f["officer"] = f["officer_id"].map(off_label)
    f["band_order"] = f["readiness"].map({b: i for i, b in enumerate(BANDS)})
    return f.sort_values(["band_order", "role_fit"], ascending=[True, False])


def req_table(officer_id: str, role_id: str) -> pd.DataFrame:
    r = D["reqs"][(D["reqs"]["officer_id"] == officer_id) & (D["reqs"]["role_id"] == role_id)].copy()
    r["competency"] = r["competency_id"].map(comp_name)
    return r


# ---------------------------------------------------------------------------
with tab_role:
    rid = st.selectbox("Role", D["roles"]["role_id"], format_func=lambda r: role_title[r], key="role_sel")
    ft = fit_table(rid)
    counts = ft["readiness"].value_counts().reindex(BANDS, fill_value=0)
    c1, c2, c3 = st.columns(3)
    for col, band in zip((c1, c2, c3), BANDS):
        col.metric(band, int(counts[band]))
    if counts[READY_NOW] <= S.scoring.thin_bench_max_ready:
        st.error(f"⚠ Thin bench: only {counts[READY_NOW]} 'Ready now' successor(s) for this role.")
    st.bar_chart(counts.rename("officers"))

    reqs = pd.read_csv(S.data_dir / "role_requirements.csv")
    rr = reqs[reqs["role_id"] == rid].assign(competency=lambda x: x["competency_id"].map(comp_name))
    with st.expander("Role requirements"):
        st.dataframe(rr[["competency", "required_level", "weight", "must_have"]], hide_index=True)

    st.subheader("Candidates (top 25 by band, then fit)")
    st.dataframe(ft.head(25)[["officer", "readiness", "role_fit", "gate_passed", "n_gaps", "n_closable_gaps"]],
                 hide_index=True, width="stretch")
    st.caption("Ordering is a convenience for reviewing evidence, not a ranking for selection.")

# ---------------------------------------------------------------------------
with tab_cand:
    c1, c2 = st.columns(2)
    rid_c = c1.selectbox("Role", D["roles"]["role_id"], format_func=lambda r: role_title[r], key="cand_role")
    ordered = fit_table(rid_c)["officer_id"].tolist()
    oid = c2.selectbox("Officer", ordered, format_func=lambda o: off_label[o], key="cand_off")
    off = D["officers"].set_index("officer_id").loc[oid]
    frow = D["fit"][(D["fit"]["officer_id"] == oid) & (D["fit"]["role_id"] == rid_c)].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Readiness", frow["readiness"])
    m2.metric("Role fit", f"{frow['role_fit']:.2f}")
    m3.metric("Must-have gate", "Passed" if frow["gate_passed"] else "Not passed")
    m4.metric("Gaps", f"{frow['n_gaps']} ({frow['n_closable_gaps']} closable)")
    st.caption(f"**Current role:** {off['current_role']} · **Years of service:** {off['years_service']}  \n"
               f"**Career history:** {off['career_history']}")

    rt = req_table(oid, rid_c)
    show = rt[["competency", "must_have", "required_level", "evidence_level", "coverage", "years_since", "recency",
               "independent_sources", "confidence", "req_score", "weight"]]
    st.subheader("Score per requirement")
    st.dataframe(show.style.format({"coverage": "{:.2f}", "recency": "{:.2f}", "confidence": "{:.2f}",
                                    "req_score": "{:.2f}", "years_since": "{:.1f}"}, na_rep="—"),
                 hide_index=True, width="stretch")

    st.subheader("Evidence behind each score")
    ev = D["evidence"]
    for r in rt.itertuples():
        ids = r.evidence_ids.split(";") if isinstance(r.evidence_ids, str) and r.evidence_ids else []
        label = f"{r.competency} — level {r.evidence_level}/{r.required_level} required · {len(ids)} quote(s)"
        with st.expander(label):
            if not ids:
                st.write("No evidence found in the officer's documents.")
            other = ev[(ev["officer_id"] == oid) & (ev["competency_id"] == r.competency_id)
                       & ~ev["evidence_id"].isin(ids)]
            for e in ev[ev["evidence_id"].isin(ids)].sort_values("date", ascending=False).itertuples():
                st.markdown(f"> {e.quote}\n\n`{e.source}` · {e.doc_type} · {e.date} · level {e.level} · {e.method}")
            if len(other):
                st.caption(f"{len(other)} further quote(s) at lower levels not used for this score.")

    sg = D["suggestions"]
    if len(sg):
        sg = sg[sg["officer_id"] == oid]
        if len(sg):
            with st.expander(f"Unverified embedding suggestions ({len(sg)}) — not scored"):
                st.caption("Sentences the rules could not map; the closest competency is suggested for a human "
                           "reviewer. These never affect scores.")
                st.dataframe(sg.assign(suggested=sg["suggested_competency_id"].map(comp_name))
                             [["quote", "suggested", "similarity", "source"]], hide_index=True)

    st.subheader("Gaps and recommended development")
    gaps = rt[rt["is_gap"]]
    if gaps.empty:
        st.success("No requirement gaps on the evidence available.")
    recs = D["recs"][(D["recs"]["officer_id"] == oid) & (D["recs"]["role_id"] == rid_c)]
    for g in gaps.itertuples():
        tag = "closable within 1–2 years" if g.closable else "larger gap"
        st.markdown(f"**{g.competency}** — evidenced level {g.evidence_level}, required {g.required_level} ({tag})")
        rg = recs[recs["competency_id"] == g.competency_id].sort_values("rank")
        for x in rg.itertuples():
            st.markdown(f"&nbsp;&nbsp;{x.rank}. {x.title} *({x.type})* — score {x.score:.2f}")

# ---------------------------------------------------------------------------
with tab_cmp:
    rid_x = st.selectbox("Role", D["roles"]["role_id"], format_func=lambda r: role_title[r], key="cmp_role")
    ordered = fit_table(rid_x)["officer_id"].tolist()
    picks = st.multiselect("Candidates (up to 4)", ordered, default=ordered[:3], max_selections=4,
                           format_func=lambda o: off_label[o])
    if picks:
        sub = D["reqs"][(D["reqs"]["role_id"] == rid_x) & D["reqs"]["officer_id"].isin(picks)].copy()
        sub["competency"] = sub["competency_id"].map(comp_name) + sub["must_have"].map({True: " ★", False: ""})
        sub["officer"] = sub["officer_id"].map(off_label)
        piv = sub.pivot_table(index="competency", columns="officer", values="req_score", sort=False)
        lv = sub.pivot_table(index="competency", columns="officer", values="evidence_level", sort=False)
        summary = D["fit"][(D["fit"]["role_id"] == rid_x) & D["fit"]["officer_id"].isin(picks)].copy()
        summary["officer"] = summary["officer_id"].map(off_label)
        st.dataframe(summary.set_index("officer")[["readiness", "role_fit", "gate_passed", "n_gaps"]].T.astype(str),
                     width="stretch")
        st.markdown("**Requirement scores** (★ = must-have)")
        st.dataframe(piv.style.format("{:.2f}"), width="stretch")
        st.markdown("**Evidenced level**")
        st.dataframe(lv, width="stretch")

# ---------------------------------------------------------------------------
with tab_audit:
    st.subheader("Adverse impact on 'Ready now' shortlists")
    rep = D["fairness_report.json"]
    if rep is None:
        st.info("Run `python -m src.fairness` to generate the audit.")
    else:
        st.caption(f"Groups A/B are a synthetic hidden attribute used only here. Flag when AIR < "
                   f"{rep['air_threshold']} (four-fifths rule). With few selections the ratio is unstable — read "
                   "the counts, not just the ratio.")
        air = pd.DataFrame(rep["air"])
        air["role"] = air["role_id"].map(role_title).fillna("All roles pooled")
        st.dataframe(air[["role", "n_A", "selected_A", "rate_A", "n_B", "selected_B", "rate_B", "air", "flag",
                          "note"]], hide_index=True, width="stretch")

        cf = rep["counterfactual"]
        st.subheader("Counterfactual name/pronoun swap")
        st.metric("Documents with unchanged evidence", f"{cf['unchanged']}/{cf['docs_tested']}",
                  f"{cf['pct_unchanged']}%", delta_color="off")
        st.caption(f"Extractor tested: {cf['extractor']}. Each document's officer name and gendered pronouns were "
                   "swapped and extraction re-run; any change in (competency, level) evidence is listed below.")
        if cf["differences"]:
            st.json(cf["differences"])

    st.subheader("Extraction quality (30 labelled documents)")
    if D["eval"] is None:
        st.info("Run `python -m src.evaluate` to generate metrics.")
    else:
        e = D["eval"]
        cols = ["competency_id", "name", "support", "competency_precision", "competency_recall", "competency_f1",
                "strict_precision", "strict_recall", "strict_f1"]
        st.dataframe(e[cols], hide_index=True, width="stretch")
        st.caption("'competency' = right competency on the right sentence; 'strict' also requires the right level.")
    if D["extraction_log.json"]:
        st.caption(f"Quotes rejected by the verbatim-substring check: {D['extraction_log.json']['rejected_quotes']}")
