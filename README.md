# Succession Planning Assistant (proof of concept)

> **Decision support only. The panel makes all decisions. All data is synthetic.**

The assistant helps a public-service HR panel assess candidates for senior roles. It reads each
officer's documents (appraisals, posting summaries, 360 feedback and course records) and
pulls out **verbatim evidence** against a competency framework. It then scores that evidence against
each role's requirements with **transparent, configurable rules**, and suggests development
for any gaps. A separate audit checks the output for adverse impact and for sensitivity to
names and pronouns.

The AI gathers evidence and the humans decide. The tool never recommends an appointment. Every number
in it traces back to quoted sentences that the panel can read and challenge.

## What's in the box

| Step | Module | Output |
|---|---|---|
| 1. Synthetic data | `src/generate_data.py` | `data/`: 18 competencies with level descriptors 1–4, 6 critical roles, role requirements, 200 fictional officers, ~870 documents, 40 development opportunities, 30 labelled ground-truth documents |
| 2. Evidence extraction | `src/extract.py` | `outputs/evidence.csv` (officer, competency, level, quote, source, date), `outputs/suggestions.csv` (unscored) |
| 3. Scoring | `src/score.py` | `outputs/requirement_scores.csv`, `outputs/role_fit.csv` (fit, gate, readiness band) |
| 4. Development | `src/recommend.py` | `outputs/recommendations.csv` (top 3 opportunities per gap) |
| 5. Fairness audit | `src/fairness.py` | `data/_audit/group_attributes.csv` (hidden), `outputs/fairness_report.json`, `outputs/fairness_air.csv` |
| 6. Evaluation | `src/evaluate.py` | `outputs/eval_metrics.csv` (precision/recall per competency) |
| App | `app.py` | Role, Candidate, Compare and Audit tabs, plus a persistent disclaimer banner |

All weights, thresholds, decay rates, model names and seeds are in **`config.toml`**.

## How to run

Requires Python 3.11+.

```bash
python -m venv .venv            # tip: put it outside OneDrive/Dropbox folders (torch is ~1 GB)
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python run_all.py               # generate -> extract -> score -> recommend -> audit -> evaluate -> tests
streamlit run app.py
```

`make demo` does the same thing where `make` is available. Useful flags:
`python run_all.py --mode mock | --mode llm | --no-embeddings | --skip-tests`.

### Without an API key (default)
Everything runs offline. Extraction uses the deterministic **mock extractor**, which applies
keyword and phrase rules per competency, a level from scope cues ("assisted" → 1, "led ... across the
department" → 3, "at the national level" → 4), and skips negated sentences ("would benefit
from ...") and sentences that are not about the officer. The first run downloads the
`all-MiniLM-L6-v2` embedding model (~90 MB) once. If that isn't possible, use `--no-embeddings`;
nothing else depends on it.

### With an API key
```bash
set ANTHROPIC_API_KEY=sk-ant-...     # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python run_all.py --mode llm
```
With `mode = "auto"` (the default), the LLM extractor is used **automatically whenever the key is set**.
Use `--mode mock` to stay offline. LLM mode sends every document (~870) to the API once. Responses
are cached in `data/cache/llm/`, so re-runs are free. The fairness counterfactual uses a smaller
sample in LLM mode (`counterfactual_sample_llm`).

LLM mode works as follows:
- The model sees a fixed competency list with level descriptors and must return JSON that matches a schema
  (structured outputs). The output is then validated again with pydantic, and unknown competency ids or
  out-of-range levels are dropped.
- **Temperature:** the brief asked for temperature 0. Current Claude models (default
  `claude-opus-5-5`, set in config) do not accept sampling parameters, so stability comes instead from a fixed prompt,
  constrained JSON output and the on-disk cache. The exact-quote check below catches what still varies.
- Note: the LLM path has been tested with a stubbed client (see `tests/`), not yet against the live API.

## Key safeguards

- **Exact-quote rule.** Any evidence whose quote is not an exact substring of its source document
  is rejected and counted. This is enforced in one place (`extract.validate_quote`) for both extractors.
- **Embeddings suggest, never score.** When a sentence looks relevant but no rule could place it
  (unusual wording such as "steadying the unit after its flagship project was cancelled"), the
  nearest competency is listed as an *unverified suggestion* for a human to review. Suggestions
  never enter the scores. Embeddings are also used to rank development opportunities.
- **No protected attributes in the model.** Officers have no race, gender, age or similar
  columns. The A/B audit attribute is generated *by the audit module itself*, stored separately
  under `data/_audit/`, and is never read by extraction or scoring.

## Scoring rules

For each officer × role requirement:

```
coverage   = min(evidence_level / required_level, 1.0)
recency    = 0.85 ** years_since_evidence
confidence = min(1.0, 0.5 + 0.25 * (independent_sources - 1))
req_score  = coverage * recency * confidence
role_fit   = Σ(weight × req_score) / Σ(weight)
gate       = every must-have requirement has coverage ≥ 1.0
```

The **supporting evidence** for a requirement is defined as follows. If any evidence reaches the required level, it is
all evidence at or above that level. Otherwise it is the evidence at the highest level shown.
`years_since_evidence` is measured from the most recent supporting item. `independent_sources` is the
number of distinct documents in the supporting set, so several quotes from one appraisal count once.

**Readiness bands**
- *Ready now*: the gate is passed and role_fit ≥ 0.8.
- *Ready in 1–2 years*: no more than 2 gaps, each no more than 1 level short, and a development opportunity that
  can reach the required level exists for each. A candidate with no gaps but stale or thin evidence also lands here.
- *Longer-term*: everything else.

Roles with 0–1 "Ready now" successors are flagged as a thin bench.

The formula is deliberately demanding. Reaching "Ready now" in practice requires 3+ independent and recent
sources for almost every requirement. That is a policy choice, and it can be tuned in `config.toml`.

## Design choices

**Rules rather than a trained model.** Senior appointments are rare (a handful a year per role
family), so there is too little data to learn from reliably. Past appointment decisions also
carry the biases the panel is trying to avoid, and a model trained to predict "who got
promoted" would learn to reproduce them. Transparent rules over quoted evidence mean that:
- every score can be explained and contested, sentence by sentence;
- the policy choices (weights, decay, thresholds) are visible in one config file and can be argued over by the panel;
- the LLM's job is narrowed to *finding and quoting* evidence, which can be verified, rather than *judging* people.

**Evidence over impressions.** Using recency and independent-source counts means one glowing
appraisal from years ago cannot carry a candidate. A thin or old record shows up as low confidence
rather than as a low rating. A thin record is not the same as low ability, and the UI shows which
one applies.

**Separation of the audit.** The fairness attribute exists only for testing. Keeping it in a separate
file and module makes it easy to show that scoring cannot use it.

## Current results (mock extractor, seed 42)

| Check | Result |
|---|---|
| Extraction, competency match (30 labelled docs) | precision ≈ 0.97, recall ≈ 0.90 |
| Extraction, competency + level | precision ≈ 0.91, recall ≈ 0.84 |
| Quotes rejected by the exact-substring check | 0 (mock quotes are sentence spans by construction) |
| Counterfactual name/pronoun swap | 120/120 documents produce identical evidence |
| Adverse impact ratio | Several roles flagged, but on fewer than 5 selections each; pooled AIR across roles ≈ 0.87 (not flagged) |

The AIR result is a useful example of why the audit reports counts alongside ratios. With 1–7
"Ready now" candidates per role, one person more or less swings the ratio between 0 and 1. The
A/B attribute is assigned at random, so these flags are noise, and the audit tab shows the counts
and an "unstable" note so the panel does not over-read them.

## Limitations

- **Synthetic everything.** The documents are template-generated and far more regular than
  real HR writing. The mock extractor's precision and recall are flattering because its rules and the
  generator share a vocabulary (the rules were written separately and some phrasings are deliberately unusual, but
  real text will be much harder). The extraction numbers do not show real-world accuracy.
- **Ground truth comes from the generator.** The 30 "hand-labelled" documents carry the labels the
  generator recorded for each sentence, which serve as a stand-in for human annotation. A real deployment needs
  independent double-annotation and agreement statistics.
- **The mock extractor is crude.** It misses evidence phrased in unexpected ways (these show up as
  suggestions), can mis-level sentences without a clear scope cue, and can be fooled by sentences
  like "worked alongside the team that led ...".
- **The formulas are policy choices.** The 0.85 decay, the confidence ramp and the 0.8 threshold are
  sensible defaults with no empirical validation. Small changes move people between bands.
- **Fairness checks are partial.** One synthetic binary attribute and a name/pronoun swap do not
  establish fairness. Real audits need real (carefully governed) demographic data, intersectional
  views, checks on *document* bias (who gets glowing appraisals in the first place), and
  significance testing on small samples.
- **The LLM mode has not been run live here.** Its schema, validation and quote checks are tested with a stub.
  It should be evaluated on the labelled set (`python run_all.py --mode llm` and then the Audit tab) before anyone relies on it.
- **Not a selection tool.** The output is a structured evidence pack for a panel. It is not a ranking
  to be thresholded automatically.

## Project layout

```
config.toml          all tunable rules and settings
run_all.py           one-command pipeline (+ tests)
app.py               Streamlit UI
src/                 generate_data, extract, score, recommend, fairness, evaluate, embeddings, config
tests/               scoring maths, quote validation, readiness bands, fairness metrics
data/                generated inputs (data/_audit/ holds the hidden audit attribute)
outputs/             pipeline outputs read by the app
```
