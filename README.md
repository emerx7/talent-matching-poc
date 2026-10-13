# Succession Planning Assistant (proof of concept)

> **Decision support only. The panel makes all decisions. All data is synthetic.**
>
> The hosted demo shows **precomputed results from one LLM run** (Claude Haiku 5.5 via OpenRouter)
> on synthetic data. It is display-only and makes no API calls.

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

Requires Python 3.11+. There are two dependency sets:

| File | For | Contents |
|---|---|---|
| `requirements.txt` | the **display-only app** (and Streamlit Community Cloud) | streamlit, pandas, pydantic |
| `requirements-full.txt` | the **full pipeline** | the above + sentence-transformers (PyTorch, ~1 GB), anthropic, openai, pytest |

### View the precomputed results (lightweight)
`data/` and `outputs/` are committed, so the app runs straight away and makes no API calls:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

**Streamlit Community Cloud:** deploy from this repo with main file `app.py`. It installs
`requirements.txt` automatically. Choose Python 3.11 or newer in the app's advanced settings.

### Run the full pipeline
```bash
pip install -r requirements-full.txt   # tip: create the venv outside OneDrive/Dropbox folders
python run_all.py --mode llm           # replays the committed LLM run from the cache - no API key needed
streamlit run app.py
```

> **Note:** `python run_all.py` *without* `--mode llm` and without an API key uses the offline
> mock extractor and **overwrites the committed LLM results** in `outputs/`. Run
> `python run_all.py --mode llm` (or `git checkout outputs/`) to get them back.

`make install` / `make demo` wrap these where `make` is available. Useful flags:
`python run_all.py --mode mock | --mode llm | --limit N | --no-embeddings | --skip-tests`.

### Mock extractor (offline, the default without a key)
Everything runs offline. Extraction uses the deterministic **mock extractor**, which applies
keyword and phrase rules per competency, a level from scope cues ("assisted" → 1, "led ... across the
department" → 3, "at the national level" → 4), and skips negated sentences ("would benefit
from ...") and sentences that are not about the officer. The first run downloads the
`all-MiniLM-L6-v2` embedding model (~90 MB) once. If that isn't possible, use `--no-embeddings`;
nothing else depends on it.

### With an API key
Two LLM providers are supported. Pick one with `provider` under `[extraction]` in `config.toml`.
API keys are read **only from environment variables** and never from a file.

| `provider` | Model setting | Key variable | How it's called |
|---|---|---|---|
| `"openrouter"` (current) | `openrouter_model` (currently `anthropic/claude-haiku-5.5`) | `OPENROUTER_API_KEY` | `openai` package against OpenRouter's OpenAI-compatible API (`https://openrouter.ai/api/v1`) |
| `"anthropic"` | `llm_model` (Anthropic IDs use hyphens, e.g. `claude-haiku-5-5`) | `ANTHROPIC_API_KEY` | `anthropic` package |

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."           # this terminal session only
python run_all.py --mode llm --limit 10         # trial run: only 10 documents first
python run_all.py --mode llm                    # full run once the trial looks right
```

- **Start with `--limit N`.** It extracts from only the first N documents and caps the fairness
  counterfactual and the evaluation at N documents each. A first trial run makes up to about 4N API calls.
  Scores and the app then cover only those documents, so do a full run afterwards.
- With `mode = "auto"` (the default), the LLM extractor is used **automatically whenever the selected
  provider's key is set**. Use `--mode mock` to stay offline. A full LLM run sends every document (~870) once.
  If the OpenRouter model is left empty, the pipeline warns and falls back to the mock extractor.
- **Caching:** responses are cached in `data/cache/llm/`, keyed by provider, model, prompt version and
  document text, so re-runs are free. Changing provider or model makes fresh calls without overwriting the old cache.
- **The committed cache lets anyone reproduce the LLM results without an API key.** With no key set,
  `python run_all.py --mode llm` replays every response from `data/cache/llm/`. The API client is only
  created on a cache miss, and a cache miss without a key stops the run with a clear error. Replaying this
  repo's cache reproduces every file in `outputs/` byte for byte. Anyone can also open the cache files
  to inspect the model's raw JSON output.
- **Speed:** OpenRouter requests run concurrently (`openrouter_workers`, default 8), with retry and
  exponential backoff on rate-limit and server errors (`openrouter_max_retries`).
- The fairness counterfactual uses a smaller sample in LLM mode (`counterfactual_sample_llm`).

LLM mode works as follows:
- The model sees a fixed competency list with level descriptors and must return JSON matching the evidence
  schema. The output is validated again with pydantic, and unknown competency ids or out-of-range
  levels are dropped. Every quote must also pass the exact-substring check.
- **OpenRouter:** the request first asks for schema-constrained JSON (`response_format: json_schema`)
  and routes only to providers that support it. If the model rejects that, the extractor falls back to describing
  the schema in the prompt and stripping any ```` ```json ```` fences. The pydantic and exact-quote checks then do the
  policing. Requests use `temperature = 0`.
- **Anthropic:** uses structured outputs. Current Claude models don't accept a `temperature` setting, so stability
  comes from the fixed prompt, constrained JSON and the cache.
- Both providers are tested with stubbed clients (`tests/test_quote_validation.py`, `tests/test_openrouter.py`).
  The committed results come from a live OpenRouter run. The Anthropic provider has not been run live.

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

## Current results (LLM run: Claude Haiku 5.5 via OpenRouter, seed 42)

These are the results committed in `outputs/` and shown in the hosted demo. The model was `anthropic/claude-haiku-5.5`
on OpenRouter, with all 873 documents extracted once.

| Check | Result |
|---|---|
| Evidence extracted | 3,501 quotes from 873 documents |
| Quotes rejected by the exact-substring check | 3 (model quotes that were not verbatim in the source) |
| Extraction, competency match (30 labelled docs, micro) | precision 0.96, recall 0.94, F1 0.95 |
| Extraction, competency + level (micro) | precision 0.81, recall 0.79, F1 0.80 |
| Weakest competencies (recall, competency match) | Resilience 0.80 (5 labels), Collaboration 0.82 (11), Systems Thinking 0.83 (6) |
| Counterfactual name/pronoun swap | **13/20 documents unchanged (65%)** |
| Adverse impact ratio ("Ready now") | **Flagged for every role and pooled (0.72)**, mostly on fewer than 5 selections per role |
| Bench ("Ready now" per role) | R1 3 · R2 1 (thin bench) · R3 4 · R4 8 · R5 8 · R6 3 |

**Reading these results**
- **Finding competencies works well, but assigning levels is harder.** The model usually spots the right competency in
  the right sentence (precision and recall around 0.95). It gives the same level as the label about 80% of the time. Most
  disagreements are off by one level, which matters because coverage depends on level.
- **The counterfactual result needs investigating before any real use.** In 7 of 20 documents, swapping the
  officer's name and pronouns changed the extracted evidence. In most cases a level moved by one (e.g. C02 from 3
  to 4), and in one case a competency changed. This test can't tell whether the cause is sensitivity
  to names and pronouns or ordinary run-to-run variation, even at temperature 0. A follow-up would re-run the
  *unchanged* documents several times to measure baseline variation and compare. The mock extractor is
  100% invariant by construction.
- **The AIR flags are consistent with chance.** The A/B attribute is assigned at random. The pooled difference
  (16 of 612 vs 11 of 588 officer-role pairs) has a Fisher exact p = 0.44. Pairs are not independent, because each
  officer appears once per role. Per-role counts are tiny, so one person more or less swings a ratio between 0 and 1.
  This is why the audit reports counts alongside ratios. A real audit needs larger samples and significance testing
  before acting on a flag.
- In LLM mode there are no embedding suggestions, because the LLM classifies every sentence itself. The
  "unverified suggestions" panel only applies to the mock extractor.

For comparison, the offline mock extractor (`python run_all.py --mode mock`) scores competency P 0.97 / R 0.90
and level P 0.91 / R 0.84 on the same labelled set. Its rules share vocabulary with the template generator,
so those numbers flatter it.

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
- **One LLM run, one model, synthetic text.** The LLM results above come from a single run of one model
  on template-generated documents. They show that the pipeline works end to end, not how accurate it would be on
  real appraisals. The counterfactual differences in particular need follow-up (see above).
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
data/cache/llm/      raw LLM responses, keyed by provider + model + document (enables keyless replay)
outputs/             pipeline outputs read by the app (committed: results of the LLM run)
requirements.txt     lightweight: display-only app
requirements-full.txt  full pipeline
```
