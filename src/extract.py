"""Evidence extraction: document text -> evidence rows.

Two interchangeable extractors produce the same row shape:
  * MockExtractor — deterministic keyword/phrase rules (default, offline).
  * LLMExtractor  — Anthropic API with a fixed competency list and a JSON schema,
                    validated with pydantic. Used only when ANTHROPIC_API_KEY is set.

Every quote, from either extractor, must be an exact substring of its source
document or the row is rejected. Embeddings are used only to *suggest* competency
mappings for sentences the rules could not place; suggestions are never scored.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from src.config import load_settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
HEADER_RE = re.compile(r"^[A-Z]+: ")
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]")
OFFICER_RE = re.compile(r"^OFFICER: (.+?) \(", re.M)
PRONOUN_RE = re.compile(r"\b(he|she|they|his|her|their)\b", re.I)


def validate_quote(quote: str, doc_text: str) -> bool:
    """Evidence is admissible only if its quote appears verbatim in the source document."""
    return bool(quote) and quote.strip() != "" and quote in doc_text


def split_sentences(text: str) -> list[str]:
    """Body sentences (header lines skipped). Each returned string is an exact substring of text."""
    out = []
    for line in text.splitlines():
        if not line.strip() or HEADER_RE.match(line):
            continue
        body = line[2:] if line.startswith("- ") else line
        for m in SENTENCE_RE.finditer(body):
            sent = m.group(0).strip()
            if sent:
                out.append(sent)
    return out


class Evidence(BaseModel):
    officer_id: str
    competency_id: str
    level: int = Field(ge=1, le=4)
    quote: str
    source: str
    date: str
    method: str


# ---------------------------------------------------------------------------
# Mock extractor — rules written independently of the generator's phrase banks
# ---------------------------------------------------------------------------
COMPETENCY_RULES: dict[str, str] = {
    "C01": r"strateg|long-term|scenario planning|foresight",
    "C02": r"crisis|emergenc|incident|outage",
    "C03": r"stakeholder|consultation|community engagement|engagement with",
    "C04": r"digital|cloud|technology|software|automation",
    "C05": r"coach|mentor|talent|capability building",
    "C06": r"\bpolic(?:y|ies)\b|legislat|cabinet paper",
    "C07": r"budget|\bcost|financ|procurement|fiscal",
    "C08": r"restructur|change management|transformation|merger",
    "C09": r"communication|\bmedia\b|\bpress\b|presentation|public speaking",
    "C10": r"collaborat|cross-agency|partnership|teamwork|taskforce",
    "C11": r"integrity|ethic|governance|\baudit|conflict-of-interest|accountab",
    "C12": r"analytic|data-driven|evidence-based|dashboard|metrics",
    "C13": r"innovat|\bpilot|prototyp|experiment|\bnovel\b",
    "C14": r"\boperations\b|process improvement|\blean\b|service-level",
    "C15": r"negotiat|settlement|contract dispute|bargain",
    "C16": r"resilien|setback|under pressure|adversity",
    "C17": r"systems thinking|ecosystem|interdependenc|end-to-end|root cause",
    "C18": r"citizen|customer|user research|residents",
}
LEVEL_RULES: list[tuple[int, str]] = [  # checked in this order
    (4, r"national level|across the public service|sector-wide|spearhead|set the direction"),
    (3, r"\bled\b|\bheaded\b|\bdrove\b|across the (?:department|division|organisation)|ownership"),
    (1, r"assisted|supported|participated|exposure|completed the"),
    (2, r"contributed|handled|delivered|managed|competent|reliabl|distinction"),
]
DEFAULT_LEVEL = 2
NEGATION_RE = re.compile(r"not yet|yet to|would benefit from|no opportunity|lacks?\b|limited exposure", re.I)

_COMP_RES = {cid: re.compile(p, re.I) for cid, p in COMPETENCY_RULES.items()}
_LEVEL_RES = [(lvl, re.compile(p, re.I)) for lvl, p in LEVEL_RULES]


def _mentions_officer(sentence: str, officer_name: str | None) -> bool:
    if PRONOUN_RE.search(sentence):
        return True
    if officer_name:
        return any(part in sentence for part in officer_name.split())
    return False


def rule_level(sentence: str) -> int:
    for lvl, rx in _LEVEL_RES:
        if rx.search(sentence):
            return lvl
    return DEFAULT_LEVEL


@dataclass
class MockExtractor:
    name: str = "mock"
    unmatched: list[str] = field(default_factory=list)  # sentences for the embedding suggester

    def extract(self, text: str) -> list[dict]:
        m = OFFICER_RE.search(text)
        officer_name = m.group(1) if m else None
        rows = []
        for sent in split_sentences(text):
            if NEGATION_RE.search(sent) or not _mentions_officer(sent, officer_name):
                continue
            comps = [cid for cid, rx in _COMP_RES.items() if rx.search(sent)]
            if not comps:
                self.unmatched.append(sent)
                continue
            lvl = rule_level(sent)
            rows.extend({"competency_id": c, "level": lvl, "quote": sent} for c in comps)
        return rows


# ---------------------------------------------------------------------------
# LLM extractor
# ---------------------------------------------------------------------------
class LLMItem(BaseModel):
    competency_id: str
    level: int = Field(ge=1, le=4)
    quote: str


class LLMExtraction(BaseModel):
    items: list[LLMItem]


PROMPT_VERSION = "v1"


@dataclass
class LLMExtractor:
    competencies: pd.DataFrame
    model: str
    effort: str
    max_tokens: int
    cache_dir: Path
    name: str = "llm"
    unmatched: list[str] = field(default_factory=list)

    def __post_init__(self):
        import anthropic  # imported lazily so mock mode needs no API package configured

        self.client = anthropic.Anthropic()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ids = list(self.competencies["competency_id"])
        self.valid_ids = set(ids)
        framework = "\n".join(
            f"{r.competency_id} {r.name}: 1={r.level_1} | 2={r.level_2} | 3={r.level_3} | 4={r.level_4}"
            for r in self.competencies.itertuples()
        )
        self.system = (
            "You extract evidence of leadership competencies from public-service HR documents for a "
            "decision-support tool. A human panel makes all decisions; your job is only to find evidence.\n\n"
            f"Competency framework (use ONLY these ids):\n{framework}\n\n"
            "Rules:\n"
            "- Return one item per piece of evidence about the named officer's own demonstrated behaviour.\n"
            "- `quote` must be copied EXACTLY, character for character, from the document (one sentence or "
            "clause). Do not paraphrase, fix typos or join sentences.\n"
            "- `level` is the level the quote demonstrates (1-4) per the descriptors above.\n"
            "- Ignore development needs, things the officer has not done, and work done by others.\n"
            "- Ignore names, pronouns and anything not about the work itself.\n"
            "- If there is no evidence, return an empty list."
        )
        self.schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {
                "type": "object",
                "properties": {"competency_id": {"type": "string", "enum": ids},
                               "level": {"type": "integer"},
                               "quote": {"type": "string"}},
                "required": ["competency_id", "level", "quote"],
                "additionalProperties": False}}},
            "required": ["items"],
            "additionalProperties": False,
        }

    def _call(self, text: str) -> str:
        # Current Claude models do not accept `temperature`; determinism comes from the fixed
        # schema (structured outputs), a fixed prompt, and the on-disk cache below.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self.schema}},
            messages=[{"role": "user", "content": f"<document>\n{text}\n</document>"}],
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"LLM stop_reason={resp.stop_reason}")
        return next(b.text for b in resp.content if b.type == "text")

    def extract(self, text: str) -> list[dict]:
        key = hashlib.sha256(f"{PROMPT_VERSION}|{self.model}|{text}".encode()).hexdigest()[:24]
        cache = self.cache_dir / f"{key}.json"
        if cache.exists():
            raw = cache.read_text(encoding="utf-8")
        else:
            raw = None
            for attempt in range(2):  # one retry on invalid output
                try:
                    raw = self._call(text)
                    LLMExtraction.model_validate_json(raw)
                    break
                except (ValidationError, RuntimeError, StopIteration) as e:
                    log.warning("LLM extraction attempt %d failed: %s", attempt + 1, e)
                    raw = None
            if raw is None:
                return []
            cache.write_text(raw, encoding="utf-8")
        return parse_llm_output(raw, self.valid_ids)


def parse_llm_output(raw: str, valid_ids: set[str]) -> list[dict]:
    """Validate LLM JSON with pydantic; drop items with unknown competencies. Quote check happens later."""
    try:
        parsed = LLMExtraction.model_validate_json(raw)
    except ValidationError:
        return []
    return [it.model_dump() for it in parsed.items if it.competency_id in valid_ids]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def choose_mode(requested: str | None = None) -> str:
    mode = requested or load_settings().extraction.mode
    if mode == "auto":
        return "llm" if os.environ.get("ANTHROPIC_API_KEY") else "mock"
    if mode == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("LLM mode requested but ANTHROPIC_API_KEY is not set; using mock extractor.")
        return "mock"
    return mode


def make_extractor(mode: str):
    s = load_settings()
    if mode == "llm":
        comps = pd.read_csv(s.data_dir / "competencies.csv")
        return LLMExtractor(comps, s.extraction.llm_model, s.extraction.llm_effort,
                            s.extraction.llm_max_tokens, s.data_dir / "cache" / "llm")
    return MockExtractor()


def extract_validated(extractor, text: str) -> tuple[list[dict], int]:
    """Run an extractor and enforce the exact-substring rule. Returns (rows, n_rejected)."""
    rows = extractor.extract(text)
    kept = [r for r in rows if validate_quote(r["quote"], text)]
    return kept, len(rows) - len(kept)


def suggest_mappings(sentences: list[str], competencies: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Embedding-based competency *suggestions* for sentences the rules could not place."""
    from src.embeddings import cosine, embed

    cols = ["quote", "suggested_competency_id", "similarity", "status"]
    if not sentences:
        return pd.DataFrame(columns=cols)
    comp_text = competencies["description"].tolist()
    s_emb, c_emb = embed(sentences), embed(comp_text)
    if s_emb is None or c_emb is None:
        return pd.DataFrame(columns=cols)
    sims = cosine(s_emb, c_emb)
    rows = []
    for i, sent in enumerate(sentences):
        j = int(sims[i].argmax())
        if sims[i, j] >= threshold:
            rows.append({"quote": sent, "suggested_competency_id": competencies.iloc[j]["competency_id"],
                         "similarity": round(float(sims[i, j]), 3),
                         "status": "needs human review (not scored)"})
    return pd.DataFrame(rows, columns=cols)


def run(mode: str | None = None, use_embeddings: bool = True) -> pd.DataFrame:
    s = load_settings()
    mode = choose_mode(mode)
    extractor = make_extractor(mode)
    docs = pd.read_csv(s.data_dir / "documents_index.csv")
    comps = pd.read_csv(s.data_dir / "competencies.csv")

    rows, rejected, suggestions = [], 0, []
    for d in docs.itertuples():
        text = (s.data_dir / d.path).read_text(encoding="utf-8")
        start = len(extractor.unmatched)
        kept, n_rej = extract_validated(extractor, text)
        rejected += n_rej
        for r in kept:
            ev = Evidence(officer_id=d.officer_id, source=d.doc_id, date=d.date, method=extractor.name, **r)
            rows.append(ev.model_dump() | {"doc_type": d.doc_type})
        for sent in extractor.unmatched[start:]:
            suggestions.append((d.officer_id, d.doc_id, d.date, sent))

    ev_df = pd.DataFrame(rows)
    ev_df.insert(0, "evidence_id", [f"E{i + 1:06d}" for i in range(len(ev_df))])
    s.outputs_dir.mkdir(exist_ok=True)
    ev_df.to_csv(s.outputs_dir / "evidence.csv", index=False)

    sug_df = pd.DataFrame(columns=["officer_id", "source", "date", "quote", "suggested_competency_id",
                                   "similarity", "status"])
    if use_embeddings and suggestions:
        meta = pd.DataFrame(suggestions, columns=["officer_id", "source", "date", "quote"])
        sug = suggest_mappings(meta["quote"].drop_duplicates().tolist(), comps, s.extraction.suggestion_threshold)
        sug_df = meta.merge(sug, on="quote")[sug_df.columns]
    sug_df.to_csv(s.outputs_dir / "suggestions.csv", index=False)

    log_info = {"mode": mode, "documents": len(docs), "evidence_rows": len(ev_df),
                "rejected_quotes": rejected, "suggestions": len(sug_df)}
    (s.outputs_dir / "extraction_log.json").write_text(json.dumps(log_info, indent=2))
    print(f"Extraction ({mode}): {len(ev_df)} evidence rows from {len(docs)} docs, "
          f"{rejected} rejected quotes, {len(sug_df)} embedding suggestions (unscored)")
    return ev_df


if __name__ == "__main__":
    run()
