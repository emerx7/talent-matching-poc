"""Evidence extraction: document text -> evidence rows.

Two interchangeable extractors produce the same row shape:
  * MockExtractor — deterministic keyword/phrase rules (default, offline).
  * LLM extractors — a fixed competency list and a JSON schema, validated with pydantic,
                    via Anthropic or OpenRouter (config.toml `provider`). Used only when that
                    provider's API key is set in the environment.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
API_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


class MissingAPIKeyError(RuntimeError):
    """A cache miss with no API key. Deliberately not retried or swallowed: the run stops."""


def strip_json_fences(raw: str) -> str:
    """Models answering without a schema sometimes wrap JSON in ```json fences."""
    m = _FENCE_RE.match(raw)
    return m.group(1) if m else raw.strip()


@dataclass
class LLMExtractor:
    """Provider-neutral LLM extraction: prompt, schema, retry and on-disk cache.

    Subclasses supply `_make_client()` and `_call(text) -> raw JSON string`. The API client is
    created lazily on the first cache miss, so a fully cached run (e.g. from the committed
    cache) needs no API key. A `client` can also be injected (tests do this).
    """
    competencies: pd.DataFrame
    model: str
    max_tokens: int
    cache_dir: Path
    client: Any = None
    provider: str = ""
    name: str = "llm"
    unmatched: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.name = f"llm-{self.provider}"
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

    def _make_client(self):
        raise NotImplementedError

    def _get_client(self):
        if self.client is None:
            self.client = self._make_client()
        return self.client

    def _call(self, text: str) -> str:
        raise NotImplementedError

    def cache_path(self, text: str) -> Path:
        key = hashlib.sha256(f"{PROMPT_VERSION}|{self.provider}|{self.model}|{text}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{key}.json"

    def extract(self, text: str) -> list[dict]:
        cache = self.cache_path(text)
        if cache.exists():
            raw = cache.read_text(encoding="utf-8")
        else:
            raw = None
            for attempt in range(2):  # one retry on invalid output
                try:
                    raw = strip_json_fences(self._call(text))
                    LLMExtraction.model_validate_json(raw)
                    break
                except MissingAPIKeyError:
                    raise
                except (ValidationError, RuntimeError, StopIteration) as e:
                    log.warning("LLM extraction attempt %d failed: %s", attempt + 1, e)
                    raw = None
            if raw is None:
                return []
            cache.write_text(raw, encoding="utf-8")
        return parse_llm_output(raw, self.valid_ids)


@dataclass
class AnthropicExtractor(LLMExtractor):
    provider: str = "anthropic"
    effort: str = "medium"

    def _make_client(self):
        import anthropic  # imported lazily so mock mode needs no API package configured

        return anthropic.Anthropic()

    def _call(self, text: str) -> str:
        # Current Claude models do not accept `temperature`; determinism comes from the fixed
        # schema (structured outputs), a fixed prompt, and the on-disk cache.
        resp = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self.schema}},
            messages=[{"role": "user", "content": f"<document>\n{text}\n</document>"}],
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"LLM stop_reason={resp.stop_reason}")
        return next(b.text for b in resp.content if b.type == "text")


@dataclass
class OpenRouterExtractor(LLMExtractor):
    """OpenRouter via its OpenAI-compatible API. The key comes only from OPENROUTER_API_KEY.

    First asks for output constrained to the JSON schema (routing only to providers that
    support it). If the model rejects that, it falls back to describing the schema in the
    prompt; pydantic validation and the exact-quote check apply either way.

    `prefetch()` fills the on-disk cache with `workers` concurrent requests. Rate-limit (429),
    server (5xx), timeout and connection errors are retried by the openai client with
    exponential backoff that honours Retry-After, up to `max_retries` times per request.
    """
    provider: str = "openrouter"
    structured: bool = True  # flips to False once the model rejects response_format
    workers: int = 8
    max_retries: int = 5
    _failed: set[str] = field(default_factory=set, init=False, repr=False)

    def _make_client(self):
        from openai import OpenAI

        key = os.environ.get(API_KEY_ENV["openrouter"])
        if not key:
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY is not set and this document is not in the cache (data/cache/llm/). "
                "Set the key to extract new documents, or replay only documents that are already cached.")
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=self.max_retries)

    def prefetch(self, texts: list[str]) -> None:
        """Extract uncached documents concurrently into the on-disk cache; cached ones are skipped."""
        unique = list(dict.fromkeys(texts))
        todo = [t for t in unique if not self.cache_path(t).exists()]
        if not todo:
            return
        print(f"OpenRouter: {len(unique) - len(todo)} documents already cached, extracting {len(todo)} "
              f"with {self.workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self.extract, t) for t in todo]
            try:
                for done, fut in enumerate(as_completed(futures), 1):
                    fut.result()  # unexpected errors (e.g. invalid key) stop the run
                    if done % 25 == 0 or done == len(todo):
                        print(f"  {done}/{len(todo)} documents", flush=True)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        # Documents still uncached failed after all retries: skip them for the rest of this run
        # (no second round of slow retries) and leave them uncached so the next run retries them.
        self._failed = {t for t in todo if not self.cache_path(t).exists()}
        if self._failed:
            log.warning("%d document(s) failed after retries and were not cached; re-run to retry them.",
                        len(self._failed))

    def extract(self, text: str) -> list[dict]:
        if text in self._failed:
            return []
        return super().extract(text)

    def _call(self, text: str) -> str:
        import openai

        try:
            return self._request(text)
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
            raise RuntimeError(f"OpenRouter request failed after {self.max_retries} retries: {e}") from e

    def _request(self, text: str) -> str:
        import openai

        system = (self.system + "\n\nRespond with ONLY a JSON object (no prose, no code fences) matching this "
                  f"JSON Schema:\n{json.dumps(self.schema)}")
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": f"<document>\n{text}\n</document>"}]}
        if self.structured:
            try:
                resp = self._get_client().chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "evidence", "strict": True, "schema": self.schema}},
                    extra_body={"provider": {"require_parameters": True}},
                )
            except (openai.BadRequestError, openai.NotFoundError) as e:
                log.warning("Model %s does not accept structured output (%s); using prompt-only JSON.",
                            self.model, e)
                self.structured = False
                return self._call(text)
        else:
            resp = self._get_client().chat.completions.create(**kwargs)
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("LLM output truncated (max_tokens)")
        if not choice.message.content:
            raise RuntimeError("LLM returned no content")
        return choice.message.content


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
    ex = load_settings().extraction
    mode = requested or ex.mode
    key_env = API_KEY_ENV[ex.provider]
    has_key = bool(os.environ.get(key_env))
    if mode == "auto":
        mode = "llm" if has_key else "mock"
    elif mode == "llm" and not has_key:
        log.warning("%s is not set: replaying LLM results from the cache (data/cache/llm/). "
                    "Any document not in the cache will stop the run.", key_env)
    if mode == "llm" and ex.provider == "openrouter" and not ex.openrouter_model:
        log.warning('provider = "openrouter" but openrouter_model is empty in config.toml; using mock extractor.')
        return "mock"
    return mode


def make_extractor(mode: str):
    s = load_settings()
    if mode == "llm":
        ex = s.extraction
        comps = pd.read_csv(s.data_dir / "competencies.csv")
        cache = s.data_dir / "cache" / "llm"  # files keyed by provider + model + document text
        if ex.provider == "openrouter":
            return OpenRouterExtractor(comps, ex.openrouter_model, ex.llm_max_tokens, cache,
                                       workers=ex.openrouter_workers, max_retries=ex.openrouter_max_retries)
        return AnthropicExtractor(comps, ex.llm_model, ex.llm_max_tokens, cache, effort=ex.llm_effort)
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


def run(mode: str | None = None, use_embeddings: bool = True, limit: int | None = None) -> pd.DataFrame:
    s = load_settings()
    mode = choose_mode(mode)
    extractor = make_extractor(mode)
    docs = pd.read_csv(s.data_dir / "documents_index.csv")
    if limit:
        docs = docs.head(limit)
    comps = pd.read_csv(s.data_dir / "competencies.csv")
    if hasattr(extractor, "prefetch"):  # concurrent providers warm the cache; the loop below then reads it
        extractor.prefetch([(s.data_dir / p).read_text(encoding="utf-8") for p in docs["path"]])

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

    ev_df = pd.DataFrame(rows, columns=[*Evidence.model_fields, "doc_type"])
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
