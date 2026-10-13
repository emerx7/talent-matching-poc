"""OpenRouter extractor tests — a stubbed OpenAI-compatible client, no network, no API key."""
import json
import threading
import time
from types import SimpleNamespace

import httpx
import openai
import pandas as pd
import pytest

from src import extract
from src.extract import AnthropicExtractor, MissingAPIKeyError, OpenRouterExtractor, extract_validated

COMPS = pd.DataFrame([{"competency_id": f"C{i:02d}", "name": f"Comp {i}", "level_1": "a", "level_2": "b",
                       "level_3": "c", "level_4": "d"} for i in range(1, 19)])
DOC = (
    "DOCUMENT: Annual appraisal comments\nOFFICER: Kara Tamberly (O9999)\nDATE: 2026-03-01\n\n"
    "Kara led the crisis response to a major system outage across the department. "
    "She was a pleasure to work with.\n"
)
GOOD = {"competency_id": "C02", "level": 3,
        "quote": "Kara led the crisis response to a major system outage across the department."}


class FakeCompletions:
    """Returns queued responses in order; an Exception in the queue is raised instead."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.queue.pop(0)
        if isinstance(r, Exception):
            raise r
        content, finish = r if isinstance(r, tuple) else (r, "stop")
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                                                        message=SimpleNamespace(content=content))])


def _extractor(tmp_path, fake, model="vendor/model-a"):
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return OpenRouterExtractor(COMPS, model, 1000, tmp_path, client=client)


def _bad_request():
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.BadRequestError("response_format not supported", response=httpx.Response(400, request=req),
                                  body=None)


def test_structured_request_and_quote_check(tmp_path):
    payload = json.dumps({"items": [GOOD,
                                    {"competency_id": "C01", "level": 4, "quote": "Kara set national strategy."},
                                    {"competency_id": "C99", "level": 2, "quote": "She was a pleasure to work with."}]})
    fake = FakeCompletions(payload)
    rows, rejected = extract_validated(_extractor(tmp_path, fake), DOC)

    assert rows == [GOOD]
    assert rejected == 1  # hallucinated quote dropped; unknown competency dropped by validation
    call = fake.calls[0]
    assert call["model"] == "vendor/model-a" and call["temperature"] == 0
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["schema"]["required"] == ["items"]


def test_cache_makes_reruns_free_and_is_keyed_by_provider_and_model(tmp_path):
    fake = FakeCompletions(json.dumps({"items": [GOOD]}))
    ex = _extractor(tmp_path, fake)
    first = ex.extract(DOC)
    second = ex.extract(DOC)  # served from disk: the stub has no more responses queued
    assert first == second == [GOOD]
    assert len(fake.calls) == 1

    other_model = _extractor(tmp_path, FakeCompletions(), model="vendor/model-b")
    anthropic_same_model = AnthropicExtractor(COMPS, "vendor/model-a", 1000, tmp_path, client=object())
    paths = {ex.cache_path(DOC), other_model.cache_path(DOC), anthropic_same_model.cache_path(DOC)}
    assert len(paths) == 3


def test_falls_back_to_prompt_json_when_structured_output_unsupported(tmp_path):
    fenced = "```json\n" + json.dumps({"items": [GOOD]}) + "\n```"
    fake = FakeCompletions(_bad_request(), fenced)
    ex = _extractor(tmp_path, fake)
    rows, rejected = extract_validated(ex, DOC)

    assert rows == [GOOD] and rejected == 0
    assert not ex.structured
    assert "response_format" not in fake.calls[1]
    assert "JSON Schema" in fake.calls[1]["messages"][0]["content"]  # schema described in the prompt


def test_truncated_or_invalid_output_is_retried_then_dropped(tmp_path):
    fake = FakeCompletions(('{"items": [', "length"), "not json at all")
    assert _extractor(tmp_path, fake).extract(DOC) == []
    assert len(fake.calls) == 2  # one retry, then give up; nothing cached
    assert not list(tmp_path.glob("*.json"))


def test_api_key_comes_only_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = OpenRouterExtractor(COMPS, "vendor/model-a", 1000, tmp_path)._get_client()
    assert "openrouter.ai/api/v1" in str(client.base_url)
    assert client.api_key == "test-key"


def test_auto_mode_needs_key_and_model(monkeypatch):
    def settings(model):
        ext = SimpleNamespace(mode="auto", provider="openrouter", openrouter_model=model)
        return lambda: SimpleNamespace(extraction=ext)

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(extract, "load_settings", settings("vendor/model-a"))
    assert extract.choose_mode() == "mock"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert extract.choose_mode() == "llm"

    monkeypatch.setattr(extract, "load_settings", settings(""))
    assert extract.choose_mode() == "mock"  # model not filled in yet


def test_explicit_llm_mode_without_key_replays_from_cache(monkeypatch):
    ext = SimpleNamespace(mode="auto", provider="openrouter", openrouter_model="vendor/model-a")
    monkeypatch.setattr(extract, "load_settings", lambda: SimpleNamespace(extraction=ext))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert extract.choose_mode("llm") == "llm"   # explicit: replay from cache
    assert extract.choose_mode("auto") == "mock"  # default stays offline


def test_replay_needs_no_key_and_cache_miss_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ex = OpenRouterExtractor(COMPS, "vendor/model-a", 1000, tmp_path, workers=2)  # no key: still constructs
    ex.cache_path(DOC).write_text(json.dumps({"items": [GOOD]}), encoding="utf-8")

    ex.prefetch([DOC])  # everything cached: no client is ever created
    assert ex.extract(DOC) == [GOOD]
    assert ex.client is None

    uncached = DOC.replace("(O9999)", "(O1234)")
    with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY"):
        ex.prefetch([DOC, uncached])
    with pytest.raises(MissingAPIKeyError):
        ex.extract(uncached)


# ---------------------------------------------------------------------------
# Concurrency, retry and backoff
# ---------------------------------------------------------------------------
def _docs(n):
    return [DOC.replace("(O9999)", f"(O{i:04d})") for i in range(n)]


class SlowConcurrentCompletions:
    """Thread-safe stub that records how many requests are in flight at once."""

    def __init__(self, delay=0.05):
        self.delay, self.lock = delay, threading.Lock()
        self.in_flight = self.max_in_flight = 0
        self.seen: list[str] = []

    def create(self, **kwargs):
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.seen.append(kwargs["messages"][1]["content"])
        time.sleep(self.delay)
        with self.lock:
            self.in_flight -= 1
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=json.dumps({"items": [GOOD]})))])


def test_prefetch_runs_concurrently_and_skips_cached_docs(tmp_path):
    docs = _docs(12)
    fake = SlowConcurrentCompletions()
    ex = OpenRouterExtractor(COMPS, "vendor/model-a", 1000, tmp_path, workers=4,
                             client=SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    ex.cache_path(docs[0]).write_text(json.dumps({"items": []}), encoding="utf-8")  # already extracted

    ex.prefetch(docs + [docs[1]])  # duplicates are only requested once

    assert len(fake.seen) == 11
    assert not any("(O0000)" in s for s in fake.seen)
    assert 1 < fake.max_in_flight <= 4
    assert all(ex.cache_path(d).exists() for d in docs)
    n_before = len(fake.seen)
    assert ex.extract(docs[5]) == [GOOD] and len(fake.seen) == n_before  # later reads come from cache


def _chat_response(content):
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "vendor/model-a",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


def _real_client(handler, max_retries):
    """A real openai client whose HTTP layer is mocked, so the SDK's retry/backoff logic actually runs."""
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key="test", max_retries=max_retries,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_rate_limit_and_server_errors_are_retried_with_backoff(tmp_path):
    statuses = iter([429, 503])
    calls = []

    def handler(request):
        calls.append(request)
        status = next(statuses, 200)
        if status != 200:
            return httpx.Response(status, headers={"retry-after-ms": "1"}, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=_chat_response(json.dumps({"items": [GOOD]})))

    ex = OpenRouterExtractor(COMPS, "vendor/model-a", 1000, tmp_path, client=_real_client(handler, 3))
    assert ex.extract(DOC) == [GOOD]
    assert len(calls) == 3  # 429 -> retry, 503 -> retry, then success


def test_persistent_failure_is_not_cached_and_not_retried_again_this_run(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, headers={"retry-after-ms": "1"}, json={"error": {"message": "down"}})

    ex = OpenRouterExtractor(COMPS, "vendor/model-a", 1000, tmp_path, max_retries=1,
                             client=_real_client(handler, 1))
    ex.prefetch([DOC])
    assert len(calls) == 4  # (1 try + 1 SDK retry) x (1 attempt + 1 extractor retry)
    assert not ex.cache_path(DOC).exists()  # next run will try again
    assert ex.extract(DOC) == [] and len(calls) == 4  # main loop doesn't re-hit the API


def test_workers_and_retries_are_applied_to_extractor_and_client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    ex = OpenRouterExtractor(COMPS, "vendor/model-a", 1000, tmp_path, workers=3, max_retries=7)
    assert ex.workers == 3 and ex._get_client().max_retries == 7
