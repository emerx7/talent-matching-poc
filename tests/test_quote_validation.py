import json

import pytest

from src.extract import LLMExtractor, MockExtractor, extract_validated, parse_llm_output, split_sentences, \
    validate_quote

DOC = (
    "DOCUMENT: Annual appraisal comments\nOFFICER: Kara Tamberly (O9999)\nDATE: 2026-03-01\n\n"
    "Kara led the crisis response to a major system outage across the department. "
    "She would benefit from more experience in stakeholder consultations for the licensing review. "
    "The division as a whole dealt with cost savings across procurement this year. "
    "She was a pleasure to work with.\n"
)


def test_exact_substring_passes():
    assert validate_quote("Kara led the crisis response to a major system outage across the department.", DOC)


@pytest.mark.parametrize("quote", [
    "Kara led the crisis response to a major outage across the department.",  # paraphrase
    "Kara  led the crisis response",                                            # whitespace change
    "kara led the crisis response",                                             # case change
    "Kara spearheaded national crisis policy.",                                 # hallucination
    "", "   ",
])
def test_non_verbatim_quotes_fail(quote):
    assert not validate_quote(quote, DOC)


def test_split_sentences_returns_exact_substrings_and_skips_header():
    sents = split_sentences(DOC)
    assert len(sents) == 4
    assert all(s in DOC for s in sents)
    assert not any(s.startswith(("DOCUMENT", "OFFICER", "DATE")) for s in sents)


def test_mock_extractor_rules():
    rows, rejected = extract_validated(MockExtractor(), DOC)
    assert rejected == 0
    assert [(r["competency_id"], r["level"]) for r in rows] == [("C02", 3)]  # negation + distractor skipped


def _stub_llm(tmp_path, payload: str) -> LLMExtractor:
    """An LLMExtractor that never touches the network."""
    ex = LLMExtractor.__new__(LLMExtractor)
    ex.model, ex.cache_dir, ex.name, ex.unmatched = "stub", tmp_path, "llm", []
    ex.valid_ids = {f"C{i:02d}" for i in range(1, 19)}
    ex._call = lambda text: payload
    return ex


def test_llm_hallucinated_quotes_are_rejected(tmp_path):
    payload = json.dumps({"items": [
        {"competency_id": "C02", "level": 3,
         "quote": "Kara led the crisis response to a major system outage across the department."},
        {"competency_id": "C01", "level": 4, "quote": "Kara set the national strategy."},  # not in the doc
        {"competency_id": "C99", "level": 2, "quote": "She was a pleasure to work with."},  # unknown competency
    ]})
    rows, rejected = extract_validated(_stub_llm(tmp_path, payload), DOC)
    assert [r["competency_id"] for r in rows] == ["C02"]
    assert rejected == 1


def test_llm_invalid_json_is_dropped(tmp_path):
    assert parse_llm_output("not json", {"C01"}) == []
    assert parse_llm_output(json.dumps({"items": [{"competency_id": "C01", "level": 9, "quote": "x"}]}),
                            {"C01"}) == []
    rows, _ = extract_validated(_stub_llm(tmp_path, '{"items": [{"competency_id": "C01"}]}'), DOC)
    assert rows == []
