import pandas as pd
import pytest

from src.extract import MockExtractor
from src.fairness import (adverse_impact_ratio, air_table, counterfactual, counterfactual_test,
                          generate_attributes, swap_name, swap_pronouns)
from src.score import LONGER, READY_NOW


def test_air_basic():
    assert adverse_impact_ratio(0.1, 0.2) == pytest.approx(0.5)
    assert adverse_impact_ratio(0.2, 0.1) == pytest.approx(0.5)
    assert adverse_impact_ratio(0.3, 0.3) == 1.0
    assert adverse_impact_ratio(0.0, 0.2) == 0.0
    assert adverse_impact_ratio(0.0, 0.0) is None  # undefined, not "fair"


def _fit(role, selections):
    return pd.DataFrame([{"officer_id": oid, "role_id": role, "readiness": READY_NOW if sel else LONGER}
                         for oid, sel in selections])


def test_air_table_flags_below_four_fifths():
    groups = pd.DataFrame({"officer_id": [f"A{i}" for i in range(10)] + [f"B{i}" for i in range(10)],
                           "group": ["A"] * 10 + ["B"] * 10})
    fit = _fit("R1", [(f"A{i}", i < 5) for i in range(10)] + [(f"B{i}", i < 2) for i in range(10)])
    row = air_table(fit, groups, 0.8).set_index("role_id").loc["R1"]
    assert row["rate_A"] == 0.5 and row["rate_B"] == 0.2
    assert row["air"] == pytest.approx(0.4)
    assert bool(row["flag"])


def test_air_table_passes_equal_rates_and_handles_no_selection():
    groups = pd.DataFrame({"officer_id": ["A1", "A2", "B1", "B2"], "group": ["A", "A", "B", "B"]})
    fit = pd.concat([_fit("R1", [("A1", True), ("A2", False), ("B1", True), ("B2", False)]),
                     _fit("R2", [("A1", False), ("A2", False), ("B1", False), ("B2", False)])])
    t = air_table(fit, groups, 0.8).set_index("role_id")
    assert t.loc["R1", "air"] == 1.0 and not t.loc["R1", "flag"]
    assert pd.isna(t.loc["R2", "air"]) and not t.loc["R2", "flag"]
    assert "undefined" in t.loc["R2", "note"]


def test_hidden_attribute_is_reproducible_and_separate():
    a = generate_attributes(["O1", "O2", "O3"], seed=7)
    b = generate_attributes(["O1", "O2", "O3"], seed=7)
    assert a.equals(b)
    assert list(a.columns) == ["officer_id", "group"]


@pytest.mark.parametrize("src,expected", [
    ("He led his team.", "She led her team."),
    ("She led her team.", "He led his team."),
    ("They led their unit.", "He led his unit."),
    ("Colleagues thanked her for it.", "Colleagues thanked him for it."),
    ("Staff supported her.", "Staff supported him."),
    ("Her work was strong.", "His work was strong."),
    ("The theme was there.", "The theme was there."),  # no partial-word replacements
])
def test_swap_pronouns(src, expected):
    assert swap_pronouns(src) == expected


def test_swap_name_replaces_full_and_first_name():
    text = "OFFICER: Kara Tamberly (O1)\n\nKara led X. Kara Tamberly drove Y. Karachi is a city."
    out = swap_name(text, "Kara Tamberly", "Lio Fenwick")
    assert "Lio Fenwick (O1)" in out and "Lio led X." in out and "Lio Fenwick drove Y." in out
    assert "Karachi" in out


DOC = (
    "DOCUMENT: Annual appraisal comments\nOFFICER: Kara Tamberly (O9999)\nDATE: 2026-03-01\n\n"
    "Kara led the crisis response to a major system outage across the department. "
    "She handled the division's budget planning competently within her team. "
    "Her colleagues said she spearheaded long-term strategy for the sector at the national level.\n"
)


def test_mock_extractor_is_invariant_to_name_and_pronoun_swap():
    swapped = counterfactual(DOC, "Kara Tamberly", "Lio Fenwick")
    assert "Kara" not in swapped and " she " not in swapped
    res = counterfactual_test(MockExtractor(), [("D1", DOC, "Kara Tamberly")], ["Lio Fenwick"])
    assert res["unchanged"] == 1 and res["differences"] == []
