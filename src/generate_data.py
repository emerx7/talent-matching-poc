"""Generate all synthetic data reproducibly (fixed seed from config.toml).

Every person, agency and document here is fictional. Officers carry NO protected
attributes. Pronouns appear in document text (as they would in real appraisals) but
are not stored anywhere as data.

Documents are composed sentence-by-sentence from level "frames" x competency
"objects". Because the generator knows what each sentence encodes, it can emit
ground-truth labels for the evaluation set.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src.config import load_settings

# ---------------------------------------------------------------------------
# Competency framework
# ---------------------------------------------------------------------------
# id, name, focus phrase (used in level descriptors), common objects (contain a
# recognisable keyword), unusual objects (deliberately avoid the obvious keywords).
COMPETENCIES = [
    ("C01", "Strategic Thinking", "long-term direction and strategy",
     ["the five-year strategic plan for the division", "long-term strategy for the sector",
      "the scenario planning exercise on future workforce needs"],
     ["a ten-year outlook for where the agency should be heading"]),
    ("C02", "Crisis Leadership", "leadership during crises and emergencies",
     ["the crisis response to a major system outage", "emergency coordination during the flood response",
      "incident management during the ransomware attack"],
     ["the overnight response when the haze situation escalated"]),
    ("C03", "Stakeholder Management", "engagement of stakeholders and partners",
     ["engagement with industry stakeholders", "stakeholder consultations for the licensing review",
      "community engagement on the estate renewal plan"],
     ["efforts to keep unions, grassroots leaders and trade associations on side"]),
    ("C04", "Digital Fluency", "use of digital technology",
     ["the migration of core services to the cloud", "the digital redesign of the permit system",
      "adoption of new technology tools for case officers"],
     ["the move of paper-based permit applications to an online portal"]),
    ("C05", "People Development", "growing and developing people",
     ["coaching and mentoring junior officers", "the talent pipeline for the branch",
      "capability building for new hires"],
     ["efforts that helped three direct reports earn promotion"]),
    ("C06", "Policy Judgement", "policy design and judgement",
     ["the policy review on housing grants", "drafting legislative amendments for the new act",
      "policy options for the cabinet paper on ageing"],
     ["the rethink of subsidy rules for elderly households"]),
    ("C07", "Financial Stewardship", "budgets and financial management",
     ["the division's budget planning", "cost savings across procurement",
      "financial controls for the grant scheme"],
     ["keeping programme spending within the approved envelope"]),
    ("C08", "Change Leadership", "leading organisational change",
     ["the organisational restructuring of the department", "change management after the merger of two units",
      "the transformation of the operating model"],
     ["the shift to a new way of working after two branches were combined"]),
    ("C09", "Communication", "communication with internal and public audiences",
     ["public communications on the new scheme", "media interviews and press conferences",
      "presentations to senior management"],
     ["explaining the new scheme on radio and at town halls"]),
    ("C10", "Collaboration", "collaboration across boundaries",
     ["cross-agency collaboration on the joint taskforce", "the partnership with sister agencies on youth services",
      "teamwork across departments on the annual review"],
     ["work with three sister agencies on a shared problem"]),
    ("C11", "Integrity & Public Values", "integrity, ethics and governance",
     ["upholding governance and audit standards", "an ethics review of conflict-of-interest declarations",
      "accountability arrangements for public funds"],
     ["the decision to decline vendor hospitality during a tender"]),
    ("C12", "Data-Driven Decision Making", "use of data and evidence in decisions",
     ["analytics to guide resource allocation", "evidence-based evaluation of programme outcomes",
      "dashboards tracking service performance metrics"],
     ["the use of survey numbers to decide where to put new clinics"]),
    ("C13", "Innovation", "innovation and experimentation",
     ["the innovation lab's experiments on service design", "piloting new approaches to grant applications",
      "prototyping novel ideas for frontline staff"],
     ["trialling a fresh approach to permit queues"]),
    ("C14", "Operational Excellence", "reliable, efficient operations",
     ["process improvement in licensing operations", "service-level targets for frontline operations",
      "lean process reviews at the service centres"],
     ["the reduction of waiting times at the licensing counter"]),
    ("C15", "Negotiation", "negotiating agreements",
     ["negotiations with contractors on the service agreement", "bilateral negotiations on the water agreement",
      "the settlement of a long-running contract dispute"],
     ["reaching a deal with the contractor after weeks of back-and-forth"]),
    ("C16", "Resilience", "resilience under pressure and setbacks",
     ["the team's recovery after a major project setback", "delivery under pressure through a demanding year-end peak",
      "steadying a team facing adversity"],
     ["steadying the unit after its flagship project was cancelled"]),
    ("C17", "Systems Thinking", "seeing whole systems and interdependencies",
     ["an end-to-end review of the service ecosystem", "mapping interdependencies across agencies",
      "root cause analysis of recurring service failures"],
     ["tracing how a delay in one agency rippled through to others"]),
    ("C18", "Customer Centricity", "putting citizens and service users first",
     ["the redesign of the citizen journey for permits", "customer feedback improvements at service centres",
      "user research with residents on the new app"],
     ["making the permit forms easier for the public to fill in"]),
]

LEVEL_NAMES = {1: "Aware", 2: "Practitioner", 3: "Leader", 4: "Shaper"}
LEVEL_TEMPLATES = {
    1: "Understands {focus} and supports others in applying it.",
    2: "Applies {focus} independently and reliably within own team.",
    3: "Leads {focus} across a department or division.",
    4: "Shapes {focus} at agency-wide, sector or national level.",
}

# Level frames. {S}=subject (name or pronoun), {pos}=possessive pronoun, {obj}=competency object.
FRAMES = {
    1: ["{S} assisted with {obj}.", "{S} supported senior colleagues on {obj}.",
        "{S} participated in {obj} as a team member.", "{S} gained initial exposure to {obj}."],
    2: ["{S} contributed effectively to {obj}.", "{S} handled {obj} competently within {pos} team.",
        "{S} delivered {pos} share of {obj} reliably.", "{S} managed {obj} for {pos} unit."],
    3: ["{S} led {obj} across the department.", "{S} drove {obj} for the whole division.",
        "{S} took ownership of {obj} and steered it to completion.",
        "{S} headed {obj}, setting priorities for several teams."],
    4: ["{S} spearheaded {obj} at the national level.", "{S} set the direction for {obj} across the public service.",
        "{S} was recognised sector-wide for {pos} leadership of {obj}.",
        "{S} championed {obj} and it became a model adopted by other agencies."],
}
COURSE_FRAMES = {
    1: ["{S} completed the {course} covering {obj}."],
    2: ["{S} achieved a distinction in the advanced module on {obj}."],
}
COURSES = ["Foundations Certificate", "Executive Module", "Leadership Milestone Programme", "Short Course"]
NEGATION_FRAMES = [
    "{S} had not yet gained exposure to {obj}.",
    "{S} would benefit from more experience in {obj}.",
]
DISTRACTOR_FRAMES = [
    "The division as a whole dealt with {obj} this year.",
    "{S} worked alongside the team that led {obj}.",
]
FILLER = [
    "{S} was a pleasure to work with.", "{S} showed good attendance and punctuality.",
    "Overall a solid year.", "{S} relocated to a new office during the period.",
    "Rating: meets expectations.", "{S} took on additional duties when a colleague was away.",
    "No concerns were raised during the period.",
]
FEEDBACK_PREFIXES = ["", "Peers observed that ", "Several respondents noted that ", "Direct reports said that "]

TRACKS = {
    "policy": ["C01", "C06", "C03", "C09", "C10", "C11", "C12"],
    "digital": ["C04", "C08", "C13", "C12", "C17", "C05"],
    "operations": ["C02", "C14", "C16", "C10", "C09", "C17", "C18"],
    "corporate": ["C07", "C11", "C12", "C15", "C03", "C01", "C08"],
}
TRACK_AREAS = {
    "policy": ["Policy Planning", "Strategy", "Legislation"],
    "digital": ["Digital Services", "Technology", "Data Office"],
    "operations": ["Operations", "Emergency Planning", "Service Delivery"],
    "corporate": ["Finance", "Corporate Services", "Procurement"],
}
AGENCIES = ["Ministry of Urban Affairs", "Ministry of Social Resilience", "National Digital Office",
            "Civic Services Board", "Public Safety Agency", "Treasury Directorate",
            "Enterprise Development Agency", "Environmental Standards Authority"]

ROLES = [
    ("R1", "Deputy Secretary (Policy)", "Ministry of Urban Affairs",
     [("C06", 4, 5, True), ("C01", 4, 5, True), ("C03", 3, 4, False), ("C09", 3, 3, False),
      ("C10", 3, 3, False), ("C11", 3, 4, False), ("C12", 2, 2, False)]),
    ("R2", "CEO, Enterprise Development Agency (statutory board)", "Enterprise Development Agency",
     [("C01", 4, 5, True), ("C08", 3, 4, True), ("C07", 3, 4, False), ("C03", 3, 4, False),
      ("C02", 3, 3, False), ("C05", 3, 3, False), ("C11", 3, 4, False)]),
    ("R3", "Director (Digital Transformation)", "National Digital Office",
     [("C04", 4, 5, True), ("C08", 3, 4, True), ("C13", 3, 4, False), ("C12", 3, 3, False),
      ("C17", 3, 3, False), ("C05", 2, 2, False)]),
    ("R4", "Director (Operations & Emergency Preparedness)", "Public Safety Agency",
     [("C02", 4, 5, True), ("C14", 3, 4, True), ("C16", 3, 3, False), ("C10", 3, 3, False),
      ("C09", 2, 2, False), ("C17", 2, 2, False)]),
    ("R5", "Chief Financial Officer", "Treasury Directorate",
     [("C07", 4, 5, True), ("C11", 4, 5, True), ("C12", 3, 4, False), ("C15", 3, 3, False),
      ("C01", 2, 2, False), ("C03", 2, 2, False)]),
    ("R6", "Director (Citizen Services)", "Civic Services Board",
     [("C18", 4, 5, True), ("C14", 3, 4, False), ("C04", 2, 3, False), ("C03", 3, 3, False),
      ("C05", 3, 3, False), ("C13", 2, 2, False)]),
]

# (type, title, primary competency, extra tags, max level reachable, months)
OPPORTUNITIES = [
    ("Posting", "Deputy Director, Strategic Futures Unit", "C01", ["C12"], 4, 24),
    ("Programme", "Strategic Foresight Fellowship", "C01", [], 3, 6),
    ("Posting", "Head, National Crisis Operations Centre", "C02", ["C16"], 4, 24),
    ("Programme", "Crisis Leadership Simulation Series", "C02", ["C09"], 3, 3),
    ("Posting", "Director, Industry Partnerships", "C03", ["C15"], 4, 24),
    ("Course", "Stakeholder Engagement Masterclass", "C03", [], 2, 1),
    ("Posting", "Product Lead, Government Digital Services", "C04", ["C13"], 4, 18),
    ("Course", "Digital Leadership Bootcamp", "C04", [], 3, 2),
    ("Programme", "Coaching Certification for Leaders", "C05", [], 3, 6),
    ("Posting", "Head, Talent & Leadership Development", "C05", ["C08"], 4, 24),
    ("Posting", "Senior Policy Adviser, Cabinet Office", "C06", ["C01"], 4, 24),
    ("Programme", "Policy Design Studio", "C06", ["C12"], 3, 4),
    ("Posting", "Deputy CFO Attachment", "C07", ["C11"], 4, 18),
    ("Course", "Public Finance for Non-Finance Leaders", "C07", [], 2, 1),
    ("Posting", "Transformation Office Lead", "C08", ["C17"], 4, 24),
    ("Programme", "Leading Change Programme", "C08", [], 3, 3),
    ("Course", "Media and Public Speaking Intensive", "C09", [], 3, 1),
    ("Posting", "Press Secretary Rotation", "C09", ["C03"], 4, 12),
    ("Posting", "Secretariat, Inter-Agency Taskforce", "C10", ["C17"], 4, 18),
    ("Programme", "Whole-of-Government Collaboration Lab", "C10", [], 3, 4),
    ("Posting", "Internal Audit Rotation", "C11", ["C07"], 4, 12),
    ("Course", "Ethics and Governance in the Public Service", "C11", [], 2, 1),
    ("Posting", "Lead, Analytics & Insights Unit", "C12", ["C04"], 4, 18),
    ("Course", "Data-Driven Decision Making for Managers", "C12", [], 3, 2),
    ("Posting", "Innovation Lab Secondment", "C13", ["C18"], 4, 12),
    ("Course", "Design Thinking Practicum", "C13", ["C18"], 3, 2),
    ("Posting", "Director, Service Operations", "C14", ["C16"], 4, 24),
    ("Course", "Lean Operations Certificate", "C14", [], 3, 2),
    ("Programme", "Negotiation Mastery Programme", "C15", [], 3, 2),
    ("Posting", "Trade Negotiations Delegation", "C15", ["C03"], 4, 18),
    ("Programme", "Resilient Leadership Retreat", "C16", [], 2, 1),
    ("Posting", "Turnaround Assignment, Struggling Unit", "C16", ["C08"], 4, 18),
    ("Programme", "Systems Leadership Programme", "C17", ["C01"], 3, 4),
    ("Posting", "Policy-Operations Review Lead", "C17", ["C06"], 4, 12),
    ("Posting", "Head, Citizen Experience Office", "C18", ["C14"], 4, 24),
    ("Course", "Service Design for Citizens", "C18", [], 3, 2),
    ("Programme", "Senior Leadership Milestone Programme", "C01", ["C05", "C08"], 3, 6),
    ("Programme", "Emerging Leaders Exchange", "C10", ["C03", "C09"], 3, 3),
    ("Posting", "Chief of Staff to Permanent Secretary", "C01", ["C06", "C09"], 4, 18),
    ("Course", "Financial Negotiation & Contracts", "C15", ["C07"], 3, 1),
]

DOC_TYPES = ["appraisal", "posting_summary", "feedback_360", "course_record"]
DOC_TITLES = {"appraisal": "Annual appraisal comments", "posting_summary": "Posting summary",
              "feedback_360": "360-degree feedback summary", "course_record": "Training and course record"}

FIRST_A = ["Ka", "Lio", "Ser", "Ta", "Mi", "Ren", "Dar", "Ve", "Jo", "Asha", "Ny", "Qui", "Bel", "Tor",
           "Hai", "Ze", "Lu", "Ori", "Pe", "Wen", "Sa", "Ilo", "Fa", "Cy"]
FIRST_B = ["ra", "n", "lin", "vi", "dre", "sa", "mon", "el", "ko", "th", "ya", "rin", "do", "ssa"]
LAST_A = ["Tam", "Kor", "Wil", "Ash", "Bren", "Har", "Lin", "Mor", "Quen", "Sel", "Val", "Yan", "Dov",
          "Fen", "Gar", "Os", "Rav", "Thal"]
LAST_B = ["berly", "dovan", "eth", "ford", "wick", "ani", "ason", "ell", "ova", "ington", "ari", "ake",
          "ira", "uen"]

PRONOUNS = [("He", "his"), ("She", "her"), ("They", "their")]


def comp_lookup() -> dict[str, tuple]:
    return {c[0]: c for c in COMPETENCIES}


def make_name_pool(rng: random.Random, n: int, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    names: list[str] = []
    seen = set(exclude)
    while len(names) < n:
        full = f"{rng.choice(FIRST_A)}{rng.choice(FIRST_B)} {rng.choice(LAST_A)}{rng.choice(LAST_B)}"
        if full not in seen:
            seen.add(full)
            names.append(full)
    return names


@dataclass
class Officer:
    officer_id: str
    name: str
    tier: str
    track: str
    years_service: int
    current_role: str
    career_history: str
    truth: dict[str, int]
    pronoun: tuple[str, str] = field(repr=False)


def _make_officers(rng: random.Random, n: int) -> list[Officer]:
    names = make_name_pool(rng, n)
    officers = []
    for i in range(n):
        r = rng.random()
        tier = "high" if r < 0.17 else ("mid" if r < 0.52 else "junior")
        track = rng.choice(list(TRACKS))
        years = {"high": rng.randint(14, 30), "mid": rng.randint(8, 18), "junior": rng.randint(3, 9)}[tier]
        area = rng.choice(TRACK_AREAS[track])
        title = {"high": rng.choice(["Director", "Deputy Director"]), "mid": "Senior Manager",
                 "junior": "Manager"}[tier]
        agency = rng.choice(AGENCIES)

        truth: dict[str, int] = {}
        track_comps = TRACKS[track]
        other = [c[0] for c in COMPETENCIES if c[0] not in track_comps]
        if tier == "high":
            for c in track_comps:
                if rng.random() < 0.9:
                    truth[c] = rng.choices([3, 4], weights=[0.4, 0.6])[0]
            for c in rng.sample(other, rng.randint(3, 6)):
                truth[c] = rng.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0]
        elif tier == "mid":
            for c in rng.sample(track_comps, rng.randint(4, len(track_comps))):
                truth[c] = rng.choices([2, 3, 4], weights=[0.35, 0.55, 0.1])[0]
            for c in rng.sample(other, rng.randint(2, 4)):
                truth[c] = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
        else:
            for c in rng.sample(track_comps, rng.randint(2, 4)):
                truth[c] = rng.choices([1, 2, 3], weights=[0.35, 0.5, 0.15])[0]
            for c in rng.sample(other, rng.randint(1, 3)):
                truth[c] = rng.choices([1, 2], weights=[0.6, 0.4])[0]

        # Talent pool: most senior officers (and some mid) are being groomed for one specific
        # role, so their profile sits at or just below that role's requirements.
        if (tier == "high" and rng.random() < 0.85) or (tier == "mid" and rng.random() < 0.3):
            _, _, _, role_reqs = rng.choice(ROLES)
            p_short = 0.12 if tier == "high" else 0.3
            for cid, req_lvl, _, _ in role_reqs:
                lvl = req_lvl - 1 if rng.random() < p_short else req_lvl
                truth[cid] = max(truth.get(cid, 0), max(lvl, 1))

        # Career history: 2-4 earlier postings, fictional agencies.
        start = 2026 - years
        n_post = rng.randint(2, 4)
        cuts = sorted(rng.sample(range(start + 1, 2026), min(n_post - 1, max(1, 2026 - start - 1))))
        bounds = [start] + cuts + [2026]
        posts = []
        for j in range(len(bounds) - 1):
            ptitle = ["Officer", "Manager", "Senior Manager", "Deputy Director"][min(j, 3)]
            parea = rng.choice(TRACK_AREAS[rng.choice(list(TRACKS))])
            posts.append(f"{bounds[j]}-{bounds[j + 1]}: {ptitle} ({parea}), {rng.choice(AGENCIES)}")
        officers.append(Officer(
            officer_id=f"O{i + 1:04d}", name=names[i], tier=tier, track=track, years_service=years,
            current_role=f"{title} ({area}), {agency}", career_history="; ".join(posts),
            truth=truth, pronoun=rng.choice(PRONOUNS),
        ))
    return officers


def _fill(frame: str, officer: Officer, rng: random.Random, obj: str = "", use_name: bool | None = None,
          course: str = "") -> str:
    first = officer.name.split()[0]
    if use_name is None:
        use_name = rng.random() < 0.45
    subj = rng.choice([first, officer.name]) if use_name else officer.pronoun[0]
    return frame.format(S=subj, pos=officer.pronoun[1], obj=obj, course=course)


def _evidence_sentence(officer: Officer, cid: str, level: int, rng: random.Random, course: bool = False,
                       use_name: bool | None = None) -> str:
    comp = comp_lookup()[cid]
    obj = rng.choice(comp[4]) if rng.random() < 0.15 else rng.choice(comp[3])
    if course:
        lvl = min(level, 2)
        return _fill(rng.choice(COURSE_FRAMES[lvl]), officer, rng, obj, use_name, rng.choice(COURSES)), lvl
    return _fill(rng.choice(FRAMES[level]), officer, rng, obj, use_name), level


def _doc_level(true_level: int, year: int, rng: random.Random, p_understate: float = 0.15) -> int:
    """Documents sometimes understate an officer's level, especially older ones."""
    lvl = true_level
    if year <= 2020 and rng.random() < 0.5:
        lvl -= 1
    if rng.random() < p_understate:
        lvl -= 1
    return max(1, lvl)


def _make_documents(officer: Officer, rng: random.Random, ref: date):
    """Return list of (doc_meta, text, labels). labels: list of (sentence, competency_id|None, level|None)."""
    n_docs = {"high": 6, "mid": rng.randint(4, 5), "junior": rng.randint(3, 4)}[officer.tier]
    earliest = max(2017, 2026 - officer.years_service)
    types = ["appraisal", "posting_summary", "feedback_360", "course_record"]
    doc_types = types[: min(n_docs, 4)] + [rng.choice(["appraisal", "feedback_360"]) for _ in range(n_docs - 4)]
    rng.shuffle(doc_types)
    comps = list(officer.truth)
    weights = [officer.truth[c] ** 2 for c in comps]
    docs = []
    for k, dtype in enumerate(doc_types):
        if officer.tier == "high":
            year = rng.randint(max(earliest, 2024), 2026)
        else:
            year = rng.randint(earliest, 2026)
        month = rng.randint(1, 9 if year == ref.year else 12)
        d = date(year, month, rng.randint(1, 28))
        labels: list[tuple[str, str | None, int | None]] = []
        if dtype == "course_record":
            chosen = list(dict.fromkeys(rng.choices(comps, weights=weights, k=rng.randint(1, 3))))
        else:
            # Each competency the officer has shows up with a probability that rises with tier
            # and level, so strong officers accumulate repeated, independent evidence.
            p_base = {"high": 0.9, "mid": 0.55, "junior": 0.45}[officer.tier]
            p_type = {"appraisal": 1.0, "feedback_360": 0.85, "posting_summary": 0.6}[dtype]
            chosen = [c for c in comps if rng.random() < p_base * p_type * (0.7 + 0.1 * officer.truth[c])]
            if not chosen:
                chosen = [rng.choices(comps, weights=weights)[0]]
        is_course = dtype == "course_record"
        is_360 = dtype == "feedback_360"
        for c in chosen:
            lvl = _doc_level(officer.truth[c], year, rng, 0.08 if officer.tier == "high" else 0.15)
            if is_360:
                prefix = rng.choice(FEEDBACK_PREFIXES)
                s, lvl = _evidence_sentence(officer, c, lvl, rng, use_name=True if prefix else None)
                s = prefix + s if prefix else s
            else:
                s, lvl = _evidence_sentence(officer, c, lvl, rng, course=is_course)
            labels.append((s, c, lvl))
        # Noise: fillers, negations, distractors.
        for _ in range(rng.randint(0, 2)):
            labels.append((_fill(rng.choice(FILLER), officer, rng), None, None))
        missing = [c for c in comp_lookup() if c not in officer.truth]
        if missing and rng.random() < 0.35:
            obj = rng.choice(comp_lookup()[rng.choice(missing)][3])
            labels.append((_fill(rng.choice(NEGATION_FRAMES), officer, rng, obj), None, None))
        if rng.random() < 0.15:
            obj = rng.choice(comp_lookup()[rng.choice(list(comp_lookup()))][3])
            labels.append((_fill(rng.choice(DISTRACTOR_FRAMES), officer, rng, obj), None, None))
        rng.shuffle(labels)
        # Terse, low-quality docs: sometimes only keep a couple of sentences.
        if dtype != "course_record" and officer.tier != "high" and rng.random() < 0.12:
            labels = labels[:2]

        if dtype == "posting_summary":
            area = officer.current_role.split("(")[1].split(")")[0] if k == 0 else rng.choice(
                TRACK_AREAS[officer.track])
            labels.insert(0, (f"Posted to {rng.choice(AGENCIES)} ({area}) from {year - 2} to {year}.", None, None))
        if is_360 or is_course:
            body = "\n".join(f"- {s}" for s, _, _ in labels)
        else:
            body = " ".join(s for s, _, _ in labels)
        doc_id = f"{officer.officer_id}_D{k + 1}"
        header = (f"DOCUMENT: {DOC_TITLES[dtype]}\nOFFICER: {officer.name} ({officer.officer_id})\n"
                  f"DATE: {d.isoformat()}\n\n")
        docs.append(({"doc_id": doc_id, "officer_id": officer.officer_id, "doc_type": dtype,
                      "date": d.isoformat(), "path": f"documents/{officer.officer_id}/{doc_id}.txt"},
                     header + body + "\n", labels))
    return docs


def main() -> None:
    s = load_settings()
    rng = random.Random(s.general.seed)
    data = s.data_dir
    # Remove old documents file-by-file (folders kept: sync clients such as OneDrive may lock them).
    for old in s.docs_dir.glob("*/*.txt"):
        old.unlink()
    s.docs_dir.mkdir(parents=True, exist_ok=True)

    comp_rows = []
    for cid, name, focus, _, _ in COMPETENCIES:
        row = {"competency_id": cid, "name": name, "description": f"{name}: {focus}."}
        for lvl in range(1, 5):
            row[f"level_{lvl}"] = f"{LEVEL_NAMES[lvl]} - " + LEVEL_TEMPLATES[lvl].format(focus=focus)
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(data / "competencies.csv", index=False)

    roles, reqs = [], []
    for rid, title, agency, rr in ROLES:
        roles.append({"role_id": rid, "title": title, "agency": agency})
        for cid, lvl, w, mh in rr:
            reqs.append({"role_id": rid, "competency_id": cid, "required_level": lvl, "weight": w,
                         "must_have": mh})
    pd.DataFrame(roles).to_csv(data / "roles.csv", index=False)
    pd.DataFrame(reqs).to_csv(data / "role_requirements.csv", index=False)

    opp_rows = []
    comps = comp_lookup()
    for i, (otype, title, primary, extra, max_lvl, months) in enumerate(OPPORTUNITIES):
        tags = [primary] + extra
        focus = " and ".join(comps[t][2] for t in tags)
        opp_rows.append({"opportunity_id": f"P{i + 1:02d}", "type": otype, "title": title,
                         "competencies": ";".join(tags), "max_level": max_lvl, "duration_months": months,
                         "description": f"{otype}: {title}. Builds {focus}."})
    pd.DataFrame(opp_rows).to_csv(data / "opportunities.csv", index=False)

    officers = _make_officers(rng, s.generation.n_officers)
    pd.DataFrame([{"officer_id": o.officer_id, "name": o.name, "current_role": o.current_role,
                   "years_service": o.years_service, "career_history": o.career_history}
                  for o in officers]).to_csv(data / "officers.csv", index=False)

    doc_index, all_labels = [], []
    for o in officers:
        (s.docs_dir / o.officer_id).mkdir(exist_ok=True)
        for meta, text, labels in _make_documents(o, rng, s.general.reference_date):
            (data / meta["path"]).write_text(text, encoding="utf-8")
            doc_index.append(meta)
            all_labels.append((meta, labels))
    pd.DataFrame(doc_index).to_csv(data / "documents_index.csv", index=False)

    # Ground truth: 30 docs, stratified across doc types, each with >= 1 evidence sentence.
    gt_rng = random.Random(s.general.seed + 1)
    by_type: dict[str, list] = {t: [] for t in DOC_TYPES}
    for meta, labels in all_labels:
        if any(c for _, c, _ in labels):
            by_type[meta["doc_type"]].append((meta, labels))
    n = s.generation.n_ground_truth_docs
    per_type = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
    gt_rows = []
    for t, k in zip(DOC_TYPES, per_type):
        for meta, labels in gt_rng.sample(by_type[t], k):
            for sent, cid, lvl in labels:
                gt_rows.append({"doc_id": meta["doc_id"], "sentence": sent,
                                "competency_id": cid or "", "level": lvl or 0})
    pd.DataFrame(gt_rows).to_csv(data / "ground_truth.csv", index=False)

    print(f"Generated {len(officers)} officers, {len(doc_index)} documents, "
          f"{len(opp_rows)} opportunities, ground truth on {n} docs -> {data}")


if __name__ == "__main__":
    main()
