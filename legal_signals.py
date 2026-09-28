"""
legal_signals.py
----------------
Classical NLP layer for BEACON: NER + POS + chunking (spaCy, no training).
Signals are extracted from the meeting sentence AND from every retrieved clause,
then compared generically. Nothing here depends on specific example sentences.
"""

import re
import spacy

_NUM = r"(?:one|two|three|four|five|six|seven|ten|fifteen|twenty|thirty|forty|sixty|ninety|\d+)"
DURATION_RE = re.compile(
    rf"\b{_NUM}(?:\s*\(\d+\))?[\s-]*(?:business\s+|calendar\s+)?(?:day|week|month|year)s?\b", re.I)
IMMEDIATE_RE = re.compile(
    r"\b(immediately|forthwith|with immediate effect|effective immediately)\b", re.I)
NO_NOTICE_RE = re.compile(
    r"\bwithout\s+(?:giving\s+)?(?:any\s+)?(?:prior\s+)?(?:written\s+)?(notice|cause|consent|approval)\b", re.I)
JURIS_RE = re.compile(
    r"\b(?i:laws?|courts?|jurisdiction)\s+of\s+(?:the\s+)?(?:State of\s+)?([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)*)")
INDIAN_LAW_RE = re.compile(
    r"\b(?:Companies Act(?:,?\s*\d{4})?|SEBI(?:\s*\(LODR\))?|LODR|MCA|Section\s+\d+[A-Za-z]*(?:\(\d+\))?)\b")

# stance patterns (work on any sentence or clause)
PROHIBIT_RE = re.compile(
    r"\b(?:shall not|shall never|may not|must not|will not|cannot|agrees? not to|"
    r"not to (?:engage|compete|solicit|sell|disclose|use)|prohibited|prohibits|"
    r"refrain from|restricted from|precluded from)\b|\bneither\b.{0,120}?\bshall\b", re.I)
PERMIT_RE = re.compile(
    r"\b(?:free to|freely|entitled to|allowed to|permitted to|at liberty to|can also|may also)\b", re.I)
UNLIMITED_RE = re.compile(
    r"\b(?:no (?:cap|limit|ceiling|maximum)|unlimited|uncapped|without (?:any )?(?:cap|limit)|"
    r"not (?:be )?(?:limited|capped))\b", re.I)
CAP_RE = re.compile(
    r"\b(?:shall not exceed|not to exceed|in no event|aggregate liability|capped at|"
    r"limited to|maximum liability|liability cap|cap on)\b", re.I)

INDIA_TERMS = {"india", "indian", "mumbai", "delhi", "maharashtra", "bengaluru", "chennai", "kolkata"}
WORD2NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10,
            "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60, "ninety": 90}
MODAL_STRENGTH = {"shall": "obligation", "must": "obligation", "will": "commitment",
                  "should": "recommendation", "may": "permission", "can": "permission",
                  "could": "permission", "might": "permission"}
NEG_WORDS = {"without", "no", "not", "never", "nor", "cannot", "neither", "n't"}


def _duration_key(text: str):
    """'thirty (30) days' -> (30, 'day')"""
    t = text.lower()
    digits = re.search(r"\d+", t)
    if digits:
        n = int(digits.group())
    else:
        n = None
        for w in sorted(WORD2NUM, key=len, reverse=True):
            if re.search(rf"\b{w}\b", t):
                n = WORD2NUM[w]
                break
    unit = re.search(r"(day|week|month|year)", t)
    return (n, unit.group(1)) if n and unit else None


class LegalSignals:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    # ------------------------------------------------------------------
    def extract(self, text: str, doc=None) -> dict:
        """Works on a meeting sentence or a contract clause."""
        doc = doc or self.nlp(text)

        # ---- NER: spaCy pretrained + legal regex layer ----
        ents = [{"text": e.text, "label": e.label_} for e in doc.ents
                if e.label_ in {"ORG", "GPE", "PERSON", "DATE", "MONEY", "PERCENT", "LAW"}]
        durations = [m.group(0) for m in DURATION_RE.finditer(text)]
        juris = [m.group(1) for m in JURIS_RE.finditer(text)]
        indian_refs = [m.group(0) for m in INDIAN_LAW_RE.finditer(text)]

        # ---- POS: modality + negation ----
        modals = [{"word": t.text, "strength": MODAL_STRENGTH.get(t.lower_, "other")}
                  for t in doc if t.tag_ == "MD"]
        negations = [t.text for t in doc if t.dep_ == "neg" or t.lower_ in NEG_WORDS]

        # ---- Chunking: actor / action / object ----
        actors = [c.text for c in doc.noun_chunks if c.root.dep_ in ("nsubj", "nsubjpass")]
        objects = [c.text for c in doc.noun_chunks if c.root.dep_ in ("dobj", "pobj", "attr")]
        actions = []
        for t in doc:
            if t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "conj"):
                aux = sorted([a for a in t.children if a.dep_ in ("aux", "auxpass", "neg")],
                             key=lambda x: x.i)
                actions.append(" ".join([a.text for a in aux] + [t.text]))

        permission_modal = any(m["strength"] == "permission" for m in modals) and not negations
        return {
            "entities": ents,
            "durations": durations,
            "jurisdictions": list(dict.fromkeys(juris + [e["text"] for e in ents if e["label"] == "GPE"])),
            "indian_law_refs": indian_refs,
            "modals": modals,
            "negations": negations,
            "immediate": bool(IMMEDIATE_RE.search(text)),
            "no_notice": bool(NO_NOTICE_RE.search(text)),
            "prohibits": bool(PROHIBIT_RE.search(text)),
            "permits": bool(PERMIT_RE.search(text)) or permission_modal,
            "unlimited": bool(UNLIMITED_RE.search(text)),
            "has_cap": bool(CAP_RE.search(text)),
            "chunks": {"actors": actors, "actions": actions, "objects": objects},
        }

    # ------------------------------------------------------------------
    def check_against_clauses(self, sig: dict, clauses: list) -> list:
        """Compare the sentence's signals with signals from each retrieved clause."""
        flags = []
        n = len(clauses)
        if n == 0:
            return flags
        clause_sigs = [self.extract(c, doc=d) for c, d in zip(clauses, self.nlp.pipe(clauses))]

        # 1. sentence GRANTS freedom, clauses PROHIBIT / restrict
        if sig["permits"] and not sig["prohibits"]:
            k = sum(cs["prohibits"] for cs in clause_sigs)
            if k:
                flags.append(f"Decision grants freedom/permission, but {k} of {n} similar "
                             f"clauses prohibit or restrict it.")

        # 2. sentence removes a limit, clauses impose a cap
        if sig["unlimited"]:
            k = sum(cs["has_cap"] for cs in clause_sigs)
            if k:
                flags.append(f"Decision removes any limit, but {k} of {n} similar clauses "
                             f"impose a cap.")

        # 3. acts immediately / without notice vs clauses requiring a notice period
        clause_durs = set()
        for cs in clause_sigs:
            for d in cs["durations"]:
                key = _duration_key(d)
                if key:
                    clause_durs.add(key)
        if (sig["immediate"] or sig["no_notice"]) and clause_durs:
            periods = ", ".join(f"{a} {u}(s)" for a, u in sorted(clause_durs))
            flags.append(f"Decision acts immediately / without notice, but similar clauses "
                         f"require a period ({periods}).")

        # 4. stated duration not found in any similar clause
        for d in sig["durations"]:
            key = _duration_key(d)
            if key and clause_durs and key not in clause_durs:
                flags.append(f"Duration '{d}' does not match durations in similar clauses.")

        # 5. non-Indian governing law
        for j in sig["jurisdictions"]:
            if j.lower() not in INDIA_TERMS:
                flags.append(f"Jurisdiction '{j}' is non-Indian; Indian statutes "
                             f"(Companies Act / SEBI) may still apply.")
                break
        return flags


if __name__ == "__main__":
    ls = LegalSignals()
    tests = [
        ("Our employee is free to start a competing business the day after leaving the company.",
         ["Employee agrees not to engage in a competitive business for two (2) years after termination."]),
        ("There is no cap on how much either party can be sued for.",
         ["In no event shall liability exceed the fees paid in the prior twelve (12) months."]),
    ]
    for s, cl in tests:
        print(s, "->", ls.check_against_clauses(ls.extract(s), cl))