"""
compliance_engine.py
---------------------
Main entry point for the Legal Compliance Engine.

    compliance(main file)
       |__ cuad_module.py
       |__ indian_module.py
       |__ legal_signals.py   (NER + POS + chunking layer)
"""

import re
from typing import List, Dict, Any

from cuad_module import CUADModule
from indian_module import IndianCorpusModule
from legal_signals import LegalSignals


class ComplianceEngine:
    def __init__(self, cuad_artifact_dir: str = None, indian_artifact_dir: str = None):
        kwargs = {"artifact_dir": cuad_artifact_dir} if cuad_artifact_dir else {}
        self.cuad = CUADModule(**kwargs)
        self.indian = IndianCorpusModule(artifact_dir=indian_artifact_dir)
        self.signals = LegalSignals()

    # ------------------------------------------------------------------
    @staticmethod
    def _merge(cuad_result: Dict[str, Any], indian_result: Dict[str, Any]) -> Dict[str, Any]:
        branches = [cuad_result, indian_result]
        verdicts = [b["compliance"] for b in branches]

        if "NON-COMPLIANT" in verdicts:
            overall = "NON-COMPLIANT"
            leading = next(b for b in branches if b["compliance"] == "NON-COMPLIANT")
        elif "NEEDS REVIEW" in verdicts:
            overall = "NEEDS REVIEW"
            leading = max(
                (b for b in branches if b["compliance"] == "NEEDS REVIEW"),
                key=lambda b: b["retrieval_score"],
            )
        else:
            overall = "COMPLIANT"
            leading = max(branches, key=lambda b: b["retrieval_score"])

        return {
            "overall_compliance": overall,
            "overall_reason": f"[{leading['source']}] {leading['reason']}",
            "cuad": cuad_result,
            "indian_corpus": indian_result,
        }

    # ------------------------------------------------------------------
    def analyze(self, meeting_sentence: str) -> Dict[str, Any]:
        cuad_result = self.cuad.analyze(meeting_sentence)
        indian_result = self.indian.analyze(meeting_sentence)
        merged = self._merge(cuad_result, indian_result)

        # classical NLP layer: NER + POS + chunking
        sig = self.signals.extract(meeting_sentence)
        clauses = [c["clause"] for c in cuad_result["candidates_checked"]]
        flags = self.signals.check_against_clauses(sig, clauses)

        merged["linguistic_signals"] = sig
        merged["rule_flags"] = flags

        if flags:
            if merged["overall_compliance"] == "COMPLIANT":
                merged["overall_compliance"] = "NEEDS REVIEW"   # rules only escalate, never relax
            merged["overall_reason"] += " | Rule check: " + flags[0]
        return merged

    def analyze_batch(self, meeting_sentences: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze(s) for s in meeting_sentences]


def split_into_sentences(paragraph: str):
    """Cuts one big paragraph into individual sentences."""
    pieces = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [p.strip() for p in pieces if p.strip()]


if __name__ == "__main__":
    engine = ComplianceEngine()

    raw_text = input("Enter meeting keypoints / Decisions: ").strip()
    sentences = split_into_sentences(raw_text)

    print(f"\nDetected {len(sentences)} separate sentences.\n")

    all_results = []
    for i, sentence in enumerate(sentences, start=1):
        print(f"\n{'#'*80}\nSENTENCE {i} of {len(sentences)}: {sentence}\n{'#'*80}")

        result = engine.analyze(sentence)
        all_results.append(result)

        print(f"\nOVERALL: {result['overall_compliance']}")
        print(f"REASON: {result['overall_reason']}")

        print(f"\n--- CUAD (contract) details ---")
        print(f"Matched category: {result['cuad']['matched_category']}")
        for j, c in enumerate(result['cuad']['candidates_checked'], start=1):
            print(f"  [{j}] {c['compliance']} (score {c['score']:.2f})")
            print(f"      {c['clause']}")

        print(f"\n--- Indian Corpus (law) details ---")
        print(f"  {result['indian_corpus']['reason']}")

        print(f"\n--- Linguistic signals (NER / POS / chunking) ---")
        sig = result["linguistic_signals"]
        print(f"  Entities:   {sig['entities']}")
        print(f"  Durations:  {sig['durations']}")
        print(f"  Jurisdictions: {sig['jurisdictions']}")
        print(f"  Modals:     {sig['modals']}")
        print(f"  Negations:  {sig['negations']}")
        print(f"  Chunks:     {sig['chunks']}")
        for f in result["rule_flags"]:
            print(f"  FLAG: {f}")
            print(f"\n--- Linguistic signals (NER / POS / chunking) ---")
            sig = result["linguistic_signals"]
            for key in ("entities", "durations", "jurisdictions", "modals", "negations", "chunks"):
                if sig[key]:
                    print(f"  {key}: {sig[key]}")
            if not result["rule_flags"]:
                print("  (no rule flags)")
            for f in result["rule_flags"]:
                print(f"  FLAG: {f}")