"""
compliance_engine.py
---------------------
Main entry point for the Legal Compliance Engine.

    compliance(main file)
       |__ cuad_module.py
       |__ indian_module.py

This is the ONLY file your teammate needs to import to plug this into the rest
of the project. It takes one meeting sentence (or a list of them, from Member
2's summariser) and returns the merged verdict across both branches.

Usage:
    from compliance_engine import ComplianceEngine

    engine = ComplianceEngine()                     # loads both modules once
    result = engine.analyze("We will terminate the agreement immediately.")
    print(result["overall_compliance"], result["overall_reason"])

    # or for a whole batch of keypoints from Member 2:
    results = engine.analyze_batch(member2_keypoints)
"""

from typing import List, Dict, Any

from cuad_module import CUADModule
from indian_module import IndianCorpusModule


class ComplianceEngine:
    def __init__(self, cuad_artifact_dir: str = None, indian_artifact_dir: str = None):
        kwargs = {"artifact_dir": cuad_artifact_dir} if cuad_artifact_dir else {}
        self.cuad = CUADModule(**kwargs)
        self.indian = IndianCorpusModule(artifact_dir=indian_artifact_dir)

    # ------------------------------------------------------------------
    @staticmethod
    def _merge(cuad_result: Dict[str, Any], indian_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge policy (adjust to taste once both branches are real):
        - either branch NON-COMPLIANT  -> overall NON-COMPLIANT
        - else either branch NEEDS REVIEW -> overall NEEDS REVIEW
        - else COMPLIANT
        Ties are broken toward whichever branch found the more confident
        (higher retrieval_score) match, for the displayed "reason".
        """
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
        return self._merge(cuad_result, indian_result)

    def analyze_batch(self, meeting_sentences: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze(s) for s in meeting_sentences]


import re

def split_into_sentences(paragraph: str):
    """Cuts one big paragraph into a list of individual sentences,
    splitting at each '.', '!' or '?' followed by a space."""
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