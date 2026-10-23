
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "pipeline" / "output" / "facts.json"
GOLD_FILE = PROJECT_ROOT / "pipeline" / "output" / "gold_facts.json"


# Fields used to identify a fact.
# Include status/owner/deadline where they are important to the fact.
CATEGORY_FIELDS = {
    "key_points": ["point"],
    "decisions": ["decision", "status"],
    "action_items": ["task", "owner", "deadline", "status"],
    "commitments": ["commitment", "committed_by", "due_date"],
    "dates_and_deadlines": ["date_or_deadline", "event"],
    "problems_and_risks": ["issue"],
    "people_mentioned": ["name_or_speaker_id", "role_or_context"],
}


def normalize(value):
    """Normalize text for comparison without changing its meaning."""
    if value is None:
        return "<null>"

    value = unicodedata.normalize("NFKC", str(value))
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def fact_signature(fact, fields):
    """Create a comparable representation of one fact."""
    return tuple(normalize(fact.get(field)) for field in fields)


def get_facts(data, category):
    facts = data.get(category, [])
    if not isinstance(facts, list):
        raise ValueError(f"'{category}' must contain a JSON array.")

    # Ignore malformed entries instead of crashing.
    return [fact for fact in facts if isinstance(fact, dict)]


def score_category(predicted, gold, fields):
    """
    Exact normalized matching, with duplicate facts counted only once.
    Returns TP, FP, FN, precision, recall, and F1.
    """
    predicted_counts = Counter(
        fact_signature(fact, fields) for fact in predicted
    )
    gold_counts = Counter(
        fact_signature(fact, fields) for fact in gold
    )

    # A matched fact is a true positive.
    tp = sum((predicted_counts & gold_counts).values())
    fp = sum(predicted_counts.values()) - tp
    fn = sum(gold_counts.values()) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Predictions not found: {PREDICTIONS_FILE}")

    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Gold reference not found: {GOLD_FILE}\n"
            "Create gold_facts.json by manually annotating the transcript."
        )

    with PREDICTIONS_FILE.open(encoding="utf-8") as f:
        predictions = json.load(f)

    with GOLD_FILE.open(encoding="utf-8") as f:
        gold = json.load(f)

    results = {}
    total_tp = total_fp = total_fn = 0
    category_f1s = []

    print("\nBEACON Fact Extraction Evaluation")
    print("=" * 76)
    print(f"{'Category':<25} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Precision':>11} {'Recall':>9} {'F1':>9}")
    print("-" * 76)

    for category, fields in CATEGORY_FIELDS.items():
        pred_facts = get_facts(predictions, category)
        gold_facts = get_facts(gold, category)

        result = score_category(pred_facts, gold_facts, fields)
        results[category] = result

        total_tp += result["TP"]
        total_fp += result["FP"]
        total_fn += result["FN"]
        category_f1s.append(result["f1"])

        print(
            f"{category:<25} "
            f"{result['TP']:>5} {result['FP']:>5} {result['FN']:>5} "
            f"{result['precision']:>10.3f} "
            f"{result['recall']:>9.3f} "
            f"{result['f1']:>9.3f}"
        )

    # Micro scores: combine TP/FP/FN across categories.
    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )

    # Macro F1: average category F1 scores equally.
    macro_f1 = sum(category_f1s) / len(category_f1s) if category_f1s else 0.0

    print("-" * 76)
    print(f"Micro precision: {micro_precision:.3f}")
    print(f"Micro recall:    {micro_recall:.3f}")
    print(f"Micro F1:        {micro_f1:.3f}")
    print(f"Macro F1:        {macro_f1:.3f}")

    # Save detailed metrics for reporting.
    report = {
        "categories": results,
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
        },
        "macro_f1": macro_f1,
        "matching_method": "exact normalized field matching",
    }

    report_path = PROJECT_ROOT / "pipeline" / "output" / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
