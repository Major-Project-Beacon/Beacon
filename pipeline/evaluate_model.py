import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PATHS
# ============================================================

BASE_MODEL = "google/flan-t5-base"

MODEL_PATH = Path(
    r"D:\BEACON\models\beacon_flan_t5_v2_1_refined_lora"
)

TEST_FILE = Path(
    r"D:\BEACON\datasets\beacon_unified"
    r"\beacon_v2_factual_augmented\test.jsonl"
)

OUTPUT_DIR = Path(
    r"D:\BEACON\pipeline\output\evaluation"
)

RESULT_FILE = OUTPUT_DIR / "v2_1_evaluation.json"


# ============================================================
# SETTINGS
# ============================================================

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 96


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s%$.-]",
        "",
        text
    )

    return text


def tokenize(text):

    return set(
        normalize(text).split()
    )


# ============================================================
# TOKEN METRICS
# ============================================================

def calculate_metrics(reference, prediction):

    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not pred_tokens:
        return 0.0, 0.0, 0.0

    if not ref_tokens:
        return 0.0, 0.0, 0.0

    overlap = len(
        ref_tokens & pred_tokens
    )

    precision = (
        overlap / len(pred_tokens)
    )

    recall = (
        overlap / len(ref_tokens)
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return precision, recall, f1


# ============================================================
# NUMERIC CHECK
# ============================================================

def extract_numbers(text):

    return re.findall(
        r"\b\d+(?:\.\d+)?%?|\$\d+(?:,\d+)*(?:\.\d+)?",
        text
    )


def numeric_preserved(reference, prediction):

    reference_numbers = extract_numbers(reference)
    prediction_numbers = extract_numbers(prediction)

    if not reference_numbers:
        return None

    return all(
        number in prediction_numbers
        for number in reference_numbers
    )


def has_extra_numbers(reference, prediction):

    reference_numbers = extract_numbers(reference)
    prediction_numbers = extract_numbers(prediction)

    return any(
        number not in reference_numbers
        for number in prediction_numbers
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    records = []

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    print("Loading base model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    print("Loading BEACON V2.1 adapter...")

    model = PeftModel.from_pretrained(
        model,
        str(MODEL_PATH)
    )

    if torch.cuda.is_available():

        model = model.cuda()

        print("Evaluation device: CUDA")

    else:

        print("Evaluation device: CPU")

    model.eval()

    return tokenizer, model


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(record):

    task = record.get(
        "task",
        "general"
    )

    fact_type = record.get(
        "fact_type",
        ""
    )

    category = record.get(
        "category",
        ""
    )

    evidence = record.get(
        "evidence",
        ""
    )

    if task == "cuad":

        prompt = f"""
You are extracting a factual statement from meeting evidence.

CONTRACT CATEGORY: {category}

TASK:
Extract ONLY the fact that is relevant to the
'{category}' category.

RULES:
1. Use only information explicitly supported by the evidence.
2. Do not invent or infer information.
3. Preserve important names, dates, numbers, amounts,
   limits, conditions, and obligations.
4. Write one concise factual statement.

EVIDENCE:
{evidence}

FACTUAL STATEMENT:
"""

    else:

        prompt = f"""
You are extracting a factual statement from meeting evidence.

FACT TYPE: {fact_type}

TASK:
Extract ONLY the fact that is relevant to the
'{fact_type}' fact type.

RULES:
1. Use only information explicitly supported by the evidence.
2. Do not invent or infer information.
3. Preserve important names, dates, numbers, amounts,
   limits, conditions, and obligations.
4. Write one concise factual statement.

EVIDENCE:
{evidence}

FACTUAL STATEMENT:
"""

    return prompt


# ============================================================
# PREDICTION
# ============================================================

def predict(
    prompt,
    tokenizer,
    model
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    if torch.cuda.is_available():

        inputs = {
            key: value.cuda()
            for key, value in inputs.items()
        }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading test dataset...")

    records = load_test_data()

    print(
        f"Test records: {len(records)}"
    )

    tokenizer, model = load_model()

    results = []

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    exact_matches = 0

    numeric_total = 0
    numeric_preserved_count = 0
    extra_number_count = 0

    print("\nStarting evaluation...\n")

    for index, record in enumerate(records, 1):

        reference = record.get(
            "target",
            ""
        )

        prompt = build_prompt(
            record
        )

        prediction = predict(
            prompt,
            tokenizer,
            model
        )

        precision, recall, f1 = calculate_metrics(
            reference,
            prediction
        )

        total_precision += precision
        total_recall += recall
        total_f1 += f1

        if normalize(reference) == normalize(prediction):

            exact_matches += 1

        numeric_result = numeric_preserved(
            reference,
            prediction
        )

        if numeric_result is not None:

            numeric_total += 1

            if numeric_result:

                numeric_preserved_count += 1

        if has_extra_numbers(
            reference,
            prediction
        ):

            extra_number_count += 1

        results.append({

            "index": index,

            "reference": reference,

            "prediction": prediction,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "exact_match":
                normalize(reference)
                == normalize(prediction),

            "category":
                record.get("category"),

            "fact_type":
                record.get("fact_type"),

            "numeric_preserved":
                numeric_result,

            "extra_numbers":
                has_extra_numbers(
                    reference,
                    prediction
                )

        })

        print(
            f"[{index:02d}/{len(records)}] "
            f"F1={f1 * 100:.2f}% | "
            f"{prediction}"
        )

    count = len(records)

    avg_precision = (
        total_precision / count
    )

    avg_recall = (
        total_recall / count
    )

    avg_f1 = (
        total_f1 / count
    )

    exact_match = (
        exact_matches / count
    )

    if numeric_total:

        numeric_preservation = (
            numeric_preserved_count
            / numeric_total
        )

    else:

        numeric_preservation = 0

    no_extra_numbers = (
        1 -
        (
            extra_number_count
            / count
        )
    )

    summary = {

        "model":
            "BEACON V2.1",

        "test_records":
            count,

        "precision":
            avg_precision,

        "recall":
            avg_recall,

        "f1":
            avg_f1,

        "exact_match":
            exact_match,

        "numeric_records":
            numeric_total,

        "numeric_preservation":
            numeric_preservation,

        "no_extra_numbers":
            no_extra_numbers

    }

    output = {

        "summary":
            summary,

        "results":
            results

    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n")
    print("=" * 55)
    print("BEACON V2.1 EVALUATION")
    print("=" * 55)

    print(
        f"Test records          : {count}"
    )

    print(
        f"Precision             : "
        f"{avg_precision * 100:.2f}%"
    )

    print(
        f"Recall                : "
        f"{avg_recall * 100:.2f}%"
    )

    print(
        f"F1 Score              : "
        f"{avg_f1 * 100:.2f}%"
    )

    print(
        f"Exact Match           : "
        f"{exact_match * 100:.2f}%"
    )

    print(
        f"Numeric Preservation  : "
        f"{numeric_preservation * 100:.2f}%"
    )

    print(
        f"No Extra Numbers      : "
        f"{no_extra_numbers * 100:.2f}%"
    )

    print("=" * 55)

    print(
        f"\nDetailed results saved to:\n"
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":

    evaluate()