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

ADAPTER_PATH = r"D:\BEACON\models\beacon_flan_t5_v2_1_refined_lora"

INPUT_FILE = Path(
    r"D:\BEACON\pipeline\output\speaker_transcript.txt"
)

OUTPUT_DIR = Path(
    r"D:\BEACON\pipeline\output"
)

OUTPUT_FILE = OUTPUT_DIR / "facts.json"


# ============================================================
# MODEL SETTINGS
# ============================================================

MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 96

CONTEXT_TURNS = 6
STEP = 3


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
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print("Loading BEACON V2.1 adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH
    )

    if torch.cuda.is_available():
        model = model.cuda()
        print("Fact extraction device: CUDA")
    else:
        print("Fact extraction device: CPU")

    model.eval()

    return tokenizer, model


# ============================================================
# LOAD SPEAKER TRANSCRIPT
# ============================================================

def load_transcript():

    pattern = re.compile(
        r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(\S+):\s*(.*)"
    )

    turns = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            match = pattern.match(line)

            if not match:
                continue

            start = float(match.group(1))
            end = float(match.group(2))
            speaker = match.group(3)
            text = match.group(4).strip()

            if not text:
                continue

            turns.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text
            })

    return turns


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(turns):

    lines = []

    for turn in turns:

        lines.append(
            f"[{turn['start']:.2f} - {turn['end']:.2f}] "
            f"{turn['speaker']}: "
            f"{turn['text']}"
        )

    return "\n".join(lines)


# ============================================================
# EXTRACT FACT
# ============================================================

def extract_fact(context, tokenizer, model):

    prompt = f"""
You are the BEACON meeting fact extraction system.

Analyze the following meeting conversation.

Extract ONE important factual statement that is explicitly supported
by the conversation.

Prioritize information involving:

- decisions
- requirements
- commitments
- deadlines
- restrictions
- permissions
- financial conditions
- risks
- compliance-related conditions
- security
- performance
- availability
- responsibilities
- important actions

Ignore:

- greetings
- introductions
- casual conversation
- jokes
- opinions without a concrete implication
- repeated statements
- irrelevant background information

Rules:
1. Use ONLY information explicitly stated in the conversation.
2. Do NOT invent information.
3. Preserve important names, dates, numbers and conditions.
4. Combine closely related statements when they express the same fact.
5. Produce ONE concise factual statement.
6. If there is no important factual information, output NONE.

MEETING CONVERSATION:

{context}

FACT:
"""

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

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    return result


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_facts():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer, model = load_model()

    print("\nLoading speaker transcript...")

    turns = load_transcript()

    print(f"Speaker turns: {len(turns)}")

    if not turns:
        raise RuntimeError(
            "No speaker turns found in speaker_transcript.txt"
        )

    results = []

    window_id = 0

    print("\nStarting BEACON fact extraction...\n")

    for start in range(
        0,
        len(turns),
        STEP
    ):

        window = turns[
            start:start + CONTEXT_TURNS
        ]

        if not window:
            break

        context = build_context(window)

        fact = extract_fact(
            context,
            tokenizer,
            model
        )

        window_id += 1

        print(
            f"[{window_id}] "
            f"{window[0]['start']:.2f}s - "
            f"{window[-1]['end']:.2f}s"
        )

        print(
            f"Fact: {fact}\n"
        )

        if not fact:
            continue

        if fact.upper() == "NONE":
            continue

        results.append({

            "id": window_id,

            "start": window[0]["start"],

            "end": window[-1]["end"],

            "speakers": sorted(
                list(
                    set(
                        turn["speaker"]
                        for turn in window
                    )
                )
            ),

            "fact": fact,

            "source_turns": [
                {
                    "start": turn["start"],
                    "end": turn["end"],
                    "speaker": turn["speaker"],
                    "text": turn["text"]
                }
                for turn in window
            ]

        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n========================================")
    print("BEACON FACT EXTRACTION COMPLETED")
    print("========================================")

    print(f"Input turns : {len(turns)}")
    print(f"Facts       : {len(results)}")
    print(f"Output      : {OUTPUT_FILE}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    extract_facts()