import re
from pathlib import Path


OUTPUT_DIR = Path(r"D:\BEACON\pipeline\output")

TRANSCRIPT_FILE = OUTPUT_DIR / "transcript.txt"
DIARIZATION_FILE = OUTPUT_DIR / "diarization.txt"
OUTPUT_FILE = OUTPUT_DIR / "speaker_transcript.txt"


def load_transcript():
    """
    Load Whisper transcript.

    Expected format:
    [start - end] text
    """

    pattern = re.compile(
        r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.*)"
    )

    segments = []

    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            match = pattern.match(line)

            if not match:
                continue

            start = float(match.group(1))
            end = float(match.group(2))
            text = match.group(3).strip()

            if text:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })

    return segments


def load_diarization():
    """
    Load pyannote diarization output.

    Expected format:
    [start - end] SPEAKER_XX
    """

    pattern = re.compile(
        r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(\S+)"
    )

    turns = []

    with open(DIARIZATION_FILE, "r", encoding="utf-8") as f:
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

            turns.append({
                "start": start,
                "end": end,
                "speaker": speaker
            })

    return turns


def find_speaker(segment, diarization):
    """
    Find the speaker with the largest temporal overlap
    with a Whisper segment.
    """

    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for turn in diarization:

        overlap_start = max(segment["start"], turn["start"])
        overlap_end = min(segment["end"], turn["end"])

        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    return best_speaker


def merge_transcript():

    if not TRANSCRIPT_FILE.exists():
        raise FileNotFoundError(
            f"Transcript not found:\n{TRANSCRIPT_FILE}"
        )

    if not DIARIZATION_FILE.exists():
        raise FileNotFoundError(
            f"Diarization file not found:\n{DIARIZATION_FILE}"
        )

    print("Loading Whisper transcript...")
    transcript = load_transcript()

    print(f"Transcript segments: {len(transcript)}")

    print("Loading diarization...")
    diarization = load_diarization()

    print(f"Diarization turns: {len(diarization)}")

    print("Merging transcript with speakers...")

    merged = []

    for segment in transcript:

        speaker = find_speaker(
            segment,
            diarization
        )

        merged.append({
            "start": segment["start"],
            "end": segment["end"],
            "speaker": speaker,
            "text": segment["text"]
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for item in merged:

            f.write(
                f"[{item['start']:.2f} - {item['end']:.2f}] "
                f"{item['speaker']}: "
                f"{item['text']}\n"
            )

    print("\nSpeaker transcript created successfully.")
    print(f"Segments: {len(merged)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_transcript()