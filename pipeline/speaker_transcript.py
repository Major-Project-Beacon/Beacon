import re
from pathlib import Path


TRANSCRIPT_PATTERN = re.compile(
    r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.*)"
)

DIARIZATION_PATTERN = re.compile(
    r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(\S+)"
)


def load_transcript(transcript_file: str | Path):
    segments = []

    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            match = TRANSCRIPT_PATTERN.match(line.strip())
            if not match:
                continue

            start, end = float(match.group(1)), float(match.group(2))
            text = match.group(3).strip()

            if text:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                })

    return segments


def load_diarization(diarization_file: str | Path):
    turns = []

    with open(diarization_file, "r", encoding="utf-8") as f:
        for line in f:
            match = DIARIZATION_PATTERN.match(line.strip())
            if not match:
                continue

            turns.append({
                "start": float(match.group(1)),
                "end": float(match.group(2)),
                "speaker": match.group(3),
            })

    return turns


def find_speaker(segment, diarization):
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


def merge_transcript(
    transcript_file: str | Path,
    diarization_file: str | Path,
    output_file: str | Path,
) -> str:
    transcript_file = Path(transcript_file)
    diarization_file = Path(diarization_file)
    output_file = Path(output_file)

    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_file}")

    if not diarization_file.exists():
        raise FileNotFoundError(f"Diarization not found: {diarization_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    transcript = load_transcript(transcript_file)
    diarization = load_diarization(diarization_file)

    merged = []

    for segment in transcript:
        speaker = find_speaker(segment, diarization)

        merged.append({
            **segment,
            "speaker": speaker,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(
                f"[{item['start']:.2f} - {item['end']:.2f}] "
                f"{item['speaker']}: {item['text']}\n"
            )

    print(f"Speaker transcript saved: {output_file}")
    print(f"Segments aligned: {len(merged)}")

    return str(output_file)


if __name__ == "__main__":
    # Temporary standalone test paths. Change these to your files.
    output_dir = Path(__file__).parent / "output"

    merge_transcript(
        transcript_file=output_dir / "transcript.txt",
        diarization_file=output_dir / "diarization.txt",
        output_file=output_dir / "speaker_transcript.txt",
    )