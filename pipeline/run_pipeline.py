import argparse
from pathlib import Path

from audio import extract_audio
from transcription import transcribe_audio
from diarization import diarize_audio
from speaker_transcript import merge_transcript


def run_pipeline(meeting_file: str, output_dir: str):
    meeting_file = Path(meeting_file)
    output_dir = Path(output_dir)

    if not meeting_file.exists():
        raise FileNotFoundError(f"Meeting file not found: {meeting_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Extracting audio...")
    audio_path = extract_audio(str(meeting_file), str(output_dir))

    print("\n[2/4] Transcribing audio...")
    transcript_path = transcribe_audio(audio_path, str(output_dir))

    print("\n[3/4] Diarizing speakers...")
    diarization_path = diarize_audio(audio_path, output_dir)

    print("\n[4/4] Aligning transcript with speakers...")
    speaker_transcript_path = merge_transcript(
        transcript_file=transcript_path,
        diarization_file=diarization_path,
        output_file=output_dir / "speaker_transcript.txt",
    )

    print("\nPipeline completed.")
    print(f"Speaker-labelled transcript: {speaker_transcript_path}")

    return speaker_transcript_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BEACON audio, transcription, diarization and alignment."
    )

    parser.add_argument(
        "meeting_file",
        help="Path to the meeting video or audio file",
    )

    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "output"),
        help="Directory for generated pipeline files",
    )

    args = parser.parse_args()

    run_pipeline(args.meeting_file, args.output)