import os
import wave
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# F_E_module/.env (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / "pipeline" / ".env")


# ---------------------------------------------------------
# LOAD WAV WITHOUT TORCHCODEC
# ---------------------------------------------------------

def load_audio(audio_path: str | Path):
    """
    Load mono, 16-bit PCM WAV manually.
    Returns a waveform tensor shaped [1, samples] and sample rate.
    """
    audio_path = Path(audio_path)

    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        raw_audio = wav.readframes(frames)

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV, got sample width: {sample_width} bytes"
        )

    if channels != 1:
        raise ValueError(
            f"Expected mono WAV, got {channels} channels. "
            "Convert the audio to mono before diarization."
        )

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
    audio /= 32768.0

    waveform = torch.from_numpy(audio).unsqueeze(0)

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


# ---------------------------------------------------------
# DIARIZATION
# ---------------------------------------------------------

def diarize_audio(
    audio_path: str | Path,
    output_dir: str | Path,
) -> str:
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add your Hugging Face token "
            "to the project-root .env file."
        )

    print("Loading diarization model...")

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        token=hf_token,
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("Diarization device: CUDA")
    else:
        print("Diarization device: CPU")

    print("Loading audio...")
    audio = load_audio(audio_path)

    print("Running speaker diarization...")
    diarization = pipeline(audio)

    output_file = output_dir / "diarization.txt"
    speakers = set()

    # Community-1 returns an output object with speaker_diarization.
    annotation = diarization.speaker_diarization

    with output_file.open("w", encoding="utf-8") as f:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            f.write(
                f"[{turn.start:.2f} - {turn.end:.2f}] {speaker}\n"
            )
            speakers.add(speaker)

    print("Diarization completed.")
    print(f"Speakers detected: {len(speakers)}")
    print(f"Saved to: {output_file}")

    return str(output_file)


# ---------------------------------------------------------
# OPTIONAL STANDALONE TEST
# ---------------------------------------------------------

if __name__ == "__main__":
    # Test with an audio file you already have.
    test_audio = PROJECT_ROOT / "pipeline" / "output" / "meeting.wav"
    test_output = PROJECT_ROOT / "pipeline" / "output"

    diarize_audio(test_audio, test_output)
    