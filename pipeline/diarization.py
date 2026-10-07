import os
import wave
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

import torch
from pyannote.audio import Pipeline


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

AUDIO_PATH = r"D:\BEACON\pipeline\output\Product Team Meeting - 2019-07-09_720p.wav"
OUTPUT_DIR = Path(r"D:\BEACON\pipeline\output")


# ---------------------------------------------------------
# LOAD AUDIO WITHOUT TORCHCODEC
# ---------------------------------------------------------

def load_audio(audio_path: str):
    """
    Load WAV audio manually so pyannote does not depend
    on TorchCodec for this Windows setup.
    """

    with wave.open(audio_path, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()

        raw_audio = wav.readframes(frames)

    if sample_width != 2:
        raise ValueError("Expected 16-bit PCM WAV audio.")

    if channels != 1:
        raise ValueError("Expected mono audio.")

    import numpy as np

    audio = np.frombuffer(
        raw_audio,
        dtype=np.int16
    ).astype(np.float32) / 32768.0

    waveform = torch.from_numpy(audio).unsqueeze(0)

    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }


# ---------------------------------------------------------
# DIARIZATION
# ---------------------------------------------------------

def diarize_audio(audio_path: str, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading diarization model...")

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set.\n"
            "Set your Hugging Face token before running diarization."
        )

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        token=hf_token
    )

    # Use GPU if available
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

    with open(output_file, "w", encoding="utf-8") as f:
        speakers = set()

        annotation = diarization.speaker_diarization

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            f.write(f"[{turn.start:.2f} - {turn.end:.2f}] {speaker}\n")
            speakers.add(speaker)

    print("\nDiarization completed.")
    print(f"Speakers detected: {len(speakers)}")
    print(f"Saved to: {output_file}")

    print(f"Speakers detected: {len(speakers)}")
    print(f"Saved to: {output_file}")

    return str(output_file)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    diarize_audio(
        AUDIO_PATH,
        OUTPUT_DIR
    )