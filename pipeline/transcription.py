import os
from pathlib import Path
import subprocess
from unittest import result
import numpy as np


# Add CUDA DLL locations so faster-whisper can find them on Windows
CUDA_DLL_DIRS = [
    r"D:\BEACON\datasets\whisper_env\Lib\site-packages\nvidia\cublas\bin",
    r"D:\BEACON\datasets\whisper_env\Lib\site-packages\nvidia\cuda_runtime\bin",
    r"D:\BEACON\datasets\whisper_env\Lib\site-packages\nvidia\cudnn\bin",
    r"D:\BEACON\datasets\whisper_env\Lib\site-packages\torch\lib",
]

for dll_dir in CUDA_DLL_DIRS:
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]


from faster_whisper import WhisperModel


MODEL_SIZE = "small"


def transcribe_audio(audio_path: str, output_dir: str):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper model...")

    model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        compute_type="float16"
    )

    print("Transcribing...")

    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", str(audio_path),
            "-f", "f32le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    audio_array = np.frombuffer(result.stdout, dtype=np.float32)

    segments, info = model.transcribe(
        audio_array,
        beam_size=5,
        vad_filter=True,
    )

    segments = list(segments)

    transcript_path = output / "transcript.txt"

    with open(transcript_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(
                f"[{segment.start:.2f} - {segment.end:.2f}] "
                f"{segment.text.strip()}\n"
            )

    print(f"\nTranscription completed.")
    print(f"Segments: {len(segments)}")
    print(f"Language: {info.language}")
    print(f"Saved to: {transcript_path}")

    return str(transcript_path)


if __name__ == "__main__":
    audio = r"D:\BEACON\pipeline\output\Product Team Meeting - 2019-07-09_720p.wav"
    output = r"D:\BEACON\pipeline\output"

    transcribe_audio(audio, output)