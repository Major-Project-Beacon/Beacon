import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_dir: str) -> str:
    video = Path(video_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    audio_path = output / f"{video.stem}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path),
    ]

    subprocess.run(command, check=True)

    return str(audio_path)


if __name__ == "__main__":
    video = r"D:\BEACON\datasets\Product Team Meeting - 2019-07-09_720p.mp4"
    output_dir = r"D:\BEACON\pipeline\output"

    audio = extract_audio(video, output_dir)

    print("\nAudio extracted successfully:")
    print(audio)