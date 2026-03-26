from __future__ import annotations

import subprocess
from pathlib import Path

from config import TMP_DIR, AUDIO


def _decode_output(data: bytes | None) -> str:
    if not data:
        return ""

    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="ignore")


def extract_audio(video_path: str) -> str:
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"영상 파일이 존재하지 않습니다: {video_path}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = TMP_DIR / f"{video_file.stem}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_file),
        "-vn",
        "-acodec", AUDIO.get("codec", "pcm_s16le"),
        "-ar", str(AUDIO.get("sample_rate", 16000)),
        "-ac", str(AUDIO.get("channels", 1)),
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    stderr_text = _decode_output(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio extraction failed\n"
            f"video_path: {video_path}\n"
            f"stderr:\n{stderr_text}"
        )

    print(f"[INFO] Audio extracted: {audio_path}")
    return str(audio_path)