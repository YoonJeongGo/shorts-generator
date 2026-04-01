# pipeline/multicat.py

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def concat_clips(
    input_paths: list[str],
    output_path: str,
    mode: str = "auto",
) -> str:
    """
    여러 클립을 순서대로 이어붙인다.

    mode:
    - "auto"     : copy concat 시도 후 실패하면 re-encode fallback
    - "copy"     : concat demuxer + -c copy (빠름, 가장 취약)
    - "reencode" : filter_complex concat + libx264/aac (느리지만 안정적)
    - "safe"     : reencode와 동일

    기본값은 auto다.
    기존 기본값 copy보다 훨씬 안전하다.
    """
    if not input_paths:
        raise ValueError("입력 클립이 없습니다.")

    if len(input_paths) == 1:
        shutil.copy2(input_paths[0], output_path)
        return output_path

    normalized_mode = mode.lower().strip()
    if normalized_mode == "safe":
        normalized_mode = "reencode"

    if normalized_mode == "copy":
        return _concat_demuxer_copy(input_paths, output_path)

    if normalized_mode == "reencode":
        return _concat_filter_reencode(input_paths, output_path)

    if normalized_mode == "auto":
        try:
            return _concat_demuxer_copy(input_paths, output_path)
        except Exception:
            return _concat_filter_reencode(input_paths, output_path)

    raise ValueError(f"지원하지 않는 concat mode: {mode}")


def _concat_demuxer_copy(input_paths: list[str], output_path: str) -> str:
    """
    가장 빠르지만 가장 취약한 방식.
    모든 클립의 스트림 파라미터가 사실상 동일해야 한다.
    """
    list_path = _write_concat_list_file(input_paths)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]
    try:
        _run_ffmpeg(cmd, "ffmpeg concat copy 실패")
        return output_path
    finally:
        Path(list_path).unlink(missing_ok=True)


def _concat_filter_reencode(input_paths: list[str], output_path: str) -> str:
    """
    더 안전한 방식.
    filter_complex concat 후 표준 코덱으로 다시 인코딩한다.
    """
    n = len(input_paths)
    cmd = ["ffmpeg", "-y"]

    for p in input_paths:
        cmd += ["-i", p]

    inputs = "".join(f"[{i}:v][{i}:a]" for i in range(n))
    filter_str = f"{inputs}concat=n={n}:v=1:a=1[v][a]"

    cmd += [
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    _run_ffmpeg(cmd, "ffmpeg concat re-encode 실패")
    return output_path


def _write_concat_list_file(input_paths: list[str]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as f:
        for path in input_paths:
            safe = str(Path(path).resolve()).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
        return f.name


def _run_ffmpeg(cmd: list[str], err_prefix: str) -> None:
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"{err_prefix}:\n{err}") from e