from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import VIDEO


def _run_command(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "명령 실행 실패\n"
            f"COMMAND: {' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )


def _run_command_capture(command: list[str]) -> str:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "명령 실행 실패\n"
            f"COMMAND: {' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout


def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"입력 영상이 존재하지 않습니다: {video_path}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]

    raw = _run_command_capture(command)
    data = json.loads(raw)

    streams = data.get("streams", [])
    video_stream = None
    for stream in streams:
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError("영상 스트림을 찾을 수 없습니다.")

    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)
    codec = str(video_stream.get("codec_name", "") or "")
    rotation = _extract_rotation(video_stream)

    display_width = width
    display_height = height
    if rotation in (90, 270):
        display_width, display_height = height, width

    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0.0) or 0.0)

    return {
        "path": video_path,
        "width": width,
        "height": height,
        "duration": duration,
        "codec": codec,
        "rotation": rotation,
        "display_width": display_width,
        "display_height": display_height,
    }


def _extract_rotation(video_stream: Dict[str, Any]) -> int:
    rotation = 0

    tags = video_stream.get("tags", {})
    side_data_list = video_stream.get("side_data_list", [])

    if isinstance(tags, dict):
        rotate_tag = tags.get("rotate")
        if rotate_tag is not None:
            try:
                rotation = int(rotate_tag) % 360
            except (TypeError, ValueError):
                pass

    if rotation == 0 and isinstance(side_data_list, list):
        for item in side_data_list:
            if not isinstance(item, dict):
                continue
            if "rotation" in item:
                try:
                    rotation = int(item["rotation"]) % 360
                    break
                except (TypeError, ValueError):
                    pass

    return rotation


def build_vertical_filter(
    width: int,
    height: int,
    rotation: int = 0,
    target_width: int = int(VIDEO.get("short_width", 720)),
    target_height: int = int(VIDEO.get("short_height", 1280)),
    crop_anchor_x: float = 0.50,
    crop_anchor_y: float = 0.50,
) -> str:
    """
    세로 쇼츠용 soft crop 필터 생성

    방식:
    1) 비율 유지한 채 target canvas보다 작지 않게 확대
       (force_original_aspect_ratio=increase)
    2) 중앙 기준으로 crop
    3) y축 anchor를 중앙 쪽으로 두어 인물/장면 집중도를 높임

    crop_anchor_x:
        0.0 = 맨 왼쪽
        0.5 = 중앙
        1.0 = 맨 오른쪽

    crop_anchor_y:
        0.0 = 맨 위
        0.5 = 중앙
        1.0 = 맨 아래
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height는 0보다 커야 합니다.")

    if not (0.0 <= crop_anchor_x <= 1.0):
        raise ValueError("crop_anchor_x는 0.0 ~ 1.0 범위여야 합니다.")

    if not (0.0 <= crop_anchor_y <= 1.0):
        raise ValueError("crop_anchor_y는 0.0 ~ 1.0 범위여야 합니다.")

    display_width = width
    display_height = height
    if rotation in (90, 270):
        display_width, display_height = height, width

    source_ratio = display_width / max(display_height, 1)

    if source_ratio <= 0.80:
        return (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        )

    crop_x_expr = f"(iw-{target_width})*{crop_anchor_x}"
    crop_y_expr = f"(ih-{target_height})*{crop_anchor_y}"

    return (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}:{crop_x_expr}:{crop_y_expr},"
        f"setsar=1"
    )


def _looks_like_sentence_end(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False

    if normalized.endswith(("?", "!", ".", "…")):
        return True

    sentence_endings = (
        "다",
        "요",
        "죠",
        "네",
        "야",
        "입니다",
        "거든",
        "맞아",
        "맞습니다",
        "했다",
        "합니다",
        "됐어",
        "됐다",
        "좋아",
        "끝",
        "라고요",
        "냐고요",
        "있습니다",
        "없습니다",
        "했죠",
        "하죠",
        "거예요",
        "거에요",
    )
    return any(normalized.endswith(ending) for ending in sentence_endings)


def _is_good_end_segment(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False

    if _looks_like_sentence_end(normalized):
        return True

    if len(normalized) <= 2:
        return False

    natural_tail_patterns = (
        "솔직히",
        "그렇죠",
        "맞죠",
        "아니죠",
        "왜요",
        "뭐예요",
        "뭐에요",
        "뭐죠",
        "알아요",
        "모르죠",
    )
    return any(normalized.endswith(pattern) for pattern in natural_tail_patterns)


def adjust_end_with_stt(
    end_sec: float,
    segments: List[Dict[str, Any]],
    max_extend_sec: float = 3.0,
) -> float:
    """
    end_sec 이후 max_extend_sec 범위 안에서
    가장 가까운 '자연스러운 문장 끝' 세그먼트의 end 시점으로 확장한다.
    없으면 원래 end_sec를 그대로 사용한다.
    """
    original_end = float(end_sec)
    best_end = original_end
    best_gap = float("inf")

    for seg in segments:
        seg_end = float(seg.get("end", 0.0) or 0.0)
        seg_text = str(seg.get("text", "") or "").strip()

        if seg_end < original_end:
            continue

        gap = seg_end - original_end
        if gap > max_extend_sec:
            continue

        if not _is_good_end_segment(seg_text):
            continue

        if gap < best_gap:
            best_gap = gap
            best_end = seg_end

    return best_end


def _clamp_clip_range(
    start_sec: float,
    end_sec: float,
    total_duration: float,
    lead_in_sec: float,
    tail_out_sec: float,
) -> tuple[float, float]:
    """
    시작/끝 보정 후 영상 길이 범위 안으로 보정
    """
    adjusted_start = max(0.0, float(start_sec) - float(lead_in_sec))
    adjusted_end = min(float(total_duration), float(end_sec) + float(tail_out_sec))

    if adjusted_end <= adjusted_start:
        adjusted_start = max(0.0, float(start_sec))
        adjusted_end = min(float(total_duration), float(end_sec))

    if adjusted_end <= adjusted_start:
        raise ValueError("보정 후 클립 구간이 유효하지 않습니다.")

    return adjusted_start, adjusted_end


def cut_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    vf_override: Optional[str] = None,
    lead_in_sec: float = 0.7,
    tail_out_sec: float = 0.7,
    max_stt_end_extend_sec: float = 3.0,
) -> str:
    """
    지정 구간을 세로 쇼츠용 mp4로 잘라 저장

    추가 기능:
    - 시작 전 lead_in_sec 만큼 앞당김
    - 끝 후 tail_out_sec 만큼 늘림
    - segments가 주어지면 end_sec 이후 가까운 자연스러운 문장 끝까지 확장
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"입력 영상이 존재하지 않습니다: {video_path}")

    if end_sec <= start_sec:
        raise ValueError("end_sec는 start_sec보다 커야 합니다.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    meta = probe_video_metadata(video_path)

    final_end_sec = float(end_sec)
    if segments:
        final_end_sec = adjust_end_with_stt(
            end_sec=float(end_sec),
            segments=segments,
            max_extend_sec=float(max_stt_end_extend_sec),
        )

    adjusted_start, adjusted_end = _clamp_clip_range(
        start_sec=float(start_sec),
        end_sec=float(final_end_sec),
        total_duration=float(meta["duration"]),
        lead_in_sec=float(lead_in_sec),
        tail_out_sec=float(tail_out_sec),
    )
    duration = adjusted_end - adjusted_start

    if vf_override:
        vf = vf_override
    else:
        vf = build_vertical_filter(
            width=int(meta["width"]),
            height=int(meta["height"]),
            rotation=int(meta.get("rotation", 0)),
            target_width=int(VIDEO.get("short_width", 720)),
            target_height=int(VIDEO.get("short_height", 1280)),
            crop_anchor_x=0.50,
            crop_anchor_y=0.50,
        )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{adjusted_start:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-vf",
        vf,
        "-c:v",
        str(VIDEO.get("video_codec", "libx264")),
        "-preset",
        str(VIDEO.get("preset", "fast")),
        "-crf",
        str(VIDEO.get("crf", "23")),
        "-c:a",
        str(VIDEO.get("audio_codec", "aac")),
        "-b:a",
        str(VIDEO.get("audio_bitrate", "128k")),
        "-movflags",
        "+faststart",
        output_path,
    ]

    _run_command(command)
    return str(output_file)


if __name__ == "__main__":
    sample_video = "test.mp4"
    if os.path.exists(sample_video):
        info = probe_video_metadata(sample_video)
        print("[INFO] video meta =", info)

        out = "output/test_vertical_softcrop.mp4"
        cut_clip(
            video_path=sample_video,
            start_sec=0.0,
            end_sec=20.0,
            output_path=out,
            segments=None,
        )
        print(f"[INFO] saved: {out}")