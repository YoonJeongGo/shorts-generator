from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _looks_like_sentence_end(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    if normalized.endswith(("?", "!", ".", "…", "~")):
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
        "됩니다",
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
        "싶어요",
        "싶습니다",
        "된 거야",
        "된 거예요",
        "맞지",
        "맞죠",
        "아니야",
        "아니에요",
        "아닙니다",
        "했다고",
        "했다고",
    )
    return any(normalized.endswith(ending) for ending in sentence_endings)


def _looks_like_incomplete_tail(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True

    if normalized.endswith(("...", "…", ",", ":", ";", "-", "—", "–", "/", "(")):
        return True

    incomplete_endings = (
        "그",
        "저",
        "이",
        "저기",
        "근데",
        "그래서",
        "그러니까",
        "그러면",
        "그리고",
        "하지만",
        "근데요",
        "그래서요",
        "그러니까요",
        "뭐",
        "뭐가",
        "뭐냐면",
        "왜",
        "왜냐",
        "왜냐면",
        "이제",
        "지금",
        "있는",
        "없는",
        "하는",
        "되는",
        "될",
        "된",
        "해서",
        "하고",
        "이며",
        "인데",
        "인데요",
        "거",
        "것",
        "때문",
        "때문에",
        "명분이",
        "권한이",
        "얘기가",
        "상황이",
        "문제가",
        "부분이",
        "느낌이",
        "말이",
        "를",
        "을",
        "은",
        "는",
        "이",
        "가",
        "에",
        "에서",
        "로",
        "으로",
        "와",
        "과",
    )
    if any(normalized.endswith(ending) for ending in incomplete_endings):
        return True

    tokens = normalized.split()
    if not tokens:
        return True

    last_token = tokens[-1]
    incomplete_tokens = {
        "그",
        "저",
        "이",
        "뭐",
        "왜",
        "근데",
        "그리고",
        "그래서",
        "그러니까",
        "있는",
        "없는",
        "하는",
        "되는",
        "인데",
        "인데요",
        "명분이",
        "권한이",
        "상황이",
        "문제가",
        "부분이",
        "얘기가",
        "말이",
    }
    if last_token in incomplete_tokens:
        return True

    if len(normalized) <= 2:
        return True

    return False


def _starts_like_continuation(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    continuation_starts = (
        "그래서",
        "그러니까",
        "그리고",
        "근데",
        "근데요",
        "근데 그",
        "근데 지금",
        "근데 이게",
        "근데 그게",
        "근데 그건",
        "그러면",
        "그럼",
        "아니",
        "아니 근데",
        "그래도",
        "그 다음에",
        "그래서 지금",
        "그래서 명분이",
        "그래서 그",
        "그래서 이",
        "그",
        "그게",
        "그건",
        "그걸",
        "그거",
        "이게",
        "이건",
        "이걸",
        "저게",
        "저건",
        "저걸",
        "또",
    )
    return any(normalized.startswith(prefix) for prefix in continuation_starts)


def _is_reaction_like(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    reaction_words = (
        "어",
        "아",
        "어?",
        "아?",
        "뭐?",
        "예?",
        "왜?",
        "아니",
        "아니야",
        "아니요",
        "뭐야",
        "뭔데",
        "진짜",
        "와",
        "헐",
        "야",
        "어이",
        "허",
    )
    if normalized in reaction_words:
        return True

    if len(normalized) <= 3 and normalized.endswith(("?", "!")):
        return True

    return False


def _find_search_start_index(end_sec: float, segments: List[Dict[str, Any]]) -> int:
    original_end = float(end_sec)
    for i, seg in enumerate(segments):
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", 0.0) or 0.0)
        if seg_start <= original_end <= seg_end:
            return i
        if seg_end >= original_end:
            return i
    return max(0, len(segments) - 1)


def _candidate_context(
    idx: int,
    segments: List[Dict[str, Any]],
) -> Tuple[str, float, Optional[str], Optional[float], Optional[float]]:
    seg = segments[idx]
    text = _normalize_text(seg.get("text", ""))
    seg_end = float(seg.get("end", 0.0) or 0.0)

    next_text: Optional[str] = None
    gap_after: Optional[float] = None
    next_seg_duration: Optional[float] = None

    if idx + 1 < len(segments):
        next_seg = segments[idx + 1]
        next_text = _normalize_text(next_seg.get("text", ""))
        next_start = float(next_seg.get("start", 0.0) or 0.0)
        next_end = float(next_seg.get("end", 0.0) or 0.0)
        gap_after = next_start - seg_end
        next_seg_duration = max(0.0, next_end - next_start)

    return text, seg_end, next_text, gap_after, next_seg_duration


def _has_close_following_dialogue(
    idx: int,
    segments: List[Dict[str, Any]],
    max_check_count: int = 2,
) -> bool:
    current_end = float(segments[idx].get("end", 0.0) or 0.0)

    for offset in range(1, max_check_count + 1):
        next_idx = idx + offset
        if next_idx >= len(segments):
            break

        next_seg = segments[next_idx]
        next_text = _normalize_text(next_seg.get("text", ""))
        next_start = float(next_seg.get("start", 0.0) or 0.0)
        gap = next_start - current_end

        if gap > 1.20:
            break

        if not next_text:
            continue

        if _is_reaction_like(next_text):
            continue

        return True

    return False


def _is_hard_reject_candidate(
    idx: int,
    segments: List[Dict[str, Any]],
    original_end: float,
) -> bool:
    text, seg_end, next_text, gap_after, next_seg_duration = _candidate_context(idx, segments)

    if seg_end < original_end:
        return True

    is_sentence_end = _looks_like_sentence_end(text)
    is_incomplete = _looks_like_incomplete_tail(text)

    if is_incomplete:
        return True

    if not is_sentence_end and gap_after is not None and gap_after < 1.00:
        return True

    if gap_after is not None and gap_after < 0.30:
        return True

    if next_text:
        if _starts_like_continuation(next_text) and gap_after is not None and gap_after < 1.25:
            return True

        if (
            not is_sentence_end
            and not _is_reaction_like(next_text)
            and gap_after is not None
            and gap_after < 1.25
        ):
            return True

        if (
            is_sentence_end
            and gap_after is not None
            and gap_after < 0.55
            and not _is_reaction_like(next_text)
        ):
            return True

        if (
            _has_close_following_dialogue(idx, segments, max_check_count=2)
            and gap_after is not None
            and gap_after < 0.90
        ):
            return True

        if (
            next_seg_duration is not None
            and next_seg_duration >= 1.00
            and gap_after is not None
            and gap_after < 0.70
            and not _is_reaction_like(next_text)
        ):
            return True

    return False


def _score_end_candidate(
    idx: int,
    segments: List[Dict[str, Any]],
    original_end: float,
    max_extend_sec: float,
) -> float:
    seg = segments[idx]
    seg_end = float(seg.get("end", 0.0) or 0.0)
    seg_start = float(seg.get("start", 0.0) or 0.0)

    if seg_end < original_end:
        return -99999.0

    gap_from_original = seg_end - original_end
    if gap_from_original > max_extend_sec:
        return -99999.0

    if _is_hard_reject_candidate(idx, segments, original_end):
        return -99999.0

    text, _, next_text, gap_after, _ = _candidate_context(idx, segments)

    score = 0.0
    is_sentence_end = _looks_like_sentence_end(text)

    if is_sentence_end:
        score += 10.0

    if seg_start <= original_end <= seg_end:
        score += 1.0

    if gap_after is None:
        score += 4.0
    else:
        if gap_after < 0.35:
            score -= 7.0
        elif gap_after < 0.55:
            score -= 3.5
        elif gap_after < 0.75:
            score -= 1.0
        elif gap_after <= 1.80:
            score += 4.5
        else:
            score += 1.5

        if next_text and _is_reaction_like(next_text):
            score += 1.0

    if gap_from_original < 0.25:
        score -= 3.0
    elif gap_from_original < 0.50:
        score -= 1.5
    elif 0.80 <= gap_from_original <= 3.20:
        score += 2.5

    score -= gap_from_original * 0.45

    if len(text) <= 3:
        score -= 2.5
    elif len(text) <= 6:
        score -= 1.0

    return score


def adjust_end_with_stt(
    end_sec: float,
    segments: List[Dict[str, Any]],
    max_extend_sec: float = 6.0,
) -> float:
    original_end = float(end_sec)
    if not segments:
        return original_end

    start_idx = _find_search_start_index(original_end, segments)

    all_candidates: List[Tuple[float, float, int]] = []
    accepted_candidates: List[Tuple[float, float, int]] = []
    last_valid_seg_end = original_end

    for i in range(start_idx, len(segments)):
        seg = segments[i]
        seg_end = float(seg.get("end", 0.0) or 0.0)

        if seg_end < original_end:
            continue

        gap_from_original = seg_end - original_end
        if gap_from_original > max_extend_sec:
            break

        last_valid_seg_end = max(last_valid_seg_end, seg_end)

        score = _score_end_candidate(
            idx=i,
            segments=segments,
            original_end=original_end,
            max_extend_sec=max_extend_sec,
        )
        all_candidates.append((score, seg_end, i))

        if score > -90000.0:
            accepted_candidates.append((score, seg_end, i))

    if accepted_candidates:
        best_score, best_end, _ = max(accepted_candidates, key=lambda x: x[0])
        if best_score >= 0.0:
            return best_end

    fallback_complete_candidates: List[Tuple[float, float, int]] = []
    for _, seg_end, idx in all_candidates:
        text, _, next_text, gap_after, _ = _candidate_context(idx, segments)

        if _looks_like_incomplete_tail(text):
            continue
        if not _looks_like_sentence_end(text):
            continue
        if gap_after is not None and gap_after < 0.45 and next_text and not _is_reaction_like(next_text):
            continue
        if gap_after is not None and gap_after < 0.80 and next_text and _starts_like_continuation(next_text):
            continue
        if _has_close_following_dialogue(idx, segments, max_check_count=2) and gap_after is not None and gap_after < 0.90:
            continue

        fallback_score = 0.0

        if gap_after is None:
            fallback_score += 4.0
        else:
            if gap_after < 0.45:
                fallback_score -= 4.0
            elif gap_after < 0.75:
                fallback_score -= 1.5
            elif gap_after <= 2.00:
                fallback_score += 3.5
            else:
                fallback_score += 1.0

            if next_text and _starts_like_continuation(next_text) and gap_after < 1.20:
                fallback_score -= 3.5

        gap_from_original = seg_end - original_end
        fallback_score -= gap_from_original * 0.40

        fallback_complete_candidates.append((fallback_score, seg_end, idx))

    if fallback_complete_candidates:
        _, fallback_end, _ = max(fallback_complete_candidates, key=lambda x: x[0])
        return fallback_end

    return last_valid_seg_end


def _clamp_clip_range(
    start_sec: float,
    end_sec: float,
    total_duration: float,
    lead_in_sec: float,
    tail_out_sec: float,
) -> tuple[float, float]:
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
    max_stt_end_extend_sec: float = 6.0,
) -> str:
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

        out = "output/test_fixed_end_v4.mp4"
        cut_clip(
            video_path=sample_video,
            start_sec=0.0,
            end_sec=20.0,
            output_path=out,
            segments=None,
        )
        print(f"[INFO] saved: {out}")


