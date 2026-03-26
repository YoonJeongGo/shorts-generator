from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

from config import OUTPUT_DIR, SUBTITLE, VIDEO


DEFAULT_FONT_FILE = Path("C:/Windows/Fonts/malgun.ttf")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    result = str(name)
    for ch in invalid_chars:
        result = result.replace(ch, "_")
    return result.strip()


def _normalize_text(text: str) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    return ascii_count / max(total, 1)


def _cleanup_subtitle_text(text: str) -> str:
    """
    자막용 최소 정리.
    STT를 완전히 교정하는 게 아니라, 자막으로 보기에 지저분한 찌꺼기만 줄인다.
    """
    text = _normalize_text(text)
    if not text:
        return ""

    # 괄호형 효과음 제거
    text = re.sub(r"\[(.*?)\]", " ", text)
    text = re.sub(r"\((.*?)\)", " ", text)

    banned_exact = {
        "uh", "um", "ah", "oh", "mm",
        "music", "bgm", "subtitle", "subtitles",
        "donkey", "produkt", "product", "million", "butcher",
    }

    tokens = text.split()
    cleaned_tokens: List[str] = []

    for token in tokens:
        raw = token.strip()
        if not raw:
            continue

        lowered = raw.lower().strip(".,!?~'\"`")
        if lowered in banned_exact:
            continue

        # 순수 ASCII 토큰 제거
        if re.fullmatch(r"[A-Za-z0-9_\-]+", raw):
            if len(raw) >= 2:
                continue

        # 한글 없이 ASCII 비율 높은 토큰 제거
        if not _contains_korean(raw):
            if _ascii_ratio(raw) >= 0.8 and len(raw) >= 2:
                continue

        cleaned_tokens.append(raw)

    text = " ".join(cleaned_tokens)
    text = _normalize_text(text)

    # 문장 전체가 거의 ASCII 덩어리면 버림
    if text and not _contains_korean(text) and _ascii_ratio(text) >= 0.7:
        return ""

    # 특수문자 과다 정리
    text = re.sub(r"[~]{2,}", "~", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"[.]{3,}", "...", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _split_long_token(token: str, limit: int) -> List[str]:
    if len(token) <= limit:
        return [token]

    parts: List[str] = []
    start = 0
    while start < len(token):
        parts.append(token[start:start + limit])
        start += limit
    return parts


def _tokenize_text(text: str, token_limit: int) -> List[str]:
    raw_tokens = text.split()
    tokens: List[str] = []

    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue

        if len(token) > token_limit:
            tokens.extend(_split_long_token(token, token_limit))
        else:
            tokens.append(token)

    return tokens


def _join_tokens(tokens: List[str]) -> str:
    return " ".join(t for t in tokens if t).strip()


def _find_best_two_line_split(tokens: List[str], max_chars_per_line: int) -> int:
    if len(tokens) <= 1:
        return -1

    total_text = _join_tokens(tokens)
    total_len = len(total_text)
    target = total_len / 2

    best_idx = -1
    best_score = float("inf")

    for i in range(1, len(tokens)):
        left = _join_tokens(tokens[:i])
        right = _join_tokens(tokens[i:])

        if not left or not right:
            continue

        left_len = len(left)
        right_len = len(right)

        overflow_penalty = 0
        if left_len > max_chars_per_line:
            overflow_penalty += (left_len - max_chars_per_line) * 10
        if right_len > max_chars_per_line:
            overflow_penalty += (right_len - max_chars_per_line) * 10

        balance_penalty = abs(left_len - target)

        punctuation_bonus = 0
        prev_token = tokens[i - 1]
        if prev_token.endswith((".", "!", "?", ",", "…", "~")):
            punctuation_bonus = -2

        score = overflow_penalty + balance_penalty + punctuation_bonus

        if score < best_score:
            best_score = score
            best_idx = i

    return best_idx


def split_subtitle_lines(text: str, threshold: int) -> List[str]:
    """
    최대 2줄로 반환.
    """
    text = _normalize_text(text)
    if not text:
        return []

    max_chars_per_line = max(8, int(threshold), int(SUBTITLE.get("max_chars_per_line", 16)))
    token_limit = max_chars_per_line

    if len(text) <= max_chars_per_line:
        return [text]

    # 1차: 구두점 우선
    punct_candidates = []
    for m in re.finditer(r"[,.!?…~]", text):
        idx = m.end()
        left = text[:idx].strip()
        right = text[idx:].strip()
        if not left or not right:
            continue

        overflow_penalty = 0
        if len(left) > max_chars_per_line:
            overflow_penalty += (len(left) - max_chars_per_line) * 10
        if len(right) > max_chars_per_line:
            overflow_penalty += (len(right) - max_chars_per_line) * 10

        balance_penalty = abs(len(left) - (len(text) / 2))
        score = overflow_penalty + balance_penalty - 2
        punct_candidates.append((score, left, right))

    if punct_candidates:
        punct_candidates.sort(key=lambda x: x[0])
        best_left, best_right = punct_candidates[0][1], punct_candidates[0][2]
        if best_left and best_right:
            return [best_left, best_right]

    # 2차: 토큰 균형 분할
    tokens = _tokenize_text(text, token_limit=token_limit)
    if len(tokens) <= 1:
        mid = min(max_chars_per_line, max(1, len(text) // 2))
        left = text[:mid].strip()
        right = text[mid:].strip()
        return [line for line in [left, right] if line]

    split_idx = _find_best_two_line_split(tokens, max_chars_per_line=max_chars_per_line)
    if split_idx == -1:
        return [text]

    line1 = _join_tokens(tokens[:split_idx])
    line2 = _join_tokens(tokens[split_idx:])

    if not line1 or not line2:
        return [text]

    return [line1, line2]


def _trim_text_for_shorts(text: str, max_total_chars: int) -> str:
    text = _normalize_text(text)
    if not text:
        return ""

    if len(text) <= max_total_chars:
        return text

    trimmed = text[:max_total_chars].rstrip()
    last_space = trimmed.rfind(" ")
    if last_space >= max_total_chars * 0.6:
        trimmed = trimmed[:last_space].rstrip()

    return trimmed + "..."


def build_clip_relative_segments(
    segments: List[Dict[str, Any]],
    clip_start_sec: float,
    clip_end_sec: float,
) -> List[Dict[str, Any]]:
    clip_duration = clip_end_sec - clip_start_sec
    if clip_duration <= 0:
        raise ValueError("clip_end_sec must be greater than clip_start_sec")

    results: List[Dict[str, Any]] = []

    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()

        if not text or seg_end <= seg_start:
            continue

        overlap_start = max(seg_start, clip_start_sec)
        overlap_end = min(seg_end, clip_end_sec)

        if overlap_end <= overlap_start:
            continue

        rel_start = max(0.0, overlap_start - clip_start_sec)
        rel_end = min(clip_duration, overlap_end - clip_start_sec)

        if rel_end <= rel_start:
            continue

        results.append({
            "start": round(rel_start, 3),
            "end": round(rel_end, 3),
            "text": text,
        })

    return results


def _decode_output(data: bytes | None) -> str:
    if not data:
        return ""

    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="ignore")


def _ffmpeg_escape_text(text: str) -> str:
    text = text.replace("\\", r"\\\\")
    text = text.replace(":", r"\:")
    text = text.replace("'", r"\'")
    text = text.replace(",", r"\,")
    text = text.replace("[", r"\[")
    text = text.replace("]", r"\]")
    text = text.replace("%", r"\%")
    return text


def _ffmpeg_escape_path(path: str | Path) -> str:
    path_str = str(Path(path).resolve())
    path_str = path_str.replace("\\", "/")
    path_str = path_str.replace(":", r"\:")
    path_str = path_str.replace("'", r"\'")
    path_str = path_str.replace(",", r"\,")
    path_str = path_str.replace("[", r"\[")
    path_str = path_str.replace("]", r"\]")
    return path_str


def _probe_video_size(input_video_path: str) -> Tuple[int, int]:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        input_video_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    if result.returncode != 0:
        stderr_text = _decode_output(result.stderr)
        raise RuntimeError(f"ffprobe failed: {stderr_text}")

    output = _decode_output(result.stdout).strip()
    match = re.match(r"(\d+)x(\d+)", output)
    if not match:
        raise RuntimeError(f"ffprobe size parse failed: {output}")

    width = int(match.group(1))
    height = int(match.group(2))
    return width, height


def _calc_subtitle_style(video_width: int, video_height: int) -> Dict[str, int]:
    """
    최종 세로 영상 기준 자막 스타일 계산.
    """
    base_width = 720
    base_height = 1280

    width_ratio = video_width / base_width
    height_ratio = video_height / base_height
    scale = min(width_ratio, height_ratio)

    config_font_size = int(SUBTITLE.get("font_size", 42))
    config_line_spacing = int(SUBTITLE.get("line_spacing", 8))
    config_borderw = int(SUBTITLE.get("borderw", 3))
    config_boxborderw = int(SUBTITLE.get("boxborderw", 12))
    config_max_chars_per_line = int(SUBTITLE.get("max_chars_per_line", 16))
    config_max_total_chars = int(SUBTITLE.get("max_total_chars", 32))

    font_size = max(26, int(config_font_size * scale))
    line_spacing = max(4, int(config_line_spacing * scale))
    borderw = max(2, int(config_borderw * scale))
    boxborderw = max(6, int(config_boxborderw * scale))

    # 아래 잘림 방지
    bottom_margin = max(110, int(video_height * 0.16))

    # 좌우 잘림 방지용: 한 줄 글자 수를 더 보수적으로 제한
    if video_width <= 720:
        max_chars_per_line = min(config_max_chars_per_line, 13)
    elif video_width <= 1080:
        max_chars_per_line = min(16, config_max_chars_per_line)
    else:
        max_chars_per_line = min(18, config_max_chars_per_line + 1)

    max_total_chars = min(config_max_total_chars, max_chars_per_line * 2)

    # 두 줄 간 실제 y 간격
    line_gap = font_size + line_spacing + 6

    return {
        "font_size": font_size,
        "line_spacing": line_spacing,
        "borderw": borderw,
        "boxborderw": boxborderw,
        "bottom_margin": bottom_margin,
        "max_chars_per_line": max_chars_per_line,
        "max_total_chars": max_total_chars,
        "line_gap": line_gap,
    }


def _build_single_drawtext(
    text: str,
    escaped_font: str,
    style: Dict[str, int],
    start_sec: float,
    end_sec: float,
    y_expr: str,
) -> str:
    escaped_text = _ffmpeg_escape_text(text)

    return (
        f"drawtext="
        f"fontfile='{escaped_font}':"
        f"text='{escaped_text}':"
        f"fontcolor=white:"
        f"fontsize={style['font_size']}:"
        f"borderw={style['borderw']}:"
        f"bordercolor=black:"
        f"box=1:"
        f"boxcolor=black@0.35:"
        f"boxborderw={style['boxborderw']}:"
        f"shadowx=0:"
        f"shadowy=0:"
        f"x=(w-text_w)/2:"
        f"y={y_expr}:"
        f"enable='between(t,{start_sec:.3f},{end_sec:.3f})'"
    )


def _build_drawtext_filter(
    segments: List[Dict[str, Any]],
    video_width: int,
    video_height: int,
) -> str:
    line_split_threshold = int(SUBTITLE.get("line_split_threshold", 16))
    style = _calc_subtitle_style(video_width, video_height)

    font_file = Path(SUBTITLE.get("font_file", DEFAULT_FONT_FILE))
    if not font_file.exists():
        if DEFAULT_FONT_FILE.exists():
            font_file = DEFAULT_FONT_FILE
        else:
            raise FileNotFoundError(f"사용 가능한 폰트 파일이 없습니다: {font_file}")

    escaped_font = _ffmpeg_escape_path(font_file)
    filters: List[str] = []

    for seg in segments:
        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", 0.0))
        raw_text = str(seg.get("text", "")).strip()

        if not raw_text or end_sec <= start_sec:
            continue

        text = _cleanup_subtitle_text(raw_text)
        if not text:
            continue

        text = _trim_text_for_shorts(text, max_total_chars=style["max_total_chars"])
        lines = split_subtitle_lines(
            text=text,
            threshold=max(line_split_threshold, style["max_chars_per_line"]),
        )

        if not lines:
            continue

        # 1줄이면 하단 중앙
        if len(lines) == 1:
            y_expr = f"h-text_h-{style['bottom_margin']}"
            filters.append(
                _build_single_drawtext(
                    text=lines[0],
                    escaped_font=escaped_font,
                    style=style,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    y_expr=y_expr,
                )
            )
            continue

        # 2줄이면 각 줄을 별도 drawtext로 그림
        line1 = lines[0]
        line2 = lines[1]

        # 둘째 줄이 기준 하단, 첫째 줄은 그 위 line_gap 만큼
        y_line2 = f"h-text_h-{style['bottom_margin']}"
        y_line1 = f"h-text_h-{style['bottom_margin']}-{style['line_gap']}"

        filters.append(
            _build_single_drawtext(
                text=line1,
                escaped_font=escaped_font,
                style=style,
                start_sec=start_sec,
                end_sec=end_sec,
                y_expr=y_line1,
            )
        )
        filters.append(
            _build_single_drawtext(
                text=line2,
                escaped_font=escaped_font,
                style=style,
                start_sec=start_sec,
                end_sec=end_sec,
                y_expr=y_line2,
            )
        )

    if not filters:
        return ""

    return ",".join(filters)


def burn_subtitles(
    input_video_path: str,
    relative_segments: List[Dict[str, Any]],
    output_video_path: str,
) -> str:
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"입력 클립 파일이 없습니다: {input_video_path}")

    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        ensure_dir(output_dir)

    video_width, video_height = _probe_video_size(input_video_path)

    vf = _build_drawtext_filter(
        segments=relative_segments,
        video_width=video_width,
        video_height=video_height,
    )
    if not vf:
        raise RuntimeError("drawtext용 자막 세그먼트가 비어 있습니다.")

    command = [
        "ffmpeg",
        "-y",
        "-i", input_video_path,
        "-vf", vf,
        "-c:v", str(VIDEO["video_codec"]),
        "-c:a", str(VIDEO["audio_codec"]),
        "-preset", str(VIDEO["preset"]),
        "-crf", str(VIDEO["crf"]),
        "-b:a", str(VIDEO["audio_bitrate"]),
        "-movflags", "+faststart",
        output_video_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    stdout_text = _decode_output(result.stdout)
    stderr_text = _decode_output(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg drawtext subtitle burn failed\n"
            f"input_video_path: {input_video_path}\n"
            f"output_video_path: {output_video_path}\n"
            f"vf:\n{vf}\n"
            f"stdout:\n{stdout_text}\n"
            f"stderr:\n{stderr_text}"
        )

    return output_video_path


def export_clip_with_subtitles(
    clip_path: str,
    clip_start_sec: float,
    clip_end_sec: float,
    segments: List[Dict[str, Any]],
    output_path: str | None = None,
    temp_srt_path: str | None = None,
) -> str:
    """
    temp_srt_path는 기존 호출 호환 때문에 남겨둠.
    현재 구현에서는 사용하지 않음.
    """
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"클립 파일이 없습니다: {clip_path}")

    clip_file = Path(clip_path)
    clip_name = sanitize_filename(clip_file.stem)

    if output_path is None:
        ensure_dir(OUTPUT_DIR)
        output_path = str(Path(OUTPUT_DIR) / f"{clip_name}_subtitled.mp4")

    relative_segments = build_clip_relative_segments(
        segments=segments,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
    )

    return burn_subtitles(
        input_video_path=clip_path,
        relative_segments=relative_segments,
        output_video_path=output_path,
    )


def export_multiple_clips_with_subtitles(
    clip_infos: List[Dict[str, Any]],
    output_dir: str | None = None,
) -> List[str]:
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)

    ensure_dir(output_dir)

    saved_files: List[str] = []

    for info in clip_infos:
        clip_path = str(info["clip_path"])
        start_sec = float(info["start"])
        end_sec = float(info["end"])
        segments = list(info.get("segments", []))

        clip_stem = sanitize_filename(Path(clip_path).stem)
        output_path = str(Path(output_dir) / f"{clip_stem}_final.mp4")

        saved_path = export_clip_with_subtitles(
            clip_path=clip_path,
            clip_start_sec=start_sec,
            clip_end_sec=end_sec,
            segments=segments,
            output_path=output_path,
        )
        saved_files.append(saved_path)

    return saved_files