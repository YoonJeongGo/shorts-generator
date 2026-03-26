from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from config import CLIP, OUTPUT_DIR
from pipeline.audio_extractor import extract_audio
from pipeline.clip_cutter import cut_clip
from pipeline.exporter import export_clip_with_subtitles
from pipeline.highlight_scorer import HighlightScorer
from pipeline.stt_engine import transcribe_audio


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        run(
            input_path=args.input,
            top_n=args.top,
            out_dir=args.out,
            subtitle_srt=args.srt,
            subtitle_txt=args.txt,
            quick_mode=args.quick,
            select_mode=args.select,
        )
    except Exception as e:
        print(f"[ERROR] {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shorts Generator MVP CLI",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="입력 영상 파일 경로",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=int(CLIP.get("top_k", CLIP.get("top_k_default", 3))),
        help="최종 하이라이트 개수",
    )
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR / "clips"),
        help="최종 결과 저장 폴더",
    )
    parser.add_argument(
        "--srt",
        default=None,
        help="SRT 자막 파일 경로",
    )
    parser.add_argument(
        "--txt",
        default=None,
        help="TXT 자막 파일 경로",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="STT/자막 없이 임시 분할 후보 추출",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="추출된 후보 중 생성할 하이라이트를 직접 선택",
    )
    return parser


def run(
    input_path: str,
    top_n: int = 3,
    out_dir: str | None = None,
    subtitle_srt: str | None = None,
    subtitle_txt: str | None = None,
    quick_mode: bool = False,
    select_mode: bool = False,
) -> None:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"입력 영상이 존재하지 않습니다: {input_path}")

    if input_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 영상 포맷입니다: {input_file.suffix}")

    if subtitle_srt and subtitle_txt:
        raise ValueError("--srt와 --txt는 동시에 사용할 수 없습니다.")

    if out_dir is None:
        out_dir = str(OUTPUT_DIR / "clips")

    out_path = Path(out_dir)
    raw_clip_dir = out_path / "raw"
    final_clip_dir = out_path / "final"
    segments_dir = OUTPUT_DIR / "segments"

    raw_clip_dir.mkdir(parents=True, exist_ok=True)
    final_clip_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 영상 정보 확인 중...")
    meta = probe_video_metadata(str(input_file))
    print(
        f"      -> {input_file.name} | "
        f"{meta['duration']:.1f}초 | "
        f"{meta['width']}x{meta['height']}"
    )

    segments: list[dict[str, Any]] = []
    saved_segments_txt: str | None = None

    if quick_mode:
        print("[2/5] 빠른 후보 추출 모드 실행 중...")
        candidates = generate_quick_windows(
            duration_sec=meta["duration"],
            window_sec=int(CLIP.get("window_sec", 30)),
            stride_sec=10,
            top_k=top_n,
        )
    else:
        print("[2/5] 세그먼트 준비 중...")
        segments, loaded_from = load_segments(
            video_path=str(input_file),
            subtitle_srt=subtitle_srt,
            subtitle_txt=subtitle_txt,
        )

        if not segments:
            raise RuntimeError("세그먼트를 하나도 만들지 못했습니다.")

        if loaded_from == "stt":
            txt_path = segments_dir / f"{sanitize_filename(input_file.stem)}_segments.txt"
            saved_segments_txt = save_segments_txt(segments, txt_path)
            print(f"      -> STT 결과 TXT 저장 완료: {saved_segments_txt}")

        print_segments_preview(segments, limit=10)

        print("[3/5] 하이라이트 추출 중...")
        candidates = extract_highlights_from_segments(
            segments=segments,
            top_n=top_n,
        )

    if not candidates:
        raise RuntimeError("후보 하이라이트를 추출하지 못했습니다.")

    assign_original_indices(candidates)
    print_highlights(candidates)

    if select_mode:
        candidates = select_highlights_interactively(candidates)
        if not candidates:
            raise RuntimeError("선택된 하이라이트가 없습니다.")

    print("[4/5] 클립 생성 중...")

    raw_saved_files: list[str] = []
    final_saved_files: list[str] = []

    for idx, clip in enumerate(candidates, start=1):
        original_idx = int(clip.get("orig_index", idx))
        start_sec = float(clip["start"])
        end_sec = float(clip["end"])

        raw_clip_path = raw_clip_dir / (
            f"{sanitize_filename(input_file.stem)}"
            f"_highlight_{original_idx}_{start_sec:.2f}_{end_sec:.2f}.mp4"
        )

        saved_raw = cut_clip(
            video_path=str(input_file),
            start_sec=start_sec,
            end_sec=end_sec,
            output_path=str(raw_clip_path),
        )
        raw_saved_files.append(saved_raw)

        if quick_mode:
            final_saved_files.append(saved_raw)
            continue

        clip_segments = clip.get("segments", [])
        if not isinstance(clip_segments, list):
            clip_segments = []

        final_clip_path = final_clip_dir / (
            f"{sanitize_filename(input_file.stem)}"
            f"_highlight_{original_idx}_final.mp4"
        )

        saved_final = export_clip_with_subtitles(
            clip_path=saved_raw,
            clip_start_sec=start_sec,
            clip_end_sec=end_sec,
            segments=clip_segments,
            output_path=str(final_clip_path),
        )
        final_saved_files.append(saved_final)

    print("[5/5] 완료")

    if saved_segments_txt:
        print("\n=== 저장된 세그먼트 TXT ===")
        print(saved_segments_txt)

    print("\n=== 생성된 RAW 클립 파일 ===")
    for path in raw_saved_files:
        print(path)

    print("\n=== 최종 출력 파일 ===")
    for path in final_saved_files:
        print(path)


def assign_original_indices(candidates: list[dict[str, Any]]) -> None:
    for i, clip in enumerate(candidates, start=1):
        clip["orig_index"] = i


def select_highlights_interactively(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    print("\n=== 하이라이트 선택 모드 ===")
    print("생성할 번호를 쉼표로 입력하세요. 예: 1,3,5")
    print("엔터만 누르면 전체 사용")

    raw = input("선택할 번호: ").strip()
    if not raw:
        print("[INFO] 전체 하이라이트를 사용합니다.")
        return candidates

    selected_indices: list[int] = []
    invalid_tokens: list[str] = []

    for token in raw.split(","):
        value = token.strip()
        if not value:
            continue

        if not value.isdigit():
            invalid_tokens.append(value)
            continue

        index = int(value)
        if index < 1 or index > len(candidates):
            invalid_tokens.append(value)
            continue

        if index not in selected_indices:
            selected_indices.append(index)

    if invalid_tokens:
        print(f"[WARN] 무시된 입력: {', '.join(invalid_tokens)}")

    if not selected_indices:
        print("[WARN] 유효한 선택이 없어 전체 하이라이트를 사용합니다.")
        return candidates

    selected = [candidates[i - 1] for i in selected_indices]
    print(f"[INFO] 선택된 하이라이트: {', '.join(str(i) for i in selected_indices)}")
    return selected


def probe_video_metadata(video_path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]

    try:
        raw = subprocess.check_output(cmd).decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"ffprobe 실행 실패: {e}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe 결과 파싱 실패: {e}")

    streams = data.get("streams", [])
    video_stream = None
    for stream in streams:
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError("영상 스트림을 찾지 못했습니다.")

    duration = 0.0
    format_info = data.get("format", {})

    if "duration" in format_info:
        try:
            duration = float(format_info["duration"])
        except Exception:
            duration = 0.0

    if duration <= 0 and "duration" in video_stream:
        try:
            duration = float(video_stream["duration"])
        except Exception:
            duration = 0.0

    return {
        "duration": duration,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
    }


def load_segments(
    video_path: str,
    subtitle_srt: str | None = None,
    subtitle_txt: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if subtitle_srt:
        subtitle_file = Path(subtitle_srt)
        if not subtitle_file.exists():
            raise FileNotFoundError(f"SRT 파일이 없습니다: {subtitle_srt}")

        print("      -> SRT 자막 사용")
        return parse_srt_file(str(subtitle_file)), "srt"

    if subtitle_txt:
        subtitle_file = Path(subtitle_txt)
        if not subtitle_file.exists():
            raise FileNotFoundError(f"TXT 파일이 없습니다: {subtitle_txt}")

        print("      -> TXT 자막 사용")
        return parse_txt_segments_file(str(subtitle_file)), "txt"

    print("      -> STT 자동 분석 사용")
    audio_path = extract_audio(video_path)
    segments = transcribe_audio(audio_path)
    return segments, "stt"


def save_segments_txt(
    segments: list[dict[str, Any]],
    output_path: Path,
) -> str:
    with open(output_path, "w", encoding="utf-8-sig") as f:
        for seg in segments:
            start_sec = float(seg.get("start", 0.0))
            end_sec = float(seg.get("end", 0.0))
            text = str(seg.get("text", "")).strip()

            if end_sec <= start_sec:
                continue
            if not text:
                continue

            f.write(f"[{start_sec:.2f} ~ {end_sec:.2f}] {text}\n")

    return str(output_path)


def extract_highlights_from_segments(
    segments: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    scorer = HighlightScorer(
        top_k=top_n,
        min_clip_sec=int(CLIP.get("min_sec", 15)),
        max_clip_sec=int(CLIP.get("max_sec", 40)),
        window_sec=int(CLIP.get("window_sec", 30)),
        overlap_gap_sec=int(CLIP.get("overlap_gap_sec", 20)),
        use_semantic=False,
    )
    highlights = scorer.extract_highlights(segments, top_k=top_n)
    return highlights


def print_segments_preview(
    segments: list[dict[str, Any]],
    limit: int = 10,
) -> None:
    print("\n=== 세그먼트 미리보기 ===")
    for seg in segments[:limit]:
        print(f"[{seg['start']:.2f} ~ {seg['end']:.2f}] {seg['text']}")

    if len(segments) > limit:
        print(f"... ({len(segments) - limit}개 세그먼트 생략)")


def print_highlights(clips: list[dict[str, Any]]) -> None:
    print("\n=== 최종 하이라이트 ===")
    for i, clip in enumerate(clips, start=1):
        print(f"\n[{i}]")
        print(f"start   : {clip.get('start')}")
        print(f"end     : {clip.get('end')}")
        print(f"score   : {clip.get('score')}")
        print(f"reasons : {clip.get('reasons')}")
        print(f"text    : {clip.get('text')}")

        score_breakdown = clip.get("score_breakdown")
        if score_breakdown:
            print(f"breakdown : {score_breakdown}")


def parse_srt_file(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip(), flags=re.MULTILINE)
    segments: list[dict[str, Any]] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        time_line = None
        text_lines: list[str] = []

        for line in lines:
            if "-->" in line:
                time_line = line
            elif not line.isdigit():
                text_lines.append(line)

        if time_line is None:
            continue

        start_sec, end_sec = parse_srt_time_line(time_line)
        if end_sec <= start_sec:
            continue

        text = " ".join(text_lines).strip()
        if not text:
            continue

        segments.append({
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "text": text,
        })

    return segments


def parse_srt_time_line(time_line: str) -> tuple[float, float]:
    parts = time_line.split("-->")
    if len(parts) != 2:
        return 0.0, 0.0

    start_str = parts[0].strip()
    end_str = parts[1].strip()
    return parse_srt_timestamp(start_str), parse_srt_timestamp(end_str)


def parse_srt_timestamp(ts: str) -> float:
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", ts)
    if not match:
        return 0.0

    hh = int(match.group(1))
    mm = int(match.group(2))
    ss = int(match.group(3))
    ms = int(match.group(4))
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def parse_txt_segments_file(file_path: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []

    pattern_bracket_float = re.compile(
        r"^\[\s*(\d+(?:\.\d+)?)\s*~\s*(\d+(?:\.\d+)?)\s*\]\s*(.+)$"
    )
    pattern_bracket_hms = re.compile(
        r"^\[\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*~\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\]\s*(.+)$"
    )
    pattern_pipe_float = re.compile(
        r"^\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(.+)$"
    )
    pattern_pipe_hms = re.compile(
        r"^\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\|\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\|\s*(.+)$"
    )

    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        start_sec = None
        end_sec = None
        text = ""

        m1 = pattern_bracket_float.match(line)
        m2 = pattern_bracket_hms.match(line)
        m3 = pattern_pipe_float.match(line)
        m4 = pattern_pipe_hms.match(line)

        if m1:
            start_sec = float(m1.group(1))
            end_sec = float(m1.group(2))
            text = m1.group(3).strip()
        elif m2:
            start_sec = parse_hms_to_seconds(m2.group(1))
            end_sec = parse_hms_to_seconds(m2.group(2))
            text = m2.group(3).strip()
        elif m3:
            start_sec = float(m3.group(1))
            end_sec = float(m3.group(2))
            text = m3.group(3).strip()
        elif m4:
            start_sec = parse_hms_to_seconds(m4.group(1))
            end_sec = parse_hms_to_seconds(m4.group(2))
            text = m4.group(3).strip()
        else:
            print(f"[WARN] TXT 파싱 실패 (줄 {idx}): {line}")
            continue

        if start_sec is None or end_sec is None:
            print(f"[WARN] 시간 해석 실패 (줄 {idx}): {line}")
            continue

        if end_sec <= start_sec:
            print(f"[WARN] 종료 시간이 시작 시간보다 작거나 같음 (줄 {idx}): {line}")
            continue

        if not text:
            print(f"[WARN] 텍스트 비어 있음 (줄 {idx}): {line}")
            continue

        segments.append({
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "text": text,
        })

    return segments


def parse_hms_to_seconds(value: str) -> float:
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$", value.strip())
    if not match:
        return 0.0

    hh = int(match.group(1))
    mm = int(match.group(2))
    ss = int(match.group(3))
    frac = match.group(4)

    fraction = 0.0
    if frac:
        fraction = float(f"0.{frac}")

    return hh * 3600 + mm * 60 + ss + fraction


def generate_quick_windows(
    duration_sec: float,
    window_sec: int,
    stride_sec: int,
    top_k: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    start = 0.0
    while start < duration_sec:
        end = min(start + window_sec, duration_sec)
        length = end - start

        if length >= 8.0:
            candidates.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "score": round(length, 2),
                "reasons": ["임시 분할 구간", "오디오 분석 미적용"],
                "text": "(텍스트 없음)",
                "segments": [],
            })

        if end >= duration_sec:
            break

        start += stride_sec

    return candidates[:top_k]


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    result = name
    for ch in invalid_chars:
        result = result.replace(ch, "_")
    return result.strip()


if __name__ == "__main__":
    main()  