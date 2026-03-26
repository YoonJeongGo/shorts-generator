from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "tmp"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4GB


# ─────────────────────────────────────
# 유틸
# ─────────────────────────────────────
def ok(data: dict[str, Any], status: int = 200):
    return jsonify(data), status


def err(message: str, status: int = 400):
    return jsonify({"error": message}), status


def safe_filename(name: str) -> str:
    name = os.path.basename(name).strip()
    if not name:
        return "uploaded_video"
    name = re.sub(r"[^\w.\-가-힣]+", "_", name)
    return name


def clip_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


# ─────────────────────────────────────
# 규칙 기반 하이라이트 분석
# 현재 확실히 동작하는 fallback
# ─────────────────────────────────────
KEYWORDS = {
    "emphasis": [
        "핵심", "중요", "결론", "요약", "포인트", "핵심은",
        "결론은", "정리하면", "한마디로", "결국", "가장 중요", "꼭", "절대"
    ],
    "exclaim": [
        "와", "진짜", "대박", "헐", "우와", "오", "이런", "세상에"
    ],
    "question": [
        "?", "죠", "인가요", "할까요", "어떨까요", "없을까요"
    ],
}

FILLERS = {"음", "어", "그", "뭐", "이제", "그냥", "아무튼", "근데", "뭔가"}

WEIGHTS = {
    "emphasis": 20,
    "exclaim": 15,
    "question": 10,
    "silence": 8,
    "too_short": -20,
    "filler": -10,
}


def score_segment(seg: dict[str, Any], prev_end: float) -> tuple[int, list[str]]:
    text = str(seg.get("text", "")).strip()
    start = float(seg.get("start", 0))
    end = float(seg.get("end", 0))
    duration = max(0.0, end - start)

    score = 0
    reasons: list[str] = []

    if any(k in text for k in KEYWORDS["emphasis"]):
        score += WEIGHTS["emphasis"]
        reasons.append(f"강조+{WEIGHTS['emphasis']}")

    if any(k in text for k in KEYWORDS["exclaim"]):
        score += WEIGHTS["exclaim"]
        reasons.append(f"감탄+{WEIGHTS['exclaim']}")

    if any(k in text for k in KEYWORDS["question"]):
        score += WEIGHTS["question"]
        reasons.append(f"질문+{WEIGHTS['question']}")

    if start - prev_end >= 1.0:
        score += WEIGHTS["silence"]
        reasons.append(f"침묵+{WEIGHTS['silence']}")

    if duration < 5.0:
        score += WEIGHTS["too_short"]
        reasons.append(f"짧음{WEIGHTS['too_short']}")

    words = text.split()
    if words:
        filler_ratio = sum(1 for w in words if w in FILLERS) / len(words)
        if filler_ratio > 0.3:
            score += WEIGHTS["filler"]
            reasons.append(f"필러{WEIGHTS['filler']}")

    return max(score, 0), reasons


def generate_dummy_segments() -> list[dict[str, Any]]:
    # 자막 없음 모드 fallback
    segments: list[dict[str, Any]] = []
    for i in range(12):
        start = float(i * 8)
        end = start + 5.0
        text = "핵심 포인트" if i % 4 == 0 else "내용"
        segments.append({
            "id": i + 1,
            "start": start,
            "end": end,
            "text": text,
        })
    return segments


def extract_highlights_rule_based(
    subtitles: list[dict[str, Any]],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    segs = subtitles[:] if subtitles else generate_dummy_segments()
    if not segs:
        return []

    segs = sorted(segs, key=lambda x: float(x.get("start", 0)))

    prev_end = 0.0
    for seg in segs:
        score, reasons = score_segment(seg, prev_end)
        seg["_score"] = score
        seg["_reasons"] = reasons
        prev_end = float(seg.get("end", 0))

    window = 30.0
    step = 5.0
    min_duration = 10.0
    total_end = float(segs[-1].get("end", 0))

    candidates: list[dict[str, Any]] = []
    t = float(segs[0].get("start", 0))

    while t < total_end:
        w_end = t + window
        inside = [
            s for s in segs
            if float(s.get("start", 0)) >= t and float(s.get("end", 0)) <= w_end
        ]

        if inside:
            duration = float(inside[-1].get("end", 0)) - float(inside[0].get("start", 0))
            if duration >= min_duration:
                total_score = sum(int(s.get("_score", 0)) for s in inside)
                reasons = sorted({r for s in inside for r in s.get("_reasons", [])})
                text = " ".join(str(s.get("text", "")).strip() for s in inside[:3]).strip()

                candidates.append({
                    "start": float(inside[0].get("start", 0)),
                    "end": float(inside[-1].get("end", 0)),
                    "score": total_score,
                    "reasons": reasons,
                    "text": text,
                })

        t += step

    candidates.sort(key=lambda x: x["score"], reverse=True)

    picked: list[dict[str, Any]] = []
    for c in candidates:
        if all(
            clip_overlap_ratio(c["start"], c["end"], p["start"], p["end"]) < 0.4
            for p in picked
        ):
            picked.append(c)
            if len(picked) >= top_n:
                break

    picked.sort(key=lambda x: x["start"])

    result: list[dict[str, Any]] = []
    for i, p in enumerate(picked, start=1):
        result.append({
            "rank": i,
            "start": p["start"],
            "end": p["end"],
            "score": min(int(p["score"]), 100),
            "reasons": p["reasons"],
            "text": p["text"],
        })
    return result


# ─────────────────────────────────────
# 라우트
# ─────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return ok({"status": "ok"})


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return err("video 필드가 없습니다.", 400)

    file = request.files["video"]
    if file.filename is None or file.filename.strip() == "":
        return err("업로드할 파일명이 없습니다.", 400)

    filename = safe_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    return ok({
        "message": "업로드 성공",
        "filename": filename,
        "path": str(save_path),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data:
        return err("JSON 본문이 없습니다.", 400)

    subtitles = data.get("subtitles", [])
    sub_mode = str(data.get("sub_mode", "none"))
    video_path = data.get("video_path")

    if sub_mode in {"manual", "file", "stt"} and not isinstance(subtitles, list):
        return err("subtitles 형식이 올바르지 않습니다.", 400)

    normalized_subs: list[dict[str, Any]] = []
    if isinstance(subtitles, list):
        for i, s in enumerate(subtitles, start=1):
            try:
                start = float(s.get("start", 0))
                end = float(s.get("end", 0))
                text = str(s.get("text", "")).strip()
            except Exception:
                return err(f"subtitles[{i}] 형식이 잘못되었습니다.", 400)

            if end <= start:
                continue

            normalized_subs.append({
                "id": int(s.get("id", i)),
                "start": start,
                "end": end,
                "text": text,
            })

    # 여기서부터는 실제 pipeline 연결 지점
    # 네 highlight_scorer.py 내부 함수명을 내가 확인하지 못했기 때문에,
    # 현재는 확실히 동작하는 fallback 분석기를 사용한다.
    # 나중에 실제 함수명을 확인하면 이 부분만 교체하면 된다.
    highlights = extract_highlights_rule_based(normalized_subs, top_n=3)

    return ok({
        "message": "분석 완료",
        "video_path": video_path,
        "sub_mode": sub_mode,
        "subtitle_count": len(normalized_subs),
        "highlights": highlights,
    })


@app.route("/cut", methods=["POST"])
def cut():
    data = request.get_json(silent=True)
    if not data:
        return err("JSON 본문이 없습니다.", 400)

    video_path = data.get("video_path")
    start = data.get("start")
    end = data.get("end")
    rank = data.get("rank", 1)

    if video_path is None or start is None or end is None:
        return err("video_path, start, end 값이 필요합니다.", 400)

    try:
        start_f = float(start)
        end_f = float(end)
    except Exception:
        return err("start/end는 숫자여야 합니다.", 400)

    if end_f <= start_f:
        return err("end는 start보다 커야 합니다.", 400)

    # 현재는 클립 생성 자리만 잡아둠.
    # 실제 ffmpeg 연결 전까지는 결과 이름만 반환.
    output_name = f"clip_{int(rank):02d}_{int(start_f)}_{int(end_f)}.mp4"
    output_path = OUTPUT_DIR / output_name

    return ok({
        "message": "클립 생성 요청 수신",
        "output_name": output_name,
        "output_path": str(output_path),
        "note": "현재 /cut은 자리만 잡아둔 상태이며 ffmpeg 실제 연결은 아직 안 붙어 있습니다.",
    })


if __name__ == "__main__":
    app.run(debug=True)