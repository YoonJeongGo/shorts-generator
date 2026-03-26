# 파일 경로: config.py

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────
# 기본 경로
# ─────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
OUTPUT_DIR = BASE_DIR / "output"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────
# STT 설정
# ─────────────────────────────────────
WHISPER_MODEL = "base"
WHISPER_LANGUAGE = "ko"

# ─────────────────────────────────────
# STT 후처리 설정
# ─────────────────────────────────────
STT_POSTPROCESS = {
    # 세그먼트 text 최소 길이
    "min_text_len": 2,

    # 한국어 비율이 이 값보다 낮으면 노이즈 가능성 큼
    "min_korean_ratio": 0.20,

    # 영문 비율이 너무 높으면 제거
    "max_english_ratio": 0.60,

    # 같은 글자가 과도하게 반복되면 제거
    "max_repeat_char_run": 4,

    # ASCII 토큰 비율이 너무 높으면 제거
    "max_ascii_token_ratio": 0.70,

    # 완전히 제거할 이상 토큰
    "banned_tokens": [
        "donkey",
        "produkt",
        "product",
        "million",
        "butcher",
        "subtitle",
        "music",
        "bgm",
        "uh",
        "um",
        "ah",
        "oh",
    ],

    # 너무 짧고 의미 없는 단독 토큰 제거용
    "noise_tokens": [
        "어",
        "음",
        "아",
        "오",
        "와",
        "뭐",
        "응",
        "흠",
        "uh",
        "um",
        "ah",
        "oh",
        "mm",
    ],

    # 짧은 세그먼트 병합 기준
    "merge_max_gap_sec": 0.80,
    "merge_max_combined_sec": 12.0,
    "merge_min_text_len": 8,
}

# ─────────────────────────────────────
# 하이라이트 규칙
# ─────────────────────────────────────
KEYWORDS = {
    "emphasis": [
        "핵심",
        "중요",
        "결론",
        "요약",
        "포인트",
        "핵심은",
        "중요한",
        "결론은",
        "정리하면",
        "한마디로",
        "결국",
        "가장 중요",
    ],
    "exclaim": [
        "와",
        "진짜",
        "대박",
        "헐",
        "맞아",
        "오",
        "아",
        "와우",
        "놀랍",
        "미쳤",
        "장난 아니",
    ],
    "sports_baseball": [
        "홈런",
        "안타",
        "삼진",
        "타석",
        "투수",
        "타자",
        "득점",
        "결정적",
        "병살",
        "볼넷",
        "스트라이크",
        "직구",
        "변화구",
        "실책",
        "역전",
        "끝내기",
        "선발",
        "불펜",
        "주자",
        "출루",
    ],
}

FILLERS = {
    "음",
    "어",
    "그",
    "뭐",
    "이제",
    "그냥",
    "아무튼",
    "근데",
    "약간",
    "진짜로",
    "사실",
    "뭔가",
}

# ─────────────────────────────────────
# 점수 가중치
# highlight_scorer.py 기준 키 이름으로 통일
# ─────────────────────────────────────
SCORE_WEIGHTS = {
    # 기본 신호
    "keyword": 1.5,
    "sports_keyword": 2.0,
    "emphasis": 1.2,
    "exclaim": 1.0,
    "question": 0.8,
    "silence_before": 0.8,
    "semantic": 1.5,

    # 감점
    "too_short_penalty": -1.2,
    "filler_heavy_penalty": -1.0,
    "repeat_penalty": -1.0,
    "low_info_penalty": -0.8,
    "bad_segment_penalty": -1.4,
    "noise_penalty": -1.8,
    "english_penalty": -1.2,
}

# ─────────────────────────────────────
# 하이라이트 필터 설정
# highlight_scorer.py 기준 키 이름으로 통일
# ─────────────────────────────────────
HIGHLIGHT_FILTER = {
    # 최종 점수가 이 값보다 낮으면 버림
    "min_final_score": 2.0,

    # rule score가 너무 낮으면 후보 생성 단계에서 제거
    "min_rule_score": -2.5,

    # 한글 비율 관련
    "good_korean_ratio": 0.55,
    "min_korean_ratio": 0.30,
    "min_window_korean_ratio": 0.25,

    # 영어/ASCII/노이즈 관련
    "max_ascii_ratio": 0.55,
    "max_noise_ratio": 0.40,

    # 반복 문자 관련
    "max_repeat_char_ratio": 0.35,
    "repeat_char_run_threshold": 4,

    # 필러 관련
    "max_filler_ratio": 0.45,
    "filler_ratio_penalty_threshold": 0.35,

    # 침묵 구간 기준
    "min_silence_gap": 1.0,

    # 짧은 문장 비중이 높으면 패널티용 참고값
    "short_segment_ratio_threshold": 0.50,
}

# ─────────────────────────────────────
# 클립 설정
# ─────────────────────────────────────
CLIP = {
    "min_sec": 15,
    "max_sec": 40,
    "window_sec": 30,
    "top_k": 3,
    "overlap_gap_sec": 20,

    # 🔥 Hook 기반 start 재조정
    "enable_hook_retrim": True,

    # 후보 탐색 범위
    "hook_search_sec": 18.0,
    "hook_search_back_sec": 6.0,

    # 좋은 훅을 찾았으면 앞의 설명형 문장을 다시 끌고 오지 않도록 0으로 유지
    "hook_pre_roll_sec": 0.0,

    # 최소 이동 / 유지 조건
    "hook_min_shift_sec": 0.5,
    "hook_keep_min_duration_sec": 10.0,

    # 현재 시작보다 확실히 좋아야 교체
    "hook_min_improve": 0.5,

    # 훅 디버그 로그
    "hook_debug": True,
    "hook_debug_top_n": 3,
}

# ─────────────────────────────────────
# 영상/오디오 처리 설정
# ─────────────────────────────────────
AUDIO = {
    "sample_rate": 16000,
    "channels": 1,
    "codec": "pcm_s16le",
}

VIDEO = {
    "short_width": 720,
    "short_height": 1280,
    "video_codec": "libx264",
    "audio_codec": "aac",
    "preset": "fast",
    "crf": "23",
    "audio_bitrate": "128k",
}

# ─────────────────────────────────────
# 자막 스타일
# exporter.py 기준
# ─────────────────────────────────────
SUBTITLE = {
    "font_file": "C:/Windows/Fonts/malgun.ttf",
    "font_size": 36,
    "line_spacing": 6,
    "borderw": 3,
    "boxborderw": 10,
    "bottom_margin": 110,
    "line_split_threshold": 12,
    "max_chars_per_line": 12,
    "max_total_chars": 24,
}