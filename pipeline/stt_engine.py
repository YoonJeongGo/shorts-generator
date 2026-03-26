from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from faster_whisper import WhisperModel

from config import FILLERS, STT_POSTPROCESS, TMP_DIR, WHISPER_LANGUAGE, WHISPER_MODEL


# 캐시 버전:
# 후처리 로직 바꾸면 이 값 올려서 예전 캐시와 분리
CACHE_VERSION = "stt_v2"


def transcribe_audio(audio_path: str) -> List[Dict[str, Any]]:
    """
    Faster-Whisper로 음성을 텍스트로 변환한 뒤,
    후처리 및 필터링을 적용한 세그먼트 리스트를 반환한다.

    반환 형식:
    [
        {
            "start": 0.0,
            "end": 3.2,
            "text": "안녕하세요",
            "quality": {
                "ascii_ratio": 0.0,
                "korean_ratio": 0.9,
                "repeat_char_ratio": 0.0,
                "noise_ratio": 0.0,
                "filler_ratio": 0.0,
                "token_count": 1,
            }
        },
        ...
    ]
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"오디오 파일이 없습니다: {audio_path}")

    cache_path = _get_cache_path(audio_file)

    if cache_path.exists():
        print(f"[INFO] STT 캐시 사용: {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            if isinstance(cached, list):
                return cached

            print("[WARN] 캐시 형식이 잘못되어 재생성합니다.")
        except Exception as e:
            print(f"[WARN] 캐시 로드 실패, 재생성합니다: {e}")

    device = "cpu"
    compute_type = "int8"

    print(
        f"[INFO] Loading Faster-Whisper model: {WHISPER_MODEL} "
        f"(device={device}, compute_type={compute_type})"
    )

    model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )

    print("[INFO] Transcribing audio...")

    segments, info = model.transcribe(
        str(audio_file),
        language=WHISPER_LANGUAGE,
        vad_filter=True,
        beam_size=5,
        best_of=5,
        word_timestamps=False,
        condition_on_previous_text=True,
    )

    raw_segments: List[Dict[str, Any]] = []
    for seg in segments:
        text = str(seg.text or "").strip()
        if not text:
            continue

        raw_segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            }
        )

    print(f"[INFO] Raw STT segments: {len(raw_segments)}")
    print(
        "[INFO] Detected language: "
        f"{getattr(info, 'language', WHISPER_LANGUAGE)} "
        f"(prob={getattr(info, 'language_probability', 'N/A')})"
    )

    processed = _postprocess_segments(raw_segments)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"[INFO] STT cache saved: {cache_path}")
    print(f"[INFO] Final segments after cleanup: {len(processed)}")

    return processed


def _postprocess_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    원본 STT 세그먼트에 대해:
    1) 텍스트 정리
    2) 노이즈 세그먼트 제거
    3) quality 정보 부착
    4) 짧은 세그먼트 병합
    5) gap_before 계산
    """
    cleaned: List[Dict[str, Any]] = []

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        raw_text = str(seg.get("text", "")).strip()

        text = _normalize_text(raw_text)
        if not text:
            continue

        quality = _analyze_text_quality(text)

        if _should_drop_segment(text, quality):
            continue

        cleaned.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "quality": quality,
            }
        )

    merged = _merge_short_segments(cleaned)

    # 병합 후 다시 한 번 정리 + 재평가
    reprocessed: List[Dict[str, Any]] = []
    for seg in merged:
        text = _normalize_text(str(seg.get("text", "")).strip())
        if not text:
            continue

        quality = _analyze_text_quality(text)
        if _should_drop_segment(text, quality):
            continue

        reprocessed.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
                "quality": quality,
            }
        )

    reprocessed = _attach_gap_before(reprocessed)
    return reprocessed


def _normalize_text(text: str) -> str:
    """
    STT 결과의 기본 정리:
    - 괄호/대괄호 안 잡음 제거
    - 불필요 공백 정리
    - 영어/ASCII 찌꺼기 제거
    - 자주 틀리는 단어 치환
    """
    text = str(text or "").strip()
    if not text:
        return ""

    # [음악], (박수), [Music] 같은 태그 제거
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*\)", " ", text)

    # 양쪽 불필요한 따옴표 제거
    text = text.strip("\"'`")

    # 특수문자만 반복되는 조각 제거
    if re.fullmatch(r"[\W_]+", text):
        return ""

    text = _remove_ascii_garbage(text)
    text = _clean_mixed_noise(text)
    text = _apply_replacements(text)

    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    # 문장 앞뒤 이상한 문장부호 정리
    text = re.sub(r"^[,.\-~!?\s]+", "", text)
    text = re.sub(r"[,.\-~!?\s]+$", "", text)

    # 최종적으로 특수문자 덩어리면 제거
    if not text or re.fullmatch(r"[\W_]+", text):
        return ""

    return text


def _remove_ascii_garbage(text: str) -> str:
    """
    한국어 STT에서 자주 섞이는 의미 없는 ASCII 토큰 제거.
    완전히 공격적으로 지우지 않고, '거의 확실히 쓰레기인 것'만 제거한다.
    """
    banned_exact = {
        str(x).strip().lower()
        for x in STT_POSTPROCESS.get("banned_tokens", [])
        if str(x).strip()
    }

    tokens = _tokenize(text)
    kept: List[str] = []

    for tok in tokens:
        stripped = tok.strip()
        lowered = stripped.lower().strip(".,!?~'\"`")

        if not stripped:
            continue

        if lowered in banned_exact:
            continue

        # 순수 ASCII인데 2글자 이상이면 일단 제거
        # 한국어 드라마/영화 대사 STT 기준에선 이게 남는 경우 대부분 쓰레기
        if re.fullmatch(r"[A-Za-z0-9_\-]+", stripped):
            if len(stripped) >= 2:
                continue

        kept.append(stripped)

    return " ".join(kept).strip()


def _clean_mixed_noise(text: str) -> str:
    """
    한글 주변에 붙는 쓸데없는 ASCII 찌꺼기 정리.
    예:
    - 인터뷰n준비 -> 인터뷰 준비
    - 뭐.n나도 -> 뭐 나도
    """
    # 한글 + ASCII 1~3글자 + 한글 -> 중간 ASCII 제거
    text = re.sub(r"([가-힣])[A-Za-z]{1,3}([가-힣])", r"\1 \2", text)

    # 한글 + 점/기호 + ASCII 짧은 토큰 + 한글
    text = re.sub(r"([가-힣])[.,/\\|:;]+[A-Za-z]{1,3}([가-힣])", r"\1 \2", text)
    text = re.sub(r"([가-힣])[A-Za-z]{1,3}[.,/\\|:;]+([가-힣])", r"\1 \2", text)

    # 점/쉼표 사이 짧은 ASCII 찌꺼기 제거
    text = re.sub(r"[.,/\\|:;]+[A-Za-z]{1,3}[.,/\\|:;]+", " ", text)

    return text


def _apply_replacements(text: str) -> str:
    """
    자주 틀리는 단어 보정.
    여기서는 '확신 높은 것'만 넣는다.
    애매한 치환은 오히려 망친다.
    """
    replacements = {
        # 드라마/스포츠에서 자주 보일 수 있는 것 위주
        "골등글러브": "골든글러브",
        "골든 글러브": "골든글러브",
        "플레어": "플레이어",
        "플레이 어": "플레이어",
        "인터부": "인터뷰",
        "인터 뷰": "인터뷰",
        "선은": "선수는",
        "인동국": "인동구",
        "인동 국": "인동구",
        "드래프트트": "드래프트",
        "메이저리그스": "메이저리그",
        "스트라익": "스트라이크",
        "스트라이 익": "스트라이크",
    }

    result = text
    for src, dst in replacements.items():
        result = result.replace(src, dst)

    # 조사 붙은 형태에서 띄어쓰기 깨진 경우 일부 정리
    result = re.sub(r"\b인터 뷰\b", "인터뷰", result)
    result = re.sub(r"\b플레이 어\b", "플레이어", result)
    result = re.sub(r"\b골든 글러브\b", "골든글러브", result)

    return result


def _analyze_text_quality(text: str) -> Dict[str, Any]:
    """
    하이라이트 scorer가 사용할 quality 정보 계산
    """
    total_chars = len(text)
    if total_chars == 0:
        return {
            "ascii_ratio": 1.0,
            "korean_ratio": 0.0,
            "repeat_char_ratio": 1.0,
            "noise_ratio": 1.0,
            "filler_ratio": 0.0,
            "token_count": 0,
        }

    korean_chars = len(re.findall(r"[가-힣]", text))
    ascii_chars = len(re.findall(r"[A-Za-z0-9]", text))

    repeat_char_count = 0
    for m in re.finditer(r"(.)\1{2,}", text):
        repeat_char_count += len(m.group(0))

    tokens = _tokenize(text)
    filler_count = sum(1 for tok in tokens if tok in FILLERS)
    noise_count = sum(1 for tok in tokens if _is_noise_token(tok))

    return {
        "ascii_ratio": ascii_chars / max(total_chars, 1),
        "korean_ratio": korean_chars / max(total_chars, 1),
        "repeat_char_ratio": repeat_char_count / max(total_chars, 1),
        "noise_ratio": noise_count / max(len(tokens), 1) if tokens else 0.0,
        "filler_ratio": filler_count / max(len(tokens), 1) if tokens else 0.0,
        "token_count": len(tokens),
    }


def _should_drop_segment(text: str, quality: Dict[str, Any]) -> bool:
    """
    세그먼트를 완전히 버릴지 판단
    """
    min_text_len = int(STT_POSTPROCESS.get("min_text_len", 2))
    min_korean_ratio = float(STT_POSTPROCESS.get("min_korean_ratio", 0.2))
    max_english_ratio = float(STT_POSTPROCESS.get("max_english_ratio", 0.6))
    max_repeat_char_run = int(STT_POSTPROCESS.get("max_repeat_char_run", 4))
    max_ascii_token_ratio = float(STT_POSTPROCESS.get("max_ascii_token_ratio", 0.7))

    banned_tokens = {
        str(x).strip().lower()
        for x in STT_POSTPROCESS.get("banned_tokens", [])
        if str(x).strip()
    }
    noise_tokens = {
        str(x).strip().lower()
        for x in STT_POSTPROCESS.get("noise_tokens", [])
        if str(x).strip()
    }

    normalized_lower = text.lower()
    tokens = _tokenize(text)
    tokens_lower = [tok.lower() for tok in tokens]

    if len(text) < min_text_len:
        return True

    # 금지 토큰 포함 시 제거
    for banned in banned_tokens:
        if banned and banned in normalized_lower:
            return True

    # 같은 글자 과도 반복
    if _has_excessive_repeat_run(text, max_repeat_char_run):
        return True

    # 한글 비율 너무 낮고, 영문 비중 높으면 제거
    korean_ratio = float(quality.get("korean_ratio", 0.0))
    ascii_ratio = float(quality.get("ascii_ratio", 0.0))

    if korean_ratio < min_korean_ratio and ascii_ratio > max_english_ratio:
        return True

    # ASCII 토큰 비율 과다
    if tokens:
        ascii_token_count = sum(1 for tok in tokens if _is_ascii_token(tok))
        ascii_token_ratio = ascii_token_count / max(len(tokens), 1)
        if ascii_token_ratio > max_ascii_token_ratio:
            return True

    # 단독 노이즈 토큰
    if len(tokens_lower) == 1 and tokens_lower[0] in noise_tokens:
        return True

    # 토큰 전부 노이즈면 제거
    if tokens_lower and all(tok in noise_tokens or _is_noise_token(tok) for tok in tokens_lower):
        return True

    return False


def _merge_short_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    너무 짧게 잘린 세그먼트를 인접 세그먼트와 병합
    """
    if not segments:
        return []

    merge_max_gap_sec = float(STT_POSTPROCESS.get("merge_max_gap_sec", 0.8))
    merge_max_combined_sec = float(STT_POSTPROCESS.get("merge_max_combined_sec", 12.0))
    merge_min_text_len = int(STT_POSTPROCESS.get("merge_min_text_len", 8))

    merged: List[Dict[str, Any]] = []
    current = dict(segments[0])

    for next_seg in segments[1:]:
        current_duration = float(current["end"]) - float(current["start"])
        next_duration = float(next_seg["end"]) - float(next_seg["start"])
        gap = float(next_seg["start"]) - float(current["end"])

        current_text = str(current["text"])
        next_text = str(next_seg["text"])

        combined_duration = float(next_seg["end"]) - float(current["start"])
        should_merge = False

        # 현재 세그먼트가 너무 짧거나 텍스트가 짧으면 다음과 합치기
        if (
            current_duration <= 2.0
            or len(current_text) < merge_min_text_len
            or next_duration <= 1.5
            or len(next_text) < merge_min_text_len
        ):
            if gap <= merge_max_gap_sec and combined_duration <= merge_max_combined_sec:
                should_merge = True

        if should_merge:
            combined_text = _merge_texts(current_text, next_text)
            current = {
                "start": float(current["start"]),
                "end": float(next_seg["end"]),
                "text": combined_text,
                "quality": _analyze_text_quality(combined_text),
            }
        else:
            merged.append(current)
            current = dict(next_seg)

    merged.append(current)
    return merged


def _merge_texts(left: str, right: str) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()

    if not left:
        return right
    if not right:
        return left

    if left == right:
        return left

    # 좌측 끝과 우측 시작이 겹치는 간단한 경우 처리
    max_overlap = min(len(left), len(right), 15)
    for size in range(max_overlap, 0, -1):
        if left.endswith(right[:size]):
            return (left + right[size:]).strip()

    return f"{left} {right}".strip()


def _attach_gap_before(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    highlight_scorer에서 침묵 구간 가점 계산용
    """
    if not segments:
        return []

    result: List[Dict[str, Any]] = []
    prev_end: Optional[float] = None

    for seg in segments:
        item = dict(seg)
        if prev_end is None:
            item["gap_before"] = 0.0
        else:
            item["gap_before"] = max(0.0, float(item["start"]) - prev_end)

        prev_end = float(item["end"])
        result.append(item)

    return result


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"\s+", str(text).strip()) if tok]


def _is_ascii_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]+", token))


def _is_noise_token(token: str) -> bool:
    token = str(token or "").strip()
    if not token:
        return True

    # 1~2글자 영문
    if re.fullmatch(r"[A-Za-z]{1,2}", token):
        return True

    # 특수문자만
    if re.fullmatch(r"[^\w가-힣]+", token):
        return True

    # 같은 문자 반복
    if re.fullmatch(r"(.)\1{2,}", token):
        return True

    # 한글도 아니고 1글자면 거의 노이즈
    if len(token) <= 1 and not re.search(r"[가-힣]", token):
        return True

    return False


def _has_excessive_repeat_run(text: str, threshold: int) -> bool:
    pattern = rf"(.)\1{{{max(threshold - 1, 1)},}}"
    return re.search(pattern, text) is not None


def _get_cache_path(audio_file: Path) -> Path:
    """
    오디오 파일 내용 + 캐시 버전 기준 해시로 캐시 파일 경로 생성
    """
    file_hash = _hash_file(audio_file)
    version_hash = hashlib.sha256(CACHE_VERSION.encode("utf-8")).hexdigest()[:8]
    return TMP_DIR / f"{file_hash}_{version_hash}_segments.json"


def _hash_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]