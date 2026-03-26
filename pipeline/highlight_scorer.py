# 파일 경로: pipeline/highlight_scorer.py

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import CLIP, FILLERS, HIGHLIGHT_FILTER, KEYWORDS, SCORE_WEIGHTS

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore


@dataclass
class HighlightCandidate:
    start: float
    end: float
    score: float
    text: str
    reasons: List[str]
    score_breakdown: Dict[str, float]
    segments: List[Dict[str, Any]]
    quality: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "score": round(self.score, 4),
            "text": self.text,
            "reasons": self.reasons,
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
            "segments": self.segments,
            "quality": self.quality,
        }


class HighlightScorer:
    DEFAULT_EMPHASIS_WORDS = [
        "진짜", "정말", "대박", "미쳤", "와", "헐", "설마", "왜", "어떻게",
        "결국", "드디어", "무조건", "확실", "반드시", "핵심", "중요", "끝내기",
        "역전", "홈런", "삼진",
    ]

    DEFAULT_NEGATIVE_START_WORDS = [
        "그리고", "근데", "그런데", "그래서", "하지만", "아니", "그러니까",
        "근데요", "근데 그", "그리고 나서",
    ]

    DEFAULT_ENDINGS_GOOD = [
        "다", "요", "죠", "네", "야", "입니다", "거든", "맞아", "맞습니다",
        "했다", "합니다", "됐어", "됐다", "좋아", "끝", "홈런", "삼진",
    ]

    DEFAULT_STRONG_PATTERNS = [
        r"\?",
        r"!",
        r"와",
        r"헐",
        r"대박",
        r"진짜",
        r"정말",
        r"어떻게",
        r"왜",
        r"뭐야",
        r"뭐라고",
        r"말도 안",
        r"미쳤",
        r"설마",
        r"역전",
        r"끝내기",
        r"홈런",
        r"삼진",
    ]

    DEFAULT_REACTION_PATTERNS = [
        r"ㅋㅋ+",
        r"하하+",
        r"웃",
        r"와",
        r"헐",
        r"어어",
        r"어\?",
        r"오",
        r"미쳤",
        r"대박",
        r"진짜",
        r"왜",
        r"뭐야",
        r"뭐라고",
        r"아니",
        r"잠깐",
        r"설마",
    ]

    DEFAULT_HOOK_PATTERNS = [
        r"\?",
        r"!",
        r"왜",
        r"뭐야",
        r"뭐라고",
        r"어떻게",
        r"아니",
        r"잠깐",
        r"설마",
        r"진짜",
        r"헐",
        r"와",
        r"미쳤",
        r"말도 안",
        r"큰일",
        r"끝났",
        r"역전",
        r"끝내기",
        r"홈런",
        r"삼진",
    ]

    DEFAULT_WEAK_OPENERS = [
        "안녕하세요",
        "여러분",
        "자 그러면",
        "자 그럼",
        "오늘은",
        "먼저",
        "일단",
        "그리고요",
        "이제",
        "예",
        "네",
    ]

    DEFAULT_EXPLAIN_PREFIXES = [
        "그냥",
        "그리고",
        "근데",
        "그런데",
        "그래서",
        "그러니까",
        "일단",
        "지금은",
        "저는",
        "우리는",
        "제가",
        "이게",
        "이건",
        "그게",
        "그 사람들",
        "그 사람은",
        "먼저",
        "자 그러면",
        "자 그럼",
        "오늘은",
        "안녕하세요",
        "여러분",
        "네",
        "예",
        "나 지금",
        "제가 지금",
        "이거 지금",
        "저번에",
        "사실은",
        "진짜 웃긴 게",
        "결국",
        "그게 뭐냐면",
        "어떻게 됐냐면",
        "말하자면",
        "쉽게 말해서",
        "정리하면",
    ]

    DEFAULT_EXPLAIN_SUBSTRINGS = [
        "행사할지",
        "설명드리",
        "말씀드리",
        "보시면",
        "같습니다",
        "생각합니다",
        "있어가지고",
        "있어서",
        "그러니까",
        "그게 아니라",
        "어떻게 할지",
        "이거 어떻게",
        "그 사람들 지금",
        "지금 저놈들",
        "지금 상황이",
        "보통은",
        "먼저",
        "일단",
        "지금은",
        "저번에",
        "진짜 웃긴 게",
        "결국 어떻게",
        "그게 뭐냐면",
        "어떻게 됐냐면",
        "사실은",
        "말하자면",
        "쉽게 말해서",
        "정리하면",
        "보면은",
        "봤더니",
        "보니까",
        "하는데",
        "하더라고",
        "하니까",
        "라고 하더라",
        "라고 해서",
        "라는 거야",
    ]

    DEFAULT_SEMANTIC_ANCHORS = [
        "감정이 강하게 드러나는 장면",
        "짧고 강한 반응이 오가는 대화 장면",
        "갈등이나 긴장감이 느껴지는 장면",
        "놀라움이나 반전이 있는 장면",
        "결정적이거나 임팩트 있는 장면",
        "웃음이나 리액션이 강한 장면",
        "야구 경기에서 중요한 순간",
        "인터뷰에서 인상적인 한마디가 나오는 장면",
    ]

    def __init__(
        self,
        top_k: Optional[int] = None,
        min_clip_sec: Optional[int] = None,
        max_clip_sec: Optional[int] = None,
        window_sec: Optional[int] = None,
        overlap_gap_sec: Optional[int] = None,
        use_semantic: bool = True,
    ) -> None:
        self.top_k = int(top_k or CLIP.get("top_k", 3))
        self.min_clip_sec = int(min_clip_sec or CLIP.get("min_sec", 15))
        self.max_clip_sec = int(max_clip_sec or CLIP.get("max_sec", 40))
        self.window_sec = int(window_sec or CLIP.get("window_sec", 30))
        self.overlap_gap_sec = int(overlap_gap_sec or CLIP.get("overlap_gap_sec", 20))

        self.min_fallback_clip_sec = max(8, self.min_clip_sec - 5)

        self.keyword_map = KEYWORDS if isinstance(KEYWORDS, dict) else {}
        self.fillers = {str(x).strip() for x in FILLERS} if isinstance(FILLERS, (set, list, tuple)) else set()

        self._semantic_enabled = bool(use_semantic and SentenceTransformer is not None)
        self._semantic_model: Optional[Any] = None
        self._semantic_anchor_vectors: Optional[Any] = None

        if self._semantic_enabled:
            try:
                self._semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                self._semantic_anchor_vectors = self._semantic_model.encode(
                    self.DEFAULT_SEMANTIC_ANCHORS,
                    normalize_embeddings=True,
                )
            except Exception:
                self._semantic_model = None
                self._semantic_anchor_vectors = None
                self._semantic_enabled = False

    # ============================================================
    # Public
    # ============================================================

    def extract_highlights(
        self,
        segments: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        segments_norm = self._normalize_segments(segments)
        if not segments_norm:
            return []

        local_top_k = int(top_k or self.top_k)

        self._prepare_segment_features(segments_norm)
        self._apply_local_contrast_scores(segments_norm)

        seeds = self._select_seed_indices(segments_norm)
        candidates = self._build_candidates_from_seeds(segments_norm, seeds)

        if len(candidates) < max(local_top_k * 2, 6):
            extra = self._build_candidates_from_seeds(
                segments_norm,
                seeds[: min(12, len(seeds))],
                allow_short_fallback=True,
            )
            candidates.extend(extra)

        deduped = self._dedupe_and_rank(candidates, local_top_k)
        rebuilt = self.rebuild_highlight_openings(deduped, segments_norm)
        reranked = self._rerank_after_rebuild(rebuilt)
        final_candidates = self._dedupe_and_rank(reranked, local_top_k)
        return [item.to_dict() for item in final_candidates[:local_top_k]]

    # ============================================================
    # Segment preparation
    # ============================================================

    def _normalize_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for seg in segments:
            if not isinstance(seg, dict):
                continue

            start = seg.get("start")
            end = seg.get("end")
            text = str(seg.get("text", "")).strip()

            if start is None or end is None or not text:
                continue

            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                continue

            if end <= start:
                continue

            quality = seg.get("quality")
            if not isinstance(quality, dict):
                quality = self._analyze_text_quality(text)

            normalized.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "gap_before": float(seg.get("gap_before", 0.0) or 0.0),
                    "quality": quality,
                }
            )

        normalized.sort(key=lambda x: x["start"])
        return normalized

    def _prepare_segment_features(self, segments: List[Dict[str, Any]]) -> None:
        for seg in segments:
            text = self._normalize_text(seg["text"])
            duration = float(seg["end"]) - float(seg["start"])
            quality = seg.get("quality", {})

            intensity = 0.0

            intensity += min(text.count("?") * 0.7, 1.4)
            intensity += min(text.count("!") * 0.6, 1.2)

            strong_hits = sum(1 for p in self.DEFAULT_STRONG_PATTERNS if re.search(p, text))
            intensity += min(strong_hits * 0.45, 1.8)

            kw_hits = self._keyword_hits(text)
            sports_hits = self._sports_keyword_hits(text)

            intensity += min(kw_hits * 0.25, 1.0)
            intensity += min(sports_hits * 0.35, 1.2)

            token_count = len(self._tokenize(text))
            if 2 <= token_count <= 8 and self._contains_strong_pattern(text):
                intensity += 1.0

            gap_before = float(seg.get("gap_before", 0.0))
            if gap_before >= float(HIGHLIGHT_FILTER.get("min_silence_gap", 1.0)):
                intensity += float(SCORE_WEIGHTS.get("silence_before", 0.8))

            korean_ratio = float(quality.get("korean_ratio", 0.0))
            ascii_ratio = float(quality.get("ascii_ratio", 0.0))
            filler_ratio = float(quality.get("filler_ratio", 0.0))
            noise_ratio = float(quality.get("noise_ratio", 0.0))

            if korean_ratio >= float(HIGHLIGHT_FILTER.get("good_korean_ratio", 0.55)):
                intensity += 0.4
            if ascii_ratio > float(HIGHLIGHT_FILTER.get("max_ascii_ratio", 0.55)):
                intensity -= 0.8
            if filler_ratio > float(HIGHLIGHT_FILTER.get("filler_ratio_penalty_threshold", 0.35)):
                intensity -= 0.7
            if noise_ratio > float(HIGHLIGHT_FILTER.get("max_noise_ratio", 0.40)):
                intensity -= 0.8

            if duration < 1.0 and token_count <= 2:
                intensity -= 0.7

            seg["_clean_text"] = text
            seg["_duration"] = duration
            seg["_token_count"] = token_count
            seg["_seed_score"] = intensity

    def _apply_local_contrast_scores(self, segments: List[Dict[str, Any]]) -> None:
        seed_scores = [float(seg.get("_seed_score", 0.0)) for seg in segments]
        if not seed_scores:
            return

        for i, seg in enumerate(segments):
            left = max(0, i - 4)
            right = min(len(segments), i + 5)

            neighborhood = seed_scores[left:right]
            if not neighborhood:
                seg["_contrast_score"] = 0.0
                continue

            local_avg = sum(neighborhood) / len(neighborhood)
            current = float(seg.get("_seed_score", 0.0))
            contrast = current - local_avg
            seg["_contrast_score"] = max(-0.8, min(1.8, contrast))

    # ============================================================
    # Seed / candidate generation
    # ============================================================

    def _select_seed_indices(self, segments: List[Dict[str, Any]]) -> List[int]:
        scored: List[Tuple[int, float]] = []
        for i, seg in enumerate(segments):
            score = float(seg.get("_seed_score", 0.0)) + float(seg.get("_contrast_score", 0.0))
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        selected: List[int] = []
        min_seed_score = 1.2

        for idx, score in scored:
            if score < min_seed_score and len(selected) >= 12:
                break

            if selected:
                too_close = any(abs(idx - prev) <= 1 for prev in selected)
                if too_close:
                    continue

            selected.append(idx)
            if len(selected) >= min(20, max(10, self.top_k * 6)):
                break

        if not selected:
            selected = [idx for idx, _ in scored[: min(10, len(scored))]]

        return selected

    def _build_candidates_from_seeds(
        self,
        segments: List[Dict[str, Any]],
        seed_indices: List[int],
        allow_short_fallback: bool = False,
    ) -> List[HighlightCandidate]:
        candidates: List[HighlightCandidate] = []

        target_windows = [12, 16, 20, 24, 28, 32]
        if self.max_clip_sec > 34:
            target_windows.append(min(self.max_clip_sec, 36))

        for seed_idx in seed_indices:
            for target_sec in target_windows:
                refined = self._build_window_around_seed(segments, seed_idx, target_sec)
                if not refined:
                    continue

                duration = refined[-1]["end"] - refined[0]["start"]
                if duration > self.max_clip_sec + 1.5:
                    continue
                if duration < self.min_clip_sec and not allow_short_fallback:
                    continue
                if duration < self.min_fallback_clip_sec:
                    continue

                candidate = self._score_window(
                    included_segments=refined,
                    seed_idx=seed_idx,
                    allow_short_fallback=allow_short_fallback,
                )
                if candidate is not None:
                    candidates.append(candidate)

        return candidates

    def _build_window_around_seed(
        self,
        segments: List[Dict[str, Any]],
        seed_idx: int,
        target_sec: int,
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []

        s_idx = seed_idx
        e_idx = seed_idx

        while True:
            start = segments[s_idx]["start"]
            end = segments[e_idx]["end"]
            duration = end - start

            if duration >= target_sec:
                break

            left_gap_cost = math.inf
            right_gap_cost = math.inf
            left_gain = -math.inf
            right_gain = -math.inf

            if s_idx > 0:
                prev_seg = segments[s_idx - 1]
                left_gap_cost = max(0.0, segments[s_idx]["start"] - prev_seg["end"])
                left_gain = float(prev_seg.get("_seed_score", 0.0)) + float(prev_seg.get("_contrast_score", 0.0))

            if e_idx < len(segments) - 1:
                next_seg = segments[e_idx + 1]
                right_gap_cost = max(0.0, next_seg["start"] - segments[e_idx]["end"])
                right_gain = float(next_seg.get("_seed_score", 0.0)) + float(next_seg.get("_contrast_score", 0.0))

            if left_gap_cost > 1.8:
                left_gain -= 1.5
            if right_gap_cost > 1.8:
                right_gain -= 1.5

            if s_idx <= 0 and e_idx >= len(segments) - 1:
                break

            choose_left = False
            if s_idx > 0 and e_idx < len(segments) - 1:
                choose_left = left_gain >= right_gain
            elif s_idx > 0:
                choose_left = True

            if choose_left and s_idx > 0:
                s_idx -= 1
            elif e_idx < len(segments) - 1:
                e_idx += 1
            else:
                break

            if segments[e_idx]["end"] - segments[s_idx]["start"] > self.max_clip_sec + 3.0:
                break

        refined = self._refine_window_segments(segments, s_idx, e_idx)
        return refined

    def _refine_window_segments(
        self,
        segments: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []

        s_idx = max(0, start_idx)
        e_idx = min(len(segments) - 1, end_idx)

        for _ in range(2):
            if s_idx <= 0:
                break

            first_seg = segments[s_idx]
            if float(first_seg.get("gap_before", 0.0)) >= 1.0:
                break
            if self._looks_like_sentence_start(first_seg["text"]):
                break

            prev_seg = segments[s_idx - 1]
            if segments[e_idx]["end"] - prev_seg["start"] > self.max_clip_sec + 2.0:
                break

            s_idx -= 1

        for _ in range(2):
            if e_idx >= len(segments) - 1:
                break

            last_seg = segments[e_idx]
            if self._looks_like_sentence_end(last_seg["text"]):
                break

            next_seg = segments[e_idx + 1]
            gap_after = max(0.0, float(next_seg["start"]) - float(last_seg["end"]))
            if gap_after > 1.2:
                break

            if next_seg["end"] - segments[s_idx]["start"] > self.max_clip_sec + 2.0:
                break

            e_idx += 1

        refined = [dict(seg) for seg in segments[s_idx: e_idx + 1]]
        refined = self._trim_large_internal_gaps(refined)
        return refined

    def _trim_large_internal_gaps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(segments) <= 1:
            return segments

        best_chunk: List[Dict[str, Any]] = []
        current_chunk = [segments[0]]

        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]
            gap = max(0.0, float(curr_seg["start"]) - float(prev_seg["end"]))

            if gap > 1.8:
                if self._chunk_value(current_chunk) > self._chunk_value(best_chunk):
                    best_chunk = current_chunk
                current_chunk = [curr_seg]
            else:
                current_chunk.append(curr_seg)

        if self._chunk_value(current_chunk) > self._chunk_value(best_chunk):
            best_chunk = current_chunk

        return best_chunk if best_chunk else segments

    def _chunk_value(self, chunk: List[Dict[str, Any]]) -> float:
        if not chunk:
            return -9999.0

        text = self._join_segment_texts(chunk)
        duration = chunk[-1]["end"] - chunk[0]["start"]
        token_count = len(self._tokenize(text))
        strong = sum(1 for seg in chunk if self._contains_strong_pattern(seg["text"]))
        return duration * 0.3 + token_count * 0.2 + strong * 0.9

    # ============================================================
    # Candidate scoring
    # ============================================================

    def _score_window(
        self,
        included_segments: List[Dict[str, Any]],
        seed_idx: int,
        allow_short_fallback: bool,
    ) -> Optional[HighlightCandidate]:
        if not included_segments:
            return None

        start = included_segments[0]["start"]
        end = included_segments[-1]["end"]
        duration = end - start

        text = self._join_segment_texts(included_segments)
        clean_text = self._normalize_text(text)
        if not clean_text:
            return None

        window_quality = self._analyze_window_quality(included_segments, clean_text)
        if window_quality["window_korean_ratio"] < float(HIGHLIGHT_FILTER.get("min_window_korean_ratio", 0.25)):
            return None

        score_breakdown: Dict[str, float] = {}
        reasons: List[str] = []

        length_score = self._score_length(duration, allow_short_fallback)
        if length_score <= -999.0:
            return None
        score_breakdown["length"] = length_score
        if length_score > 2.5:
            reasons.append("길이 적합")
        elif length_score > 0:
            reasons.append("길이 보통")

        hook_score, hook_reasons = self._score_hook(included_segments)
        score_breakdown["hook"] = hook_score
        reasons.extend(hook_reasons)

        intensity_score, intensity_reasons = self._score_intensity(included_segments, clean_text)
        score_breakdown["intensity"] = intensity_score
        reasons.extend(intensity_reasons)

        dialog_score, dialog_reasons = self._score_dialog_activity(included_segments, clean_text)
        score_breakdown["dialog"] = dialog_score
        reasons.extend(dialog_reasons)

        boundary_score, boundary_reasons = self._score_boundaries(included_segments)
        score_breakdown["boundary"] = boundary_score
        reasons.extend(boundary_reasons)

        contrast_score = self._score_contrast(included_segments)
        score_breakdown["contrast"] = contrast_score
        if contrast_score >= 0.8:
            reasons.append("주변 대비 강함")

        quality_score, quality_reasons = self._score_quality(window_quality)
        score_breakdown["quality"] = quality_score
        reasons.extend(quality_reasons)

        keyword_score, keyword_reasons = self._score_keywords(clean_text)
        score_breakdown["keywords"] = keyword_score
        reasons.extend(keyword_reasons)

        semantic_score = self._score_semantic(clean_text)
        score_breakdown["semantic"] = semantic_score
        if semantic_score >= 0.5:
            reasons.append("의미적 임팩트")

        final_score = sum(score_breakdown.values())
        min_final_score = float(HIGHLIGHT_FILTER.get("min_final_score", 2.0))
        if final_score < min_final_score:
            return None

        return HighlightCandidate(
            start=start,
            end=end,
            score=final_score,
            text=clean_text,
            reasons=self._unique_keep_order(reasons),
            score_breakdown=score_breakdown,
            segments=included_segments,
            quality=window_quality,
        )

    def _score_length(self, duration: float, allow_short_fallback: bool) -> float:
        if self.min_clip_sec <= duration <= self.max_clip_sec:
            midpoint = (self.min_clip_sec + self.max_clip_sec) / 2.0
            distance = abs(duration - midpoint)
            return 3.0 + max(0.0, 1.2 - (distance / 10.0))

        if allow_short_fallback and self.min_fallback_clip_sec <= duration < self.min_clip_sec:
            distance = self.min_clip_sec - duration
            return max(0.4, 1.2 - distance * 0.15)

        return -9999.0

    def _score_hook(
        self,
        segments: List[Dict[str, Any]],
    ) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        hook_text = self._get_hook_text(segments)
        if not hook_text:
            return score, reasons

        token_count = len(self._tokenize(hook_text))
        pattern_hits = sum(1 for p in self.DEFAULT_HOOK_PATTERNS if re.search(p, hook_text))

        if pattern_hits > 0:
            score += min(pattern_hits * 0.55, 1.8)
            reasons.append("초반 훅 존재")

        if "?" in hook_text:
            score += 0.6
            reasons.append("초반 질문형")

        if "!" in hook_text:
            score += 0.4

        if 2 <= token_count <= 8 and self._contains_strong_pattern(hook_text):
            score += 0.8
            reasons.append("짧고 강한 시작")

        lowered = hook_text.lower()
        if any(lowered.startswith(prefix) for prefix in self.DEFAULT_WEAK_OPENERS):
            score -= 0.8
            reasons.append("초반 설명형")

        return score, reasons

    def _get_hook_text(self, segments: List[Dict[str, Any]]) -> str:
        if not segments:
            return ""

        collected: List[str] = []
        start_time = float(segments[0]["start"])

        for i, seg in enumerate(segments):
            text = self._normalize_text(seg.get("text", ""))
            if not text:
                continue

            collected.append(text)

            elapsed = float(seg["end"]) - start_time
            if i >= 1 or elapsed >= 3.2:
                break

        return self._normalize_text(" ".join(collected))

    def _score_intensity(
        self,
        segments: List[Dict[str, Any]],
        text: str,
    ) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        seed_scores = [float(seg.get("_seed_score", 0.0)) for seg in segments]
        avg_seed = sum(seed_scores) / max(len(seed_scores), 1)
        max_seed = max(seed_scores) if seed_scores else 0.0

        score += min(avg_seed * 0.9, 3.0)

        if max_seed >= 2.5:
            score += 0.9
            reasons.append("강한 한방 대사")

        q_count = text.count("?")
        ex_count = text.count("!")
        if q_count >= 1:
            score += min(q_count * 0.4, 0.8)
            reasons.append("질문/궁금증")
        if ex_count >= 1:
            score += min(ex_count * 0.35, 0.7)
            reasons.append("강한 반응")

        short_strong = sum(
            1 for seg in segments
            if 2 <= int(seg.get("_token_count", 0)) <= 8
            and self._contains_strong_pattern(seg.get("_clean_text", seg["text"]))
        )
        if short_strong >= 1:
            score += min(short_strong * 0.45, 1.2)
            reasons.append("짧은 임팩트 대사")

        return score, reasons

    def _score_dialog_activity(
        self,
        segments: List[Dict[str, Any]],
        text: str,
    ) -> Tuple[float, List[str]]:
        if len(segments) <= 1:
            single_bonus = 0.5 if self._contains_strong_pattern(text) else 0.0
            return single_bonus, ["단일 강한 발화"] if single_bonus > 0 else []

        score = 0.0
        reasons: List[str] = []

        durations = [float(s["end"]) - float(s["start"]) for s in segments]
        gaps = [
            max(0.0, float(segments[i]["start"]) - float(segments[i - 1]["end"]))
            for i in range(1, len(segments))
        ]

        avg_seg_dur = sum(durations) / max(len(durations), 1)
        avg_gap = sum(gaps) / max(len(gaps), 1) if gaps else 0.0

        short_turn_ratio = sum(1 for d in durations if d <= 2.4) / max(len(durations), 1)
        reaction_hits = sum(1 for p in self.DEFAULT_REACTION_PATTERNS if re.search(p, text))

        if avg_gap <= 0.7:
            score += 1.0
            reasons.append("대화 템포 빠름")
        elif avg_gap <= 1.1:
            score += 0.4

        if short_turn_ratio >= 0.45:
            score += 0.8
            reasons.append("짧은 주고받기")

        if 0.9 <= avg_seg_dur <= 4.5:
            score += 0.4

        if len(segments) >= 5:
            score += 0.6
            reasons.append("대사 턴 충분")

        if reaction_hits >= 2:
            score += 0.6
            reasons.append("리액션 존재")

        return score, reasons

    def _score_boundaries(self, segments: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        if not segments:
            return score, reasons

        first_text = segments[0]["text"]
        last_text = segments[-1]["text"]

        if self._looks_like_sentence_start(first_text):
            score += 1.0
            reasons.append("시작 자연스러움")
        else:
            score -= 0.8
            reasons.append("시작 어색함")

        if self._looks_like_sentence_end(last_text):
            score += 1.0
            reasons.append("끝맺음 자연스러움")
        else:
            score -= 0.4

        first_gap = float(segments[0].get("gap_before", 0.0))
        if first_gap >= float(HIGHLIGHT_FILTER.get("min_silence_gap", 1.0)):
            score += 0.5
            reasons.append("장면 시작 구분감")

        return score, reasons

    def _score_contrast(self, segments: List[Dict[str, Any]]) -> float:
        contrast_scores = [float(seg.get("_contrast_score", 0.0)) for seg in segments]
        if not contrast_scores:
            return 0.0

        positive = [x for x in contrast_scores if x > 0]
        if not positive:
            return min(sum(contrast_scores) / len(contrast_scores), 0.0)

        return min((sum(positive) / len(positive)) * 1.1, 1.8)

    def _score_quality(self, quality: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        korean_ratio = float(quality.get("window_korean_ratio", 0.0))
        ascii_ratio = float(quality.get("window_ascii_ratio", 0.0))
        noise_ratio = float(quality.get("window_noise_ratio", 0.0))
        filler_ratio = float(quality.get("window_filler_ratio", 0.0))
        repeat_ratio = float(quality.get("window_repeat_char_ratio", 0.0))

        if korean_ratio >= float(HIGHLIGHT_FILTER.get("good_korean_ratio", 0.55)):
            score += 0.9
            reasons.append("한글 비율 양호")
        elif korean_ratio < float(HIGHLIGHT_FILTER.get("min_korean_ratio", 0.30)):
            score += float(SCORE_WEIGHTS.get("bad_segment_penalty", -1.4))
            reasons.append("한글 비율 낮음")

        if ascii_ratio > float(HIGHLIGHT_FILTER.get("max_ascii_ratio", 0.55)):
            score += float(SCORE_WEIGHTS.get("english_penalty", -1.2))
            reasons.append("영문 비중 높음")

        if noise_ratio > float(HIGHLIGHT_FILTER.get("max_noise_ratio", 0.40)):
            score += float(SCORE_WEIGHTS.get("noise_penalty", -1.8))
            reasons.append("노이즈 높음")

        if filler_ratio > float(HIGHLIGHT_FILTER.get("filler_ratio_penalty_threshold", 0.35)):
            score += float(SCORE_WEIGHTS.get("filler_heavy_penalty", -1.0))
            reasons.append("필러 과다")

        if repeat_ratio > float(HIGHLIGHT_FILTER.get("max_repeat_char_ratio", 0.35)):
            score += float(SCORE_WEIGHTS.get("repeat_penalty", -1.0))
            reasons.append("반복 과다")

        return score, reasons

    def _score_keywords(self, text: str) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        kw_hits = self._keyword_hits(text)
        sports_hits = self._sports_keyword_hits(text)

        if kw_hits > 0:
            score += min(kw_hits * float(SCORE_WEIGHTS.get("keyword", 1.5)) * 0.28, 1.3)
            reasons.append("키워드 포함")

        if sports_hits > 0:
            score += min(sports_hits * float(SCORE_WEIGHTS.get("sports_keyword", 2.0)) * 0.3, 1.5)
            reasons.append("스포츠 핵심어")

        return score, reasons

    def _score_semantic(self, text: str) -> float:
        if not self._semantic_enabled or self._semantic_model is None or self._semantic_anchor_vectors is None:
            return 0.0

        try:
            query_vec = self._semantic_model.encode([text], normalize_embeddings=True)
            sims = (query_vec @ self._semantic_anchor_vectors.T)[0]
            best = float(max(sims))
            if best < 0.35:
                return 0.0
            return min((best - 0.35) * 2.2, float(SCORE_WEIGHTS.get("semantic", 1.5)))
        except Exception:
            return 0.0

    # ============================================================
    # Hook retrim / opening rebuild
    # ============================================================

    def refine_highlight_starts(
        self,
        candidates: List[HighlightCandidate],
        segments: List[Dict[str, Any]],
    ) -> List[HighlightCandidate]:
        return self.rebuild_highlight_openings(candidates, segments)

    def rebuild_highlight_openings(
        self,
        candidates: List[HighlightCandidate],
        segments: List[Dict[str, Any]],
    ) -> List[HighlightCandidate]:
        if not CLIP.get("enable_hook_retrim", False):
            return candidates

        results: List[HighlightCandidate] = []

        search_back_sec = max(float(CLIP.get("hook_search_back_sec", 6.0)), 5.0)
        search_forward_sec = max(float(CLIP.get("hook_search_sec", 18.0)), 15.0)
        pre_roll_sec = max(0.0, min(float(CLIP.get("hook_pre_roll_sec", 0.4)), 0.5))
        keep_min_duration_sec = float(CLIP.get("hook_keep_min_duration_sec", 10.0))
        max_extend_after_sec = float(CLIP.get("hook_max_extend_after_sec", 2.0))
        hook_min_improve = max(float(CLIP.get("hook_min_improve", 0.35)), 0.25)
        max_shift_sec = max(float(CLIP.get("hook_max_shift_sec", 8.0)), 6.0)
        debug_top_n = int(CLIP.get("hook_debug_top_n", 3))
        debug_enabled = bool(CLIP.get("hook_debug", False))

        for cand in candidates:
            original_start = float(cand.start)
            original_end = float(cand.end)
            original_duration = max(0.0, original_end - original_start)
            original_score = float(cand.score)
            original_reasons = list(cand.reasons)
            original_breakdown = dict(cand.score_breakdown)

            current_bundle = self._build_opening_bundle_by_time(
                segments=cand.segments,
                anchor_idx=0,
                max_bundle_sec=3.4,
            )
            current_eval = self._score_opening_bundle(
                bundle=current_bundle,
                original_start=original_start,
                anchor_start=original_start,
                max_shift_sec=max_shift_sec,
                relaxed=False,
            )

            strict_valid, strict_rejected = self._collect_opening_candidates(
                segments=segments,
                original_start=original_start,
                original_end=original_end,
                search_back_sec=search_back_sec,
                search_forward_sec=search_forward_sec,
                max_extend_after_sec=max_extend_after_sec,
                max_shift_sec=max_shift_sec,
                relaxed=False,
            )

            valid_candidates = strict_valid
            rejected_candidates = strict_rejected
            rescue_used = False

            if current_eval.get("is_explain", False) or not valid_candidates:
                rescue_valid, rescue_rejected = self._collect_opening_candidates(
                    segments=segments,
                    original_start=original_start,
                    original_end=original_end,
                    search_back_sec=max(search_back_sec, 8.0),
                    search_forward_sec=max(search_forward_sec, 22.0),
                    max_extend_after_sec=max_extend_after_sec + 1.0,
                    max_shift_sec=max(max_shift_sec + 4.0, 10.0),
                    relaxed=True,
                )
                if rescue_valid:
                    valid_candidates = rescue_valid
                    rescue_used = True
                rejected_candidates.extend(rescue_rejected)

            if debug_enabled:
                self._print_hook_debug(
                    current_eval=current_eval,
                    opening_candidates=valid_candidates,
                    rejected_candidates=rejected_candidates,
                    top_n=debug_top_n,
                    original_start=original_start,
                    rescue_used=rescue_used,
                )

            if not valid_candidates:
                if current_eval.get("is_explain", False):
                    cand.reasons = self._unique_keep_order(
                        cand.reasons + ["설명형 시작 유지됨(대체 후보 없음)"]
                    )
                    cand.score_breakdown["opening_penalty"] = round(
                        float(cand.score_breakdown.get("opening_penalty", 0.0)) - 1.25, 4
                    )
                    cand.score = max(0.0, cand.score - 1.25)
                if current_eval.get("reject_reason") == "no_hook_signal":
                    cand.score_breakdown["opening_penalty"] = round(
                        float(cand.score_breakdown.get("opening_penalty", 0.0)) - 1.5, 4
                    )
                    cand.score = max(0.0, cand.score - 1.5)
                    cand.reasons = self._unique_keep_order(cand.reasons + ["최종 후보 패널티: no_hook_signal"])
                results.append(cand)
                continue

            best_eval = valid_candidates[0]

            if not self._is_better_opening_than_original(
                best_eval=best_eval,
                current_eval=current_eval,
                improve_threshold=hook_min_improve,
                rescue_used=rescue_used,
            ):
                if current_eval.get("is_explain", False):
                    cand.reasons = self._unique_keep_order(
                        cand.reasons + ["설명형 시작 잔류(교체 기준 미달)"]
                    )
                    cand.score_breakdown["opening_penalty"] = round(
                        float(cand.score_breakdown.get("opening_penalty", 0.0)) - 1.0, 4
                    )
                    cand.score = max(0.0, cand.score - 1.0)
                if current_eval.get("reject_reason") == "no_hook_signal":
                    cand.score_breakdown["opening_penalty"] = round(
                        float(cand.score_breakdown.get("opening_penalty", 0.0)) - 1.2, 4
                    )
                    cand.score = max(0.0, cand.score - 1.2)
                    cand.reasons = self._unique_keep_order(cand.reasons + ["최종 후보 패널티: no_hook_signal"])
                results.append(cand)
                continue

            applied = self._apply_rebuilt_opening(
                cand=cand,
                all_segments=segments,
                best_eval=best_eval,
                original_start=original_start,
                original_end=original_end,
                original_duration=original_duration,
                keep_min_duration_sec=keep_min_duration_sec,
                max_extend_after_sec=max_extend_after_sec,
                pre_roll_sec=pre_roll_sec,
                original_score=original_score,
                original_reasons=original_reasons,
                original_breakdown=original_breakdown,
                current_eval=current_eval,
                rescue_used=rescue_used,
            )

            if applied is None:
                if current_eval.get("is_explain", False):
                    cand.reasons = self._unique_keep_order(
                        cand.reasons + ["설명형 시작 잔류(적용 실패)"]
                    )
                    cand.score_breakdown["opening_penalty"] = round(
                        float(cand.score_breakdown.get("opening_penalty", 0.0)) - 0.8, 4
                    )
                    cand.score = max(0.0, cand.score - 0.8)
                results.append(cand)
                continue

            results.append(applied)

        return results

    def _collect_opening_candidates(
        self,
        segments: List[Dict[str, Any]],
        original_start: float,
        original_end: float,
        search_back_sec: float,
        search_forward_sec: float,
        max_extend_after_sec: float,
        max_shift_sec: float,
        relaxed: bool,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        search_start = max(0.0, original_start - search_back_sec)
        search_end = min(original_end + max_extend_after_sec, original_start + search_forward_sec)

        valid_candidates: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []

        for idx, seg in enumerate(segments):
            seg_start = float(seg["start"])
            if seg_start < search_start or seg_start > search_end:
                continue

            bundle = self._build_opening_bundle_by_time(
                segments=segments,
                anchor_idx=idx,
                max_bundle_sec=3.8 if relaxed else 3.4,
            )
            if not bundle:
                continue

            anchor_start = float(bundle[0]["start"])

            eval_result = self._score_opening_bundle(
                bundle=bundle,
                original_start=original_start,
                anchor_start=anchor_start,
                max_shift_sec=max_shift_sec,
                relaxed=relaxed,
            )

            if eval_result.get("valid", False):
                valid_candidates.append(eval_result)
            else:
                rejected_candidates.append(eval_result)

        valid_candidates.sort(key=lambda x: x["score"], reverse=True)
        rejected_candidates.sort(
            key=lambda x: (
                1 if x.get("is_explain", False) else 0,
                float(x.get("distance", 999.0)),
                -float(x.get("raw_score", -999.0)),
            )
        )

        return valid_candidates, rejected_candidates

    def _build_opening_bundle_by_time(
        self,
        segments: List[Dict[str, Any]],
        anchor_idx: int,
        max_bundle_sec: float = 3.4,
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []

        if anchor_idx < 0 or anchor_idx >= len(segments):
            return []

        bundle: List[Dict[str, Any]] = []
        anchor_seg = segments[anchor_idx]
        anchor_start = float(anchor_seg["start"])

        for idx in range(anchor_idx, len(segments)):
            seg = segments[idx]
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])

            if bundle:
                prev_end = float(bundle[-1]["end"])
                if max(0.0, seg_start - prev_end) > 1.1:
                    break

            if seg_start - anchor_start > max_bundle_sec:
                break

            bundle.append(seg)

            elapsed = seg_end - anchor_start

            if elapsed >= max_bundle_sec:
                break

            if len(bundle) >= 4:
                break

            if elapsed >= 2.6 and len(bundle) >= 2:
                break

        return bundle

    def _score_opening_bundle(
        self,
        bundle: List[Dict[str, Any]],
        original_start: float,
        anchor_start: float,
        max_shift_sec: float,
        relaxed: bool = False,
    ) -> Dict[str, Any]:
        invalid_result = {
            "valid": False,
            "score": -999.0,
            "raw_score": -999.0,
            "anchor_start": anchor_start,
            "bundle": bundle,
            "text": "",
            "first_line": "",
            "is_explain": False,
            "breakdown": {},
            "distance": abs(anchor_start - original_start),
            "reject_reason": "unknown",
            "relaxed": relaxed,
        }

        if not bundle:
            result = dict(invalid_result)
            result["reject_reason"] = "empty_bundle"
            return result

        text = self._join_segment_texts(bundle)
        if not text:
            result = dict(invalid_result)
            result["reject_reason"] = "empty_text"
            return result

        first_line = self._normalize_text(str(bundle[0].get("text", "")))
        if not first_line:
            result = dict(invalid_result)
            result["text"] = text
            result["reject_reason"] = "empty_first_line"
            return result

        token_count = len(self._tokenize(first_line))
        total_tokens = len(self._tokenize(text))
        quality = self._analyze_text_quality(text)
        distance = abs(anchor_start - original_start)

        result_base = dict(invalid_result)
        result_base["text"] = text
        result_base["first_line"] = first_line
        result_base["distance"] = distance

        if distance > max_shift_sec:
            result_base["reject_reason"] = "shift_too_far"
            return result_base

        is_explain = self._looks_like_explain_opening(first_line, text)
        if is_explain and not relaxed:
            result_base["is_explain"] = True
            result_base["reject_reason"] = "explain_opening"
            return result_base

        max_first_tokens = 12 if relaxed else 8
        if token_count == 0 or token_count > max_first_tokens:
            result_base["reject_reason"] = "first_line_too_long"
            return result_base

        question_count = first_line.count("?") + text.count("?")
        exclaim_count = first_line.count("!") + text.count("!")
        hook_hits = sum(1 for p in self.DEFAULT_HOOK_PATTERNS if re.search(p, text))
        strong_hits = sum(1 for p in self.DEFAULT_STRONG_PATTERNS if re.search(p, text))
        reaction_hits = sum(1 for p in self.DEFAULT_REACTION_PATTERNS if re.search(p, text))

        has_strong_hook = (
            question_count > 0
            or exclaim_count > 0
            or strong_hits >= 2
            or hook_hits >= 1
            or reaction_hits >= 1
            or (token_count <= 6 and self._contains_strong_pattern(first_line))
            or (relaxed and self._contains_strong_pattern(text))
        )
        if not has_strong_hook:
            result_base["reject_reason"] = "no_hook_signal"
            return result_base

        bundle_start = float(bundle[0]["start"])
        bundle_end = float(bundle[-1]["end"])
        bundle_duration = max(0.0, bundle_end - bundle_start)

        first_seg_duration = float(bundle[0]["end"]) - float(bundle[0]["start"])
        first_two_sec_text = self._collect_text_within_seconds(bundle, seconds=2.0)
        first_two_sec_tokens = len(self._tokenize(first_two_sec_text))
        first_two_sec_hook_hits = sum(1 for p in self.DEFAULT_HOOK_PATTERNS if re.search(p, first_two_sec_text))
        first_two_sec_reaction_hits = sum(1 for p in self.DEFAULT_REACTION_PATTERNS if re.search(p, first_two_sec_text))

        breakdown: Dict[str, float] = {}

        if token_count <= 2:
            breakdown["short_first_line"] = 3.0
        elif token_count <= 4:
            breakdown["short_first_line"] = 2.5
        elif token_count <= 5:
            breakdown["short_first_line"] = 1.8
        elif token_count <= 6:
            breakdown["short_first_line"] = 1.0
        elif relaxed and token_count <= 10:
            breakdown["short_first_line"] = 0.2
        else:
            breakdown["short_first_line"] = -0.8

        breakdown["question"] = min(question_count * 1.8, 3.0)
        breakdown["exclaim"] = min(exclaim_count * 1.5, 2.5)
        breakdown["hook_hits"] = min(hook_hits * 0.6, 1.8)
        breakdown["strong_hits"] = min(strong_hits * 0.8, 2.4)
        breakdown["reaction_hits"] = min(reaction_hits * 0.5, 1.5)
        breakdown["first_two_sec_focus"] = min((first_two_sec_hook_hits + first_two_sec_reaction_hits) * 0.6, 2.0)

        if bundle_duration <= 3.6:
            breakdown["bundle_duration_fit"] = 1.2
        elif bundle_duration <= 4.2:
            breakdown["bundle_duration_fit"] = 0.4
        else:
            breakdown["bundle_duration_fit"] = -0.8

        if first_seg_duration <= 2.8:
            breakdown["quick_first_turn"] = 0.8
        else:
            breakdown["quick_first_turn"] = -0.3

        if 1 <= len(bundle) <= 2:
            breakdown["tight_bundle"] = 0.9
        elif len(bundle) == 3:
            breakdown["tight_bundle"] = 0.3
        else:
            breakdown["tight_bundle"] = -0.4

        if first_two_sec_tokens <= 10:
            breakdown["compact_first_two_sec"] = 0.9
        elif first_two_sec_tokens <= 14:
            breakdown["compact_first_two_sec"] = 0.2
        else:
            breakdown["compact_first_two_sec"] = -0.7

        if self._looks_like_sentence_start(first_line):
            breakdown["sentence_start"] = 0.6
        else:
            breakdown["sentence_start"] = -0.9

        if float(quality.get("korean_ratio", 0.0)) >= 0.55:
            breakdown["korean_ratio"] = 0.5
        else:
            breakdown["korean_ratio"] = -0.3

        if float(quality.get("ascii_ratio", 0.0)) > 0.55:
            breakdown["ascii_penalty"] = -1.0
        else:
            breakdown["ascii_penalty"] = 0.0

        if float(quality.get("filler_ratio", 0.0)) > 0.35:
            breakdown["filler_penalty"] = -1.2
        else:
            breakdown["filler_penalty"] = 0.0

        if float(quality.get("noise_ratio", 0.0)) > 0.40:
            breakdown["noise_penalty"] = -1.2
        else:
            breakdown["noise_penalty"] = 0.0

        if total_tokens >= 15:
            breakdown["narrative_penalty"] = -1.0
        elif total_tokens >= 12:
            breakdown["narrative_penalty"] = -0.5
        else:
            breakdown["narrative_penalty"] = 0.0

        if is_explain and relaxed:
            breakdown["explain_penalty_relaxed"] = -1.2

        if relaxed:
            breakdown["rescue_bonus"] = 0.4

        raw_score = sum(breakdown.values())
        score = max(0.0, raw_score) * 0.8

        min_required = 2.2 if relaxed else 2.8
        if score < min_required:
            result = dict(result_base)
            result["reject_reason"] = "score_too_low"
            result["breakdown"] = breakdown
            result["raw_score"] = raw_score
            result["score"] = score
            result["is_explain"] = is_explain
            return result

        return {
            "valid": True,
            "score": score,
            "raw_score": raw_score,
            "anchor_start": anchor_start,
            "bundle": bundle,
            "text": text,
            "first_line": first_line,
            "is_explain": is_explain,
            "breakdown": breakdown,
            "distance": distance,
            "bundle_duration": bundle_duration,
            "token_count": token_count,
            "total_tokens": total_tokens,
            "reject_reason": "",
            "relaxed": relaxed,
        }

    def _is_better_opening_than_original(
        self,
        best_eval: Dict[str, Any],
        current_eval: Dict[str, Any],
        improve_threshold: float,
        rescue_used: bool = False,
    ) -> bool:
        if not best_eval.get("valid", False):
            return False

        best_score = float(best_eval.get("score", -999.0))
        current_score = float(current_eval.get("score", -999.0))
        current_explain = bool(current_eval.get("is_explain", False))
        best_relaxed = bool(best_eval.get("relaxed", False))

        if best_score < 0.0:
            return False

        if current_explain and rescue_used:
            return True

        if current_explain and best_score > 0.0:
            return True

        if current_score < 0.0:
            return best_score >= improve_threshold

        best_first = str(best_eval.get("first_line", ""))
        current_first = str(current_eval.get("first_line", ""))
        best_tokens = len(self._tokenize(best_first))
        current_tokens = len(self._tokenize(current_first))

        best_has_signal = self._contains_strong_pattern(best_first) or ("?" in best_first) or ("!" in best_first)
        current_has_signal = self._contains_strong_pattern(current_first) or ("?" in current_first) or ("!" in current_first)

        effective_threshold = improve_threshold
        if rescue_used or best_relaxed:
            effective_threshold = max(0.05, improve_threshold - 0.25)

        if (best_score - current_score) >= effective_threshold:
            return True

        if best_has_signal and not current_has_signal and (best_score - current_score) >= max(0.0, effective_threshold - 0.15):
            return True

        if best_tokens <= current_tokens and best_has_signal and (best_score - current_score) >= max(0.0, effective_threshold - 0.10):
            return True

        return False

    def _apply_rebuilt_opening(
        self,
        cand: HighlightCandidate,
        all_segments: List[Dict[str, Any]],
        best_eval: Dict[str, Any],
        original_start: float,
        original_end: float,
        original_duration: float,
        keep_min_duration_sec: float,
        max_extend_after_sec: float,
        pre_roll_sec: float,
        original_score: float,
        original_reasons: List[str],
        original_breakdown: Dict[str, float],
        current_eval: Dict[str, Any],
        rescue_used: bool,
    ) -> Optional[HighlightCandidate]:
        if not best_eval.get("valid", False):
            return None

        bundle = best_eval.get("bundle")
        if not isinstance(bundle, list) or not bundle:
            return None

        old_first_line = ""
        if cand.segments:
            old_first_line = self._normalize_text(str(cand.segments[0].get("text", "")))

        opening_bundle_start = float(bundle[0]["start"])
        new_start = max(0.0, opening_bundle_start - pre_roll_sec)

        new_end = max(original_end, new_start + original_duration)
        new_end = min(new_end, original_end + max_extend_after_sec)

        if (new_end - new_start) < keep_min_duration_sec:
            new_end = min(new_start + keep_min_duration_sec, original_end + max_extend_after_sec)

        if (new_end - new_start) > (self.max_clip_sec + 2.0):
            new_end = new_start + self.max_clip_sec + 2.0

        if new_end <= new_start:
            return None

        eligible = [seg for seg in all_segments if float(seg["start"]) >= new_start and float(seg["start"]) <= new_end]
        if not eligible:
            eligible = [seg for seg in all_segments if float(seg["end"]) > new_start and float(seg["start"]) <= new_end]

        if not eligible:
            return None

        first_idx = None
        first_seg = eligible[0]
        for i, seg in enumerate(all_segments):
            if seg is first_seg:
                first_idx = i
                break

        if first_idx is None:
            return None

        new_segments: List[Dict[str, Any]] = []
        for seg in all_segments[first_idx:]:
            seg_start = float(seg["start"])
            if seg_start > new_end:
                break
            new_segments.append(seg)

        if not new_segments:
            return None

        new_opening = self._build_opening_bundle_by_time(
            segments=new_segments,
            anchor_idx=0,
            max_bundle_sec=3.8 if rescue_used else 3.4,
        )
        new_opening_eval = self._score_opening_bundle(
            bundle=new_opening,
            original_start=original_start,
            anchor_start=float(new_opening[0]["start"]) if new_opening else new_start,
            max_shift_sec=max(float(CLIP.get("hook_max_shift_sec", 8.0)), 10.0 if rescue_used else 6.0),
            relaxed=rescue_used,
        )
        if not new_opening_eval.get("valid", False):
            return None

        new_text = self._join_segment_texts(new_segments)
        if not new_text:
            return None

        cand.start = new_start
        cand.end = new_end
        cand.text = new_text
        cand.segments = new_segments
        cand.quality = self._analyze_window_quality(new_segments, new_text)
        cand.score_breakdown = dict(original_breakdown)
        cand.score_breakdown["hook_retrim"] = round(float(best_eval.get("score", 0.0)), 4)

        current_score = max(0.0, float(current_eval.get("score", 0.0)))
        best_score = max(0.0, float(best_eval.get("score", 0.0)))
        improve_delta = max(0.0, best_score - current_score)
        cand.score = original_score + min(2.4, improve_delta * 0.45 + (0.25 if rescue_used else 0.0))

        new_first_line = self._normalize_text(str(best_eval.get("first_line", "")))
        rebuilt_reason = f"오프닝 재구성: {old_first_line[:24]} -> {new_first_line[:24]}"

        extra_reasons = ["강한 훅 시작 선택", "설명형 시작 제거", rebuilt_reason]
        if rescue_used:
            extra_reasons.append("오프닝 구조 rescue 적용")

        cand.reasons = self._unique_keep_order(original_reasons + extra_reasons)

        return cand

    def _rerank_after_rebuild(self, candidates: List[HighlightCandidate]) -> List[HighlightCandidate]:
        """
        델타 가산 방식이 아니라 강제 재채점 + 하드 필터 방식으로 최종 랭킹을 다시 만든다.
        """
        reranked: List[HighlightCandidate] = []

        for cand in candidates:
            first_line = ""
            if cand.segments:
                first_line = self._normalize_text(str(cand.segments[0].get("text", "")))

            hook_retrim_score = float(cand.score_breakdown.get("hook_retrim", 0.0))
            opening_penalty = float(cand.score_breakdown.get("opening_penalty", 0.0))
            base_score = float(cand.score)

            opening_is_explain = self._looks_like_explain_opening(first_line, first_line)
            opening_has_signal = self._contains_strong_pattern(first_line) or ("?" in first_line) or ("!" in first_line)
            hook_component = float(cand.score_breakdown.get("hook", 0.0))
            intensity_component = float(cand.score_breakdown.get("intensity", 0.0))
            boundary_component = float(cand.score_breakdown.get("boundary", 0.0))

            final_score = base_score

            if hook_retrim_score > 0.0:
                final_score += min(3.0, 0.8 + hook_retrim_score * 0.35)

            if any(reason == "오프닝 구조 rescue 적용" for reason in cand.reasons):
                final_score += 0.9

            if any(reason == "강한 훅 시작 선택" for reason in cand.reasons):
                final_score += 0.8

            final_score += opening_penalty

            if opening_is_explain:
                final_score -= 3.0
                cand.reasons = self._unique_keep_order(cand.reasons + ["최종 랭킹 패널티: 설명형 시작"])

            if not opening_has_signal and hook_component < 0.8:
                final_score -= 2.7
                cand.reasons = self._unique_keep_order(cand.reasons + ["최종 랭킹 패널티: 훅 약함"])

            if "최종 후보 패널티: no_hook_signal" in cand.reasons:
                final_score -= 2.5

            if intensity_component < 1.2 and hook_component < 0.8:
                final_score -= 1.2

            if boundary_component < 0.2:
                final_score -= 0.6

            hard_fail = False
            if opening_is_explain and hook_retrim_score <= 0.0:
                hard_fail = True
            if "최종 후보 패널티: no_hook_signal" in cand.reasons and hook_retrim_score <= 0.0:
                hard_fail = True

            if hard_fail:
                final_score -= 4.0
                cand.reasons = self._unique_keep_order(cand.reasons + ["최종 하드 필터 적용"])

            cand.score_breakdown["final_rerank_score"] = round(final_score, 4)
            cand.score = max(0.0, final_score)

            reranked.append(cand)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    def _collect_text_within_seconds(self, segments: List[Dict[str, Any]], seconds: float) -> str:
        if not segments:
            return ""

        start_time = float(segments[0]["start"])
        collected: List[str] = []

        for seg in segments:
            seg_start = float(seg["start"])
            if seg_start - start_time > seconds:
                break
            text = self._normalize_text(seg.get("text", ""))
            if text:
                collected.append(text)

        return self._normalize_text(" ".join(collected))

    def _looks_like_explain_opening(self, first_line: str, full_text: str) -> bool:
        first = self._normalize_text(first_line).lower()
        full = self._normalize_text(full_text).lower()

        if not first:
            return True

        if any(first.startswith(prefix) for prefix in self.DEFAULT_EXPLAIN_PREFIXES):
            return True

        if any(fragment in first for fragment in self.DEFAULT_EXPLAIN_SUBSTRINGS):
            return True

        if any(fragment in full for fragment in self.DEFAULT_EXPLAIN_SUBSTRINGS):
            return True

        if re.search(r"(있어가지고|있어서|하려고|해가지고|말씀드리|설명드리|생각합니다|같습니다)", first):
            return True

        if re.search(r"^(나|저|우리|제가|이거|그거|그 사람들)\s+지금", first):
            return True

        if re.search(r"^(근데|그리고|그래서|그러니까)\b", first):
            return True

        if re.search(r"^(저번에|진짜 웃긴|결국|사실은|그게 뭐냐면|어떻게 됐냐면)", first):
            return True

        if re.search(r"(뭐냐면|됐냐면|하더라고|하니까|봤더니|보니까|라고 하더라|라고 해서)", first):
            return True

        if len(self._tokenize(first)) >= 7 and not self._contains_strong_pattern(first):
            return True

        return False

    def _print_hook_debug(
        self,
        current_eval: Dict[str, Any],
        opening_candidates: List[Dict[str, Any]],
        rejected_candidates: List[Dict[str, Any]],
        top_n: int,
        original_start: float,
        rescue_used: bool,
    ) -> None:
        print("\n[HOOK DEBUG]")
        print(
            f"  current_start={original_start:.2f} "
            f"current_score={float(current_eval.get('score', -999.0)):.3f} "
            f"current_text={str(current_eval.get('first_line', ''))} "
            f"current_reject={str(current_eval.get('reject_reason', ''))} "
            f"rescue_used={rescue_used}"
        )

        valid_top = opening_candidates[:max(1, top_n)]
        rejected_top = rejected_candidates[:max(1, top_n)]

        if valid_top:
            for rank, item in enumerate(valid_top, start=1):
                print(
                    f"  cand#{rank} "
                    f"start={float(item.get('anchor_start', 0.0)):.2f} "
                    f"score={float(item.get('score', -999.0)):.3f} "
                    f"relaxed={bool(item.get('relaxed', False))} "
                    f"text={str(item.get('first_line', ''))}"
                )
        else:
            print("  cand#none valid candidate 없음")

        if rejected_top:
            for rank, item in enumerate(rejected_top, start=1):
                print(
                    f"  reject#{rank} "
                    f"start={float(item.get('anchor_start', 0.0)):.2f} "
                    f"reason={str(item.get('reject_reason', ''))} "
                    f"raw={float(item.get('raw_score', -999.0)):.3f} "
                    f"relaxed={bool(item.get('relaxed', False))} "
                    f"text={str(item.get('first_line', ''))}"
                )

    # ============================================================
    # Dedupe / ranking
    # ============================================================

    def _dedupe_and_rank(
        self,
        candidates: List[HighlightCandidate],
        top_k: int,
    ) -> List[HighlightCandidate]:
        if not candidates:
            return []

        candidates_sorted = sorted(candidates, key=lambda x: x.score, reverse=True)

        deduped: List[HighlightCandidate] = []

        for cand in candidates_sorted:
            keep = True
            for chosen in deduped:
                if self._is_near_duplicate(cand, chosen):
                    keep = False
                    break
            if keep:
                deduped.append(cand)
            if len(deduped) >= max(top_k * 3, 8):
                break

        if len(deduped) < top_k:
            for cand in candidates_sorted:
                if cand in deduped:
                    continue
                deduped.append(cand)
                if len(deduped) >= top_k:
                    break

        return deduped

    def _is_near_duplicate(self, a: HighlightCandidate, b: HighlightCandidate) -> bool:
        overlap = self._time_overlap_ratio(a.start, a.end, b.start, b.end)
        text_sim = self._jaccard_similarity(a.text, b.text)

        if overlap >= 0.75:
            return True
        if overlap >= 0.55 and text_sim >= 0.45:
            return True
        if text_sim >= 0.78:
            return True

        return False

    def _time_overlap_ratio(self, a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
        union = max(a_end, b_end) - min(a_start, b_start)
        if union <= 0:
            return 0.0
        return inter / union

    def _jaccard_similarity(self, a: str, b: str) -> float:
        a_set = set(self._tokenize(a))
        b_set = set(self._tokenize(b))
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    # ============================================================
    # Text / language heuristics
    # ============================================================

    def _normalize_text(self, text: str) -> str:
        text = str(text or "").replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _join_segment_texts(self, segments: List[Dict[str, Any]]) -> str:
        return self._normalize_text(
            " ".join(str(seg.get("text", "")).strip() for seg in segments if seg.get("text"))
        )

    def _tokenize(self, text: str) -> List[str]:
        return [tok for tok in re.split(r"\s+", str(text).strip()) if tok]

    def _contains_strong_pattern(self, text: str) -> bool:
        text = str(text or "")
        return any(re.search(pattern, text) for pattern in self.DEFAULT_STRONG_PATTERNS)

    def _looks_like_sentence_start(self, text: str) -> bool:
        text = self._normalize_text(text)
        if not text:
            return False

        lower = text.lower()
        for bad in self.DEFAULT_NEGATIVE_START_WORDS:
            if lower.startswith(bad):
                return False

        if re.match(r"^[\"'“‘]?[가-힣A-Za-z0-9]", text):
            return True

        return False

    def _looks_like_sentence_end(self, text: str) -> bool:
        text = self._normalize_text(text)
        if not text:
            return False

        if text.endswith(("?", "!", ".", "…")):
            return True

        return any(text.endswith(end) for end in self.DEFAULT_ENDINGS_GOOD)

    def _keyword_hits(self, text: str) -> int:
        hits = 0
        for category, words in self.keyword_map.items():
            if category == "sports_baseball":
                continue
            for word in words:
                if word and word in text:
                    hits += 1
        return hits

    def _sports_keyword_hits(self, text: str) -> int:
        hits = 0
        for word in self.keyword_map.get("sports_baseball", []):
            if word and word in text:
                hits += 1
        return hits

    def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
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
        repeat_char_count = sum(len(m.group(0)) for m in re.finditer(r"(.)\1{2,}", text))

        tokens = self._tokenize(text)
        filler_count = sum(1 for tok in tokens if tok in self.fillers)
        noise_count = sum(1 for tok in tokens if self._is_noise_token(tok))

        return {
            "ascii_ratio": ascii_chars / max(total_chars, 1),
            "korean_ratio": korean_chars / max(total_chars, 1),
            "repeat_char_ratio": repeat_char_count / max(total_chars, 1),
            "noise_ratio": noise_count / max(len(tokens), 1) if tokens else 0.0,
            "filler_ratio": filler_count / max(len(tokens), 1) if tokens else 0.0,
            "token_count": len(tokens),
        }

    def _analyze_window_quality(
        self,
        segments: List[Dict[str, Any]],
        clean_text: str,
    ) -> Dict[str, Any]:
        base = self._analyze_text_quality(clean_text)

        qualities = [seg.get("quality", {}) for seg in segments if isinstance(seg.get("quality"), dict)]
        if not qualities:
            return {
                "window_korean_ratio": base["korean_ratio"],
                "window_ascii_ratio": base["ascii_ratio"],
                "window_repeat_char_ratio": base["repeat_char_ratio"],
                "window_noise_ratio": base["noise_ratio"],
                "window_filler_ratio": base["filler_ratio"],
                "window_token_count": base["token_count"],
            }

        def avg(key: str) -> float:
            vals = [float(q.get(key, 0.0)) for q in qualities]
            return sum(vals) / max(len(vals), 1)

        return {
            "window_korean_ratio": max(base["korean_ratio"], avg("korean_ratio")),
            "window_ascii_ratio": min(base["ascii_ratio"], max(base["ascii_ratio"], avg("ascii_ratio"))),
            "window_repeat_char_ratio": max(base["repeat_char_ratio"], avg("repeat_char_ratio")),
            "window_noise_ratio": max(base["noise_ratio"], avg("noise_ratio")),
            "window_filler_ratio": max(base["filler_ratio"], avg("filler_ratio")),
            "window_token_count": base["token_count"],
        }

    def _is_noise_token(self, token: str) -> bool:
        token = str(token or "").strip()
        if not token:
            return True

        if re.fullmatch(r"[A-Za-z]{1,2}", token):
            return True

        if re.fullmatch(r"[^\w가-힣]+", token):
            return True

        if re.fullmatch(r"(.)\1{2,}", token):
            return True

        if len(token) <= 1 and not re.search(r"[가-힣]", token):
            return True

        return False

    def _unique_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result