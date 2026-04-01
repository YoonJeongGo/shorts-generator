# ui/app_state.py

from __future__ import annotations

from copy import deepcopy
from typing import Any

from PySide6.QtCore import QObject, Signal


class AppState(QObject):
    """
    앱 전체의 단일 진실 소스.
    MainWindow만 이 상태를 갱신하고,
    각 위젯은 이 상태를 기준으로 refresh 된다.
    """

    state_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._clip_seq = 0
        self._multicut_seq = 0
        self.reset()

    def reset(self) -> None:
        self.video_path: str | None = None
        self.segments: list[dict[str, Any]] = []
        self.highlights: list[dict[str, Any]] = []
        self.timeline_clips: list[dict[str, Any]] = []
        self.multicut_clips: list[dict[str, Any]] = []
        self.current_position_ms: int = 0
        self.in_point_ms: int = 0
        self.out_point_ms: int = 0
        self.duration_ms: int = 0
        self.crop_params: dict[str, Any] = {
            "enabled": True,
            "mode": "auto",
            "y_offset": 0.0,
            "x_offset": 0.0,
            "font_size": 36,
        }
        self.state_changed.emit("reset")

    def set_video_path(self, path: str | None) -> None:
        self.video_path = path
        self.state_changed.emit("video_path")

    def set_duration_ms(self, ms: int) -> None:
        self.duration_ms = max(0, int(ms))
        self.state_changed.emit("duration_ms")

    def set_segments(self, segments: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = str(seg.get("text", "")).strip()
            if end <= start:
                continue
            normalized.append({
                "start": start,
                "end": end,
                "text": text,
            })
        self.segments = normalized
        self.state_changed.emit("segments")

    def set_highlights(self, highlights: list[dict[str, Any]]) -> None:
        self.highlights = [self._normalize_clip(c) for c in highlights]
        self.timeline_clips = deepcopy(self.highlights)
        self.state_changed.emit("highlights")

    def add_manual_clip(self, clip: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_clip({
            **clip,
            "reasons": clip.get("reasons", ["수동 추가"]),
        })
        self.highlights.append(deepcopy(normalized))
        self.timeline_clips.append(deepcopy(normalized))
        self.state_changed.emit("manual_clip_added")
        return deepcopy(normalized)

    def update_clip_range_by_index(self, index: int, start_sec: float, end_sec: float) -> None:
        if not (0 <= index < len(self.timeline_clips)):
            return

        start_sec = max(0.0, float(start_sec))
        end_sec = max(start_sec, float(end_sec))
        clip_id = self.timeline_clips[index].get("clip_id")

        for collection in (self.timeline_clips, self.highlights):
            for clip in collection:
                if clip.get("clip_id") == clip_id:
                    clip["start"] = start_sec
                    clip["end"] = end_sec

        for clip in self.multicut_clips:
            if clip.get("source_clip_id") == clip_id:
                clip["start"] = start_sec
                clip["end"] = end_sec

        self.state_changed.emit("clip_range")

    def append_multicut_clip(self, clip: dict[str, Any]) -> dict[str, Any]:
        source_clip_id = clip.get("clip_id") or clip.get("source_clip_id")
        if source_clip_id is None:
            source_clip_id = self._next_clip_id()

        item = {
            **deepcopy(clip),
            "source_clip_id": source_clip_id,
            "multicut_id": clip.get("multicut_id") or self._next_multicut_id(),
        }
        if "clip_id" not in item:
            item["clip_id"] = source_clip_id

        item["start"] = float(item.get("start", 0.0))
        item["end"] = float(item.get("end", 0.0))
        self.multicut_clips.append(item)
        self.state_changed.emit("multicut_append")
        return deepcopy(item)

    def set_multicut_clips(self, clips: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for clip in clips:
            source_clip_id = clip.get("source_clip_id") or clip.get("clip_id")
            if source_clip_id is None:
                source_clip_id = self._next_clip_id()

            normalized.append({
                **deepcopy(clip),
                "clip_id": source_clip_id,
                "source_clip_id": source_clip_id,
                "multicut_id": clip.get("multicut_id") or self._next_multicut_id(),
                "start": float(clip.get("start", 0.0)),
                "end": float(clip.get("end", 0.0)),
            })

        self.multicut_clips = normalized
        self.state_changed.emit("multicut_clips")

    def set_current_position_ms(self, ms: int) -> None:
        self.current_position_ms = max(0, int(ms))
        self.state_changed.emit("current_position_ms")

    def set_in_point_ms(self, ms: int) -> None:
        self.in_point_ms = max(0, int(ms))
        self.state_changed.emit("in_point_ms")

    def set_out_point_ms(self, ms: int) -> None:
        self.out_point_ms = max(0, int(ms))
        self.state_changed.emit("out_point_ms")

    def set_crop_params(self, params: dict[str, Any]) -> None:
        merged = deepcopy(self.crop_params)
        merged.update(params)
        self.crop_params = merged
        self.state_changed.emit("crop_params")

    def build_export_payload(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "segments": deepcopy(self.segments),
            "highlights": deepcopy(self.highlights),
            "timeline_clips": deepcopy(self.timeline_clips),
            "multicut_clips": deepcopy(self.multicut_clips),
            "current_position_ms": self.current_position_ms,
            "in_point_ms": self.in_point_ms,
            "out_point_ms": self.out_point_ms,
            "crop_params": deepcopy(self.crop_params),
            "duration_ms": self.duration_ms,
        }

    def _normalize_clip(self, clip: dict[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(clip)
        normalized["clip_id"] = clip.get("clip_id") or self._next_clip_id()
        normalized["start"] = float(clip.get("start", 0.0))
        normalized["end"] = float(clip.get("end", 0.0))
        normalized["score"] = float(clip.get("score", 0.0))
        normalized["text"] = str(clip.get("text", ""))
        normalized["reasons"] = list(clip.get("reasons", []))
        return normalized

    def _next_clip_id(self) -> str:
        self._clip_seq += 1
        return f"clip_{self._clip_seq:05d}"

    def _next_multicut_id(self) -> str:
        self._multicut_seq += 1
        return f"mc_{self._multicut_seq:05d}"