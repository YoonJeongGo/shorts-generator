# ui/workers.py

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot


class SttWorker(QObject):
    """
    영상 -> 오디오 추출 -> STT -> 세그먼트 반환
    """

    finished = Signal(list)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, video_path: str) -> None:
        super().__init__()
        self._video_path = video_path

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("오디오 추출 중...")
            from pipeline.audio_extractor import extract_audio
            audio_path = extract_audio(self._video_path)

            self.progress.emit("STT 분석 중...")
            from pipeline.stt_engine import transcribe_audio
            segments = transcribe_audio(audio_path)

            self.finished.emit(segments)
        except Exception as e:
            self.error.emit(f"STT 오류: {e}")


class HighlightWorker(QObject):
    """
    세그먼트 -> HighlightScorer -> 하이라이트 반환
    """

    finished = Signal(list)
    error = Signal(str)

    def __init__(self, segments: list[dict[str, Any]]) -> None:
        super().__init__()
        self._segments = segments

    @Slot()
    def run(self) -> None:
        try:
            from config import CLIP
            from pipeline.highlight_scorer import HighlightScorer

            scorer = HighlightScorer(
                top_k=int(CLIP.get("top_k", 3)),
                min_clip_sec=int(CLIP.get("min_sec", 15)),
                max_clip_sec=int(CLIP.get("max_sec", 40)),
                window_sec=int(CLIP.get("window_sec", 30)),
                overlap_gap_sec=int(CLIP.get("overlap_gap_sec", 20)),
                use_semantic=False,
            )
            highlights = scorer.extract_highlights(
                self._segments,
                top_k=int(CLIP.get("top_k", 3)),
            )
            self.finished.emit(highlights)
        except Exception as e:
            self.error.emit(f"하이라이트 추출 오류: {e}")


class ExportWorker(QObject):
    """
    AppState 스냅샷 하나만 받아 최종 내보내기를 수행한다.

    export_state 필수 키:
    - video_path
    - segments
    - multicut_clips
    - crop_params
    """

    finished = Signal(str)
    progress = Signal(int)
    error = Signal(str)

    def __init__(
        self,
        export_state: dict[str, Any],
        out_dir: str,
        concat_mode: str = "auto",
    ) -> None:
        super().__init__()
        self._state = export_state
        self._out_dir = out_dir
        self._concat_mode = concat_mode

    @Slot()
    def run(self) -> None:
        try:
            video_path = str(self._state.get("video_path") or "")
            clips = list(self._state.get("multicut_clips", []))
            segments = list(self._state.get("segments", []))
            crop_params = dict(self._state.get("crop_params", {}))

            if not video_path:
                raise ValueError("video_path가 없습니다.")
            if not clips:
                raise ValueError("내보낼 multicut_clips가 없습니다.")

            out_dir = Path(self._out_dir)
            raw_dir = out_dir / "raw"
            final_dir = out_dir / "final"
            raw_dir.mkdir(parents=True, exist_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(video_path).stem
            raw_paths: list[str] = []

            from pipeline.clip_cutter import cut_clip

            total_clips = len(clips)
            for i, clip in enumerate(clips, start=1):
                start = float(clip.get("start", 0.0))
                end = float(clip.get("end", 0.0))
                if end <= start:
                    continue

                raw_path = str(raw_dir / f"{stem}_raw_{i:02d}.mp4")
                cut_clip(
                    video_path=video_path,
                    start_sec=start,
                    end_sec=end,
                    output_path=raw_path,
                )
                raw_paths.append(raw_path)
                self.progress.emit(int(i / max(1, total_clips) * 40))

            if not raw_paths:
                raise ValueError("잘라낸 클립이 없습니다.")

            if len(raw_paths) == 1:
                concat_path = str(raw_dir / f"{stem}_concat.mp4")
                shutil.copy2(raw_paths[0], concat_path)
            else:
                from pipeline.multicat import concat_clips
                concat_path = str(raw_dir / f"{stem}_concat.mp4")
                concat_clips(
                    input_paths=raw_paths,
                    output_path=concat_path,
                    mode=self._concat_mode,
                )

            self.progress.emit(65)

            final_segments = self._build_concat_segments(
                source_segments=segments,
                multicut_clips=clips,
            )

            final_path = str(final_dir / f"{stem}_shortcut.mp4")
            crop_mode = str(crop_params.get("mode", "auto"))

            if crop_mode == "none":
                from pipeline.exporter import export_clip_with_subtitles
                total_duration = sum(
                    max(0.0, float(c["end"]) - float(c["start"]))
                    for c in clips
                )
                export_clip_with_subtitles(
                    clip_path=concat_path,
                    clip_start_sec=0.0,
                    clip_end_sec=total_duration,
                    segments=final_segments,
                    output_path=final_path,
                )
            else:
                from pipeline.exporter import export_clip_with_crop_and_subtitles
                export_clip_with_crop_and_subtitles(
                    clip_path=concat_path,
                    segments=final_segments,
                    crop_params=crop_params,
                    output_path=final_path,
                )

            self.progress.emit(100)
            self.finished.emit(final_path)

        except Exception as e:
            self.error.emit(f"내보내기 오류: {e}")

    def _build_concat_segments(
        self,
        source_segments: list[dict[str, Any]],
        multicut_clips: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        concat 이후 시간축으로 자막 재배치.

        기존 코드보다 안전하게:
        - 클립 내부에 완전히 포함된 세그먼트만이 아니라
        - 클립과 겹치는 세그먼트도 잘라서 포함한다.
        """
        result: list[dict[str, Any]] = []
        time_offset = 0.0

        for clip in multicut_clips:
            clip_start = float(clip.get("start", 0.0))
            clip_end = float(clip.get("end", 0.0))
            if clip_end <= clip_start:
                continue

            for seg in source_segments:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                seg_text = str(seg.get("text", "")).strip()

                overlap_start = max(seg_start, clip_start)
                overlap_end = min(seg_end, clip_end)

                if overlap_end <= overlap_start:
                    continue
                if not seg_text:
                    continue

                result.append({
                    "start": (overlap_start - clip_start) + time_offset,
                    "end": (overlap_end - clip_start) + time_offset,
                    "text": seg_text,
                })

            time_offset += (clip_end - clip_start)

        return result