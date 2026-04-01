# ui/timeline_widget.py

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal, Slot, QRect, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollBar, QLabel, QSlider
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont


COLOR_BG = QColor(30, 30, 46)
COLOR_RULER_BG = QColor(20, 20, 35)
COLOR_RULER_TEXT = QColor(160, 160, 180)
COLOR_CLIP_AI = QColor(46, 117, 182)
COLOR_CLIP_MANUAL = QColor(29, 122, 46)
COLOR_CLIP_SEL = QColor(255, 200, 50)
COLOR_CLIP_HANDLE = QColor(255, 255, 255, 120)
COLOR_SUB_BLOCK = QColor(181, 212, 244)
COLOR_SUB_TEXT = QColor(12, 68, 124)
COLOR_PLAYHEAD = QColor(255, 255, 255)
COLOR_INOUT = QColor(255, 140, 0, 180)

H_RULER = 20
H_CUT = 36
H_SUB = 20
H_PADDING = 4
HANDLE_W = 6


class TimelineCanvas(QWidget):
    seek_requested = Signal(int)
    clip_selected = Signal(dict)
    clip_range_changed = Signal(dict)
    subtitle_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(H_RULER + H_CUT + H_SUB + H_PADDING * 3)
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

        self._duration_ms: int = 0
        self._playhead_ms: int = 0
        self._in_ms: int = 0
        self._out_ms: int = 0
        self._clips: list[dict[str, Any]] = []
        self._subtitles: list[dict[str, Any]] = []
        self._selected_idx: int = -1

        self._zoom: float = 1.0
        self._scroll_px: int = 0

        self._drag_mode: str = "none"
        self._drag_clip_idx: int = -1

    def _ms_to_px(self, ms: int) -> int:
        if self._duration_ms == 0:
            return 0
        ratio = ms / self._duration_ms
        return int(ratio * self.width() * self._zoom) - self._scroll_px

    def _px_to_ms(self, px: int) -> int:
        if self._duration_ms == 0 or self.width() == 0:
            return 0
        return int((px + self._scroll_px) / (self.width() * self._zoom) * self._duration_ms)

    @Slot(int)
    def set_duration_ms(self, ms: int) -> None:
        self._duration_ms = ms
        self.update()

    @Slot(int)
    def set_playhead_ms(self, ms: int) -> None:
        self._playhead_ms = ms
        self.update()

    def set_clips(self, clips: list[dict[str, Any]]) -> None:
        self._clips = [dict(c) for c in clips]
        self._selected_idx = -1
        self.update()

    def add_clip(self, clip: dict[str, Any]) -> None:
        self._clips.append(dict(clip))
        self.update()

    def set_subtitles(self, segs: list[dict[str, Any]]) -> None:
        self._subtitles = segs
        self.update()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(1.0, min(20.0, zoom))
        self.update()

    def set_scroll_px(self, px: int) -> None:
        self._scroll_px = px
        self.update()

    def set_in_point_ms(self, ms: int) -> None:
        self._in_ms = ms
        self.update()

    def set_out_point_ms(self, ms: int) -> None:
        self._out_ms = ms
        self.update()

    def in_point_ms(self) -> int:
        return self._in_ms

    def out_point_ms(self) -> int:
        return self._out_ms

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, COLOR_BG)

        if self._duration_ms == 0:
            painter.setPen(QColor(100, 100, 120))
            painter.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "영상을 불러오세요")
            return

        y_ruler = 0
        y_cut = y_ruler + H_RULER + H_PADDING
        y_sub = y_cut + H_CUT + H_PADDING

        self._draw_ruler(painter, y_ruler, w)
        self._draw_in_out(painter, y_ruler, h)
        self._draw_clips(painter, y_cut)
        self._draw_subtitles(painter, y_sub)
        self._draw_playhead(painter, h)

        painter.end()

    def _draw_ruler(self, p: QPainter, y: int, w: int) -> None:
        p.fillRect(0, y, w, H_RULER, COLOR_RULER_BG)
        p.setPen(QPen(COLOR_RULER_TEXT))
        p.setFont(QFont("monospace", 8))

        dur_s = self._duration_ms / 1000
        step_s = self._auto_step_sec(dur_s)

        t = 0.0
        while t <= dur_s:
            x = self._ms_to_px(int(t * 1000))
            if 0 <= x <= w:
                p.drawLine(x, y + H_RULER - 5, x, y + H_RULER)
                p.drawText(x + 2, y + H_RULER - 3, self._fmt_sec(t))
            t += step_s

    def _draw_in_out(self, p: QPainter, y: int, h: int) -> None:
        if self._in_ms > 0:
            x = self._ms_to_px(self._in_ms)
            p.setPen(QPen(COLOR_INOUT, 1, Qt.DashLine))
            p.drawLine(x, y, x, h)
        if self._out_ms > 0:
            x = self._ms_to_px(self._out_ms)
            p.setPen(QPen(COLOR_INOUT, 1, Qt.DashLine))
            p.drawLine(x, y, x, h)
        if self._in_ms > 0 and self._out_ms > self._in_ms:
            x1 = self._ms_to_px(self._in_ms)
            x2 = self._ms_to_px(self._out_ms)
            p.fillRect(x1, y, x2 - x1, h, QColor(255, 140, 0, 30))

    def _draw_clips(self, p: QPainter, y: int) -> None:
        p.setFont(QFont("Arial", 9))
        for i, clip in enumerate(self._clips):
            x1 = self._ms_to_px(int(float(clip["start"]) * 1000))
            x2 = self._ms_to_px(int(float(clip["end"]) * 1000))
            rect_w = max(x2 - x1, 4)

            is_manual = "수동 추가" in clip.get("reasons", [])
            base_color = COLOR_CLIP_MANUAL if is_manual else COLOR_CLIP_AI
            color = COLOR_CLIP_SEL if i == self._selected_idx else base_color

            p.fillRect(x1, y, rect_w, H_CUT, color)
            p.fillRect(x1, y, HANDLE_W, H_CUT, COLOR_CLIP_HANDLE)
            p.fillRect(x2 - HANDLE_W, y, HANDLE_W, H_CUT, COLOR_CLIP_HANDLE)

            if rect_w > 40:
                p.setPen(QColor(255, 255, 255, 220))
                dur = float(clip["end"]) - float(clip["start"])
                p.drawText(
                    QRect(x1 + HANDLE_W + 2, y, rect_w - HANDLE_W * 2 - 4, H_CUT),
                    Qt.AlignVCenter | Qt.AlignLeft,
                    f"#{i+1}  {dur:.0f}s",
                )

    def _draw_subtitles(self, p: QPainter, y: int) -> None:
        p.setFont(QFont("Arial", 8))
        for i, seg in enumerate(self._subtitles):
            x1 = self._ms_to_px(int(float(seg["start"]) * 1000))
            x2 = self._ms_to_px(int(float(seg["end"]) * 1000))
            rect_w = max(x2 - x1, 2)
            p.fillRect(x1, y, rect_w, H_SUB, COLOR_SUB_BLOCK)
            if rect_w > 20:
                p.setPen(COLOR_SUB_TEXT)
                p.drawText(
                    QRect(x1 + 2, y, rect_w - 4, H_SUB),
                    Qt.AlignVCenter | Qt.AlignLeft,
                    str(seg.get("text", ""))[:12],
                )

    def _draw_playhead(self, p: QPainter, h: int) -> None:
        x = self._ms_to_px(self._playhead_ms)
        p.setPen(QPen(COLOR_PLAYHEAD, 1.5))
        p.drawLine(x, 0, x, h)

        p.setBrush(QBrush(COLOR_PLAYHEAD))
        p.setPen(Qt.NoPen)
        from PySide6.QtGui import QPolygon
        tri = QPolygon([QPoint(x - 5, 0), QPoint(x + 5, 0), QPoint(x, 8)])
        p.drawPolygon(tri)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        px = event.x()
        py = event.y()

        head_x = self._ms_to_px(self._playhead_ms)
        if abs(px - head_x) <= 8:
            self._drag_mode = "head"
            return

        y_cut = H_RULER + H_PADDING
        if y_cut <= py <= y_cut + H_CUT:
            for i, clip in enumerate(self._clips):
                x1 = self._ms_to_px(int(float(clip["start"]) * 1000))
                x2 = self._ms_to_px(int(float(clip["end"]) * 1000))
                if abs(px - x1) <= HANDLE_W:
                    self._drag_mode = "handle_l"
                    self._drag_clip_idx = i
                    return
                if abs(px - x2) <= HANDLE_W:
                    self._drag_mode = "handle_r"
                    self._drag_clip_idx = i
                    return
                if x1 < px < x2:
                    self._selected_idx = i
                    self.clip_selected.emit(clip)
                    self.update()
                    return

        y_sub = H_RULER + H_PADDING + H_CUT + H_PADDING
        if y_sub <= py <= y_sub + H_SUB:
            for i, seg in enumerate(self._subtitles):
                x1 = self._ms_to_px(int(float(seg["start"]) * 1000))
                x2 = self._ms_to_px(int(float(seg["end"]) * 1000))
                if x1 <= px <= x2:
                    self.subtitle_clicked.emit(i)
                    return

        ms = self._px_to_ms(px)
        self.seek_requested.emit(ms)

    def mouseMoveEvent(self, event) -> None:
        px = event.x()
        ms = max(0, min(self._duration_ms, self._px_to_ms(px)))

        if self._drag_mode == "head":
            self.seek_requested.emit(ms)
        elif self._drag_mode in ("handle_l", "handle_r") and self._drag_clip_idx >= 0:
            i = self._drag_clip_idx
            clip = self._clips[i]
            sec = ms / 1000.0
            if self._drag_mode == "handle_l":
                new_start = max(0.0, min(sec, float(clip["end"]) - 1.0))
                self._clips[i]["start"] = new_start
            else:
                new_end = min(self._duration_ms / 1000.0, max(sec, float(clip["start"]) + 1.0))
                self._clips[i]["end"] = new_end
            self.clip_range_changed.emit({
                "index": i,
                "start": self._clips[i]["start"],
                "end": self._clips[i]["end"],
            })
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_mode = "none"
        self._drag_clip_idx = -1

    @staticmethod
    def _auto_step_sec(dur_s: float) -> float:
        if dur_s <= 30:
            return 5.0
        if dur_s <= 120:
            return 10.0
        if dur_s <= 600:
            return 30.0
        if dur_s <= 3600:
            return 60.0
        return 300.0

    @staticmethod
    def _fmt_sec(s: float) -> str:
        s = int(s)
        return f"{s // 60}:{s % 60:02}"


class TimelineWidget(QWidget):
    seek_requested = Signal(int)
    clip_selected = Signal(dict)
    clip_range_changed = Signal(dict)
    subtitle_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(130)
        self._build_ui()
        self._connect_internal()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._canvas = TimelineCanvas(self)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(4, 0, 4, 2)
        self._scrollbar = QScrollBar(Qt.Horizontal)
        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(10, 200)
        self._zoom_slider.setValue(10)
        self._zoom_slider.setFixedWidth(90)
        self._zoom_label = QLabel("1x")
        self._zoom_label.setFixedWidth(24)
        self._zoom_label.setStyleSheet("font-size: 10px;")

        bottom.addWidget(self._scrollbar)
        bottom.addWidget(QLabel("줌"))
        bottom.addWidget(self._zoom_slider)
        bottom.addWidget(self._zoom_label)

        layout.addWidget(self._canvas, stretch=1)
        layout.addLayout(bottom)

    def _connect_internal(self) -> None:
        self._canvas.seek_requested.connect(self.seek_requested)
        self._canvas.clip_selected.connect(self.clip_selected)
        self._canvas.clip_range_changed.connect(self.clip_range_changed)
        self._canvas.subtitle_clicked.connect(self.subtitle_clicked)

        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self._scrollbar.valueChanged.connect(self._canvas.set_scroll_px)

    def _on_zoom_changed(self, value: int) -> None:
        zoom = value / 10.0
        self._zoom_label.setText(f"{zoom:.1f}x")
        self._canvas.set_zoom(zoom)
        self._scrollbar.setMaximum(max(0, int(self._canvas.width() * (zoom - 1))))

    @Slot(int)
    def set_duration_ms(self, ms: int) -> None:
        self._canvas.set_duration_ms(ms)

    @Slot(int)
    def set_playhead_ms(self, ms: int) -> None:
        self._canvas.set_playhead_ms(ms)

    def set_clips(self, clips: list[dict]) -> None:
        self._canvas.set_clips(clips)

    def add_clip(self, clip: dict) -> None:
        self._canvas.add_clip(clip)

    def set_subtitles(self, segs: list[dict]) -> None:
        self._canvas.set_subtitles(segs)

    def set_in_point_ms(self, ms: int) -> None:
        self._canvas.set_in_point_ms(ms)

    def set_out_point_ms(self, ms: int) -> None:
        self._canvas.set_out_point_ms(ms)

    def in_point_ms(self) -> int:
        return self._canvas.in_point_ms()

    def out_point_ms(self) -> int:
        return self._canvas.out_point_ms()