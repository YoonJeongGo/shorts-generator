# ui/crop_overlay.py

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class CropGuideOverlay(QWidget):
    """
    9:16 크롭 영역을 시각적으로 표시하는 투명 오버레이 위젯.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._params: dict[str, Any] = {
            "enabled": True,
            "y_offset": 0.0,
            "x_offset": 0.0,
            "mode": "auto",
        }

    def set_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)
        self.update()

    def set_enabled(self, enabled: bool) -> None:
        self._params["enabled"] = enabled
        self.update()

    def set_y_offset(self, value: float) -> None:
        self._params["y_offset"] = max(-0.5, min(0.5, value))
        self.update()

    def set_x_offset(self, value: float) -> None:
        self._params["x_offset"] = max(-0.3, min(0.3, value))
        self.update()

    def paintEvent(self, event) -> None:
        if self._params.get("mode") == "none":
            return
        if not self._params.get("enabled", True):
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        w = self.width()
        h = self.height()

        crop_w = int(h * 9 / 16)
        crop_w = min(crop_w, w)

        cx = w // 2 + int(self._params.get("x_offset", 0.0) * w)
        left = cx - crop_w // 2
        right = left + crop_w

        left = max(0, left)
        right = min(w, right)

        mask_color = QColor(0, 0, 0, 150)
        painter.fillRect(0, 0, left, h, mask_color)
        painter.fillRect(right, 0, w - right, h, mask_color)

        pen = QPen(QColor(255, 255, 255, 200), 1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(left, 0, right - left, h)

        pen_guide = QPen(QColor(255, 255, 255, 60), 0.5, Qt.DashLine)
        painter.setPen(pen_guide)
        painter.drawLine(left, h // 3, right, h // 3)
        painter.drawLine(left, h * 2 // 3, right, h * 2 // 3)
        painter.drawLine(left + (right - left) // 3, 0, left + (right - left) // 3, h)
        painter.drawLine(left + (right - left) * 2 // 3, 0, left + (right - left) * 2 // 3, h)

        painter.end()