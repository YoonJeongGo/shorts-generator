# ui/crop_editor.py

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class CropEditor(QWidget):
    crop_params_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_internal()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        mode_group = QGroupBox("크롭 모드")
        mode_layout = QHBoxLayout(mode_group)
        self._radio_auto = QRadioButton("9:16 자동 (중앙)")
        self._radio_manual = QRadioButton("9:16 수동 조정")
        self._radio_none = QRadioButton("원본 그대로")
        self._radio_auto.setChecked(True)

        self._btn_group = QButtonGroup(self)
        self._btn_group.addButton(self._radio_auto, 0)
        self._btn_group.addButton(self._radio_manual, 1)
        self._btn_group.addButton(self._radio_none, 2)

        mode_layout.addWidget(self._radio_auto)
        mode_layout.addWidget(self._radio_manual)
        mode_layout.addWidget(self._radio_none)
        layout.addWidget(mode_group)

        pos_group = QGroupBox("위치 조정 (수동 모드)")
        pos_layout = QVBoxLayout(pos_group)

        y_row = QHBoxLayout()
        y_row.addWidget(QLabel("Y축 (위↑ ↓아래):"))
        self._slider_y = QSlider(Qt.Horizontal)
        self._slider_y.setRange(-50, 50)
        self._slider_y.setValue(0)
        self._lbl_y = QLabel("0%")
        self._lbl_y.setFixedWidth(36)
        y_row.addWidget(self._slider_y)
        y_row.addWidget(self._lbl_y)

        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("X축 (좌← →우):"))
        self._slider_x = QSlider(Qt.Horizontal)
        self._slider_x.setRange(-30, 30)
        self._slider_x.setValue(0)
        self._lbl_x = QLabel("0%")
        self._lbl_x.setFixedWidth(36)
        x_row.addWidget(self._slider_x)
        x_row.addWidget(self._lbl_x)

        pos_layout.addLayout(y_row)
        pos_layout.addLayout(x_row)
        layout.addWidget(pos_group)

        overlay_row = QHBoxLayout()
        self._btn_overlay_toggle = QPushButton("크롭 가이드 OFF")
        self._overlay_enabled = True
        self._btn_overlay_toggle.clicked.connect(self._toggle_overlay)
        overlay_row.addWidget(self._btn_overlay_toggle)
        overlay_row.addStretch()
        layout.addLayout(overlay_row)

        sub_group = QGroupBox("자막 설정")
        sub_layout = QVBoxLayout(sub_group)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("폰트 크기:"))
        self._spin_font_size = QSpinBox()
        self._spin_font_size.setRange(12, 72)
        self._spin_font_size.setValue(36)
        font_row.addWidget(self._spin_font_size)
        font_row.addStretch()

        sub_layout.addLayout(font_row)
        layout.addWidget(sub_group)

        layout.addWidget(QLabel("출력: 720 × 1280  (9:16)"))
        layout.addStretch()

    def _connect_internal(self) -> None:
        self._btn_group.idClicked.connect(self._on_mode_changed)
        self._slider_y.valueChanged.connect(self._on_slider_changed)
        self._slider_x.valueChanged.connect(self._on_slider_changed)
        self._spin_font_size.valueChanged.connect(self._on_slider_changed)

    def get_crop_params(self) -> dict[str, Any]:
        return {
            "enabled": self._overlay_enabled,
            "mode": self._get_mode(),
            "y_offset": self._slider_y.value() / 100.0,
            "x_offset": self._slider_x.value() / 100.0,
            "font_size": self._spin_font_size.value(),
        }

    def _get_mode(self) -> str:
        idx = self._btn_group.checkedId()
        return ["auto", "manual", "none"][idx]

    @Slot(int)
    def _on_mode_changed(self, idx: int) -> None:
        manual = (idx == 1)
        self._slider_y.setEnabled(manual)
        self._slider_x.setEnabled(manual)
        self._emit()

    @Slot()
    def _on_slider_changed(self) -> None:
        self._lbl_y.setText(f"{self._slider_y.value():+d}%")
        self._lbl_x.setText(f"{self._slider_x.value():+d}%")
        self._emit()

    def _toggle_overlay(self) -> None:
        self._overlay_enabled = not self._overlay_enabled
        self._btn_overlay_toggle.setText(
            "크롭 가이드 OFF" if self._overlay_enabled else "크롭 가이드 ON"
        )
        self._emit()

    def _emit(self) -> None:
        self.crop_params_changed.emit(self.get_crop_params())