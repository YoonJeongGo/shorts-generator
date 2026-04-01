# ui/left_panel.py

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem,
    QLabel, QPushButton, QSplitter,
)

from ui.multicut_editor import MiniTimelineBar


class LeftPanel(QWidget):
    clip_selected = Signal(int)
    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(240)
        self._clips: list[dict[str, Any]] = []
        self._build_ui()
        self._connect_internal()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Vertical)

        clip_widget = QWidget()
        clip_layout = QVBoxLayout(clip_widget)
        clip_layout.setContentsMargins(0, 0, 0, 0)
        clip_layout.setSpacing(2)

        lbl1 = QLabel("컷 목록")
        lbl1.setStyleSheet("font-weight: bold; font-size: 11px;")

        top_btns = QHBoxLayout()
        self._btn_ai = QPushButton("AI 추천")
        self._btn_add = QPushButton("+ 수동")
        self._btn_ai.setStyleSheet("background: #2E75B6; color: white;")
        top_btns.addWidget(self._btn_ai)
        top_btns.addWidget(self._btn_add)

        self._clip_list = QListWidget()
        self._clip_list.setAlternatingRowColors(True)

        clip_layout.addWidget(lbl1)
        clip_layout.addLayout(top_btns)
        clip_layout.addWidget(self._clip_list)
        splitter.addWidget(clip_widget)

        mc_widget = QWidget()
        mc_layout = QVBoxLayout(mc_widget)
        mc_layout.setContentsMargins(0, 0, 0, 0)
        mc_layout.setSpacing(2)

        lbl2 = QLabel("멀티컷 순서")
        lbl2.setStyleSheet("font-weight: bold; font-size: 11px;")

        self._mini_bar = MiniTimelineBar()

        self._mc_list = QListWidget()
        self._mc_list.setDragDropMode(QListWidget.InternalMove)
        self._mc_list.model().rowsMoved.connect(self._on_mc_order_changed)

        total_row = QHBoxLayout()
        self._lbl_total = QLabel("총: 0.00초")
        self._lbl_total.setStyleSheet("font-size: 10px;")
        self._btn_export = QPushButton("내보내기")
        self._btn_export.setStyleSheet(
            "background: #1D7A2E; color: white; font-size: 10px;"
        )
        total_row.addWidget(self._lbl_total)
        total_row.addStretch()
        total_row.addWidget(self._btn_export)

        mc_layout.addWidget(lbl2)
        mc_layout.addWidget(self._mini_bar)
        mc_layout.addWidget(self._mc_list)
        mc_layout.addLayout(total_row)
        splitter.addWidget(mc_widget)

        splitter.setSizes([300, 200])
        layout.addWidget(splitter)

    def _connect_internal(self) -> None:
        self._clip_list.itemClicked.connect(self._on_clip_clicked)
        self._btn_export.clicked.connect(self.export_requested)

    @Slot(list)
    def load_clips(self, clips: list[dict[str, Any]]) -> None:
        self._clips = clips
        self._clip_list.clear()
        for i, clip in enumerate(clips):
            self._clip_list.addItem(self._make_clip_label(i, clip))

    @Slot(dict)
    def add_clip(self, clip: dict[str, Any]) -> None:
        i = self._clip_list.count()
        self._clips.append(clip)
        self._clip_list.addItem(self._make_clip_label(i, clip))

    @Slot(dict)
    def add_multicut_clip(self, clip: dict[str, Any]) -> None:
        start = float(clip.get("start", 0))
        end = float(clip.get("end", 0))
        dur = max(0.0, end - start)
        item = QListWidgetItem(
            f"≡  {self._fmt(start)}~{self._fmt(end)}  ({dur:.0f}초)"
        )
        item.setData(Qt.UserRole, clip)
        self._mc_list.addItem(item)
        self._refresh_mc()

    @Slot(list)
    def update_mini_timeline(self, clips: list[dict[str, Any]]) -> None:
        self._mini_bar.set_clips(clips)

    @Slot(QListWidgetItem)
    def _on_clip_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data:
            ms = int(float(data.get("start", 0)) * 1000)
            self.clip_selected.emit(ms)

    def _on_mc_order_changed(self) -> None:
        self._refresh_mc()

    def _refresh_mc(self) -> None:
        clips: list[dict[str, Any]] = []
        for i in range(self._mc_list.count()):
            data = self._mc_list.item(i).data(Qt.UserRole)
            if data:
                clips.append(data)
        total = sum(max(0.0, c["end"] - c["start"]) for c in clips)
        self._lbl_total.setText(f"총: {total:.1f}초")
        self._mini_bar.set_clips(clips)

    @staticmethod
    def _make_clip_label(i: int, clip: dict[str, Any]) -> QListWidgetItem:
        start = float(clip.get("start", 0))
        end = float(clip.get("end", 0))
        score = float(clip.get("score", 0.0))
        label = f"#{i+1}  {LeftPanel._fmt(start)}~{LeftPanel._fmt(end)}  ★{score:.1f}"
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, clip)
        return item

    @staticmethod
    def _fmt(sec: float) -> str:
        s = int(sec)
        return f"{s // 60}:{s % 60:02}"