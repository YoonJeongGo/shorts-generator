# ui/multicut_editor.py

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QLabel,
    QPushButton, QMessageBox,
)
from PySide6.QtGui import QPainter, QColor


class MiniTimelineBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)
        self._clips: list[dict[str, Any]] = []

    def set_clips(self, clips: list[dict[str, Any]]) -> None:
        self._clips = clips
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor(30, 30, 46))

        if not self._clips:
            painter.setPen(QColor(100, 100, 120))
            painter.drawText(0, 0, w, h, Qt.AlignCenter, "멀티컷 구간 없음")
            painter.end()
            return

        total = sum(max(0.0, c["end"] - c["start"]) for c in self._clips)
        if total <= 0:
            painter.end()
            return

        colors = [
            QColor(46, 117, 182),
            QColor(29, 122, 46),
            QColor(180, 80, 50),
            QColor(120, 50, 160),
        ]

        x = 0
        for i, clip in enumerate(self._clips):
            dur = max(0.0, float(clip["end"]) - float(clip["start"]))
            block_w = int(dur / total * w)
            color = colors[i % len(colors)]
            painter.fillRect(x, 0, block_w, h, color)
            if block_w > 24:
                painter.setPen(QColor(255, 255, 255, 200))
                painter.drawText(
                    x + 2, 0, block_w - 4, h,
                    Qt.AlignVCenter | Qt.AlignLeft,
                    f"#{i+1}",
                )
            x += block_w

        painter.end()


class MulticutEditor(QWidget):
    clips_changed = Signal(list)
    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_internal()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        self._lbl_total = QLabel("총 길이: 0.00 초")
        self._lbl_total.setStyleSheet("font-weight: bold; color: #1F4E79;")
        self._btn_export = QPushButton("🎬 내보내기")
        self._btn_export.setStyleSheet(
            "background: #1D7A2E; color: white; font-weight: bold; padding: 4px 12px;"
        )
        top_row.addWidget(self._lbl_total)
        top_row.addStretch()
        top_row.addWidget(self._btn_export)
        layout.addLayout(top_row)

        self._mini_timeline = MiniTimelineBar(self)
        layout.addWidget(self._mini_timeline)

        self._list = QListWidget()
        self._list.setDragDropMode(QListWidget.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setToolTip("≡ 아이콘을 드래그하여 순서 변경")
        layout.addWidget(self._list, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._btn_up = QPushButton("▲")
        self._btn_down = QPushButton("▼")
        self._btn_del = QPushButton("삭제")
        self._btn_manual = QPushButton("+ 직접 추가")
        self._btn_clear = QPushButton("전체 삭제")
        self._btn_clear.setStyleSheet("color: #CC0000;")

        for b in (self._btn_up, self._btn_down):
            b.setFixedWidth(36)

        btn_row.addWidget(self._btn_up)
        btn_row.addWidget(self._btn_down)
        btn_row.addWidget(self._btn_del)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_manual)
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

    def _connect_internal(self) -> None:
        self._btn_export.clicked.connect(self.export_requested)
        self._btn_up.clicked.connect(self._move_up)
        self._btn_down.clicked.connect(self._move_down)
        self._btn_del.clicked.connect(self._delete_selected)
        self._btn_manual.clicked.connect(self._add_manual)
        self._btn_clear.clicked.connect(self._clear_all)
        self._list.model().rowsMoved.connect(self._on_order_changed)

    @Slot(dict)
    def add_clip(self, clip: dict[str, Any]) -> None:
        item = self._make_item(clip)
        self._list.addItem(item)
        self._refresh()

    def get_clips(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for i in range(self._list.count()):
            data = self._list.item(i).data(Qt.UserRole)
            if data:
                result.append(data)
        return result

    def _make_item(self, clip: dict[str, Any]) -> QListWidgetItem:
        start = float(clip.get("start", 0))
        end = float(clip.get("end", 0))
        dur = max(0.0, end - start)
        label = f"≡  [{self._list.count() + 1}]  {self._fmt(start)} ~ {self._fmt(end)}  ({dur:.1f}초)"
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, {**clip})
        return item

    def _refresh(self) -> None:
        clips = self.get_clips()
        total = sum(max(0.0, c["end"] - c["start"]) for c in clips)
        minutes = int(total // 60)
        seconds = total % 60
        self._lbl_total.setText(f"총 길이: {minutes:02}:{seconds:04.1f}  ({len(clips)}개 컷)")
        self._mini_timeline.set_clips(clips)
        self.clips_changed.emit(clips)

    @Slot()
    def _on_order_changed(self) -> None:
        self._relabel()
        self._refresh()

    def _relabel(self) -> None:
        for i in range(self._list.count()):
            data = self._list.item(i).data(Qt.UserRole)
            if not data:
                continue
            start = float(data.get("start", 0))
            end = float(data.get("end", 0))
            dur = max(0.0, end - start)
            self._list.item(i).setText(
                f"≡  [{i + 1}]  {self._fmt(start)} ~ {self._fmt(end)}  ({dur:.1f}초)"
            )

    @Slot()
    def _move_up(self) -> None:
        row = self._list.currentRow()
        if row <= 0:
            return
        item = self._list.takeItem(row)
        self._list.insertItem(row - 1, item)
        self._list.setCurrentRow(row - 1)
        self._relabel()
        self._refresh()

    @Slot()
    def _move_down(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= self._list.count() - 1:
            return
        item = self._list.takeItem(row)
        self._list.insertItem(row + 1, item)
        self._list.setCurrentRow(row + 1)
        self._relabel()
        self._refresh()

    @Slot()
    def _delete_selected(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        self._list.takeItem(row)
        self._relabel()
        self._refresh()

    @Slot()
    def _add_manual(self) -> None:
        self.add_clip({
            "start": 0.0,
            "end": 30.0,
            "score": 0.0,
            "reasons": ["수동 추가"],
            "text": "",
        })

    @Slot()
    def _clear_all(self) -> None:
        if self._list.count() == 0:
            return
        if QMessageBox.question(self, "확인", "전체 삭제하시겠습니까?") != QMessageBox.Yes:
            return
        self._list.clear()
        self._refresh()

    @staticmethod
    def _fmt(sec: float) -> str:
        s = int(sec)
        return f"{s // 60}:{s % 60:02}"