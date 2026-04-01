# ui/subtitle_editor.py

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)


class SubtitleEditor(QWidget):
    row_seek_requested = Signal(int)
    segments_changed = Signal()

    COL_IDX = 0
    COL_START = 1
    COL_END = 2
    COL_TEXT = 3
    COLUMNS = ["#", "시작(초)", "끝(초)", "자막 텍스트"]

    COLOR_CURRENT = QColor(255, 248, 180)
    COLOR_NORMAL = QColor(0, 0, 0, 0)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_highlighted_row: int = -1
        self._segments: list[dict[str, Any]] = []
        self._current_player_ms: int = 0
        self._block_change_signal = False
        self._build_ui()
        self._connect_internal()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        action_row = QHBoxLayout()
        self._btn_set_start = QPushButton("현재위치로 시작↓")
        self._btn_set_end = QPushButton("현재위치로 끝↓")
        self._btn_add_row = QPushButton("+ 행 추가")
        self._btn_del_row = QPushButton("삭제")

        self._btn_set_start.setStyleSheet("background: #2E75B6; color: white;")
        self._btn_set_end.setStyleSheet("background: #2E75B6; color: white;")

        action_row.addWidget(self._btn_set_start)
        action_row.addWidget(self._btn_set_end)
        action_row.addStretch()
        action_row.addWidget(self._btn_add_row)
        action_row.addWidget(self._btn_del_row)

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(self.COL_TEXT, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(self.COL_IDX, QHeaderView.Fixed)
        self._table.setColumnWidth(self.COL_IDX, 32)
        self._table.setColumnWidth(self.COL_START, 72)
        self._table.setColumnWidth(self.COL_END, 72)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.AnyKeyPressed
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)

        bottom_row = QHBoxLayout()
        self._btn_save_srt = QPushButton("💾 SRT 저장")
        self._btn_load_srt = QPushButton("📂 SRT 불러오기")
        bottom_row.addWidget(self._btn_save_srt)
        bottom_row.addStretch()
        bottom_row.addWidget(self._btn_load_srt)

        layout.addLayout(action_row)
        layout.addWidget(self._table, stretch=1)
        layout.addLayout(bottom_row)

    def _connect_internal(self) -> None:
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.itemChanged.connect(self._on_item_changed)
        self._btn_set_start.clicked.connect(self._on_set_start)
        self._btn_set_end.clicked.connect(self._on_set_end)
        self._btn_add_row.clicked.connect(self._add_row)
        self._btn_del_row.clicked.connect(self._delete_selected_rows)
        self._btn_save_srt.clicked.connect(self._save_srt)
        self._btn_load_srt.clicked.connect(self._load_srt_dialog)

    @Slot(list)
    def load_segments(self, segments: list[dict[str, Any]]) -> None:
        self._segments = [dict(s) for s in segments]
        self._block_change_signal = True
        try:
            self._table.setRowCount(0)
            for i, seg in enumerate(self._segments):
                self._insert_row(
                    row=i,
                    idx=i + 1,
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=str(seg.get("text", "")).strip(),
                )
        finally:
            self._block_change_signal = False

    @Slot(int)
    def highlight_row_at_ms(self, ms: int) -> None:
        sec = ms / 1000.0
        target = -1
        for row in range(self._table.rowCount()):
            start = self._get_float(row, self.COL_START)
            end = self._get_float(row, self.COL_END)
            if start <= sec <= end:
                target = row
                break

        if target == self._current_highlighted_row:
            return

        self._set_row_bg(self._current_highlighted_row, self.COLOR_NORMAL)
        self._set_row_bg(target, self.COLOR_CURRENT)
        self._current_highlighted_row = target

        if target >= 0 and self._table.item(target, self.COL_TEXT):
            self._table.scrollToItem(self._table.item(target, self.COL_TEXT))

    @Slot(int)
    def select_row_by_index(self, idx: int) -> None:
        if 0 <= idx < self._table.rowCount():
            self._table.selectRow(idx)
            item = self._table.item(idx, self.COL_TEXT)
            if item:
                self._table.scrollToItem(item)

    def set_player_position_ms(self, ms: int) -> None:
        self._current_player_ms = ms

    def get_segments(self) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for row in range(self._table.rowCount()):
            try:
                start = float(self._table.item(row, self.COL_START).text())
                end = float(self._table.item(row, self.COL_END).text())
                text = self._table.item(row, self.COL_TEXT).text().strip()
            except (AttributeError, ValueError):
                continue

            if end > start and text:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                })
        return segments

    @Slot(int, int)
    def _on_cell_clicked(self, row: int, col: int) -> None:
        start_sec = self._get_float(row, self.COL_START)
        self.row_seek_requested.emit(int(start_sec * 1000))

    @Slot()
    def _on_item_changed(self) -> None:
        if self._block_change_signal:
            return
        self._segments = self.get_segments()
        self.segments_changed.emit()

    @Slot()
    def _on_set_start(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        sec = self._current_player_ms / 1000.0
        item = self._table.item(row, self.COL_START)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(row, self.COL_START, item)
        item.setText(f"{sec:.2f}")

    @Slot()
    def _on_set_end(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        sec = self._current_player_ms / 1000.0
        item = self._table.item(row, self.COL_END)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(row, self.COL_END, item)
        item.setText(f"{sec:.2f}")

    @Slot()
    def _add_row(self) -> None:
        row = self._table.rowCount()
        self._block_change_signal = True
        try:
            self._insert_row(row, idx=row + 1, start=0.0, end=0.0, text="")
        finally:
            self._block_change_signal = False
        text_item = self._table.item(row, self.COL_TEXT)
        if text_item:
            self._table.setCurrentCell(row, self.COL_TEXT)
            self._table.editItem(text_item)
        self.segments_changed.emit()

    @Slot()
    def _delete_selected_rows(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            return

        self._block_change_signal = True
        try:
            for row in rows:
                self._table.removeRow(row)
            self._renumber()
        finally:
            self._block_change_signal = False

        self._segments = self.get_segments()
        self.segments_changed.emit()

    @Slot()
    def _save_srt(self) -> None:
        segments = self.get_segments()
        if not segments:
            QMessageBox.warning(self, "경고", "저장할 자막이 없습니다.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "SRT 저장",
            "output.srt",
            "SRT 파일 (*.srt)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8-sig") as f:
                for i, seg in enumerate(segments, start=1):
                    f.write(f"{i}\n")
                    f.write(
                        f"{self._to_srt_ts(float(seg['start']))} --> "
                        f"{self._to_srt_ts(float(seg['end']))}\n"
                    )
                    f.write(f"{seg['text']}\n\n")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"SRT 저장 실패:\n{e}")
            return

        QMessageBox.information(self, "완료", f"SRT 저장 완료:\n{path}")

    @Slot()
    def _load_srt_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "SRT 불러오기",
            "",
            "SRT 파일 (*.srt)",
        )
        if not path:
            return

        try:
            from main import parse_srt_file
            segments = parse_srt_file(path)
            self.load_segments(segments)
            self.segments_changed.emit()
        except Exception as e:
            QMessageBox.critical(self, "오류", f"SRT 불러오기 실패:\n{e}")

    def _insert_row(self, row: int, idx: int, start: float, end: float, text: str) -> None:
        self._table.insertRow(row)

        num_item = QTableWidgetItem(str(idx))
        num_item.setFlags(num_item.flags() & ~Qt.ItemIsEditable)
        num_item.setTextAlignment(Qt.AlignCenter)

        start_item = QTableWidgetItem(f"{start:.2f}")
        start_item.setTextAlignment(Qt.AlignCenter)

        end_item = QTableWidgetItem(f"{end:.2f}")
        end_item.setTextAlignment(Qt.AlignCenter)

        text_item = QTableWidgetItem(text)

        self._table.setItem(row, self.COL_IDX, num_item)
        self._table.setItem(row, self.COL_START, start_item)
        self._table.setItem(row, self.COL_END, end_item)
        self._table.setItem(row, self.COL_TEXT, text_item)

    def _renumber(self) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self.COL_IDX)
            if item:
                item.setText(str(row + 1))

    def _get_float(self, row: int, col: int) -> float:
        item = self._table.item(row, col)
        if item is None:
            return 0.0
        try:
            return float(item.text())
        except ValueError:
            return 0.0

    def _set_row_bg(self, row: int, color: QColor) -> None:
        if row < 0 or row >= self._table.rowCount():
            return
        brush = QBrush(color)
        for col in range(self._table.columnCount()):
            item = self._table.item(row, col)
            if item:
                item.setBackground(brush)

    @staticmethod
    def _to_srt_ts(sec: float) -> str:
        total_ms = int(sec * 1000)
        h = total_ms // 3_600_000
        m = (total_ms % 3_600_000) // 60_000
        s = (total_ms % 60_000) // 1000
        ms = total_ms % 1000
        return f"{h:02}:{m:02}:{s:02},{ms:03}"