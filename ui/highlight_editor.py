# ui/highlight_editor.py

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QPushButton, QCheckBox, QDoubleSpinBox,
    QFrame, QMessageBox,
)


class HighlightCard(QFrame):
    seek_requested = Signal(int)      # ms
    add_clicked = Signal(dict)
    start_changed = Signal(int, float)
    end_changed = Signal(int, float)

    def __init__(
        self,
        index: int,
        clip: dict[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._clip = dict(clip)
        self._player_ms: int = 0

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 5px; }")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()

        self._chk = QCheckBox()
        self._chk.setChecked(True)

        rank_text = f"<b>#{self._index + 1}</b>"
        self._lbl_rank = QLabel(rank_text)
        self._lbl_rank.setCursor(Qt.PointingHandCursor)
        self._lbl_rank.setToolTip("클릭하면 해당 구간으로 이동")
        self._lbl_rank.mousePressEvent = lambda e: self.seek_requested.emit(
            int(float(self._clip.get("start", 0.0)) * 1000)
        )

        score = float(self._clip.get("score", 0.0))
        self._lbl_score = QLabel(f"★ {score:.1f}")
        self._lbl_score.setStyleSheet("color: #E67E22; font-weight: bold;")

        header.addWidget(self._chk)
        header.addWidget(self._lbl_rank)
        header.addStretch()
        header.addWidget(self._lbl_score)
        layout.addLayout(header)

        start_row = QHBoxLayout()
        start_row.setSpacing(3)
        start_row.addWidget(QLabel("시작:"))

        self._spin_start = QDoubleSpinBox()
        self._spin_start.setRange(0, 99999)
        self._spin_start.setDecimals(2)
        self._spin_start.setSuffix(" 초")
        self._spin_start.setValue(float(self._clip.get("start", 0.0)))
        self._spin_start.setFixedWidth(90)

        btn_s_minus = QPushButton("-1s")
        btn_s_plus = QPushButton("+1s")
        btn_s_now = QPushButton("현재위치")
        for b in (btn_s_minus, btn_s_plus, btn_s_now):
            b.setFixedWidth(52)
            b.setFocusPolicy(Qt.NoFocus)
        btn_s_now.setStyleSheet("background: #2E75B6; color: white;")

        btn_s_minus.clicked.connect(
            lambda: self._spin_start.setValue(self._spin_start.value() - 1.0)
        )
        btn_s_plus.clicked.connect(
            lambda: self._spin_start.setValue(self._spin_start.value() + 1.0)
        )
        btn_s_now.clicked.connect(self._on_set_start_now)

        start_row.addWidget(btn_s_minus)
        start_row.addWidget(self._spin_start)
        start_row.addWidget(btn_s_plus)
        start_row.addWidget(btn_s_now)
        layout.addLayout(start_row)

        end_row = QHBoxLayout()
        end_row.setSpacing(3)
        end_row.addWidget(QLabel("끝:   "))

        self._spin_end = QDoubleSpinBox()
        self._spin_end.setRange(0, 99999)
        self._spin_end.setDecimals(2)
        self._spin_end.setSuffix(" 초")
        self._spin_end.setValue(float(self._clip.get("end", 0.0)))
        self._spin_end.setFixedWidth(90)

        btn_e_minus = QPushButton("-1s")
        btn_e_plus = QPushButton("+1s")
        btn_e_now = QPushButton("현재위치")
        for b in (btn_e_minus, btn_e_plus, btn_e_now):
            b.setFixedWidth(52)
            b.setFocusPolicy(Qt.NoFocus)
        btn_e_now.setStyleSheet("background: #2E75B6; color: white;")

        btn_e_minus.clicked.connect(
            lambda: self._spin_end.setValue(self._spin_end.value() - 1.0)
        )
        btn_e_plus.clicked.connect(
            lambda: self._spin_end.setValue(self._spin_end.value() + 1.0)
        )
        btn_e_now.clicked.connect(self._on_set_end_now)

        end_row.addWidget(btn_e_minus)
        end_row.addWidget(self._spin_end)
        end_row.addWidget(btn_e_plus)
        end_row.addWidget(btn_e_now)
        layout.addLayout(end_row)

        reasons = self._clip.get("reasons", [])
        if reasons:
            tags_layout = QHBoxLayout()
            tags_layout.setSpacing(3)
            for r in reasons[:4]:
                tag = QLabel(str(r))
                tag.setStyleSheet(
                    "background: #EEE; border-radius: 8px; "
                    "padding: 1px 6px; font-size: 10px; color: #555;"
                )
                tags_layout.addWidget(tag)
            tags_layout.addStretch()
            layout.addLayout(tags_layout)

        text = str(self._clip.get("text", ""))
        if text:
            preview = text[:55] + ("…" if len(text) > 55 else "")
            lbl_text = QLabel(preview)
            lbl_text.setStyleSheet(
                "color: #555; font-size: 11px; font-style: italic;"
            )
            lbl_text.setWordWrap(True)
            layout.addWidget(lbl_text)

        btn_add = QPushButton("▶ 멀티컷에 추가")
        btn_add.setStyleSheet(
            "background: #1F4E79; color: white; font-weight: bold;"
        )
        btn_add.clicked.connect(self._on_add_clicked)
        layout.addWidget(btn_add)

        self._spin_start.valueChanged.connect(
            lambda v: self.start_changed.emit(self._index, float(v))
        )
        self._spin_end.valueChanged.connect(
            lambda v: self.end_changed.emit(self._index, float(v))
        )

    def is_checked(self) -> bool:
        return self._chk.isChecked()

    def set_start_sec(self, sec: float) -> None:
        self._spin_start.blockSignals(True)
        self._spin_start.setValue(sec)
        self._spin_start.blockSignals(False)
        self._clip["start"] = sec

    def set_end_sec(self, sec: float) -> None:
        self._spin_end.blockSignals(True)
        self._spin_end.setValue(sec)
        self._spin_end.blockSignals(False)
        self._clip["end"] = sec

    def get_clip(self) -> dict[str, Any]:
        return {
            **self._clip,
            "start": float(self._spin_start.value()),
            "end": float(self._spin_end.value()),
        }

    def _on_set_start_now(self) -> None:
        self._spin_start.setValue(self._player_ms / 1000.0)

    def _on_set_end_now(self) -> None:
        self._spin_end.setValue(self._player_ms / 1000.0)

    def _on_add_clicked(self) -> None:
        self.add_clicked.emit(self.get_clip())


class HighlightEditor(QWidget):
    add_to_multicut = Signal(dict)
    clip_seek_requested = Signal(int)
    highlights_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: list[HighlightCard] = []
        self._current_player_ms: int = 0
        self._suppress_emit: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        self._btn_manual = QPushButton("➕ 수동 구간 추가")
        self._btn_all = QPushButton("▶▶ 전체 선택 → 멀티컷")
        self._btn_all.setStyleSheet("background: #2E75B6; color: white;")
        self._btn_manual.clicked.connect(self._add_manual)
        self._btn_all.clicked.connect(self._send_all_checked)
        top_row.addWidget(self._btn_manual)
        top_row.addStretch()
        top_row.addWidget(self._btn_all)
        layout.addLayout(top_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setAlignment(Qt.AlignTop)
        self._card_layout.setSpacing(6)

        self._empty_lbl = QLabel("STT 실행 후 하이라이트가 표시됩니다.")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet("color: #aaa; padding: 30px;")
        self._card_layout.addWidget(self._empty_lbl)

        scroll.setWidget(self._card_container)
        layout.addWidget(scroll, stretch=1)

    def set_highlights(self, highlights: list[dict[str, Any]]) -> None:
        self.load_highlights(highlights)

    # 🔥 핵심 추가
    def set_current_player_ms(self, ms: int) -> None:
        self._current_player_ms = ms
        for card in self._cards:
            card._player_ms = ms

    @Slot(list)
    def load_highlights(self, highlights: list[dict[str, Any]]) -> None:
        self._suppress_emit = True
        try:
            self._clear_cards()
            self._empty_lbl.setVisible(len(highlights) == 0)
            for i, h in enumerate(highlights):
                self._add_card(i, h)
        finally:
            self._suppress_emit = False
        self._emit_highlights_changed()

    def _add_card(self, idx: int, clip: dict[str, Any]) -> None:
        card = HighlightCard(idx, clip, self._card_container)
        card._player_ms = self._current_player_ms
        card.seek_requested.connect(self.clip_seek_requested)
        card.add_clicked.connect(self.add_to_multicut)
        card.start_changed.connect(self._on_card_start_changed)
        card.end_changed.connect(self._on_card_end_changed)
        self._cards.append(card)
        self._card_layout.addWidget(card)

    def _clear_cards(self) -> None:
        for card in self._cards:
            card.deleteLater()
        self._cards.clear()

    @Slot(int, float)
    def _on_card_start_changed(self, idx: int, value: float) -> None:
        if 0 <= idx < len(self._cards):
            self._cards[idx]._clip["start"] = float(value)
        self._emit_highlights_changed()

    @Slot(int, float)
    def _on_card_end_changed(self, idx: int, value: float) -> None:
        if 0 <= idx < len(self._cards):
            self._cards[idx]._clip["end"] = float(value)
        self._emit_highlights_changed()

    def _emit_highlights_changed(self) -> None:
        if self._suppress_emit:
            return
        self.highlights_changed.emit(self.get_highlights())

    def get_highlights(self) -> list[dict[str, Any]]:
        return [c.get_clip() for c in self._cards]

    @Slot()
    def _add_manual(self) -> None:
        clip = {
            "start": 0.0,
            "end": 30.0,
            "score": 0.0,
            "reasons": ["수동 추가"],
            "text": "",
        }
        self._add_card(len(self._cards), clip)
        self._emit_highlights_changed()

    @Slot()
    def _send_all_checked(self) -> None:
        clips = [c.get_clip() for c in self._cards if c.is_checked()]
        if not clips:
            QMessageBox.warning(self, "경고", "선택된 구간이 없습니다.")
            return
        for clip in clips:
            self.add_to_multicut.emit(clip)