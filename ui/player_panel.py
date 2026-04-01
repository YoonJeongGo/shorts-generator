# ui/player_panel.py

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from ui.crop_overlay import CropGuideOverlay


class PlayerControlBar(QWidget):
    play_pause_clicked = Signal()
    seek_requested = Signal(int)
    step_requested = Signal(int)
    set_in_requested = Signal()
    set_out_requested = Signal()
    volume_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(62)
        self._duration_ms = 0
        self._is_playing = False
        self._seeking = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        seek_row = QHBoxLayout()
        seek_row.setSpacing(6)

        self._lbl_pos = QLabel("0:00")
        self._lbl_pos.setFixedWidth(40)
        self._lbl_pos.setStyleSheet("font-family: monospace; font-size: 11px;")

        self._seek_slider = QSlider(Qt.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setTracking(False)

        self._lbl_dur = QLabel("0:00")
        self._lbl_dur.setFixedWidth(40)
        self._lbl_dur.setAlignment(Qt.AlignRight)
        self._lbl_dur.setStyleSheet("font-family: monospace; font-size: 11px;")

        seek_row.addWidget(self._lbl_pos)
        seek_row.addWidget(self._seek_slider)
        seek_row.addWidget(self._lbl_dur)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        def _btn(label: str, width: int = 52) -> QPushButton:
            b = QPushButton(label)
            b.setFixedWidth(width)
            b.setFocusPolicy(Qt.NoFocus)
            return b

        self._btn_minus5 = _btn("◀◀ 5s", 58)
        self._btn_minus1 = _btn("◀ 1s", 52)
        self._btn_play = _btn("▶ 재생", 68)
        self._btn_plus1 = _btn("1s ▶", 52)
        self._btn_plus5 = _btn("5s ▶▶", 58)
        self._btn_in = _btn("[I]", 36)
        self._btn_out = _btn("[O]", 36)

        self._btn_in.setToolTip("현재 위치를 구간 시작점으로 (I)")
        self._btn_out.setToolTip("현재 위치를 구간 끝점으로 (O)")
        self._btn_play.setStyleSheet("font-weight: bold;")

        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setFixedWidth(70)
        self._vol_slider.setToolTip("볼륨")

        btn_row.addWidget(self._btn_minus5)
        btn_row.addWidget(self._btn_minus1)
        btn_row.addWidget(self._btn_play)
        btn_row.addWidget(self._btn_plus1)
        btn_row.addWidget(self._btn_plus5)
        btn_row.addSpacing(12)
        btn_row.addWidget(self._btn_in)
        btn_row.addWidget(self._btn_out)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("🔊"))
        btn_row.addWidget(self._vol_slider)

        layout.addLayout(seek_row)
        layout.addLayout(btn_row)

        self._seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self._seek_slider.sliderReleased.connect(self._on_slider_released)
        self._seek_slider.sliderMoved.connect(self._on_slider_moved)
        self._btn_play.clicked.connect(self.play_pause_clicked)
        self._btn_minus5.clicked.connect(lambda: self.step_requested.emit(-5000))
        self._btn_minus1.clicked.connect(lambda: self.step_requested.emit(-1000))
        self._btn_plus1.clicked.connect(lambda: self.step_requested.emit(1000))
        self._btn_plus5.clicked.connect(lambda: self.step_requested.emit(5000))
        self._btn_in.clicked.connect(self.set_in_requested)
        self._btn_out.clicked.connect(self.set_out_requested)
        self._vol_slider.valueChanged.connect(
            lambda v: self.volume_changed.emit(v / 100.0)
        )

    def set_position_ms(self, ms: int) -> None:
        if self._seeking:
            return
        self._seek_slider.setValue(ms)
        self._lbl_pos.setText(self._fmt(ms))

    def set_duration_ms(self, ms: int) -> None:
        self._duration_ms = ms
        self._seek_slider.setRange(0, ms)
        self._lbl_dur.setText(self._fmt(ms))

    def set_playing(self, playing: bool) -> None:
        self._is_playing = playing
        self._btn_play.setText("⏸ 일시정지" if playing else "▶ 재생")

    def _on_slider_pressed(self) -> None:
        self._seeking = True

    def _on_slider_released(self) -> None:
        self._seeking = False
        self.seek_requested.emit(self._seek_slider.value())

    def _on_slider_moved(self, value: int) -> None:
        self._lbl_pos.setText(self._fmt(value))

    @staticmethod
    def _fmt(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60}:{s % 60:02}"


class SubtitleOverlayLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.setWordWrap(True)
        self.setStyleSheet(
            "color: white;"
            "font-size: 18px;"
            "font-weight: bold;"
            "padding-bottom: 16px; padding-left: 8px; padding-right: 8px;"
        )
        self.setText("")


class PlayerPanel(QWidget):
    position_changed = Signal(int)
    duration_changed = Signal(int)
    set_in_point = Signal(int)
    set_out_point = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._setup_player()
        self._connect_internal()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._video_container = QWidget()
        self._video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._video_widget = QVideoWidget(self._video_container)
        self._crop_overlay = CropGuideOverlay(self._video_container)
        self._sub_overlay = SubtitleOverlayLabel(self._video_container)

        self._control_bar = PlayerControlBar(self)

        layout.addWidget(self._video_container, stretch=1)
        layout.addWidget(self._control_bar)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w = self._video_container.width()
        h = self._video_container.height()
        self._video_widget.setGeometry(0, 0, w, h)
        self._crop_overlay.setGeometry(0, 0, w, h)
        self._sub_overlay.setGeometry(0, 0, w, h)

    def _setup_player(self) -> None:
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._audio.setVolume(0.8)

    def _connect_internal(self) -> None:
        self._player.positionChanged.connect(self._control_bar.set_position_ms)
        self._player.durationChanged.connect(self._control_bar.set_duration_ms)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)

        self._control_bar.play_pause_clicked.connect(self.toggle_play_pause)
        self._control_bar.seek_requested.connect(self._player.setPosition)
        self._control_bar.step_requested.connect(self.step_ms)
        self._control_bar.volume_changed.connect(self._audio.setVolume)

        self._control_bar.set_in_requested.connect(
            lambda: self.set_in_point.emit(self._player.position())
        )
        self._control_bar.set_out_requested.connect(
            lambda: self.set_out_point.emit(self._player.position())
        )

    @Slot(str)
    def load_video(self, path: str) -> None:
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.pause()

    @Slot(int)
    def seek_to_ms(self, ms: int) -> None:
        self._player.setPosition(ms)

    @Slot(int)
    def step_ms(self, delta: int) -> None:
        pos = max(0, self._player.position() + delta)
        self._player.setPosition(pos)

    @Slot()
    def toggle_play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    @Slot(dict)
    def update_crop_overlay(self, params: dict[str, Any]) -> None:
        self._crop_overlay.set_params(params)

    @Slot(str)
    def set_subtitle_text(self, text: str) -> None:
        self._sub_overlay.setText(text)

    def current_position_ms(self) -> int:
        return self._player.position()

    @Slot(int)
    def _on_position_changed(self, ms: int) -> None:
        self.position_changed.emit(ms)

    @Slot(int)
    def _on_duration_changed(self, ms: int) -> None:
        self.duration_changed.emit(ms)

    @Slot(QMediaPlayer.PlaybackState)
    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self._control_bar.set_playing(state == QMediaPlayer.PlayingState)