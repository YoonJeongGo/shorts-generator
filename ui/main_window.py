# ui/main_window.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ui.app_state import AppState
from ui.crop_editor import CropEditor
from ui.highlight_editor import HighlightEditor
from ui.left_panel import LeftPanel
from ui.multicut_editor import MulticutEditor
from ui.player_panel import PlayerPanel
from ui.subtitle_editor import SubtitleEditor
from ui.timeline_widget import TimelineWidget
from ui.workers import ExportWorker, HighlightWorker, SttWorker


class MainWindow(QMainWindow):
    """
    앱 최상위 윈도우.
    모든 상태 변경은 MainWindow -> AppState를 거치고,
    각 위젯은 MainWindow가 state 기준으로 refresh 한다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ShortsCut")
        self.setMinimumSize(1200, 760)

        self.state = AppState(self)

        self._stt_thread: QThread | None = None
        self._hl_thread: QThread | None = None
        self._export_thread: QThread | None = None

        self._syncing_subtitles = False
        self._syncing_highlights = False
        self._syncing_multicut = False

        self._build_ui()
        self._connect_signals()

        self.state.set_crop_params(self._crop_editor.get_crop_params())
        self._player_panel.update_crop_overlay(self.state.crop_params)

        self._refresh_all_from_state()
        self._update_actions()

    # ── UI 구성 ──────────────────────────────

    def _build_ui(self) -> None:
        self._build_menubar()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

    def _build_menubar(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("파일(&F)")
        self._act_open = file_menu.addAction("영상 열기(&O)")
        self._act_open.setShortcut("Ctrl+O")
        self._act_load_srt = file_menu.addAction("SRT 불러오기(&S)")
        file_menu.addSeparator()
        self._act_save_project = file_menu.addAction("프로젝트 저장")
        self._act_save_project.setShortcut("Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("종료(&Q)", self.close, "Ctrl+Q")

        edit_menu = mb.addMenu("편집(&E)")
        self._act_add_cut = edit_menu.addAction("컷 추가 (I~O 구간)")
        self._act_add_cut.setShortcut("Return")

        settings_menu = mb.addMenu("설정(&S)")
        settings_menu.addAction("환경 설정", self._on_preferences)

        help_menu = mb.addMenu("도움말(&H)")
        help_menu.addAction("버전 정보", self._on_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("메인 툴바", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        self._act_tb_open = tb.addAction("📂 영상 열기")
        tb.addSeparator()
        self._act_tb_stt = tb.addAction("🎙 STT 실행")
        tb.addSeparator()
        self._act_tb_cut = tb.addAction("✂ 컷 추가")
        tb.addSeparator()
        self._act_tb_export = tb.addAction("🎬 내보내기")

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        self._lbl_timecode = QLabel("00:00:00")
        self._lbl_timecode.setStyleSheet(
            "font-family: monospace; font-size: 14px; padding-right: 8px;"
        )
        tb.addWidget(self._lbl_timecode)

    def _build_central(self) -> None:
        self._player_panel = PlayerPanel(self)
        self._timeline_widget = TimelineWidget(self)
        self._subtitle_editor = SubtitleEditor(self)
        self._highlight_editor = HighlightEditor(self)
        self._multicut_editor = MulticutEditor(self)
        self._crop_editor = CropEditor(self)
        self._left_panel = LeftPanel(self)

        self._right_tabs = QTabWidget()
        self._right_tabs.setMinimumWidth(340)
        self._right_tabs.setMaximumWidth(420)
        self._right_tabs.addTab(self._subtitle_editor, "① 자막")
        self._right_tabs.addTab(self._highlight_editor, "② 하이라이트")
        self._right_tabs.addTab(self._multicut_editor, "③ 멀티컷")
        self._right_tabs.addTab(self._crop_editor, "④ 크롭")

        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.addWidget(self._left_panel)
        h_splitter.addWidget(self._player_panel)
        h_splitter.addWidget(self._right_tabs)
        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        h_splitter.setStretchFactor(2, 0)
        h_splitter.setSizes([240, 820, 360])

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        main_layout.addWidget(h_splitter, stretch=1)
        main_layout.addWidget(self._timeline_widget)

        self.setCentralWidget(main_widget)

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)

        self._lbl_status = QLabel("준비")
        self._lbl_duration = QLabel("영상: --")
        self._lbl_segment = QLabel("세그먼트: --")
        self._lbl_multicut = QLabel("멀티컷: --")

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(180)
        self._progress_bar.setVisible(False)

        sb.addWidget(self._lbl_status)
        sb.addPermanentWidget(self._lbl_duration)
        sb.addPermanentWidget(self._lbl_segment)
        sb.addPermanentWidget(self._lbl_multicut)
        sb.addPermanentWidget(self._progress_bar)

    # ── 연결 ─────────────────────────────────

    def _connect_signals(self) -> None:
        self._act_open.triggered.connect(self._on_open_video)
        self._act_tb_open.triggered.connect(self._on_open_video)
        self._act_load_srt.triggered.connect(self._on_load_srt)
        self._act_tb_stt.triggered.connect(self._on_run_stt)
        self._act_tb_cut.triggered.connect(self._on_add_cut)
        self._act_tb_export.triggered.connect(self._on_export)
        self._act_add_cut.triggered.connect(self._on_add_cut)

        self._player_panel.position_changed.connect(self._on_player_position_changed)
        self._player_panel.duration_changed.connect(self._on_duration_changed)
        self._player_panel.set_in_point.connect(self._on_set_in_point)
        self._player_panel.set_out_point.connect(self._on_set_out_point)

        self._timeline_widget.seek_requested.connect(self._player_panel.seek_to_ms)
        self._timeline_widget.clip_selected.connect(self._on_timeline_clip_selected)
        self._timeline_widget.clip_range_changed.connect(self._on_timeline_clip_range_changed)
        self._timeline_widget.subtitle_clicked.connect(self._subtitle_editor.select_row_by_index)

        self._subtitle_editor.row_seek_requested.connect(self._player_panel.seek_to_ms)
        self._subtitle_editor.segments_changed.connect(self._on_subtitles_edited)

        self._highlight_editor.clip_seek_requested.connect(self._player_panel.seek_to_ms)
        self._highlight_editor.add_to_multicut.connect(self._on_add_highlight_to_multicut)
        self._highlight_editor.highlights_changed.connect(self._on_highlights_edited)

        self._multicut_editor.clips_changed.connect(self._on_multicut_changed)
        self._multicut_editor.export_requested.connect(self._on_export)

        self._left_panel.clip_selected.connect(self._on_left_panel_clip_selected)
        self._left_panel.export_requested.connect(self._on_export)

        self._crop_editor.crop_params_changed.connect(self._on_crop_params_changed)

    # ── 키 이벤트 ────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        mod = event.modifiers()

        if key == Qt.Key_Space:
            self._player_panel.toggle_play_pause()
            return

        if key == Qt.Key_Left:
            delta = 5000 if mod & Qt.ShiftModifier else 1000
            self._player_panel.step_ms(-delta)
            return

        if key == Qt.Key_Right:
            delta = 5000 if mod & Qt.ShiftModifier else 1000
            self._player_panel.step_ms(delta)
            return

        if key == Qt.Key_I:
            self._on_set_in_point(self.state.current_position_ms)
            return

        if key == Qt.Key_O:
            self._on_set_out_point(self.state.current_position_ms)
            return

        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._on_add_cut()
            return

        super().keyPressEvent(event)

    # ── 파일 로드 ────────────────────────────

    @Slot()
    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 선택",
            "",
            "영상 파일 (*.mp4 *.mov *.avi *.mkv *.webm)",
        )
        if not path:
            return

        current_crop = self._crop_editor.get_crop_params()

        self.state.reset()
        self.state.set_video_path(path)
        self.state.set_crop_params(current_crop)

        self._player_panel.load_video(path)
        self._player_panel.update_crop_overlay(self.state.crop_params)

        self._refresh_all_from_state()
        self._set_status(f"영상 로드: {Path(path).name}")
        self._update_actions()

    @Slot()
    def _on_load_srt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "SRT 파일 선택",
            "",
            "자막 파일 (*.srt)",
        )
        if not path:
            return

        try:
            from main import parse_srt_file
            segments = parse_srt_file(path)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"SRT 파싱 실패:\n{e}")
            return

        self.state.set_segments(segments)
        self._refresh_subtitles_from_state()
        self._set_status(f"SRT 로드 완료: {len(self.state.segments)}개 자막")

    # ── STT / 하이라이트 ────────────────────

    @Slot()
    def _on_run_stt(self) -> None:
        if not self.state.video_path:
            QMessageBox.warning(self, "경고", "영상을 먼저 불러오세요.")
            return

        self._start_progress("STT 분석 중...")
        self._act_tb_stt.setEnabled(False)

        self._stt_thread = QThread(self)
        self._stt_worker = SttWorker(self.state.video_path)
        self._stt_worker.moveToThread(self._stt_thread)

        self._stt_thread.started.connect(self._stt_worker.run)
        self._stt_worker.finished.connect(self._on_stt_finished)
        self._stt_worker.error.connect(self._on_worker_error)
        self._stt_worker.finished.connect(self._stt_thread.quit)
        self._stt_thread.finished.connect(self._stt_thread.deleteLater)

        self._stt_thread.start()

    @Slot(list)
    def _on_stt_finished(self, segments: list[dict[str, Any]]) -> None:
        self.state.set_segments(segments)
        self._refresh_subtitles_from_state()

        self._stop_progress()
        self._act_tb_stt.setEnabled(True)
        self._set_status(f"STT 완료 — {len(self.state.segments)}개 세그먼트")

        self._run_highlight_worker(self.state.segments)

    def _run_highlight_worker(self, segments: list[dict[str, Any]]) -> None:
        self._start_progress("하이라이트 분석 중...")

        self._hl_thread = QThread(self)
        self._hl_worker = HighlightWorker(segments)
        self._hl_worker.moveToThread(self._hl_thread)

        self._hl_thread.started.connect(self._hl_worker.run)
        self._hl_worker.finished.connect(self._on_highlight_finished)
        self._hl_worker.error.connect(self._on_worker_error)
        self._hl_worker.finished.connect(self._hl_thread.quit)
        self._hl_thread.finished.connect(self._hl_thread.deleteLater)

        self._hl_thread.start()

    @Slot(list)
    def _on_highlight_finished(self, highlights: list[dict[str, Any]]) -> None:
        self.state.set_highlights(highlights)
        self._refresh_highlights_from_state()
        self._refresh_timeline_clips_from_state()
        self._refresh_left_panel_from_state()

        self._stop_progress()
        self._set_status(f"하이라이트 {len(self.state.highlights)}개 추출 완료")
        self._right_tabs.setCurrentIndex(1)

    # ── 재생 / 자막 / 타임라인 동기화 ───────

    @Slot(int)
    def _on_player_position_changed(self, ms: int) -> None:
        self.state.set_current_position_ms(ms)

        h = ms // 3_600_000
        m = (ms % 3_600_000) // 60_000
        s = (ms % 60_000) // 1000
        self._lbl_timecode.setText(f"{h:02}:{m:02}:{s:02}")

        self._timeline_widget.set_playhead_ms(ms)
        self._subtitle_editor.highlight_row_at_ms(ms)
        self._subtitle_editor.set_player_position_ms(ms)
        self._highlight_editor.set_current_player_ms(ms)

        self._sync_player_subtitle_overlay()

    @Slot(int)
    def _on_duration_changed(self, ms: int) -> None:
        self.state.set_duration_ms(ms)
        self._timeline_widget.set_duration_ms(ms)

        h = ms // 3_600_000
        m = (ms % 3_600_000) // 60_000
        s = (ms % 60_000) // 1000
        self._lbl_duration.setText(f"길이: {h:02}:{m:02}:{s:02}")

    @Slot()
    def _on_subtitles_edited(self) -> None:
        if self._syncing_subtitles:
            return

        self.state.set_segments(self._subtitle_editor.get_segments())
        self._timeline_widget.set_subtitles(self.state.segments)
        self._lbl_segment.setText(f"세그먼트: {len(self.state.segments)}개")
        self._sync_player_subtitle_overlay()

    @Slot(int)
    def _on_set_in_point(self, ms: int) -> None:
        self.state.set_in_point_ms(ms)
        self._timeline_widget.set_in_point_ms(ms)

    @Slot(int)
    def _on_set_out_point(self, ms: int) -> None:
        self.state.set_out_point_ms(ms)
        self._timeline_widget.set_out_point_ms(ms)

    @Slot()
    def _on_add_cut(self) -> None:
        in_ms = self.state.in_point_ms
        out_ms = self.state.out_point_ms

        if out_ms <= in_ms:
            QMessageBox.warning(self, "경고", "I/O 구간이 올바르지 않습니다.")
            return

        clip = {
            "start": in_ms / 1000.0,
            "end": out_ms / 1000.0,
            "score": 0.0,
            "reasons": ["수동 추가"],
            "text": "",
        }
        self.state.add_manual_clip(clip)

        self._refresh_highlights_from_state()
        self._refresh_timeline_clips_from_state()
        self._refresh_left_panel_from_state()
        self._set_status("수동 컷 추가 완료")

    @Slot(dict)
    def _on_timeline_clip_selected(self, clip: dict[str, Any]) -> None:
        ms = int(float(clip.get("start", 0.0)) * 1000)
        self._player_panel.seek_to_ms(ms)
        self._highlight_editor.select_clip_by_start_ms(ms)

    @Slot(dict)
    def _on_timeline_clip_range_changed(self, data: dict[str, Any]) -> None:
        index = int(data.get("index", -1))
        start = float(data.get("start", 0.0))
        end = float(data.get("end", 0.0))

        self.state.update_clip_range_by_index(index, start, end)

        self._refresh_highlights_from_state()
        self._refresh_left_panel_from_state()
        self._refresh_multicut_from_state()
        self._timeline_widget.set_clips(self.state.timeline_clips)

    # ── 하이라이트 / 멀티컷 ─────────────────

    @Slot(list)
    def _on_highlights_edited(self, highlights: list[dict[str, Any]]) -> None:
        if self._syncing_highlights:
            return

        self.state.set_highlights(highlights)
        self._refresh_timeline_clips_from_state()
        self._refresh_left_panel_from_state()

    @Slot(dict)
    def _on_add_highlight_to_multicut(self, clip: dict[str, Any]) -> None:
        self.state.append_multicut_clip(clip)
        self._refresh_multicut_from_state()
        self._set_status("멀티컷에 구간 추가됨")

    @Slot(list)
    def _on_multicut_changed(self, clips: list[dict[str, Any]]) -> None:
        if self._syncing_multicut:
            return

        self.state.set_multicut_clips(clips)
        self._refresh_left_panel_multicut_only()

    @Slot(int)
    def _on_left_panel_clip_selected(self, ms: int) -> None:
        self._player_panel.seek_to_ms(ms)
        self._highlight_editor.select_clip_by_start_ms(ms)

    # ── 크롭 / 내보내기 ─────────────────────

    @Slot(dict)
    def _on_crop_params_changed(self, params: dict[str, Any]) -> None:
        self.state.set_crop_params(params)
        self._player_panel.update_crop_overlay(self.state.crop_params)

    @Slot()
    def _on_export(self) -> None:
        if not self.state.video_path:
            QMessageBox.warning(self, "경고", "영상을 먼저 불러오세요.")
            return

        self._pull_editor_state_into_app_state()

        if not self.state.multicut_clips:
            QMessageBox.warning(self, "경고", "멀티컷 구간이 없습니다.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "저장 폴더 선택")
        if not out_dir:
            return

        export_payload = self.state.build_export_payload()

        self._start_progress("내보내기 중...")

        self._export_thread = QThread(self)
        self._export_worker = ExportWorker(
            export_state=export_payload,
            out_dir=out_dir,
            concat_mode="auto",
        )
        self._export_worker.moveToThread(self._export_thread)

        self._export_thread.started.connect(self._export_worker.run)
        self._export_worker.progress.connect(self._progress_bar.setValue)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_worker_error)
        self._export_worker.finished.connect(self._export_thread.quit)
        self._export_thread.finished.connect(self._export_thread.deleteLater)

        self._progress_bar.setRange(0, 100)
        self._export_thread.start()

    @Slot(str)
    def _on_export_finished(self, output_path: str) -> None:
        self._stop_progress()
        self._set_status(f"내보내기 완료: {output_path}")
        QMessageBox.information(self, "완료", f"저장 완료:\n{output_path}")

    # ── 에러 ────────────────────────────────

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        self._stop_progress()
        self._act_tb_stt.setEnabled(True)
        self._set_status(f"오류: {msg}")
        QMessageBox.critical(self, "오류", msg)

    # ── refresh helpers ─────────────────────

    def _refresh_all_from_state(self) -> None:
        self._refresh_subtitles_from_state()
        self._refresh_highlights_from_state()
        self._refresh_timeline_clips_from_state()
        self._refresh_multicut_from_state()
        self._player_panel.update_crop_overlay(self.state.crop_params)
        self._timeline_widget.set_in_point_ms(self.state.in_point_ms)
        self._timeline_widget.set_out_point_ms(self.state.out_point_ms)
        self._timeline_widget.set_duration_ms(self.state.duration_ms)
        self._timeline_widget.set_playhead_ms(self.state.current_position_ms)
        self._sync_player_subtitle_overlay()

    def _refresh_subtitles_from_state(self) -> None:
        self._syncing_subtitles = True
        try:
            self._subtitle_editor.load_segments(self.state.segments)
            self._timeline_widget.set_subtitles(self.state.segments)
            self._lbl_segment.setText(f"세그먼트: {len(self.state.segments)}개")
        finally:
            self._syncing_subtitles = False

    def _refresh_highlights_from_state(self) -> None:
        self._syncing_highlights = True
        try:
            self._highlight_editor.set_highlights(self.state.highlights)
            self._highlight_editor.set_current_player_ms(self.state.current_position_ms)
        finally:
            self._syncing_highlights = False

    def _refresh_timeline_clips_from_state(self) -> None:
        self._timeline_widget.set_clips(self.state.timeline_clips)
        self._timeline_widget.set_in_point_ms(self.state.in_point_ms)
        self._timeline_widget.set_out_point_ms(self.state.out_point_ms)

    def _refresh_left_panel_from_state(self) -> None:
        self._left_panel.load_clips(self.state.timeline_clips)
        self._refresh_left_panel_multicut_only()

    def _refresh_left_panel_multicut_only(self) -> None:
        self._left_panel.set_multicut_clips(self.state.multicut_clips)
        total = sum(max(0.0, c["end"] - c["start"]) for c in self.state.multicut_clips)
        self._lbl_multicut.setText(
            f"멀티컷: {len(self.state.multicut_clips)}개 / {total:.1f}초"
        )

    def _refresh_multicut_from_state(self) -> None:
        self._syncing_multicut = True
        try:
            self._multicut_editor.set_clips(self.state.multicut_clips)
            self._refresh_left_panel_multicut_only()
        finally:
            self._syncing_multicut = False

    def _pull_editor_state_into_app_state(self) -> None:
        self.state.set_segments(self._subtitle_editor.get_segments())
        self.state.set_highlights(self._highlight_editor.get_highlights())
        self.state.set_multicut_clips(self._multicut_editor.get_clips())
        self.state.set_crop_params(self._crop_editor.get_crop_params())

    def _sync_player_subtitle_overlay(self) -> None:
        current_sec = self.state.current_position_ms / 1000.0
        text = ""
        for seg in self.state.segments:
            if float(seg["start"]) <= current_sec <= float(seg["end"]):
                text = str(seg.get("text", ""))
                break
        self._player_panel.set_subtitle_text(text)

    # ── misc ────────────────────────────────

    def _set_status(self, text: str) -> None:
        self._lbl_status.setText(text)

    def _start_progress(self, msg: str) -> None:
        self._set_status(msg)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(True)

    def _stop_progress(self) -> None:
        self._progress_bar.setVisible(False)

    def _update_actions(self) -> None:
        has_video = self.state.video_path is not None
        self._act_tb_stt.setEnabled(has_video)
        self._act_tb_cut.setEnabled(has_video)
        self._act_tb_export.setEnabled(has_video)

    def _on_preferences(self) -> None:
        QMessageBox.information(self, "설정", "환경 설정은 추후 구현 예정입니다.")

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "ShortsCut",
            "ShortsCut v0.2\nAI 보조 쇼츠 편집기\nPySide6 Desktop MVP",
        )