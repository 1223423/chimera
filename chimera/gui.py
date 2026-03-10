from __future__ import annotations

import sys
import time
from typing import Callable

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from .app import PreparedRender, build_fragment_glob, prepare_render
from .device import best_available_accelerator, format_bytes, list_accelerators, system_total_memory_gb
from .renderer import ChunkedMosaicRenderer, ProgressUpdate, RenderCancelled, RenderConfig, RenderSummary


APP_STYLE = """
QWidget {
    background: #f3f4f6;
    color: #1f2933;
}
QMainWindow {
    background: #f3f4f6;
    color: #1f2933;
}
QFrame#Card {
    background: #f8fafc;
    border: 1px solid #d7dee7;
    border-radius: 6px;
}
QLabel#Title {
    background: transparent;
    font-size: 24px;
    font-weight: 700;
    color: #111827;
}
QLabel#Badge {
    background: #eef2f7;
    color: #243b53;
    border: 1px solid #c9d3df;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: 600;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background: #f8fafc;
    border: none;
}
QGroupBox {
    background: #f8fafc;
    font-weight: 700;
    border: 1px solid #d7dee7;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    background: #f8fafc;
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLabel {
    background: transparent;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 16px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4c6f8c;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background: #eef2f6;
    color: #7b8794;
}
QPushButton {
    background: #dfe7ef;
    color: #102a43;
    border: 1px solid #bcccdc;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background: #d2dce7;
}
QPushButton:disabled {
    background: #edf2f7;
    color: #9aa5b1;
    border-color: #d9e2ec;
}
QPushButton#SecondaryButton {
    background: #ffffff;
    color: #243b53;
}
QPushButton#SecondaryButton:hover {
    background: #f3f6f9;
}
QPushButton#PrimaryButton {
    background: #486581;
    color: #ffffff;
    border-color: #486581;
}
QPushButton#PrimaryButton:hover {
    background: #3f5d79;
}
QPushButton#DangerButton {
    background: #b23a30;
    color: #ffffff;
    border-color: #b23a30;
}
QPushButton#DangerButton:hover {
    background: #9f342a;
}
QCheckBox {
    background: transparent;
    spacing: 6px;
}
QProgressBar {
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    border-radius: 4px;
    min-height: 14px;
    text-align: center;
}
QProgressBar::chunk {
    background: #486581;
    border-radius: 3px;
}
QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #d7dee7;
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #bcccdc;
}
"""


def rgba_to_qimage(rgba: object) -> QtGui.QImage:
    array = rgba
    if not hasattr(array, "shape"):
        raise TypeError("Expected a numpy RGBA array.")
    height, width, _ = array.shape
    bytes_per_line = width * 4
    return QtGui.QImage(array.data, width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888).copy()


def format_plan(prepared: PreparedRender) -> str:
    plan = prepared.resource_plan
    lines = [
        f"Device: {prepared.device_label}",
        f"Fragments: {prepared.library.count:,}",
        f"Source size: {prepared.source_size[0]}x{prepared.source_size[1]}",
        f"Output size: {prepared.output_size[0]}x{prepared.output_size[1]}",
        f"Chunk size: {plan.chunk_size_original} px",
        f"Placement batch: {plan.placement_batch_size:,}",
        f"Fragment block: {plan.fragment_block_size}",
        f"Fragment residency: {format_bytes(plan.memory_budget.fragment_bytes)}",
        f"Working memory budget: {format_bytes(plan.memory_budget.working_bytes)}",
    ]
    return "\n".join(lines)


class PreviewLabel(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__("Preview will appear here once rendering starts.")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(360, 360)
        self.setStyleSheet(
            "background: #ffffff; border: 1px solid #d7dee7; border-radius: 4px; color: #52606d; padding: 8px;"
        )
        self._pixmap: QtGui.QPixmap | None = None

    def set_preview(self, image: QtGui.QImage) -> None:
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self._apply_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size() - QtCore.QSize(16, 16),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class RenderWorker(QtCore.QObject):
    prepared = QtCore.Signal(object)
    progress = QtCore.Signal(int, int, str)
    preview = QtCore.Signal(QtGui.QImage)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, config: RenderConfig) -> None:
        super().__init__()
        self.config = config
        self._cancel_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        try:
            prepared = prepare_render(self.config)
            self.prepared.emit(prepared)

            renderer = ChunkedMosaicRenderer(
                config=prepared.config,
                source_image=prepared.source_image,
                library=prepared.library,
                device=prepared.device,
                resource_plan=prepared.resource_plan,
                progress_callback=self._emit_progress,
                preview_callback=self._emit_preview,
                cancel_callback=self.is_cancel_requested,
            )

            started_at = time.perf_counter()
            renderer.run()
            summary = RenderSummary(
                output_path=prepared.config.output_path,
                device_label=prepared.device_label,
                fragment_count=prepared.library.count,
                source_size=prepared.source_size,
                output_size=prepared.output_size,
                elapsed_seconds=time.perf_counter() - started_at,
                resource_plan=prepared.resource_plan,
            )
            self.finished.emit(summary)
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def _emit_progress(self, update: ProgressUpdate) -> None:
        self.progress.emit(update.completed_chunks, update.total_chunks, update.message)

    def _emit_preview(self, rgba: object) -> None:
        self.preview.emit(rgba_to_qimage(rgba))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Chimera")
        self.resize(1260, 860)
        self.setMinimumSize(1120, 780)

        self._thread: QtCore.QThread | None = None
        self._worker: RenderWorker | None = None
        self._browse_buttons: list[QtWidgets.QPushButton] = []
        self._render_running = False
        self._cancel_pending = False
        self._estimate_timer = QtCore.QTimer(self)
        self._estimate_timer.setSingleShot(True)
        self._estimate_timer.timeout.connect(self._update_output_estimate)
        self._last_canvas_path = ""
        self._last_canvas_size: tuple[int, int] | None = None

        self._build_ui()
        self._apply_defaults()
        self._wire_signals()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        header = QtWidgets.QFrame(objectName="Card")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(10)

        title = QtWidgets.QLabel("Chimera", objectName="Title")
        header_layout.addWidget(title, 1)

        self.accelerator_badge = QtWidgets.QLabel(objectName="Badge")
        self.accelerator_badge.setAlignment(QtCore.Qt.AlignCenter)
        header_layout.addWidget(self.accelerator_badge, 0, QtCore.Qt.AlignTop)
        outer.addWidget(header)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, 1)

        controls_card = QtWidgets.QFrame(objectName="Card")
        controls_layout = QtWidgets.QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        controls_layout.addWidget(scroll)

        controls_content = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QVBoxLayout(controls_content)
        self.form_layout.setSpacing(10)
        scroll.setWidget(controls_content)

        self._build_input_group()
        self._build_render_group()
        self._build_performance_group()
        self.form_layout.addStretch(1)

        body.addWidget(controls_card, 1)

        preview_card = QtWidgets.QFrame(objectName="Card")
        preview_layout = QtWidgets.QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(8)

        preview_title = QtWidgets.QLabel("Render Preview")
        preview_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #16252a;")
        preview_layout.addWidget(preview_title)

        self.preview_label = PreviewLabel()
        preview_layout.addWidget(self.preview_label, 1)

        self.plan_text = QtWidgets.QPlainTextEdit()
        self.plan_text.setReadOnly(True)
        self.plan_text.setPlaceholderText("Resolved device and plan details will appear here.")
        self.plan_text.setMaximumBlockCount(32)
        preview_layout.addWidget(self.plan_text)

        body.addWidget(preview_card, 1)

        footer = QtWidgets.QFrame(objectName="Card")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 10, 10, 10)
        footer_layout.setSpacing(10)

        footer_left = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setStyleSheet("font-weight: 600; color: #23333a;")
        footer_left.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        footer_left.addWidget(self.progress_bar)
        footer_layout.addLayout(footer_left, 1)

        button_box = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start Render")
        self.start_button.setObjectName("PrimaryButton")
        button_box.addWidget(self.start_button)
        footer_layout.addLayout(button_box)

        outer.addWidget(footer)

    def _build_input_group(self) -> None:
        group = QtWidgets.QGroupBox("Inputs")
        layout = QtWidgets.QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.canvas_edit = QtWidgets.QLineEdit()
        self.fragment_folder_edit = QtWidgets.QLineEdit()
        self.fragment_pattern_edit = QtWidgets.QLineEdit("*.png")
        self.output_edit = QtWidgets.QLineEdit()

        self._add_path_row(layout, 0, "Source image", self.canvas_edit, self._choose_canvas_file)
        self._add_path_row(layout, 1, "Fragment folder", self.fragment_folder_edit, self._choose_fragment_folder)
        layout.addWidget(QtWidgets.QLabel("Fragment pattern"), 2, 0)
        layout.addWidget(self.fragment_pattern_edit, 2, 1, 1, 2)
        self._add_path_row(layout, 3, "Output image", self.output_edit, self._choose_output_file)

        self.form_layout.addWidget(group)

    def _build_render_group(self) -> None:
        group = QtWidgets.QGroupBox("Render")
        layout = QtWidgets.QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.scaling_spin = QtWidgets.QDoubleSpinBox()
        self.scaling_spin.setRange(0.1, 100.0)
        self.scaling_spin.setDecimals(2)
        self.scaling_spin.setSingleStep(0.5)
        self.scaling_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.fragment_width_spin = QtWidgets.QSpinBox()
        self.fragment_width_spin.setRange(1, 4096)
        self.fragment_width_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.fragment_height_spin = QtWidgets.QSpinBox()
        self.fragment_height_spin.setRange(1, 4096)
        self.fragment_height_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.coverage_spin = QtWidgets.QDoubleSpinBox()
        self.coverage_spin.setRange(0.01, 1.0)
        self.coverage_spin.setDecimals(3)
        self.coverage_spin.setSingleStep(0.05)
        self.coverage_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.output_estimate_label = QtWidgets.QLabel()
        self.output_estimate_label.setStyleSheet("color: #607076;")
        self.output_estimate_label.setWordWrap(True)

        layout.addWidget(QtWidgets.QLabel("Scaling"), 0, 0)
        layout.addWidget(self.scaling_spin, 0, 1, 1, 3)
        layout.addWidget(self.output_estimate_label, 1, 1, 1, 3)

        layout.addWidget(QtWidgets.QLabel("Fragment width"), 2, 0)
        layout.addWidget(self.fragment_width_spin, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Fragment height"), 2, 2)
        layout.addWidget(self.fragment_height_spin, 2, 3)

        layout.addWidget(QtWidgets.QLabel("Target coverage"), 3, 0)
        layout.addWidget(self.coverage_spin, 3, 1, 1, 2)

        self.form_layout.addWidget(group)

    def _build_performance_group(self) -> None:
        group = QtWidgets.QGroupBox("Performance")
        layout = QtWidgets.QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])

        self.memory_spin = QtWidgets.QDoubleSpinBox()
        self.memory_spin.setRange(0.5, max(1.0, round(system_total_memory_gb(), 1)))
        self.memory_spin.setDecimals(1)
        self.memory_spin.setSingleStep(0.5)
        self.memory_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)

        self.temp_dir_edit = QtWidgets.QLineEdit()
        self.keep_temp_check = QtWidgets.QCheckBox("Keep temporary chunk buffer on disk")

        layout.addWidget(QtWidgets.QLabel("Device"), 0, 0)
        layout.addWidget(self.device_combo, 0, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Memory limit (GB)"), 1, 0)
        layout.addWidget(self.memory_spin, 1, 1, 1, 2)

        self._add_path_row(layout, 2, "Temp directory", self.temp_dir_edit, self._choose_temp_directory)
        layout.addWidget(self.keep_temp_check, 3, 0, 1, 3)

        self.device_hint_label = QtWidgets.QLabel()
        self.device_hint_label.setStyleSheet("color: #607076;")
        layout.addWidget(self.device_hint_label, 4, 0, 1, 3)

        self.form_layout.addWidget(group)

    def _wire_signals(self) -> None:
        self.start_button.clicked.connect(self._handle_action_button)
        self.device_combo.currentTextChanged.connect(self._update_device_hint)
        self.canvas_edit.textChanged.connect(self._schedule_output_estimate)
        self.scaling_spin.valueChanged.connect(self._schedule_output_estimate)

    def _apply_defaults(self) -> None:
        self.canvas_edit.setText("./canvas/puppy.jpeg")
        self.fragment_folder_edit.setText("./fragments")
        self.output_edit.setText("./output.png")

        self.scaling_spin.setValue(10.0)
        self.fragment_width_spin.setValue(32)
        self.fragment_height_spin.setValue(32)
        self.coverage_spin.setValue(1.0)

        default_memory = min(25.0, max(1.0, round(system_total_memory_gb(), 1)))
        self.memory_spin.setValue(default_memory)

        accelerator = best_available_accelerator()
        if accelerator.key == "cpu":
            self.accelerator_badge.setText("Accelerator: CPU only")
            self.accelerator_badge.setStyleSheet(
                "background: #f2f4f7; color: #52606d; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 8px; font-weight: 600;"
            )
        else:
            self.accelerator_badge.setText(f"Accelerator: {accelerator.label} detected")

        self._update_device_hint()
        self._update_output_estimate()

    def _set_form_enabled(self, enabled: bool) -> None:
        for widget in (
            self.canvas_edit,
            self.fragment_folder_edit,
            self.fragment_pattern_edit,
            self.output_edit,
            self.scaling_spin,
            self.fragment_width_spin,
            self.fragment_height_spin,
            self.coverage_spin,
            self.device_combo,
            self.memory_spin,
            self.temp_dir_edit,
            self.keep_temp_check,
        ):
            widget.setEnabled(enabled)

        for button in self._browse_buttons:
            button.setEnabled(enabled)

    def _update_device_hint(self) -> None:
        requested = self.device_combo.currentText()
        available = {status.key: status for status in list_accelerators()}

        if requested == "auto":
            chosen = best_available_accelerator()
            self.device_hint_label.setText(f"Auto will use {chosen.label}.")
            return

        if available[requested].available:
            self.device_hint_label.setText(f"{available[requested].label} is available.")
        else:
            self.device_hint_label.setText(f"{available[requested].label} is not available on this machine.")

    def _choose_canvas_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Source Image",
            self.canvas_edit.text() or ".",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.avif)",
        )
        if path:
            self.canvas_edit.setText(path)

    def _choose_fragment_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Fragment Folder",
            self.fragment_folder_edit.text() or ".",
        )
        if path:
            self.fragment_folder_edit.setText(path)

    def _choose_output_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Output Image",
            self.output_edit.text() or "output.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)",
        )
        if path:
            self.output_edit.setText(path)

    def _choose_temp_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Temporary Directory",
            self.temp_dir_edit.text() or ".",
        )
        if path:
            self.temp_dir_edit.setText(path)

    def _add_path_row(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        label_text: str,
        line_edit: QtWidgets.QLineEdit,
        chooser: Callable[[], None] | None,
    ) -> None:
        layout.addWidget(QtWidgets.QLabel(label_text), row, 0)
        layout.addWidget(line_edit, row, 1)
        button = QtWidgets.QPushButton("Browse")
        button.setObjectName("SecondaryButton")
        if chooser is not None:
            button.clicked.connect(chooser)
        layout.addWidget(button, row, 2)
        self._browse_buttons.append(button)

    def _handle_action_button(self) -> None:
        if self._render_running:
            self._cancel_render()
            return
        self._start_render()

    def _set_action_button(self, *, text: str, object_name: str, enabled: bool) -> None:
        self.start_button.setEnabled(enabled)
        self.start_button.setText(text)
        if self.start_button.objectName() != object_name:
            self.start_button.setObjectName(object_name)
            self.start_button.style().unpolish(self.start_button)
            self.start_button.style().polish(self.start_button)
            self.start_button.update()

    def _schedule_output_estimate(self) -> None:
        self._estimate_timer.start(150)

    def _load_canvas_size(self) -> tuple[int, int] | None:
        canvas_path = self.canvas_edit.text().strip()
        if not canvas_path:
            self._last_canvas_path = ""
            self._last_canvas_size = None
            return None

        if canvas_path == self._last_canvas_path:
            return self._last_canvas_size

        try:
            with Image.open(canvas_path) as image_handle:
                canvas_size = image_handle.size
        except Exception:
            canvas_size = None

        self._last_canvas_path = canvas_path
        self._last_canvas_size = canvas_size
        return canvas_size

    def _update_output_estimate(self) -> None:
        canvas_size = self._load_canvas_size()
        if canvas_size is None:
            self.output_estimate_label.setText("Resulting image resolution will appear once the source image is available.")
            return

        output_width = int(round(canvas_size[0] * self.scaling_spin.value()))
        output_height = int(round(canvas_size[1] * self.scaling_spin.value()))
        self.output_estimate_label.setText(
            f"Resulting image will be {output_width:,} x {output_height:,}."
        )

    def _build_config(self) -> RenderConfig:
        return RenderConfig(
            canvas_path=self.canvas_edit.text().strip(),
            fragments_glob=build_fragment_glob(
                self.fragment_folder_edit.text().strip(),
                self.fragment_pattern_edit.text().strip() or "*.png",
            ),
            output_path=self.output_edit.text().strip(),
            scaling=self.scaling_spin.value(),
            target_coverage=self.coverage_spin.value(),
            fragment_size=(self.fragment_width_spin.value(), self.fragment_height_spin.value()),
            memory_limit_gb=self.memory_spin.value(),
            device_request=self.device_combo.currentText(),
            temp_dir=self.temp_dir_edit.text().strip() or None,
            keep_temp=self.keep_temp_check.isChecked(),
        )

    def _start_render(self) -> None:
        try:
            config = self._build_config()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Configuration Error", str(exc))
            return

        self.progress_bar.setValue(0)
        self.status_label.setText("Preparing render...")
        self.plan_text.clear()
        self._render_running = True
        self._cancel_pending = False
        self._set_form_enabled(False)
        self._set_action_button(text="Cancel", object_name="DangerButton", enabled=True)

        self._thread = QtCore.QThread(self)
        self._worker = RenderWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.prepared.connect(self._on_prepared)
        self._worker.progress.connect(self._on_progress)
        self._worker.preview.connect(self._on_preview)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)

        self._thread.start()

    def _cancel_render(self) -> None:
        if self._worker is None or self._cancel_pending:
            return
        self._cancel_pending = True
        self.status_label.setText("Cancelling...")
        self._worker.cancel()
        self._set_action_button(text="Cancelling...", object_name="DangerButton", enabled=False)

    def _on_prepared(self, prepared: object) -> None:
        if not isinstance(prepared, PreparedRender):
            return
        self.plan_text.setPlainText(format_plan(prepared))
        self.status_label.setText(f"Rendering on {prepared.device_label}...")

    def _on_progress(self, completed: int, total: int, message: str) -> None:
        percent = 0 if total <= 0 else int((completed / total) * 100)
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_preview(self, image: QtGui.QImage) -> None:
        self.preview_label.set_preview(image)

    def _on_finished(self, summary: object) -> None:
        if not isinstance(summary, RenderSummary):
            return

        self.progress_bar.setValue(100)
        self.status_label.setText(f"Finished in {summary.elapsed_seconds:.2f}s")
        self.plan_text.appendPlainText("")
        self.plan_text.appendPlainText(f"Saved to {summary.output_path}")
        self.plan_text.appendPlainText(f"Output size: {summary.output_size[0]}x{summary.output_size[1]}")
        self.plan_text.appendPlainText(f"Fragments used: {summary.fragment_count:,}")
        QtWidgets.QMessageBox.information(self, "Render Complete", f"Saved output to:\n{summary.output_path}")

    def _on_failed(self, message: str) -> None:
        self.progress_bar.setValue(0)
        self.status_label.setText("Render failed.")
        QtWidgets.QMessageBox.critical(self, "Render Failed", message)

    def _on_cancelled(self) -> None:
        self.progress_bar.setValue(0)
        self.status_label.setText("Render cancelled.")

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

        self._render_running = False
        self._cancel_pending = False
        self._set_form_enabled(True)
        self._set_action_button(text="Start Render", object_name="PrimaryButton", enabled=True)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)
    font.setPointSize(11)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
