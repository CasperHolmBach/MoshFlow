"""ExportBar: output settings, the Export button, progress and Cancel.

The render runs in a QThread, so the UI stays responsive and can keep editing.
Projects are immutable, so the render keeps using the version from when Export
was pressed.
"""

import os

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QWidget,
)

from moshflow.app.render_job import JobProgress, RenderResult, render_project
from moshflow.core.project import OutputSettings, Project
from moshflow.core.renderer import RenderCancelled
from moshflow.ui.model import ProjectModel

MAX_DIMENSION = 7680
MAX_FPS = 240.0


class RenderWorker(QThread):
    progressed = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, project: Project, output: str, parent=None):
        super().__init__(parent)
        self._project = project
        self._output = output
        self._job = JobProgress(self.progressed.emit)

    def cancel(self) -> None:
        self._job.cancel()

    def run(self) -> None:
        try:
            self.succeeded.emit(render_project(self._project, self._job, self._output))
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))


def _auto_spin(maximum: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(0, maximum)
    spin.setSpecialValueText("auto")
    return spin


class ExportBar(QWidget):
    def __init__(self, model: ProjectModel, parent=None):
        super().__init__(parent)
        self._model = model
        self.worker: RenderWorker | None = None

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("output.mp4")
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse)
        self.width_spin = _auto_spin(MAX_DIMENSION)
        self.height_spin = _auto_spin(MAX_DIMENSION)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.0, MAX_FPS)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSpecialValueText("auto")
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.start_export)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_export)
        self.progress = QProgressBar()
        self.status = QLabel("Ready")

        layout = QHBoxLayout(self)
        layout.addWidget(QLabel("Output"))
        layout.addWidget(self.path_edit, 2)
        layout.addWidget(browse)
        layout.addWidget(QLabel("W"))
        layout.addWidget(self.width_spin)
        layout.addWidget(QLabel("H"))
        layout.addWidget(self.height_spin)
        layout.addWidget(QLabel("FPS"))
        layout.addWidget(self.fps_spin)
        layout.addWidget(self.export_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.progress, 1)
        layout.addWidget(self.status)

        self.path_edit.textEdited.connect(self._store_settings)
        self.width_spin.valueChanged.connect(self._store_settings)
        self.height_spin.valueChanged.connect(self._store_settings)
        self.fps_spin.valueChanged.connect(self._store_settings)
        self._storing = False
        model.changed.connect(self._load_settings)      # also after Undo / Redo and Open
        self._load_settings()

    # --- settings --------------------------------------------------------
    def _load_settings(self) -> None:
        if self._storing:
            return      # the change came from these widgets; leave the text and cursor alone
        out = self._model.project.output
        for widget in (self.path_edit, self.width_spin, self.height_spin, self.fps_spin):
            widget.blockSignals(True)
        self.path_edit.setText(out.path)
        self.width_spin.setValue(out.width or 0)
        self.height_spin.setValue(out.height or 0)
        self.fps_spin.setValue(out.fps or 0.0)
        for widget in (self.path_edit, self.width_spin, self.height_spin, self.fps_spin):
            widget.blockSignals(False)

    def _settings_from_widgets(self) -> OutputSettings:
        return OutputSettings(
            path=self.path_edit.text().strip() or "output.mp4",
            width=self.width_spin.value() or None,
            height=self.height_spin.value() or None,
            fps=self.fps_spin.value() or None,
        )

    def _store_settings(self, *_) -> None:
        self._storing = True
        try:
            self._model.set_output(self._settings_from_widgets())
        finally:
            self._storing = False

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export video", self.resolved_output(), "MP4 video (*.mp4)")
        if path:
            self.path_edit.setText(path)
            self._store_settings()

    def resolved_output(self) -> str:
        """A relative output path is placed next to the first clip."""
        path = self._model.project.output.path
        clips = self._model.project.clips
        if os.path.isabs(path) or not clips:
            return path
        return os.path.join(os.path.dirname(clips[0].path), path)

    # --- export ----------------------------------------------------------
    def start_export(self) -> None:
        project = self._model.project
        if not project.clips:
            QMessageBox.information(self, "Export", "Add at least one clip to the timeline first.")
            return
        if bool(project.output.width) != bool(project.output.height):
            QMessageBox.information(self, "Export", "Set both width and height, or leave both on auto.")
            return
        output = self.resolved_output()
        if os.path.exists(output):
            answer = QMessageBox.question(self, "Export", f"{output}\nalready exists. Replace it?")
            if answer != QMessageBox.Yes:
                return
        self.run_worker(output)

    def run_worker(self, output: str) -> None:
        self.worker = RenderWorker(self._model.project, output, self)
        self.worker.progressed.connect(self._on_progress)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_failure)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self._on_finished)
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText("Rendering…")
        self.worker.start()

    def cancel_export(self) -> None:
        if self.worker:
            self.status.setText("Cancelling…")
            self.worker.cancel()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)

    def _on_success(self, result: RenderResult) -> None:
        self.status.setText(f"Done in {result.seconds:.1f} s: {os.path.basename(result.path)}")

    def _on_failure(self, message: str) -> None:
        self.status.setText("Export failed")
        QMessageBox.critical(self, "Export failed", message)

    def _on_cancelled(self) -> None:
        self.progress.setValue(0)
        self.status.setText("Cancelled")

    def _on_finished(self) -> None:
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None
