"""MainWindow: puts the views together and handles the File menu."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QSplitter, QVBoxLayout, QWidget

from moshflow.app.catalog import EffectCatalog
from moshflow.app.store import EXTENSION
from moshflow.ui.export_bar import ExportBar
from moshflow.ui.inspector import Inspector
from moshflow.ui.media_bin import MediaBin
from moshflow.ui.model import ProjectModel
from moshflow.ui.timeline_view import TimelineView

PROJECT_FILTER = f"MoshFlow project (*{EXTENSION})"
CLOSE_WAIT_MS = 10_000


class MainWindow(QMainWindow):
    def __init__(self, catalog: EffectCatalog | None = None):
        super().__init__()
        self.catalog = catalog or EffectCatalog()
        self.model = ProjectModel(self)
        self.media_bin = MediaBin(self.model)
        self.inspector = Inspector(self.model, self.catalog)
        self.timeline = TimelineView(self.model, self.catalog)
        self.export_bar = ExportBar(self.model)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.media_bin)
        top.addWidget(self.inspector)
        top.setStretchFactor(0, 2)
        top.setStretchFactor(1, 3)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(top, 1)
        layout.addWidget(self.timeline)
        layout.addWidget(self.export_bar)
        self.setCentralWidget(central)
        self.resize(1100, 680)

        self._build_menu()
        self.media_bin.errors.connect(self._show_errors)
        self.timeline.errors.connect(self._show_errors)
        self.model.changed.connect(self._update_title)
        self._update_title()

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        for text, shortcut, handler in [
            ("&New", QKeySequence.New, self.new_project),
            ("&Open…", QKeySequence.Open, self.open_project),
            ("&Save", QKeySequence.Save, self.save_project),
            ("Save &As…", QKeySequence.SaveAs, self.save_project_as),
            ("&Quit", QKeySequence.Quit, self.close),
        ]:
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _=False, h=handler: h())
            menu.addAction(action)
        render = self.menuBar().addMenu("&Render")
        export = QAction("&Export", self)
        export.setShortcut("Ctrl+E")
        export.triggered.connect(lambda _=False: self.export_bar.start_export())
        render.addAction(export)

    def show_load_warnings(self) -> None:
        """Tells the user about effect files that could not be loaded."""
        if self.catalog.failures:
            details = "\n\n".join(f"{f.module}:\n{f.error}" for f in self.catalog.failures)
            QMessageBox.warning(self, "Some effects could not be loaded", details)

    def _show_errors(self, errors: list[str]) -> None:
        QMessageBox.warning(self, "MoshFlow", "\n".join(errors))

    def _update_title(self) -> None:
        name = self.model.path or "Untitled"
        self.setWindowTitle(f"{name}{' *' if self.model.dirty else ''} - MoshFlow")

    # --- project files ---------------------------------------------------
    def new_project(self) -> None:
        if self._confirm_discard():
            self.model.new_project()

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
        if not path:
            return
        try:
            errors = self.model.open_project(path)
        except (OSError, ValueError, TypeError, KeyError) as e:
            QMessageBox.critical(self, "Open project", f"Cannot open {path}:\n{e}")
            return
        if errors:
            self._show_errors(errors)

    def save_project(self) -> bool:
        if not self.model.path:
            return self.save_project_as()
        return self._save_to(self.model.path)

    def save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.model.path or f"project{EXTENSION}", PROJECT_FILTER)
        return self._save_to(path) if path else False

    def _save_to(self, path: str) -> bool:
        try:
            self.model.save_project(path)
        except OSError as e:
            QMessageBox.critical(self, "Save project", f"Cannot save {path}:\n{e}")
            return False
        self._update_title()
        return True

    def _confirm_discard(self) -> bool:
        if not self.model.dirty:
            return True
        answer = QMessageBox.question(self, "MoshFlow", "Save changes to the current project?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self.save_project()
        return answer == QMessageBox.Discard

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        worker = self.export_bar.worker
        if worker:
            worker.cancel()
            worker.wait(CLOSE_WAIT_MS)
        event.accept()


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MoshFlow")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    window.show_load_warnings()
    return app.exec()
