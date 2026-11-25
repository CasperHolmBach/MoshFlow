"""MediaBin: the imported videos. Clips are dragged from here onto the timeline."""

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from moshflow.ui.model import ProjectModel

CLIP_MIME = "application/x-moshflow-clip"
VIDEO_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)"


def local_paths(mime: QMimeData) -> list[str]:
    return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]


class _MediaList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SingleSelection)

    def mimeTypes(self):
        return [CLIP_MIME]

    def mimeData(self, items):
        mime = QMimeData()
        if items:
            mime.setData(CLIP_MIME, items[0].data(Qt.UserRole).encode("utf-8"))
        return mime


class MediaBin(QWidget):
    errors = Signal(list)

    def __init__(self, model: ProjectModel, parent=None):
        super().__init__(parent)
        self._model = model
        self.setAcceptDrops(True)

        self.list = _MediaList()
        self.list.itemDoubleClicked.connect(lambda item: model.add_clip(item.data(Qt.UserRole)))
        import_button = QPushButton("+ Import")
        import_button.clicked.connect(self._choose_files)
        add_button = QPushButton("Add to timeline")
        add_button.clicked.connect(self._add_selected)

        buttons = QHBoxLayout()
        buttons.addWidget(import_button)
        buttons.addWidget(add_button)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Media Bin</b>"))
        layout.addWidget(self.list)
        layout.addLayout(buttons)

        model.mediaChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for info in self._model.media:
            # Size and frame rate are set for the whole project, in the export bar.
            item = QListWidgetItem(f"{info.name}\n{info.duration:.1f} s")
            item.setData(Qt.UserRole, info.path)
            self.list.addItem(item)

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import videos", "", VIDEO_FILTER)
        self._import(paths)

    def _import(self, paths: list[str]) -> None:
        errors = self._model.import_media(paths)
        if errors:
            self.errors.emit(errors)

    def _add_selected(self) -> None:
        item = self.list.currentItem()
        if item:
            self._model.add_clip(item.data(Qt.UserRole))

    def dragEnterEvent(self, event) -> None:
        if local_paths(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self._import(local_paths(event.mimeData()))
        event.acceptProposedAction()
