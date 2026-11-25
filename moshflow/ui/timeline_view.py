"""TimelineView: clips from left to right with a transition chip on every cut.

Click selects a clip or a chip, dragging a clip reorders it, Delete removes it,
and clips can be dropped in from the media bin or the file manager.
"""

import os

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView

from moshflow.app.catalog import EffectCatalog
from moshflow.ui.media_bin import CLIP_MIME, local_paths
from moshflow.ui.model import CLIP, TRANSITION, ProjectModel

PIXELS_PER_SECOND = 28
MIN_CLIP_WIDTH = 110
MAX_CLIP_WIDTH = 420
CHIP_WIDTH = 104
CHIP_GAP = 4
CLIP_HEIGHT = 56
CHIP_HEIGHT = 38

CLIP_COLOR = QColor("#3D3850")
CHIP_COLOR = QColor("#C2286A")
HARD_CUT_COLOR = QColor("#5A5468")
SELECTED_COLOR = QColor("#2DE2E6")
TEXT_COLOR = QColor("#FFFFFF")
HINT_COLOR = QColor("#8F8AA0")


class TimelineView(QGraphicsView):
    errors = Signal(list)

    def __init__(self, model: ProjectModel, catalog: EffectCatalog, parent=None):
        super().__init__(parent)
        self._model = model
        self._catalog = catalog
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setRenderHint(QPainter.Antialiasing)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setFixedHeight(112)
        self.setBackgroundBrush(QBrush(QColor("#1C1924")))

        self.regions: list[tuple[QRectF, str, int]] = []
        self._clip_rects: list[QRectF] = []
        self._press_pos: QPointF | None = None
        self._drag_clip: int | None = None
        self._marker = None

        model.changed.connect(self.rebuild)
        model.selectionChanged.connect(self.rebuild)
        self.rebuild()

    # --- drawing ---------------------------------------------------------
    def rebuild(self) -> None:
        self._scene.clear()
        self._marker = None
        self.regions, self._clip_rects = [], []
        project = self._model.project
        if not project.clips:
            hint = QGraphicsSimpleTextItem("Drag clips here from the media bin")
            hint.setBrush(QBrush(HINT_COLOR))
            self._scene.addItem(hint)
            self._scene.setSceneRect(self._scene.itemsBoundingRect())
            return

        x = 0.0
        for i, clip in enumerate(project.clips):
            seconds = self._model.clip_duration(i)
            width = min(MAX_CLIP_WIDTH, max(MIN_CLIP_WIDTH, seconds * PIXELS_PER_SECOND))
            rect = QRectF(x, -CLIP_HEIGHT / 2, width, CLIP_HEIGHT)
            self._clip_rects.append(rect)
            self._add_block(rect, CLIP_COLOR, (CLIP, i), f"{os.path.basename(clip.path)}\n{seconds:.1f} s")
            x += width + CHIP_GAP
            if i < len(project.transitions):
                transition = project.transitions[i]
                chip = QRectF(x, -CHIP_HEIGHT / 2, CHIP_WIDTH, CHIP_HEIGHT)
                color = HARD_CUT_COLOR if transition.effect == "hard_cut" else CHIP_COLOR
                self._add_block(chip, color, (TRANSITION, i),
                                f"{self._effect_name(transition.effect)}\n{transition.duration:g} s")
                x += CHIP_WIDTH + CHIP_GAP
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-10, -8, 10, 8))

    def _add_block(self, rect: QRectF, color: QColor, target: tuple[str, int], label: str) -> None:
        selected = self._model.selection == target
        pen = QPen(SELECTED_COLOR, 2.5) if selected else QPen(Qt.NoPen)
        block = self._scene.addRect(rect, pen, QBrush(color))
        metrics = QFontMetrics(self.font())
        lines = [metrics.elidedText(line, Qt.ElideMiddle, int(rect.width()) - 14) for line in label.split("\n")]
        text = QGraphicsSimpleTextItem("\n".join(lines), block)
        text.setBrush(QBrush(TEXT_COLOR))
        box = text.boundingRect()
        text.setPos(rect.left() + 8, rect.center().y() - box.height() / 2)
        self.regions.append((rect, *target))

    def _effect_name(self, effect_id: str) -> str:
        try:
            return self._catalog.get(effect_id).name
        except KeyError:
            return f"{effect_id}?"

    # --- hit testing -----------------------------------------------------
    def _hit(self, scene_pos: QPointF) -> tuple[str, int] | None:
        for rect, kind, index in self.regions:
            if rect.contains(scene_pos):
                return kind, index
        return None

    def _insert_index(self, x: float) -> int:
        return sum(1 for rect in self._clip_rects if rect.center().x() < x)

    def _marker_x(self, index: int) -> float:
        if not self._clip_rects:
            return 0.0
        if index >= len(self._clip_rects):
            return self._clip_rects[-1].right() + CHIP_GAP
        return self._clip_rects[index].left() - CHIP_GAP / 2

    def _show_marker(self, index: int) -> None:
        self._hide_marker()
        x = self._marker_x(index)
        self._marker = self._scene.addLine(x, -CLIP_HEIGHT / 2 - 4, x, CLIP_HEIGHT / 2 + 4, QPen(SELECTED_COLOR, 3))

    def _hide_marker(self) -> None:
        if self._marker is not None:
            self._scene.removeItem(self._marker)
            self._marker = None

    # --- mouse and keyboard ----------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        self.setFocus()
        hit = self._hit(self.mapToScene(event.position().toPoint()))
        self._press_pos = event.position()
        self._drag_clip = hit[1] if hit and hit[0] == CLIP else None
        if hit:
            self._model.select(*hit)
        else:
            self._model.select(None)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_clip is None or self._press_pos is None:
            return
        if (event.position() - self._press_pos).manhattanLength() < QApplication.startDragDistance():
            return
        scene_x = self.mapToScene(event.position().toPoint()).x()
        self._show_marker(self._insert_index(scene_x))

    def mouseReleaseEvent(self, event) -> None:
        moving = self._marker is not None and self._drag_clip is not None
        self._hide_marker()
        src, self._drag_clip, self._press_pos = self._drag_clip, None, None
        if moving:
            target = self._insert_index(self.mapToScene(event.position().toPoint()).x())
            if target > src:
                target -= 1
            if target != src:
                self._model.move_clip(src, target)

    def keyPressEvent(self, event) -> None:
        selection = self._model.selection
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and selection and selection[0] == CLIP:
            self._model.remove_clip(selection[1])
        else:
            super().keyPressEvent(event)

    # --- drops -----------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(CLIP_MIME) or local_paths(event.mimeData()):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        self._show_marker(self._insert_index(self.mapToScene(event.position().toPoint()).x()))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._hide_marker()

    def dropEvent(self, event) -> None:
        self._hide_marker()
        index = self._insert_index(self.mapToScene(event.position().toPoint()).x())
        mime = event.mimeData()
        if mime.hasFormat(CLIP_MIME):
            paths = [bytes(mime.data(CLIP_MIME)).decode("utf-8")]
        else:
            paths = local_paths(mime)
            errors = self._model.import_media(paths)
            if errors:
                self.errors.emit(errors)
            paths = [p for p in paths if self._model.info(p)]
        for offset, path in enumerate(paths):
            self._model.add_clip(path, index + offset)
        event.acceptProposedAction()
