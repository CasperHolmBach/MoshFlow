"""Inspector: settings of the selected transition or clip.

The controls of an effect are generated from its param_specs(), so a new effect
gets its UI without any change here.
"""

import os
from dataclasses import replace
from typing import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from moshflow.api import BoolParam, ChoiceParam, FloatParam, IntParam, ParamSpec, validate_params
from moshflow.app.catalog import EffectCatalog
from moshflow.core.project import ALIGNMENTS, Transition
from moshflow.ui.model import CLIP, TRANSITION, ProjectModel

ALIGNMENT_LABELS = {"before": "Before the cut", "center": "Centred on the cut", "after": "After the cut"}
SLIDER_STEPS = 1000
MAX_DURATION = 60.0
MAX_SEED = 999_999


def make_param_widget(spec: ParamSpec, value: object, on_change: Callable[[object], None]) -> QWidget:
    """Widget factory: one control per parameter type."""
    if isinstance(spec, BoolParam):
        box = QCheckBox()
        box.setChecked(bool(value))
        box.toggled.connect(on_change)
        return box
    if isinstance(spec, ChoiceParam):
        combo = QComboBox()
        combo.addItems(spec.options)
        combo.setCurrentText(str(value))
        combo.currentTextChanged.connect(on_change)
        return combo
    if isinstance(spec, IntParam):
        spin = QSpinBox()
        spin.setRange(spec.min, spec.max)
        spin.setValue(int(value))
        spin.valueChanged.connect(on_change)
        return spin
    if isinstance(spec, FloatParam):
        return _make_float_widget(spec, float(value), on_change)
    raise TypeError(f"unknown parameter type: {spec!r}")


def _make_float_widget(spec: FloatParam, value: float, on_change: Callable[[object], None]) -> QWidget:
    span = (spec.max - spec.min) or 1.0
    spin = QDoubleSpinBox()
    spin.setDecimals(3)
    spin.setRange(spec.min, spec.max)
    spin.setSingleStep(span / 100)
    spin.setValue(value)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, SLIDER_STEPS)
    slider.setValue(round((value - spec.min) / span * SLIDER_STEPS))

    def from_slider(position: int) -> None:
        spin.setValue(spec.min + position / SLIDER_STEPS * span)

    def from_spin(new_value: float) -> None:
        slider.blockSignals(True)
        slider.setValue(round((new_value - spec.min) / span * SLIDER_STEPS))
        slider.blockSignals(False)
        on_change(new_value)

    slider.valueChanged.connect(from_slider)
    spin.valueChanged.connect(from_spin)
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(slider, 1)
    layout.addWidget(spin)
    return row


class Inspector(QWidget):
    def __init__(self, model: ProjectModel, catalog: EffectCatalog, parent=None):
        super().__init__(parent)
        self._model = model
        self._catalog = catalog
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Inspector</b>"))
        layout.addWidget(self._scroll)
        model.selectionChanged.connect(self.rebuild)
        self.rebuild()

    def rebuild(self) -> None:
        selection = self._model.selection
        if selection and selection[0] == TRANSITION:
            content = self._transition_page(selection[1])
        elif selection and selection[0] == CLIP:
            content = self._clip_page(selection[1])
        else:
            content = QLabel("Select a clip or a transition on the timeline.")
            content.setAlignment(Qt.AlignCenter)
            content.setWordWrap(True)
        self._show_page(content)

    def _show_page(self, page: QWidget) -> None:
        # The old page may be sending the signal that got us here (its combo box
        # or Remove button), so it must not be deleted until that signal returns.
        old = self._scroll.takeWidget()
        self._scroll.setWidget(page)
        if old is not None:
            old.deleteLater()

    # --- transition ------------------------------------------------------
    def _transition_page(self, index: int) -> QWidget:
        transition = self._model.project.transitions[index]
        page = QWidget()
        form = QFormLayout(page)
        form.addRow(QLabel(f"<b>Transition {index + 1} → {index + 2}</b>"))

        effect_box = QComboBox()
        for entry in self._catalog.entries():
            effect_box.addItem(entry.name, entry.id)
        position = effect_box.findData(transition.effect)
        if position < 0:
            effect_box.addItem(f"{transition.effect} (not available)", transition.effect)
            position = effect_box.count() - 1
        effect_box.setCurrentIndex(position)
        effect_box.currentIndexChanged.connect(lambda _: self._set_effect(index, effect_box.currentData()))
        form.addRow("Effect", effect_box)

        duration = QDoubleSpinBox()
        duration.setRange(0.0, MAX_DURATION)
        duration.setDecimals(2)
        duration.setSingleStep(0.1)
        duration.setSuffix(" s")
        duration.setValue(transition.duration)
        duration.valueChanged.connect(lambda v: self._update_transition(index, duration=v))
        form.addRow("Duration", duration)

        alignment = QComboBox()
        for value in ALIGNMENTS:
            alignment.addItem(ALIGNMENT_LABELS[value], value)
        alignment.setCurrentIndex(ALIGNMENTS.index(transition.alignment))
        alignment.currentIndexChanged.connect(lambda _: self._update_transition(index, alignment=alignment.currentData()))
        form.addRow("Alignment", alignment)

        seed = QSpinBox()
        seed.setRange(0, MAX_SEED)
        seed.setValue(transition.seed)
        seed.valueChanged.connect(lambda v: self._update_transition(index, seed=v))
        form.addRow("Seed", seed)

        try:
            entry = self._catalog.get(transition.effect)
        except KeyError:
            return page
        if entry.description:
            note = QLabel(entry.description)
            note.setWordWrap(True)
            form.addRow(note)
        values = self._current_params(transition, entry.specs)
        for spec in entry.specs:
            setter = self._param_setter(index, spec, entry.specs)
            form.addRow(spec.label, make_param_widget(spec, values[spec.id], setter))
        return page

    @staticmethod
    def _current_params(transition: Transition, specs: Sequence[ParamSpec]) -> dict[str, object]:
        """The transition's parameter values, or the defaults if they do not fit its effect."""
        try:
            return validate_params(specs, transition.params)
        except ValueError:
            return validate_params(specs, {})

    def _param_setter(self, index: int, spec: ParamSpec, specs: Sequence[ParamSpec]) -> Callable[[object], None]:
        def setter(value: object) -> None:
            transition = self._model.project.transitions[index]
            params = {**self._current_params(transition, specs), spec.id: value}
            self._model.set_transition(index, replace(transition, params=params), spec.label)
        return setter

    def _update_transition(self, index: int, **changes: object) -> None:
        transition = self._model.project.transitions[index]
        self._model.set_transition(index, replace(transition, **changes), next(iter(changes)))

    def _set_effect(self, index: int, effect_id: str) -> None:
        self._update_transition(index, effect=effect_id, params={})
        self._show_page(self._transition_page(index))   # controls of the new effect

    # --- clip ------------------------------------------------------------
    def _clip_page(self, index: int) -> QWidget:
        clip = self._model.project.clips[index]
        info = self._model.info(clip.path)
        total = info.duration if info else 0.0
        page = QWidget()
        form = QFormLayout(page)
        form.addRow(QLabel(f"<b>Clip {index + 1}: {os.path.basename(clip.path)}</b>"))
        if info:
            form.addRow(QLabel(f"Length: {total:.2f} s"))
        else:
            form.addRow(QLabel("The file cannot be read."))

        start = QDoubleSpinBox()
        start.setRange(0.0, max(total, clip.start))
        start.setDecimals(2)
        start.setSuffix(" s")
        start.setValue(clip.start)
        start.valueChanged.connect(lambda v: self._update_clip(index, "trim start", start=v))
        form.addRow("Trim start", start)

        end = QDoubleSpinBox()
        end.setRange(0.0, max(total, clip.end or 0.0))
        end.setDecimals(2)
        end.setSuffix(" s")
        end.setSpecialValueText("End of clip")     # shown at the minimum value, 0
        end.setValue(clip.end or 0.0)
        end.valueChanged.connect(lambda v: self._update_clip(index, "trim end", end=v or None))
        form.addRow("Trim end", end)

        remove = QPushButton("Remove from timeline")
        remove.clicked.connect(lambda: self._model.remove_clip(index))
        form.addRow(remove)
        return page

    def _update_clip(self, index: int, what: str, **changes: object) -> None:
        clip = self._model.project.clips[index]
        self._model.set_clip(index, replace(clip, **changes), what)
