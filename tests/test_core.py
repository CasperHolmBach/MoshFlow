"""Parameters, toolkit and timeline."""

import numpy as np
import pytest

from helpers import ArraySource, moving_clip
from moshflow.api import BoolParam, ChoiceParam, FloatParam, IntParam, validate_params
from moshflow.core.discovery import discover_effects
from moshflow.core.project import Clip, Project, Transition
from moshflow.core.timeline import ClipNode, TimelineError, TransitionNode, resolve
from moshflow.toolkit import color, resize

SPECS = [IntParam("n", "N", 4, 1, 8), FloatParam("x", "X", 0.5, 0.0, 1.0),
         ChoiceParam("c", "C", "a", ("a", "b")), BoolParam("b", "B", False)]


def test_params_fill_defaults_and_convert():
    assert validate_params(SPECS, {"n": 2.0, "x": 1}) == {"n": 2, "x": 1.0, "c": "a", "b": False}


@pytest.mark.parametrize("values", [{"n": 9}, {"n": 1.5}, {"x": "big"}, {"c": "z"}, {"b": 1}, {"nope": 1}])
def test_params_reject_bad_values(values):
    with pytest.raises(ValueError):
        validate_params(SPECS, values)


def test_bilinear_keeps_a_constant_image_and_the_same_size():
    image = np.full((10, 20, 3), 77, dtype=np.uint8)
    assert resize.bilinear(image, (20, 10)) is image
    out = resize.bilinear(image, (7, 31))
    assert out.shape == (31, 7, 3) and out.dtype == np.uint8 and (out == 77).all()


def test_bilinear_interpolates_known_values():
    image = np.array([[[0, 0, 0], [100, 100, 100]]], dtype=np.uint8)
    out = resize.bilinear(image, (4, 1))
    assert out[0, :, 0].tolist() == [0, 25, 75, 100]


def test_nearest_picks_existing_pixels():
    image = np.array([[[10, 10, 10], [200, 200, 200]]], dtype=np.uint8)
    out = resize.nearest(image, (4, 2))
    assert out.shape == (2, 4, 3) and out[0, :, 0].tolist() == [10, 10, 200, 200]


def test_luma_weights():
    frame = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert color.bgr_to_y(frame)[0].tolist() == pytest.approx([29.07, 149.685, 76.245], abs=0.01)


def test_to_uint8_rounds_and_saturates():
    assert color.to_uint8(np.array([-5.0, 1.6, 300.0])).tolist() == [0, 2, 255]


def graph_for(duration, alignment, lengths=(20, 20)):
    sources = [ArraySource(moving_clip(n)) for n in lengths]
    project = Project()
    for i in range(len(lengths)):
        project.add_clip(Clip(f"clip{i}.mp4"))
    for t in project.transitions:
        t.duration, t.alignment = duration, alignment
    return resolve(project, sources)


@pytest.mark.parametrize("alignment, a_len, b_len", [("center", 5, 5), ("before", 10, 0), ("after", 0, 10)])
def test_alignment_windows(alignment, a_len, b_len):
    graph = graph_for(1.0, alignment)         # 1 s at 10 fps = 10 frames
    first, transition, last = graph.nodes
    assert isinstance(first, ClipNode) and isinstance(transition, TransitionNode) and isinstance(last, ClipNode)
    assert first.frames == range(0, 20 - a_len)
    assert transition.a_window == range(20 - a_len, 20)
    assert transition.b_window == range(0, b_len)
    assert last.frames == range(b_len, 20)
    assert graph.total_frames == 40            # a transition replaces frames, it adds none


def test_zero_length_transition_is_skipped():
    assert [type(n) for n in graph_for(0.0, "center").nodes] == [ClipNode, ClipNode]


def test_clip_fully_used_by_its_transitions_has_no_clip_node():
    graph = graph_for(1.0, "center", lengths=(20, 10, 20))
    assert [type(n) for n in graph.nodes] == [ClipNode, TransitionNode, TransitionNode, ClipNode]


def test_clip_too_short_is_a_clear_error():
    with pytest.raises(TimelineError, match="clip 2"):
        graph_for(1.0, "center", lengths=(20, 4, 20))


def test_project_needs_one_transition_per_cut():
    project = Project(clips=[Clip("a"), Clip("b")], transitions=[])
    with pytest.raises(ValueError):
        project.validate()


def test_project_add_remove_and_move():
    project = Project()
    for name in "abc":
        project.add_clip(Clip(name))
    assert len(project.transitions) == 2
    project.move_clip(2, 0)
    assert [c.path for c in project.clips] == ["c", "a", "b"]
    project.remove_clip(1)
    assert [c.path for c in project.clips] == ["c", "b"] and len(project.transitions) == 1
    project.validate()


def test_project_dict_round_trip():
    project = Project()
    project.add_clip(Clip("a.mp4", start=1.0))
    project.add_clip(Clip("b.mp4", end=3.0))
    project.transitions[0] = Transition("crossfade", 2.0, "after", {"ease": 2.0}, 5)
    assert Project.from_dict(project.to_dict()) == project


def test_broken_effect_file_is_reported_and_does_not_stop_the_others(tmp_path, monkeypatch):
    package = tmp_path / "some_effects"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "good.py").write_text("LOADED = True\n")
    (package / "broken.py").write_text("raise RuntimeError('boom')\n")
    (package / "_skipped.py").write_text("raise RuntimeError('must not be imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    failures = discover_effects("some_effects")
    assert [f.module for f in failures] == ["some_effects.broken"]
    assert "boom" in failures[0].error
