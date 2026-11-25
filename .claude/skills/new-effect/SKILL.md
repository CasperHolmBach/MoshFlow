---
name: new-effect
description: Create, write or finish a MoshFlow transition effect. Use when a team member wants to add their own datamosh effect, turn an idea into an effect file, finish one of the sample skeletons, or fix an effect that fails the contract test.
---

# Write a MoshFlow effect

Paths are relative to the repository root. The member decides what the effect does;
it only has to follow the API in `docs/API.md`. Read `docs/API.md` and the library
rule in `CLAUDE.md` first.

## 1. Agree on the idea

If the member has not said what the transition should look like, ask: what happens
to clip A's picture, what happens to clip B's, and which values they want to tune.
Describe the idea at the pixel level (what happens to one pixel value), because the
professor asks questions that way.

## 2. Create the file

```bash
cp moshflow/effects/_template.py moshflow/effects/<your_effect>.py
```

Or start from a sample skeleton (`keyframe_drop.py`, `motion_bloom.py`,
`liquid_melt.py`). Then:

- Rename the class and set `info = EffectInfo("<your_effect>", "<Name>", "<author>", "<one sentence>")`.
  The id must be unique and stay the same once a project file uses it.
- Declare every tunable value in `param_specs()` with `IntParam`, `FloatParam`,
  `ChoiceParam` or `BoolParam`. The Inspector builds the controls from it.
- In `render(ctx, progress)`, walk the output frames with
  `toolkit.frames.window_frames(ctx)`, start a datamosh from
  `toolkit.frames.last_clean_a(ctx)`, and do the maths in float32. Use
  `toolkit.motion` and `toolkit.warp` for motion; see "Toolkit" in `docs/API.md`.
- Yield exactly `ctx.length` frames with `color.to_uint8(...)`, check
  `progress.cancelled` in the loop, and use `ctx.seed` for any randomness.

## 3. Check it

```bash
.venv/bin/python -m pytest tests/test_effect_contract.py tests/test_compliance.py
.venv/bin/python .claude/skills/run-moshflow/driver.py --out shots --effect <your_effect> --export
```

Look at `shots/frame_*.png` (see the `run-moshflow` skill). Then run the whole suite:

```bash
.venv/bin/python -m pytest
```

## 4. Finish

- Remove `(draft)` from the name once it works.
- New `.py` file: run `.venv/bin/python tools/update_vs_project.py` so Visual Studio lists it.
- Do not change `moshflow/api/`, the toolkit's signatures, or other members' effects.

## Contract test failures

| Failure | Usual cause |
| --- | --- |
| wrong number of frames | A loop over `range(ctx.length)` skipped frames, or an early `return` without cancel |
| shape or dtype | Yielded float32, or a frame of the wrong size; end with `color.to_uint8` |
| input changed | Wrote into a frame from `.frame(i)`; call `.astype(np.float32)` first |
| different output for the same seed | Used `np.random` without `np.random.default_rng(ctx.seed)` |
| fails on `before`, `after` or `one_frame` | Assumed both windows are non-empty, or read B frame 1 when B has one frame |
