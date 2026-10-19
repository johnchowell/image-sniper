---
name: image-blast-local
description: Run the full IMAGE-BLAST pass on local open-weight models (no FAL or World Labs credit needed) - lighting, layout, clean plate, environment mesh, object meshes, and one assembled scene GLB. Use when the user wants a model from an image locally, or when FAL or World Labs are unavailable.
argument-hint: [world-name] [--skip light|layout|plate|environment|objects|scene]
allowed-tools: Read Write Glob Bash(ls *) Bash(node .claude/scripts/project/project-state.mjs *) Bash(bash .claude/scripts/local/setup.sh) Bash(bash .claude/scripts/bench/setup.sh) Bash(.venv/bin/python .claude/scripts/local/*) Bash(.venv-render/bin/python .claude/scripts/local/*)
context: fork
agent: image-blast-local
---

Run one full local pass for project `$0`.

## Instructions

- If `$0` is missing, ask for the world slug.
- Use `ls -a` before reading generated state. Objects must be confirmed first: `worlds/$0/output/<object>/object.json` (from `image-blast-uncover`). The pass uses the original photo (lowest source index).
- The local runtime lives in `.venv` (or `IMAGE_BLAST_PYTHON`) with pinned GitHub repos in `third_party/`. If `.venv/bin/python` is missing, run the setup once:

```bash
bash .claude/scripts/local/setup.sh
```

- First run downloads about 5 GB of weights from Hugging Face. On a 4-core CPU the pass takes about 10 minutes; the slow steps are TripoSR (about 30 s per object) and the lighting model (about 90 s).

```bash
node .claude/scripts/project/project-state.mjs --world "$0"
```

Run:

```bash
.venv/bin/python .claude/scripts/local/local_pass.py --world "$0"
```

Add `--skip <step>` to reuse the latest output of a step (for example `--skip light --skip layout` to only rebuild plate, meshes, and scene).

## Steps and outputs

| step | model | output |
|---|---|---|
| light | Marigold-IID-Lighting v1.1 | `output/light/N-light*` (see `image-blast-light`) |
| layout | MoGe-2, Grounding DINO (name -> box, all names compete), SigLIP 2 (crop -> name check), SAM 2.1 (box -> mask), then `build-layout.mjs` | `output/layout/N-layout*` (see `image-blast-layout`) |
| plate | big-lama | `source/N-<slug>-plate.png`; the removal mask is hidden metadata `source/.N-<slug>-plate-mask.png` |
| environment | MoGe-2 on the plate, fov locked to the layout camera, scale fit to the original depth on unchanged pixels | `output/scene/N-scene-environment.glb` (textured, layout frame) |
| objects | TripoSR (geometry from the photo crop, vertex colors from the albedo crop, 30k faces) | `output/<object>/N-<object>.glb` (canonical: meters, +Y up, front +Z, base at y=0) and `N-<object>.png` (input crop); placement in `.N-<object>__model-request.json` |
| scene | assembly | `output/scene/N-scene.glb` (environment + placed objects + source camera + KHR_lights_punctual lights) and `N-scene.json` (manifest) |

## Primitive scene (PBR primitives instead of TripoSR meshes)

After the light and layout steps, the layout can also become a scene of textured primitives: a closed room shell and panels with base color (lighting-free albedo), roughness, normal (relief above the depth noise) and window emission maps, and each object and each unlisted part (clutter) carved from the view's depth and textured by projection.

```bash
.venv/bin/python .claude/scripts/local/primitive_scene.py --world "$0"
.venv-render/bin/python .claude/scripts/local/render_check.py --world "$0"
```

Each object mesh comes from these steps:
1. The yaw comes from the depth normals of the visible faces.
2. The box comes from the observed points. Its support comes from the layout (an object whose base the view cannot show stands on the floor, `support_basis: hidden_base`).
3. A learned shape prior (`.claude/models/shape-prior.pt`, a 3D U-Net trained on simulated views of ModelNet40 indoor shapes, each with the box the layout fits to its visible points, so a truncated object's thin box still completes to a solid) completes the hidden part of the occupancy. The observed surface and the free space stay exact. Closed cavities become full, because the view cannot see them.
4. If the object has a generated asset (`output/<object>/N-<object>.glb` from the objects step: the mask cut-out through TripoSR), the asset is registered to the view (yaw, scale, position; base on the support) so that the observed points lie on it and none of it lies in seen-through space or outside the prior's cube. The registered asset then adds evidence to the prior's log-odds (`--generated fuse`, weight 1). An asset whose depth is less than a quarter of its width (a single-view depth collapse) can also stretch along its view axis (`--generated-stretch 1`, the default). Without an asset, the prior is used alone. Run the objects step (`local_pass.py`, or `object_mesh.py --world "$0"`) before this script to use it.
5. The surface is decimated to an error bound, so flat faces become a few large triangles.

To compare the variants against the true meshes of the benchmark, use `.claude/scripts/bench/shape_eval.py score --split train --mode gt|pred --name <name> --args "<primitive_scene.py options>"`, then `shape_eval.py compare <a> <b>`. Tune on train, confirm on val, and read test only once.

The first command writes `output/scene/N-primitive-scene.glb`. The second (Blender runtime: `bash .claude/scripts/bench/setup.sh` once) fits the light powers and the camera response to the photo and writes `N-primitive-scene-check.json` (PSNR against the photo), `-check.png` (photo, render, difference, two orbit views) and `-lights.json` (the fitted light rig).

Do not load the PNG outputs into context to do quality checks. Use the printed summary and the JSON files.

Final response: report the timings per step, the layout instances per object, the environment depth scale and its spread, each object's size against its layout box, the scene file and its size, and any step that failed with its error.
