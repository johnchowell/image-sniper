---
name: image-blast-light
description: Estimate the lighting of a world's source photo with a local intrinsic-decomposition model (Marigold-IID-Lighting). Splits the photo into albedo, diffuse shading, and non-diffuse light, and finds light sources such as windows. Use after objects are confirmed and before the layout and 3D object generation.
argument-hint: [world-name] [optional image path] [--steps 4] [--ensemble 1] [--resolution 768] [--regenerate]
allowed-tools: Read Write Glob Bash(ls *) Bash(node .claude/scripts/project/project-state.mjs *) Bash(node .claude/scripts/light/generate-light.mjs *)
context: fork
agent: image-blast-light
---

Create one light estimate for project `$0`.

## Instructions

- If `$0` is missing, ask for the world slug.
- Use `ls -a` before reading generated state.
- Without `--image`, the helper uses the lowest-index source image (the original photo), which is the photo the layout and the 3D references are measured from. Do not pass a clean plate unless the user asks.
- The model runs locally (fal does not host it). It needs a Python environment with torch and diffusers. The helper uses `IMAGE_BLAST_PYTHON` (environment or `.env`), else `.venv/bin/python`. If it is missing, the helper prints the setup commands; report them to the user and stop.
- First run downloads about 2 GB of weights from Hugging Face. CPU inference takes about 1.5 minutes at the default 768 px processing resolution. More `--steps` or `--ensemble` members reduce noise and cost time linearly.
- The helper skips when the latest `N-light.json` exists. Pass `--regenerate` for a new estimate.

```bash
node .claude/scripts/project/project-state.mjs --world "$0"
```

Run:

```bash
node .claude/scripts/light/generate-light.mjs --world "$0"
```

## Output Contract

Files in `worlds/$0/output/light/` with index `N`:

- `N-light.json`:
  - `decomposition`: `photo_linear = albedo * shading + residual`.
  - `scales`: fitted scales that put shading and residual in image units. `reconstruction_r2`: how well the decomposition rebuilds the unclipped photo.
  - `light_color`: linear RGB and correlated color temperature in Kelvin.
  - `nondiffuse_fraction`: share of reflected light that is not diffuse (gloss, glass) outside light sources.
  - `emitters[]`: bright regions the model assigns to non-diffuse light. `kind: light_source` (mostly overexposed, such as a window or a lamp) or `kind: reflection` (within sensor range, such as glass reflections and highlights).
- `N-light-albedo.png`: 8-bit sRGB albedo with the lighting removed. The 3D generator uses it as a color reference.
- `N-light-shading.png`, `N-light-residual.png`: 16-bit RGB linear maps (decode with the scale in `encoding`).
- `N-light-shading-preview.png`: tone-mapped shading for viewing or as a lighting reference for image models.
- `N-light-emitters.png`: 8-bit region ids matching `emitters[].id`.

The layout stage reads this estimate for the same photo. It fits the light direction against the depth normals and places light sources in 3D. The 3D stage adds the albedo to the object reference edit.

Do not load the PNG outputs into context to do quality checks. Use the JSON values.

Final response: report the source image, the index, `reconstruction_r2`, the light color in Kelvin, `nondiffuse_fraction`, the light sources and reflections with their image boxes, the inference time, and the output paths.
