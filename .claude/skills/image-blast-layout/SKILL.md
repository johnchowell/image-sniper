---
name: image-blast-layout
description: Estimate metric depth for a world's source image and fit 3D primitives (floor, walls, ceiling, surfaces, one box per confirmed object instance). Use after objects are confirmed and before world generation, or when a downstream model needs scene scale, camera pose, or a spatial blockout.
argument-hint: [world-name] [optional image path] [--fov-x deg] [--mask-threshold 0-1] [--regenerate]
allowed-tools: Read Write Glob Bash(ls *) Bash(node .claude/scripts/project/project-state.mjs *) Bash(node .claude/scripts/project/ensure-local-assets.mjs *) Bash(node .claude/scripts/layout/generate-layout.mjs *) Bash(node .claude/scripts/layout/build-layout.mjs *)
context: fork
agent: image-blast-layout
---

Create one depth-based primitive layout for project `$0`.

## Instructions

- If `$0` is missing, ask for the world slug.
- Use `ls -a` before reading generated state.
- Run `Agent(image-blast-light)` first when possible; the layout reads its estimate for the same source image and adds the lighting.
- Confirmed objects are `worlds/$0/output/<object>/object.json`. Each one gets one segmentation request with `object.name` as the text prompt. Make sure the object files exist before you run the layout; objects added later need `--regenerate`.
- Without `--image`, the helper uses the lowest-index source image (the original photo). The objects must be visible in that image, so do not pass a clean plate unless the user asks for a structure-only layout.
- The helper sends the image to `fal-ai/moge-2` for metric depth, camera intrinsics, and a point cloud. At the same time, it sends one `fal-ai/sam-3/image` request per confirmed object for instance masks. Then it fits primitives locally. It resumes unfinished requests and skips when the latest `N-layout.json` exists.
- Pass `--fov-x <deg>` only when the user knows the lens field of view. Lower `--mask-threshold` (default `0.4`) only when the result says a confirmed object was `not_found`.

```bash
node .claude/scripts/project/project-state.mjs --world "$0"
```

Run:

```bash
node .claude/scripts/layout/generate-layout.mjs --world "$0"
```

To refit primitives from the files that are already downloaded, with no provider calls and no new index:

```bash
node .claude/scripts/layout/build-layout.mjs --world "$0" --index <N>
```

If request metadata records provider URLs but local files are missing, fill them, then rebuild:

```bash
node .claude/scripts/project/ensure-local-assets.mjs --from "worlds/$0/output/layout/.<N>-layout__depth-request.json"
node .claude/scripts/project/ensure-local-assets.mjs --from "worlds/$0/output/layout/.<N>-layout__mask-<object-id>-request.json"
```

## Output Contract

All files are in `worlds/$0/output/layout/` with index `N`:

- `N-layout.json`: the primitive layout. All units are meters. The frame is right-handed: +Y up, camera forward -Z, +X right, origin on the floor below the source camera.
  - `camera`: image size, pixel intrinsics, FOV, height above floor, pitch, roll, pose quaternion (OpenGL camera), and the `matrix_world_from_camera_opencv` matrix.
  - `structure[]`: `plane` primitives with `class` `floor | ceiling | wall | vertical_surface | horizontal_surface | inclined_surface`, plus `center`, `normal`, `u_axis`, `v_axis`, `size`, `corners`, and `rotation_quaternion`. Extents cover the visible points only.
  - `objects[]`: one entry per confirmed object, with `instances[]` of `box` primitives (`center`, `size`, `yaw_deg`, `rotation_quaternion`, `corners`, `support`, `image_bbox_px`, `mask_file`). `support` is `floor`, a structure plane id, another instance id (for example a laptop on `l-shaped-desk-1`), or `none_detected`. In `size_basis`, the axis along the view direction is `visible_lower_bound`, because the back of the object is not visible. When the mask touches an image edge, `truncated` is true and `truncated_edges` names the edges; the axes the frame cuts are `visible_lower_bound` too.
  - `scale`: the metric scale applied to the depth model's points (`applied`, 1 = as predicted). With `--params '{"scale_anchors":1}'` it combines real-world anchors (door, ceiling and camera height, table heights) with the depth model's own scale; `evidence[]` lists each anchor.
  - `lighting`: present when `Agent(image-blast-light)` ran on the same photo first. Contains `dominant_light` (direction toward the light, azimuth, elevation, `fit_r2`, `confidence`, `uncertainty_deg`), `irradiance_sh` (9-term spherical-harmonic irradiance coefficients per RGB channel, for renderers), `emitters[]` (`area_light` primitives such as windows, with corners and color; each sits on the plane that holds the depth around it, a region spanning a corner splits between the two walls, and parts on one plane merge only across mullion-sized gaps), `highlights_on_objects[]`, `sunlit_patches[]` (bright patches on floors and tabletops: sunlight, not sources), and `unplaced_light_regions[]` (no surface and no depth to place them). With `confidence: low`, one distant light does not explain the shading; use the emitters, not the direction.
  - `prompts.structure`: an empty-environment spatial description (camera, floor, walls, ceiling, surfaces). It names no objects. `prompts.objects`: per-instance size and placement text.
  - `labels.palette`: label map colors. Floor, wall, and ceiling use the ADE20K colors.
- `N-layout.glb`: blockout with one named node per primitive (`extras` carry `id`, `class`, `object_id`), emissive quads for light sources, a `KHR_lights_punctual` directional light when the direction confidence is not low, and a `source-camera` node.
- `N-layout-labels.png`: flat per-primitive label colors, for segmentation-conditioned models.
- `N-layout-guide.png`: dimmed source with label tint and projected primitive wireframes, for vision models and humans.
- `N-layout-depth.png`: uint16 depth in millimeters along the optical axis. `N-layout-depth-control.png`: 8-bit inverse depth, near = white.
- Provider files: `N-layout-points.ply`, `N-layout-mesh.glb` (camera frame), `N-layout-valid-mask.png`, `N-layout-mask-<object-id>-<k>.png`, and previews.

Do not load the PNG outputs into context to do quality checks. Use the JSON values and `warnings`.

```bash
node .claude/scripts/project/project-state.mjs --world "$0"
```

Final response: report the source image, the layout index, the camera height/pitch/FOV, the structure primitives with sizes, the instance count per object (name each `not_found` object), the `warnings`, the `prompts.structure` text, and the output file paths.
