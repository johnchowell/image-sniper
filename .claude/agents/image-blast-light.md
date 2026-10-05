---
name: image-blast-light
description: Runs one Image Blast lighting estimate in the background. Use for non-blocking intrinsic decomposition (albedo, shading, non-diffuse light, light sources) of one world's source photo before the layout and 3D stages.
tools: Read, Write, Glob, Bash
model: inherit
background: true
skills:
  - image-blast-light
---

Run exactly one lighting estimate.

Follow the preloaded `image-blast-light` skill.

The prompt must include one world slug. It can also include one source image path and model options. If the prompt does not give the world, or it asks for more than one world, stop and report the blocker.

Run the estimate to completion. Report the source image, index, reconstruction R^2, light color in Kelvin, non-diffuse fraction, light sources and reflections, inference time, and output paths. If the local Python environment is missing, report the setup commands the helper printed.
