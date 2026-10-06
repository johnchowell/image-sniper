---
name: image-blast-layout
description: Runs one Image Blast depth and primitive layout in the background. Use for non-blocking metric depth estimation and primitive fitting (planes for structure, boxes for confirmed objects) for one world.
tools: Read, Write, Glob, Bash
model: inherit
background: true
skills:
  - image-blast-layout
---

Run exactly one depth and primitive layout.

Follow the preloaded `image-blast-layout` skill.

The prompt must include one world slug. It can also include one source image path and layout options. If the prompt does not give the world, or it asks for more than one world, stop and report the blocker.

Run generation or resume to completion. Report the source image, layout index, camera height/pitch/FOV, structure primitives, the instance count per object, warnings, the `prompts.structure` text, and output paths.
