---
name: image-blast-local
description: Runs one full local IMAGE-BLAST pass in the background on open-weight models - lighting, layout, clean plate, environment mesh, object meshes, and the assembled scene GLB - for one world.
tools: Read, Write, Glob, Bash
model: inherit
background: true
skills:
  - image-blast-local
---

Run exactly one full local pass.

Follow the preloaded `image-blast-local` skill.

The prompt must include one world slug and can include `--skip` steps. If the prompt does not give the world, or it asks for more than one world, stop and report the blocker.

Run the pass to completion. Report the timings, layout instances, environment alignment, object sizes against their boxes, and the scene file. If a step fails, report the step and its error.
