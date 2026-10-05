#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ensureDir, loadDotEnv, one, parseArgs, pathExists, readJson, writeJson } from "../asset-pipeline/fal-queue.mjs";
import { artifactPath, isVisibleFile, latestIndexed, nextIndex, parseIndexedName, requestPath } from "../asset-pipeline/request-metadata.mjs";

const PROVIDER = "local/marigold-iid-lighting-v1-1";
const IMAGE_EXTENSIONS = new Set([".avif", ".gif", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".webp"]);
const RUNNER = path.join(path.dirname(fileURLToPath(import.meta.url)), "estimate_light.py");

export function lightDir(world) {
  return path.join("worlds", world, "output", "light");
}

// The original photo (lowest index): the layout and object references are measured from it.
async function originalSourceImage(world) {
  const sourceDir = path.join("worlds", world, "source");
  const entries = await readdir(sourceDir, { withFileTypes: true }).catch(() => []);
  return entries
    .filter((entry) => entry.isFile() && isVisibleFile(entry.name) && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase()))
    .map((entry) => parseIndexedName(path.join(sourceDir, entry.name)) || { index: 0, slug: entry.name, path: path.join(sourceDir, entry.name) })
    .sort((a, b) => a.index - b.index || a.slug.localeCompare(b.slug))[0]?.path;
}

async function resolvePython(explicit) {
  await loadDotEnv();
  const python = explicit || process.env.IMAGE_BLAST_PYTHON || ".venv/bin/python";
  if (!(await pathExists(python))) {
    throw new Error(
      `Python for local models not found at ${python}. Set it up once with:\n` +
      "  python3 -m venv .venv\n" +
      "  .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision\n" +
      "  .venv/bin/pip install -r .claude/scripts/light/requirements.txt\n" +
      "or point IMAGE_BLAST_PYTHON (env or .env) at an existing interpreter."
    );
  }
  return python;
}

export async function generateLight({ world, image, python, regenerate = false, steps = 4, ensemble = 1, resolution = 768 }) {
  if (!world) throw new Error("world is required.");
  const dir = lightDir(world);
  await ensureDir(dir);

  const latest = await latestIndexed(dir, "light");
  if (latest && !regenerate) {
    return { world, index: latest.index, skipped: true, skip_reason: `${latest.path} already exists. Pass --regenerate for a new estimate.`, light_json: latest.path };
  }

  const selectedImage = image || await originalSourceImage(world);
  if (!selectedImage) throw new Error(`No source image found in worlds/${world}/source.`);
  const interpreter = await resolvePython(python);
  const index = await nextIndex(dir);
  const metadataPath = requestPath(dir, index, "light");
  const settings = { steps: Number(steps), ensemble: Number(ensemble), processing_resolution: Number(resolution) };
  const base = { schema_version: 1, kind: "light", provider: PROVIDER, endpoint: PROVIDER, index, input_files: [selectedImage], input: settings };
  await writeJson(metadataPath, { ...base, status: "running", submitted_at: new Date().toISOString() });

  const run = spawnSync(interpreter, [
    RUNNER, "--image", selectedImage, "--out-dir", dir, "--index", String(index),
    "--steps", String(settings.steps), "--ensemble", String(settings.ensemble), "--resolution", String(settings.processing_resolution)
  ], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], maxBuffer: 64 * 1024 * 1024 });

  const lightJson = artifactPath(dir, index, "light", ".json");
  if (run.status !== 0 || !(await pathExists(lightJson))) {
    const error = (run.stderr || run.error?.message || "").trim().split("\n").slice(-12).join("\n");
    await writeJson(metadataPath, { ...base, status: "failed", completed_at: new Date().toISOString(), error });
    throw new Error(`Light estimation failed:\n${error}`);
  }

  const summary = await readJson(lightJson);
  await writeJson(metadataPath, {
    ...base,
    status: "completed",
    completed_at: new Date().toISOString(),
    output_files: [lightJson, ...Object.values(summary.files)],
    result: { reconstruction_r2: summary.reconstruction_r2, scales: summary.scales, emitters: summary.emitters.length }
  });

  return {
    world,
    index,
    source_image: selectedImage,
    light_json: lightJson,
    reconstruction_r2: summary.reconstruction_r2,
    light_color: summary.light_color,
    nondiffuse_fraction: summary.nondiffuse_fraction,
    emitters: summary.emitters.map((emitter) => ({ id: emitter.id, image_bbox_px: emitter.image_bbox_px, area_fraction: emitter.area_fraction })),
    files: summary.files,
    inference_seconds: summary.inference_seconds
  };
}

async function main() {
  const { flags } = parseArgs();
  const world = one(flags, "world");
  if (!world) {
    throw new Error("Usage: node .claude/scripts/light/generate-light.mjs --world <world> [--image <path>] [--python <path>] [--steps 4] [--ensemble 1] [--resolution 768] [--regenerate]");
  }
  const result = await generateLight({
    world,
    image: one(flags, "image"),
    python: one(flags, "python"),
    regenerate: Boolean(flags.regenerate),
    steps: one(flags, "steps", 4),
    ensemble: one(flags, "ensemble", 1),
    resolution: one(flags, "resolution", 768)
  });
  console.log(JSON.stringify(result, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}
