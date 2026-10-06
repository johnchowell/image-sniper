#!/usr/bin/env node
import { readdir } from "node:fs/promises";
import path from "node:path";
import {
  callFalQueue,
  downloadFile,
  ensureDir,
  getFalQueueResult,
  one,
  parseArgs,
  pathExists,
  pollFalQueue,
  readJson,
  sanitizeForMetadata,
  slugify,
  toModelInputUrl,
  writeJson
} from "../asset-pipeline/fal-queue.mjs";
import {
  artifactPath,
  isVisibleFile,
  nextIndex,
  parseIndexedName,
  requestMetadataFiles,
  requestPath
} from "../asset-pipeline/request-metadata.mjs";
import { buildLayout, layoutDir } from "./build-layout.mjs";

const DEPTH_ENDPOINT = "fal-ai/moge-2";
const MASK_ENDPOINT = "fal-ai/sam-3/image";
const IMAGE_EXTENSIONS = new Set([".avif", ".gif", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".webp"]);
const RESERVED_OUTPUT_DIRS = new Set(["world", "sfx", "light", "layout", "scene"]);
const DONE = new Set(["completed", "failed", "cancelled", "canceled"]);

async function readJsonIfExists(filePath) {
  return (await pathExists(filePath)) ? readJson(filePath) : undefined;
}

// The original photo (lowest index) still shows every object, which the layout needs to measure.
async function originalSourceImage(world) {
  const sourceDir = path.join("worlds", world, "source");
  const entries = await readdir(sourceDir, { withFileTypes: true }).catch(() => []);
  const images = entries
    .filter((entry) => entry.isFile() && isVisibleFile(entry.name) && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase()))
    .map((entry) => parseIndexedName(path.join(sourceDir, entry.name)) || { index: 0, slug: entry.name, path: path.join(sourceDir, entry.name) });
  return images.sort((a, b) => a.index - b.index || a.slug.localeCompare(b.slug))[0]?.path;
}

async function confirmedObjects(world) {
  const outputDir = path.join("worlds", world, "output");
  const entries = await readdir(outputDir, { withFileTypes: true }).catch(() => []);
  const objects = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || RESERVED_OUTPUT_DIRS.has(entry.name)) continue;
    const json = await readJsonIfExists(path.join(outputDir, entry.name, "object.json"));
    if (!json) continue;
    const object = json.object || json;
    objects.push({
      id: slugify(object.id || entry.name),
      name: object.name || entry.name,
      count: Number(object.count_estimate) || 1
    });
  }
  return objects.sort((a, b) => a.id.localeCompare(b.id));
}

async function runFalStep({ endpoint, input, metadataPath, metadata, pollIntervalMs }) {
  const previous = await readJsonIfExists(metadataPath);
  const status = String(previous?.status || "").toLowerCase();
  if (status === "completed" && previous.result) return previous.result;
  if (previous?.request_id && !DONE.has(status)) {
    await pollFalQueue(endpoint, previous.request_id, { statusUrl: previous.status_url, metadataPath, pollIntervalMs });
    return (await getFalQueueResult(endpoint, previous.request_id, { responseUrl: previous.response_url, metadataPath })).data;
  }
  const result = await callFalQueue(endpoint, input, {
    metadataPath,
    metadata: { ...metadata, input: sanitizeForMetadata(input) },
    pollIntervalMs
  });
  return result.data;
}

async function downloadNamed(metadataPath, targets) {
  const downloaded = [];
  for (const { label, file, outputPath } of targets) {
    if (!file?.url) continue;
    if (!(await pathExists(outputPath))) await downloadFile(file.url, outputPath);
    downloaded.push({ label, path: outputPath, source: sanitizeForMetadata(file) });
  }
  const previous = await readJson(metadataPath);
  await writeJson(metadataPath, {
    ...previous,
    output_files: downloaded.map((item) => item.path),
    downloaded_files: downloaded,
    updated_at: new Date().toISOString()
  });
  return downloaded;
}

async function runDepth({ dir, index, image, imageUrl, fovX, pollIntervalMs }) {
  const metadataPath = requestPath(dir, index, "layout", "depth");
  const input = {
    image_url: imageUrl,
    model: "vitl-normal",
    resolution_level: 9,
    apply_mask: true,
    export_glb: true,
    export_ply: true,
    ...(fovX ? { fov_x: Number(fovX) } : {})
  };
  const result = await runFalStep({
    endpoint: DEPTH_ENDPOINT,
    input,
    metadataPath,
    metadata: { schema_version: 1, kind: "layout-depth", provider: DEPTH_ENDPOINT, index, input_files: [image] },
    pollIntervalMs
  });
  if (!result?.point_cloud?.url || !result?.mask?.url || !result?.intrinsics) {
    throw new Error(`Depth result is missing point_cloud, mask, or intrinsics. See ${metadataPath}.`);
  }
  await downloadNamed(metadataPath, [
    { label: "point_cloud", file: result.point_cloud, outputPath: artifactPath(dir, index, "layout-points", ".ply") },
    { label: "mask", file: result.mask, outputPath: artifactPath(dir, index, "layout-valid-mask", ".png") },
    { label: "model_mesh", file: result.model_mesh, outputPath: artifactPath(dir, index, "layout-mesh", ".glb") },
    { label: "depth_map", file: result.depth_map, outputPath: artifactPath(dir, index, "layout-depth-preview", ".png") },
    { label: "normal_map", file: result.normal_map, outputPath: artifactPath(dir, index, "layout-normals-preview", ".png") }
  ]);
  return metadataPath;
}

async function runMask({ dir, index, image, imageUrl, object, maskThreshold, pollIntervalMs }) {
  const scope = `mask-${object.id}`;
  const metadataPath = requestPath(dir, index, "layout", scope);
  const input = {
    image_url: imageUrl,
    prompt: object.name,
    apply_mask: false,
    output_format: "png",
    return_multiple_masks: true,
    max_masks: Math.min(32, Math.max(3, object.count * 2)),
    include_scores: true,
    include_boxes: true
  };
  const result = await runFalStep({
    endpoint: MASK_ENDPOINT,
    input,
    metadataPath,
    metadata: {
      schema_version: 1,
      kind: "layout-mask",
      provider: MASK_ENDPOINT,
      index,
      object_id: object.id,
      object_name: object.name,
      prompt: object.name,
      mask_threshold: maskThreshold,
      input_files: [image]
    },
    pollIntervalMs
  });
  await downloadNamed(metadataPath, (result?.masks || []).map((file, k) => ({
    label: `mask-${k + 1}`,
    file,
    outputPath: artifactPath(dir, index, `layout-${scope}-${k + 1}`, ".png")
  })));
  return metadataPath;
}

export async function generateLayout({ world, image, fovX, maskThreshold = 0.4, regenerate = false, pollIntervalMs = 5000 }) {
  if (!world) throw new Error("world is required.");
  const dir = layoutDir(world);
  await ensureDir(dir);

  const depthRequests = await requestMetadataFiles(dir, { slug: "layout", scope: "depth" });
  const latest = depthRequests[0];
  const latestBuilt = latest && await pathExists(artifactPath(dir, latest.index, "layout", ".json"));
  if (latestBuilt && !regenerate) {
    return {
      world,
      index: latest.index,
      skipped: true,
      skip_reason: `${artifactPath(dir, latest.index, "layout", ".json")} already exists. Pass --regenerate for a new layout.`
    };
  }

  const resume = latest && !latestBuilt && !regenerate ? latest : undefined;
  const index = resume?.index ?? await nextIndex(dir);
  const selectedImage = resume?.data?.input_files?.[0] || image || await originalSourceImage(world);
  if (!selectedImage) throw new Error(`No source image found in worlds/${world}/source.`);
  const imageUrl = await toModelInputUrl(selectedImage);
  const objects = await confirmedObjects(world);

  await Promise.all([
    runDepth({ dir, index, image: selectedImage, imageUrl, fovX, pollIntervalMs }),
    ...objects.map((object) => runMask({ dir, index, image: selectedImage, imageUrl, object, maskThreshold, pollIntervalMs }))
  ]);

  return {
    ...(await buildLayout({ world, index })),
    source_image: selectedImage,
    resumed: Boolean(resume)
  };
}

async function main() {
  const { flags } = parseArgs();
  const world = one(flags, "world");
  if (!world) {
    throw new Error("Usage: node .claude/scripts/layout/generate-layout.mjs --world <world> [--image <path>] [--fov-x <deg>] [--mask-threshold 0.4] [--regenerate]");
  }
  const result = await generateLayout({
    world,
    image: one(flags, "image"),
    fovX: one(flags, "fov-x"),
    maskThreshold: Number(one(flags, "mask-threshold", 0.4)),
    regenerate: Boolean(flags.regenerate),
    pollIntervalMs: Number(one(flags, "poll-interval-ms", 5000))
  });
  console.log(JSON.stringify(result, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}
