#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { one, parseArgs, pathExists, readJson, writeJson } from "../asset-pipeline/fal-queue.mjs";
import { artifactPath, requestMetadataFiles, requestPath } from "../asset-pipeline/request-metadata.mjs";
import {
  add,
  cross,
  degrees,
  dot,
  extractPlanes,
  minAreaRect,
  mulMat3Vec,
  normalize,
  percentile,
  quaternionFromBasis,
  scale,
  sortedValues,
  sub,
  transpose3
} from "./geometry.mjs";
import { writeLayoutGlb } from "./glb.mjs";
import { analyzeLighting, lightingPrompt, loadLightEstimate } from "./lighting.mjs";
import { decodePng, encodePng, maskFromPng } from "./png.mjs";
import { parsePly } from "./ply.mjs";

const MAX_GRID_SIDE = 1024;
// Solver parameters; build-layout.mjs --params '<json>' overrides any of them (used by benchmark calibration).
export const DEFAULT_PARAMS = {
  horizontal_deg: 15,
  up_search_deg: 40,
  plane_min_fraction: 0.003,
  plane_normal_deg: 25,
  plane_base_tolerance_m: 0.01,
  plane_relative_tolerance: 0.01,
  region_reach_cells: 2,
  region_min_fraction: 0.25,
  region_dominant_fraction: 0.5,
  wall_min_height_m: 1.5,
  support_snap_min_m: 0.08,
  support_snap_fraction: 0.1,
  flat_mask_min_extent_m: 1.5,
  flat_mask_max_height_ratio: 0.1
};
const CLASS_COLORS = {
  floor: [80, 50, 50],
  wall: [120, 120, 120],
  ceiling: [120, 120, 80],
  vertical_surface: [180, 120, 120],
  horizontal_surface: [255, 184, 6],
  inclined_surface: [140, 140, 200]
};

export function layoutDir(world) {
  return path.join("worlds", world, "output", "layout");
}

const round = (value, digits = 3) => Number(value.toFixed(digits));
const roundVec = (vector, digits = 3) => vector.map((value) => round(value, digits));

function objectColor(order) {
  const hue = (order * 0.618033988749895 + 0.11) % 1;
  const f = (n) => {
    const k = (n + hue * 6) % 6;
    return Math.round(255 * (1 - 0.75 * Math.max(0, Math.min(k, 4 - k, 1))));
  };
  return [f(5), f(3), f(1)];
}

// Picks the axis signs that map the PLY point frame to OpenCV camera coordinates (x right, y down, z forward).
function selectPointFrame(ply, intrinsics, validMask, width, height) {
  const stride = Math.max(1, Math.floor(ply.count / 20000));
  const invalidFraction = validMask.reduce((sum, value) => sum + (value ? 0 : 1), 0) / validMask.length;
  const candidates = [];
  for (const sy of [1, -1]) {
    for (const sz of [1, -1]) {
      let samples = 0, positive = 0, inBounds = 0, maskAgree = 0;
      let sumI = 0, sumV = 0, sumII = 0, sumVV = 0, sumIV = 0, n = 0;
      for (let i = 0; i < ply.count; i += stride) {
        samples += 1;
        const x = ply.positions[i * 3];
        const y = sy * ply.positions[i * 3 + 1];
        const z = sz * ply.positions[i * 3 + 2];
        if (!(z > 0)) continue;
        positive += 1;
        const u = (intrinsics.fx * x) / z + intrinsics.cx;
        const v = (intrinsics.fy * y) / z + intrinsics.cy;
        if (u < 0 || v < 0 || u >= width || v >= height) continue;
        inBounds += 1;
        if (validMask[Math.floor(v) * width + Math.floor(u)]) maskAgree += 1;
        n += 1; sumI += i; sumV += v; sumII += i * i; sumVV += v * v; sumIV += i * v;
      }
      const covariance = sumIV / n - (sumI / n) * (sumV / n);
      const spread = Math.sqrt(Math.max(1e-12, (sumII / n - (sumI / n) ** 2) * (sumVV / n - (sumV / n) ** 2)));
      candidates.push({
        signs: [1, sy, sz],
        positive_fraction: round(positive / samples, 4),
        in_bounds_fraction: round(inBounds / samples, 4),
        mask_agreement: round(n ? maskAgree / inBounds : 0, 4),
        row_order_correlation: round(n ? covariance / spread : 0, 4)
      });
    }
  }
  const passing = candidates.filter((c) => c.positive_fraction >= 0.9 && c.in_bounds_fraction >= 0.95);
  const ordered = passing.filter((c) => c.row_order_correlation > 0.3);
  let chosen;
  let basis;
  if (ordered.length === 1) {
    chosen = ordered[0];
    basis = "row_order";
  } else if (passing.length && invalidFraction > 0.01) {
    chosen = [...passing].sort((a, b) => b.mask_agreement - a.mask_agreement)[0];
    basis = "validity_mask";
    if (chosen.mask_agreement < 0.95) chosen = undefined;
  }
  if (!chosen) {
    throw new Error(`Cannot map the point cloud to the image frame. Candidates: ${JSON.stringify(candidates)}`);
  }
  return { signs: chosen.signs, basis, candidates };
}

function buildGrid(ply, signs, intrinsics, width, height) {
  const step = Math.max(1, Math.ceil(Math.max(width, height) / MAX_GRID_SIDE));
  const gw = Math.ceil(width / step);
  const gh = Math.ceil(height / step);
  const cells = gw * gh;
  const points = new Float32Array(cells * 3);
  const colors = new Uint8Array(cells * 3);
  const valid = new Uint8Array(cells);
  for (let i = 0; i < ply.count; i += 1) {
    const x = signs[0] * ply.positions[i * 3];
    const y = signs[1] * ply.positions[i * 3 + 1];
    const z = signs[2] * ply.positions[i * 3 + 2];
    if (!(z > 0) || !Number.isFinite(x) || !Number.isFinite(y)) continue;
    const u = (intrinsics.fx * x) / z + intrinsics.cx;
    const v = (intrinsics.fy * y) / z + intrinsics.cy;
    if (u < 0 || v < 0 || u >= width || v >= height) continue;
    const cell = Math.floor(v / step) * gw + Math.floor(u / step);
    if (valid[cell]) continue;
    valid[cell] = 1;
    points[cell * 3] = x;
    points[cell * 3 + 1] = y;
    points[cell * 3 + 2] = z;
    if (ply.colors) {
      colors[cell * 3] = ply.colors[i * 3];
      colors[cell * 3 + 1] = ply.colors[i * 3 + 1];
      colors[cell * 3 + 2] = ply.colors[i * 3 + 2];
    }
  }
  return { step, gw, gh, points, colors, valid, hasColor: Boolean(ply.colors) };
}

// The stencil spans ~0.6% of the image so normals average out per-pixel depth noise at any resolution.
// At surface edges it falls back to one-sided and shorter differences, so thin foreshortened surfaces keep normals.
function computeNormals(grid) {
  const { gw, gh, points, valid } = grid;
  const span = Math.max(2, Math.round(Math.max(gw, gh) / 160));
  const spans = [...new Set([span, Math.ceil(span / 2), 1])];
  const normals = new Float32Array(gw * gh * 3);
  const hasNormal = new Uint8Array(gw * gh);
  const at = (cell) => [points[cell * 3], points[cell * 3 + 1], points[cell * 3 + 2]];
  const continuous = (center, cell, distance) =>
    valid[cell] && Math.abs(points[cell * 3 + 2] - center[2]) < 0.04 * distance * center[2];
  const difference = (cell, gx, gy, dx, dy) => {
    const center = at(cell);
    for (const s of spans) {
      for (const [a, b] of [[-s, s], [0, s], [-s, 0]]) {
        const ax = gx + dx * a, ay = gy + dy * a, bx = gx + dx * b, by = gy + dy * b;
        if (ax < 0 || ay < 0 || bx >= gw || by >= gh) continue;
        const ca = ay * gw + ax;
        const cb = by * gw + bx;
        if ((a === 0 || continuous(center, ca, s)) && (b === 0 || continuous(center, cb, s))) return sub(at(cb), at(ca));
      }
    }
    return undefined;
  };
  for (let gy = 0; gy < gh; gy += 1) {
    for (let gx = 0; gx < gw; gx += 1) {
      const cell = gy * gw + gx;
      if (!valid[cell]) continue;
      const horizontal = difference(cell, gx, gy, 1, 0);
      const vertical = horizontal && difference(cell, gx, gy, 0, 1);
      if (!vertical) continue;
      let normal = normalize(cross(horizontal, vertical));
      if (dot(normal, at(cell)) > 0) normal = scale(normal, -1);
      normals.set(normal, cell * 3);
      hasNormal[cell] = 1;
    }
  }
  return { normals, hasNormal };
}

function maskToGrid(maskImage, grid, width, height) {
  const { gw, gh, step, valid } = grid;
  const result = new Uint8Array(gw * gh);
  for (let gy = 0; gy < gh; gy += 1) {
    const py = Math.min(height - 1, gy * step + Math.floor(step / 2));
    const my = Math.min(maskImage.height - 1, Math.floor((py * maskImage.height) / height));
    for (let gx = 0; gx < gw; gx += 1) {
      const px = Math.min(width - 1, gx * step + Math.floor(step / 2));
      const mx = Math.min(maskImage.width - 1, Math.floor((px * maskImage.width) / width));
      result[gy * gw + gx] = maskImage.mask[my * maskImage.width + mx] && valid[gy * gw + gx] ? 1 : 0;
    }
  }
  return result;
}

function morph(mask, gw, gh, erode) {
  const out = new Uint8Array(mask.length);
  for (let gy = 0; gy < gh; gy += 1) {
    for (let gx = 0; gx < gw; gx += 1) {
      const cell = gy * gw + gx;
      const neighbors = [
        gx > 0 ? mask[cell - 1] : 0,
        gx < gw - 1 ? mask[cell + 1] : 0,
        gy > 0 ? mask[cell - gw] : 0,
        gy < gh - 1 ? mask[cell + gw] : 0
      ];
      out[cell] = erode
        ? (mask[cell] && neighbors.every(Boolean) ? 1 : 0)
        : (mask[cell] || neighbors.some(Boolean) ? 1 : 0);
    }
  }
  return out;
}

async function loadObjectMasks(dir, index, grid, width, height) {
  const requests = (await requestMetadataFiles(dir, { slug: "layout" }))
    .filter((request) => request.index === index && request.scope?.startsWith("mask-"));
  const objects = [];
  for (const request of requests.sort((a, b) => a.scope.localeCompare(b.scope))) {
    const data = request.data;
    const scores = data.result?.scores || [];
    const files = (data.downloaded_files || []).map((file) => file.path).filter((file) => /\.png$/i.test(file));
    const threshold = data.mask_threshold ?? 0.4;
    const instances = [];
    for (const [k, file] of files.entries()) {
      const score = typeof scores[k] === "number" ? scores[k] : undefined;
      if (score !== undefined && score < threshold) continue;
      if (!(await pathExists(file))) continue;
      const gridMask = maskToGrid(maskFromPng(decodePng(await readFile(file))), grid, width, height);
      const area = gridMask.reduce((sum, value) => sum + value, 0);
      if (area < 20) continue;
      const duplicate = instances.some((other) => {
        let intersection = 0;
        for (let i = 0; i < gridMask.length; i += 1) if (gridMask[i] && other.gridMask[i]) intersection += 1;
        return intersection / (area + other.area - intersection) > 0.6;
      });
      if (!duplicate) instances.push({ file, score, gridMask, area });
    }
    objects.push({
      object_id: data.object_id,
      name: data.object_name || data.object_id,
      prompt: data.prompt,
      request: request.path,
      instances
    });
  }
  return objects;
}

function planeBasis(normal) {
  const horizontal = Math.abs(normal[1]) > 0.999;
  const u = horizontal ? [1, 0, 0] : normalize(cross([0, 1, 0], normal));
  return { u, v: cross(normal, u) };
}

function directionWords(point) {
  const ahead = -point[2];
  const lateral = point[0];
  const side = Math.abs(lateral) < 0.15 ? "centered" : `${round(Math.abs(lateral), 1)} m ${lateral < 0 ? "left" : "right"}`;
  return `${round(ahead, 1)} m ahead, ${side}`;
}

function wallFacing(normal) {
  if (normal[2] > 0.7) return "facing the camera";
  if (normal[0] > 0.7) return "on the left side, running away from the camera";
  if (normal[0] < -0.7) return "on the right side, running away from the camera";
  if (normal[2] < -0.7) return "behind the camera";
  return `at an angle (normal yaw ${round(degrees(Math.atan2(normal[0], normal[2])), 0)} deg)`;
}

function drawLine(image, gw, gh, a, b, color) {
  let [x0, y0] = a.map(Math.round);
  const [x1, y1] = b.map(Math.round);
  const dx = Math.abs(x1 - x0);
  const dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let error = dx + dy;
  for (let guard = 0; guard < 8192; guard += 1) {
    if (x0 >= 0 && y0 >= 0 && x0 < gw && y0 < gh) image.set(color, (y0 * gw + x0) * 3);
    if (x0 === x1 && y0 === y1) break;
    const e2 = 2 * error;
    if (e2 >= dy) { error += dy; x0 += sx; }
    if (e2 <= dx) { error += dx; y0 += sy; }
  }
}

export async function buildLayout({ world, index, params: overrides = {} }) {
  const params = { ...DEFAULT_PARAMS, ...overrides };
  const HORIZONTAL_DEG = params.horizontal_deg;
  const UP_SEARCH_DEG = params.up_search_deg;
  const dir = layoutDir(world);
  const depthRequestPath = requestPath(dir, index, "layout", "depth");
  if (!(await pathExists(depthRequestPath))) throw new Error(`Missing depth request metadata: ${depthRequestPath}`);
  const depthRequest = await readJson(depthRequestPath);
  const depthResult = depthRequest.result;
  if (!depthResult?.intrinsics) throw new Error(`Depth request has no intrinsics: ${depthRequestPath}`);

  const files = {
    points: artifactPath(dir, index, "layout-points", ".ply"),
    valid_mask: artifactPath(dir, index, "layout-valid-mask", ".png"),
    depth_mesh: artifactPath(dir, index, "layout-mesh", ".glb"),
    layout_glb: artifactPath(dir, index, "layout", ".glb"),
    labels_png: artifactPath(dir, index, "layout-labels", ".png"),
    guide_png: artifactPath(dir, index, "layout-guide", ".png"),
    depth_png: artifactPath(dir, index, "layout-depth", ".png"),
    depth_control_png: artifactPath(dir, index, "layout-depth-control", ".png")
  };
  for (const key of ["points", "valid_mask"]) {
    if (!(await pathExists(files[key]))) throw new Error(`Missing ${key} file: ${files[key]}. Run ensure-local-assets on ${depthRequestPath}.`);
  }

  const validPng = maskFromPng(decodePng(await readFile(files.valid_mask)));
  const { width, height } = validPng;
  const K = depthResult.intrinsics;
  const intrinsics = { fx: K[0][0] * width, fy: K[1][1] * height, cx: K[0][2] * width, cy: K[1][2] * height };
  const ply = parsePly(await readFile(files.points));
  const frameCheck = selectPointFrame(ply, intrinsics, validPng.mask, width, height);
  const grid = buildGrid(ply, frameCheck.signs, intrinsics, width, height);
  const { gw, gh, step, points, valid } = grid;
  const cellCount = gw * gh;
  const validCount = valid.reduce((sum, value) => sum + value, 0);
  const warnings = [];
  const cellPoint = (cell) => [points[cell * 3], points[cell * 3 + 1], points[cell * 3 + 2]];

  // Objects first: their pixels are excluded from structural plane fitting.
  const objectMasks = await loadObjectMasks(dir, index, grid, width, height);
  const { normals, hasNormal } = computeNormals(grid);
  // Overexposed cells (all channels at the sensor limit) carry no depth signal for a monocular model:
  // a blown-out window would otherwise form a plane of its own at the wrong depth.
  const overexposed = new Uint8Array(cellCount);
  if (grid.hasColor) {
    for (let cell = 0; cell < cellCount; cell += 1) {
      if (grid.colors[cell * 3] >= 250 && grid.colors[cell * 3 + 1] >= 250 && grid.colors[cell * 3 + 2] >= 250) overexposed[cell] = 1;
    }
  }
  const planeOptions = {
    minPoints: Math.max(200, Math.round(params.plane_min_fraction * validCount)),
    iterations: 400,
    sampleSize: 15000,
    normalAngleDeg: params.plane_normal_deg,
    baseTolerance: params.plane_base_tolerance_m,
    relativeTolerance: params.plane_relative_tolerance,
    regionReach: params.region_reach_cells,
    minRegionFraction: params.region_min_fraction,
    dominantRegionFraction: params.region_dominant_fraction,
    gridWidth: gw
  };

  // A mask that is a flat surface meters across is floor (or a rug), not an object: excluding it would hide
  // the floor from plane fitting. Up comes from a first plane pass over all cells.
  {
    const all = [];
    for (let cell = 0; cell < cellCount; cell += 1) if (valid[cell] && hasNormal[cell] && !overexposed[cell]) all.push(cell);
    const firstPass = extractPlanes(points, normals, all, { ...planeOptions, maxPlanes: 4 });
    const level = firstPass
      .filter((plane) => Math.abs(plane.normal[1]) > Math.cos((params.up_search_deg * Math.PI) / 180))
      .sort((a, b) => b.inliers.length - a.inliers.length)[0];
    const up0 = level ? (level.normal[1] < 0 ? level.normal : scale(level.normal, -1)) : [0, -1, 0];
    const side = normalize(cross(up0, Math.abs(up0[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1]));
    const side2 = cross(up0, side);
    for (const object of objectMasks) {
      object.instances = object.instances.filter((instance) => {
        const cells = [];
        for (let cell = 0; cell < cellCount; cell += 1) if (instance.gridMask[cell] && valid[cell]) cells.push(cell);
        if (cells.length < 30) return true;
        const spread = (axis) => {
          const values = sortedValues(cells.map((cell) => dot(cellPoint(cell), axis)));
          return percentile(values, 0.98) - percentile(values, 0.02);
        };
        const extent = Math.max(spread(side), spread(side2));
        const height = spread(up0);
        const flat = extent > params.flat_mask_min_extent_m && height < params.flat_mask_max_height_ratio * extent;
        if (flat) warnings.push(`Dropped a mask of "${object.object_id}": a flat surface ${round(extent, 2)} m across and ${round(height, 2)} m tall (floor or rug, not the object).`);
        return !flat;
      });
    }
  }

  const occupied = new Uint8Array(cellCount);
  for (const object of objectMasks) {
    for (const instance of object.instances) {
      const dilated = morph(morph(instance.gridMask, gw, gh, false), gw, gh, false);
      for (let i = 0; i < cellCount; i += 1) if (dilated[i]) occupied[i] = 1;
    }
  }

  const candidates = [];
  for (let cell = 0; cell < cellCount; cell += 1) {
    if (valid[cell] && hasNormal[cell] && !occupied[cell] && !overexposed[cell]) candidates.push(cell);
  }
  const rawPlanes = extractPlanes(points, normals, candidates, planeOptions).map((plane) => (plane.d < 0 ? { ...plane, normal: scale(plane.normal, -1), d: -plane.d } : plane));

  // Gravity: the best-supported near-horizontal plane defines up; otherwise assume a level camera.
  const cameraUp = [0, -1, 0];
  const horizontalCandidates = rawPlanes
    .filter((plane) => Math.abs(dot(plane.normal, cameraUp)) > Math.cos((UP_SEARCH_DEG * Math.PI) / 180))
    .sort((a, b) => b.inliers.length - a.inliers.length);
  let up = cameraUp;
  let gravitySource = "assumed_level_camera";
  if (horizontalCandidates.length) {
    const reference = horizontalCandidates[0];
    up = dot(reference.normal, cameraUp) > 0 ? reference.normal : scale(reference.normal, -1);
    const aligned = rawPlanes.filter((plane) => Math.abs(dot(plane.normal, up)) > Math.cos((5 * Math.PI) / 180));
    if (aligned.length > 1) {
      let sum = [0, 0, 0];
      for (const plane of aligned) sum = add(sum, scale(dot(plane.normal, up) > 0 ? plane.normal : scale(plane.normal, -1), plane.inliers.length));
      up = normalize(sum);
    }
    gravitySource = "horizontal_planes";
  } else {
    warnings.push("No horizontal plane found; gravity assumes a level camera.");
  }

  const sinHorizontal = Math.sin((HORIZONTAL_DEG * Math.PI) / 180);
  const cosHorizontal = Math.cos((HORIZONTAL_DEG * Math.PI) / 180);
  const heightOf = (plane) => -plane.d * Math.sign(dot(plane.normal, up));
  const horizontalPlanes = rawPlanes.filter((plane) => Math.abs(dot(plane.normal, up)) > cosHorizontal);
  const below = horizontalPlanes.filter((plane) => heightOf(plane) < -0.2);
  const maxBelow = Math.max(0, ...below.map((plane) => plane.inliers.length));
  const floorPlane = below
    .filter((plane) => plane.inliers.length >= 0.25 * maxBelow)
    .sort((a, b) => heightOf(a) - heightOf(b))[0];
  let cameraHeight;
  let floorSource;
  if (floorPlane) {
    cameraHeight = -heightOf(floorPlane);
    floorSource = "plane";
  } else {
    const heights = [];
    for (let cell = 0; cell < cellCount; cell += 1) if (valid[cell]) heights.push(dot(cellPoint(cell), up));
    cameraHeight = -percentile(sortedValues(heights), 0.02);
    floorSource = "lowest_points";
    warnings.push("No floor plane found; floor height uses the lowest 2% of points.");
  }

  // Layout frame: +Y up, -Z camera forward on the floor, +X right, origin on the floor below the camera.
  let forward = sub([0, 0, 1], scale(up, dot([0, 0, 1], up)));
  if (Math.hypot(...forward) < 1e-3) forward = sub([0, 1, 0], scale(up, dot([0, 1, 0], up)));
  forward = normalize(forward);
  const axisY = up;
  const axisZ = scale(forward, -1);
  const axisX = cross(axisY, axisZ);
  const R = [axisX, axisY, axisZ];
  const translation = [0, cameraHeight, 0];
  const toLayout = (p) => add(mulMat3Vec(R, p), translation);
  const toCamera = (p) => mulMat3Vec(transpose3(R), sub(p, translation));
  const cameraPosition = translation;

  const project = (pLayout) => {
    const p = toCamera(pLayout);
    return [((intrinsics.fx * p[0]) / p[2] + intrinsics.cx) / step, ((intrinsics.fy * p[1]) / p[2] + intrinsics.cy) / step, p[2]];
  };

  // Structure primitives.
  const aboveHeights = horizontalPlanes.filter((plane) => heightOf(plane) > 0.3);
  const maxAbove = Math.max(0, ...aboveHeights.map((plane) => plane.inliers.length));
  const ceilingPlane = aboveHeights
    .filter((plane) => plane.inliers.length >= 0.25 * maxAbove)
    .sort((a, b) => heightOf(b) - heightOf(a))[0];

  const counters = {};
  const structure = [];
  const labels = new Uint16Array(cellCount);
  const palette = [];
  for (const plane of rawPlanes) {
    const vertical = Math.abs(dot(plane.normal, up));
    let normal = mulMat3Vec(R, plane.normal);
    let kind;
    if (vertical > cosHorizontal) {
      kind = plane === floorPlane ? "floor" : plane === ceilingPlane ? "ceiling" : "horizontal_surface";
      normal = [0, Math.sign(normal[1]) || 1, 0];
    } else if (vertical < sinHorizontal) {
      kind = "vertical_surface";
      normal = normalize([normal[0], 0, normal[2]]);
    } else {
      kind = "inclined_surface";
    }

    const layoutPoints = Array.from(plane.inliers, (cell) => toLayout(cellPoint(cell)));
    const centroid = scale(layoutPoints.reduce((sum, p) => add(sum, p), [0, 0, 0]), 1 / layoutPoints.length);
    const offset = kind === "floor" ? 0 : -dot(normal, centroid);
    const anchor = kind === "floor" ? [centroid[0], 0, centroid[2]] : sub(centroid, scale(normal, dot(normal, centroid) + offset));
    const { u, v } = planeBasis(normal);
    const us = sortedValues(layoutPoints.map((p) => dot(sub(p, anchor), u)));
    const vs = sortedValues(layoutPoints.map((p) => dot(sub(p, anchor), v)));
    const u0 = percentile(us, 0.01), u1 = percentile(us, 0.99);
    const v0 = percentile(vs, 0.01), v1 = percentile(vs, 0.99);
    const center = add(anchor, add(scale(u, (u0 + u1) / 2), scale(v, (v0 + v1) / 2)));
    const size = [u1 - u0, v1 - v0];
    const residuals = layoutPoints.map((p) => (dot(normal, p) + offset) ** 2);
    const rms = Math.sqrt(residuals.reduce((sum, value) => sum + value, 0) / residuals.length);

    if (kind === "vertical_surface" && size[1] >= params.wall_min_height_m) kind = "wall";
    counters[kind] = (counters[kind] || 0) + 1;
    const id = kind === "floor" || kind === "ceiling" ? kind : `${kind.replace("_surface", "")}-${counters[kind]}`;
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) =>
      add(center, add(scale(u, (a * size[0]) / 2), scale(v, (b * size[1]) / 2))));
    const labelId = palette.length + 1;
    palette.push({ label: labelId, id, class: kind, color: CLASS_COLORS[kind] });
    for (const cell of plane.inliers) labels[cell] = labelId;

    structure.push({
      id,
      type: "plane",
      class: kind,
      center: roundVec(center),
      normal: roundVec(normal, 4),
      u_axis: roundVec(u, 4),
      v_axis: roundVec(v, 4),
      size: roundVec(size),
      rotation_quaternion: roundVec(quaternionFromBasis(u, v, normal), 5),
      corners: corners.map((corner) => roundVec(corner)),
      height_above_floor_m: kind === "floor" || vertical > cosHorizontal ? round(center[1]) : undefined,
      distance_from_camera_m: round(Math.abs(dot(normal, cameraPosition) + offset)),
      image_coverage: round(plane.inliers.length / cellCount, 4),
      fit_rms_m: round(rms, 4),
      extent_basis: "visible_points"
    });
  }

  // Object primitives: one gravity-aligned oriented box per mask instance.
  const objects = [];
  for (const [order, object] of objectMasks.entries()) {
    const color = objectColor(order);
    const labelId = palette.length + 1;
    palette.push({ label: labelId, id: object.object_id, class: "object", color });
    const instances = [];
    for (const instance of object.instances) {
      let mask = morph(instance.gridMask, gw, gh, true);
      if (mask.reduce((sum, value) => sum + value, 0) < 30) mask = instance.gridMask;
      const cells = [];
      for (let cell = 0; cell < cellCount; cell += 1) if (mask[cell] && valid[cell]) cells.push(cell);
      if (cells.length < 10) continue;
      for (let cell = 0; cell < cellCount; cell += 1) if (instance.gridMask[cell]) labels[cell] = labelId;

      const distances = cells.map((cell) => Math.hypot(...cellPoint(cell)));
      const sortedDistances = sortedValues(distances);
      const median = percentile(sortedDistances, 0.5);
      const mad = percentile(sortedValues(distances.map((d) => Math.abs(d - median))), 0.5);
      const limit = Math.max(3 * 1.4826 * mad, 0.05);
      const kept = cells.filter((_, k) => Math.abs(distances[k] - median) <= limit);
      const layoutPoints = kept.map((cell) => toLayout(cellPoint(cell)));
      const sampleStride = Math.max(1, Math.floor(layoutPoints.length / 4000));
      const sample = layoutPoints.filter((_, k) => k % sampleStride === 0);
      const rect = minAreaRect(sample.map((p) => p[0]), sample.map((p) => p[2]));
      const ys = sortedValues(layoutPoints.map((p) => p[1]));
      let bottom = percentile(ys, 0.02);
      const top = percentile(ys, 0.98);
      const objectHeight = top - bottom;
      const snap = Math.max(params.support_snap_min_m, params.support_snap_fraction * objectHeight);
      let support = "none_detected";
      if (bottom < snap) {
        bottom = 0;
        support = floorPlane ? "floor" : "floor_estimate";
      } else {
        const surface = structure.find((plane) =>
          plane.class === "horizontal_surface" &&
          Math.abs(bottom - plane.center[1]) < snap &&
          Math.abs(rect.center[0] - plane.center[0]) <= plane.size[0] / 2 &&
          Math.abs(rect.center[1] - plane.center[2]) <= plane.size[1] / 2);
        if (surface) {
          bottom = surface.center[1];
          support = surface.id;
        }
      }

      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let cell = 0; cell < cellCount; cell += 1) {
        if (!instance.gridMask[cell]) continue;
        const gx = cell % gw;
        const gy = Math.floor(cell / gw);
        x0 = Math.min(x0, gx * step); y0 = Math.min(y0, gy * step);
        x1 = Math.max(x1, Math.min(width, (gx + 1) * step)); y1 = Math.max(y1, Math.min(height, (gy + 1) * step));
      }

      instances.push({
        id: `${object.object_id}-${instances.length + 1}`,
        object_id: object.object_id,
        rect, bottom, top, snap, support, median,
        image_bbox_px: [x0, y0, x1, y1],
        gridMask: instance.gridMask,
        file: instance.file,
        score: instance.score,
        point_count: kept.length
      });
    }
    if (!instances.length) warnings.push(`No mask instance found for object "${object.object_id}".`);
    objects.push({
      object_id: object.object_id,
      name: object.name,
      segmentation_prompt: object.prompt,
      label: labelId,
      color,
      status: instances.length ? "found" : "not_found",
      instances
    });
  }

  // Objects resting on other objects (a laptop on a desk): the desk is excluded from plane fitting,
  // so its box top is the support surface. Lower boxes resolve first so stacks chain correctly.
  const drafts = objects.flatMap((object) => object.instances).sort((a, b) => a.bottom - b.bottom);
  const insideFootprint = (base, x, z, margin = 0.05) => {
    const c = Math.cos(base.rect.yaw);
    const s = Math.sin(base.rect.yaw);
    const dx = x - base.rect.center[0];
    const dz = z - base.rect.center[1];
    return Math.abs(dx * c - dz * s) <= base.rect.sizeX / 2 + margin && Math.abs(dx * s + dz * c) <= base.rect.sizeZ / 2 + margin;
  };
  // Image contact: below the object's resting band (its lowest rows, so an overhanging lamp shade
  // does not count), the next cells belong to the base's mask.
  const contactBelow = (top, isBase, reach = 4) => {
    const lowestByColumn = [];
    let topRow = gh, bottomRow = -1;
    for (let gx = 0; gx < gw; gx += 1) {
      for (let gy = gh - 1; gy >= 0; gy -= 1) {
        if (!top.gridMask[gy * gw + gx]) continue;
        lowestByColumn.push([gx, gy]);
        bottomRow = Math.max(bottomRow, gy);
        break;
      }
      for (let gy = 0; gy < gh; gy += 1) if (top.gridMask[gy * gw + gx]) { topRow = Math.min(topRow, gy); break; }
    }
    const band = Math.max(3, Math.round(0.1 * (bottomRow - topRow + 1)));
    const resting = lowestByColumn.filter(([, gy]) => gy >= bottomRow - band);
    const contacts = [];
    const touching = resting.filter(([gx, gy]) => {
      for (let dy = 1; dy <= reach && gy + dy < gh; dy += 1) {
        const cell = (gy + dy) * gw + gx;
        if (isBase(cell)) { contacts.push(cell); return true; }
      }
      return false;
    });
    return { touching: resting.length > 0 && touching.length / resting.length >= 0.5, resting, contacts };
  };
  const restsOnInImage = (top, base) => contactBelow(top, (cell) => base.gridMask[cell]).touching;
  // A contact counts only if the support cells are near the object's resting band in 3D (not a far background patch).
  const nearInPlan = (draft, contact) => {
    if (!contact.contacts.length) return false;
    const centroid = (cells) => scale(cells.reduce((sum, cell) => add(sum, toLayout(cellPoint(cell))), [0, 0, 0]), 1 / cells.length);
    const a = centroid(contact.resting.map(([gx, gy]) => gy * gw + gx).filter((cell) => valid[cell]));
    const b = centroid(contact.contacts.filter((cell) => valid[cell]));
    return Math.hypot(a[0] - b[0], a[2] - b[2]) < Math.max(0.5, 0.75 * Math.max(draft.rect.sizeX, draft.rect.sizeZ));
  };
  const floorLabel = palette.find((entry) => entry.class === "floor")?.label;
  const surfaceLabels = palette.filter((entry) => entry.class === "horizontal_surface");
  for (const draft of drafts) {
    if (draft.support !== "none_detected") continue;
    const base = drafts
      .filter((other) => other !== draft && other.top < draft.top &&
        Math.abs(draft.bottom - other.top) < draft.snap &&
        (insideFootprint(other, draft.rect.center[0], draft.rect.center[1]) || restsOnInImage(draft, other)))
      .sort((a, b) => Math.abs(draft.bottom - a.top) - Math.abs(draft.bottom - b.top))[0];
    if (base) {
      draft.bottom = base.top;
      draft.support = base.id;
      draft.support_basis = "object_contact";
      continue;
    }
    // Thin or hidden legs leave the lowest visible points above the support; the image contact below the
    // resting band still shows what the object stands on.
    if (floorLabel) {
      const contact = contactBelow(draft, (cell) => labels[cell] === floorLabel);
      if (contact.touching && nearInPlan(draft, contact)) {
        draft.bottom = 0;
        draft.support = "floor";
        draft.support_basis = "image_contact";
        continue;
      }
    }
    for (const surface of surfaceLabels) {
      const plane = structure.find((p) => p.id === surface.id);
      const contact = contactBelow(draft, (cell) => labels[cell] === surface.label);
      if (plane && plane.center[1] < draft.top && contact.touching && nearInPlan(draft, contact)) {
        draft.bottom = plane.center[1];
        draft.support = plane.id;
        draft.support_basis = "image_contact";
        break;
      }
    }
  }

  const finalizeBox = (draft) => {
    const { rect, bottom, top, support } = draft;
    const center = [rect.center[0], (bottom + top) / 2, rect.center[1]];
    const size = [rect.sizeX, top - bottom, rect.sizeZ];
    const localX = [Math.cos(rect.yaw), 0, -Math.sin(rect.yaw)];
    const localZ = [Math.sin(rect.yaw), 0, Math.cos(rect.yaw)];
    const view = normalize([center[0] - cameraPosition[0], 0, center[2] - cameraPosition[2]]);
    const depthAxis = Math.abs(dot(view, localX)) > Math.abs(dot(view, localZ)) ? "x" : "z";
    const corners = [];
    for (const sy of [-1, 1]) {
      for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        corners.push(add(center, add(add(scale(localX, (sx * size[0]) / 2), [0, (sy * size[1]) / 2, 0]), scale(localZ, (sz * size[2]) / 2))));
      }
    }
    return {
      id: draft.id,
      type: "box",
      class: "object",
      object_id: draft.object_id,
      center: roundVec(center),
      size: roundVec(size),
      yaw_deg: round(degrees(rect.yaw), 1),
      rotation_quaternion: roundVec([0, Math.sin(rect.yaw / 2), 0, Math.cos(rect.yaw / 2)], 5),
      size_basis: {
        x: depthAxis === "x" ? "visible_lower_bound" : "observed",
        y: support === "none_detected" ? "observed" : "observed_to_support",
        z: depthAxis === "z" ? "visible_lower_bound" : "observed"
      },
      bottom_y_m: round(bottom),
      support,
      support_basis: draft.support_basis || (support === "none_detected" ? null : "height"),
      distance_from_camera_m: round(draft.median),
      corners: corners.map((corner) => roundVec(corner)),
      image_bbox_px: draft.image_bbox_px,
      mask_file: draft.file,
      mask_score: draft.score === undefined ? undefined : round(draft.score, 3),
      point_count: draft.point_count
    };
  };
  for (const object of objects) object.instances = object.instances.map(finalizeBox);

  // Lighting from the intrinsic light estimate of the same photo, when one exists.
  const lightEstimate = await loadLightEstimate(world, depthRequest.input_files?.[0]);
  const planeNormalOf = new Map();
  rawPlanes.forEach((plane, k) => { for (const cell of plane.inliers) planeNormalOf.set(cell, structure[k].normal); });
  const lightResult = lightEstimate
    ? analyzeLighting({ light: lightEstimate, grid, normals, hasNormal, planeNormalOf, R, toLayout, labels, palette, structure, width, height, cameraPosition, intrinsics })
    : undefined;
  const lighting = lightResult?.lighting || { status: "no_light_estimate", hint: "Run image-blast-light before the layout for lighting." };
  const emitterPrimitives = lightResult?.emitters || [];

  // Raster outputs at grid resolution.
  const labelImage = new Uint8Array(cellCount * 3);
  const guideImage = new Uint8Array(cellCount * 3);
  const depthImage = new Uint16Array(cellCount);
  const inverse = [];
  for (let cell = 0; cell < cellCount; cell += 1) if (valid[cell]) inverse.push(1 / points[cell * 3 + 2]);
  const inverseSorted = sortedValues(inverse);
  const inverseLow = percentile(inverseSorted, 0.01);
  const inverseHigh = percentile(inverseSorted, 0.99);
  const controlImage = new Uint8Array(cellCount);
  for (let cell = 0; cell < cellCount; cell += 1) {
    const color = labels[cell] ? palette[labels[cell] - 1].color : [0, 0, 0];
    labelImage.set(color, cell * 3);
    const base = grid.hasColor ? [grid.colors[cell * 3], grid.colors[cell * 3 + 1], grid.colors[cell * 3 + 2]] : [128, 128, 128];
    for (let c = 0; c < 3; c += 1) {
      guideImage[cell * 3 + c] = !valid[cell] ? 0 : labels[cell] ? Math.round(0.45 * base[c] + 0.55 * color[c]) : Math.round(0.6 * base[c]);
    }
    if (valid[cell]) {
      const z = points[cell * 3 + 2];
      depthImage[cell] = Math.min(65535, Math.round(z * 1000));
      const t = (1 / z - inverseLow) / Math.max(1e-9, inverseHigh - inverseLow);
      controlImage[cell] = Math.round(255 * Math.max(0, Math.min(1, t)));
    }
  }
  const drawPolyline = (cornersLayout, edges, color) => {
    for (const [a, b] of edges) {
      let p = toCamera(cornersLayout[a]);
      let q = toCamera(cornersLayout[b]);
      const near = 0.05;
      if (p[2] < near && q[2] < near) continue;
      if (p[2] < near) p = add(p, scale(sub(q, p), (near - p[2]) / (q[2] - p[2])));
      if (q[2] < near) q = add(q, scale(sub(p, q), (near - q[2]) / (p[2] - q[2])));
      const pa = project(toLayout(p));
      const pb = project(toLayout(q));
      drawLine(guideImage, gw, gh, pa, pb, color);
    }
  };
  for (const plane of structure) drawPolyline(plane.corners, [[0, 1], [1, 2], [2, 3], [3, 0]], CLASS_COLORS[plane.class]);
  const boxEdges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (const emitter of emitterPrimitives) drawPolyline(emitter.corners, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], [255, 236, 120]);
  for (const object of objects) {
    const bright = object.color.map((c) => Math.min(255, Math.round(c * 0.5 + 128)));
    for (const instance of object.instances) drawPolyline(instance.corners, boxEdges, bright);
  }
  await writeFile(files.labels_png, encodePng({ width: gw, height: gh, channels: 3, data: labelImage }));
  await writeFile(files.guide_png, encodePng({ width: gw, height: gh, channels: 3, data: guideImage }));
  await writeFile(files.depth_png, encodePng({ width: gw, height: gh, channels: 1, data: depthImage }));
  await writeFile(files.depth_control_png, encodePng({ width: gw, height: gh, channels: 1, data: controlImage }));

  // Camera pose. OpenGL camera axes (right, up, back) expressed in the layout frame.
  const camRight = mulMat3Vec(R, [1, 0, 0]);
  const camUpGl = mulMat3Vec(R, [0, -1, 0]);
  const camBack = mulMat3Vec(R, [0, 0, -1]);
  const cameraQuaternion = quaternionFromBasis(camRight, camUpGl, camBack);
  const fovX = 2 * Math.atan(width / (2 * intrinsics.fx));
  const fovY = 2 * Math.atan(height / (2 * intrinsics.fy));
  const pitch = degrees(Math.asin(Math.max(-1, Math.min(1, dot(mulMat3Vec(R, [0, 0, 1]), [0, 1, 0])))));
  const roll = degrees(Math.atan2(camRight[1], camUpGl[1]));
  const worldFromCameraOpenCv = [
    [R[0][0], R[0][1], R[0][2], translation[0]],
    [R[1][0], R[1][1], R[1][2], translation[1]],
    [R[2][0], R[2][1], R[2][2], translation[2]],
    [0, 0, 0, 1]
  ].map((row) => roundVec(row, 6));

  // Blockout GLB with the source camera.
  const materialIndex = new Map();
  const materials = [];
  const materialFor = (name, color, doubleSided) => {
    if (!materialIndex.has(name)) {
      materials.push({ name, color: color.map((c) => c / 255), doubleSided });
      materialIndex.set(name, materials.length - 1);
    }
    return materialIndex.get(name);
  };
  const nodes = [
    ...structure.map((plane) => ({
      name: plane.id,
      shape: "quad",
      material: materialFor(plane.class, CLASS_COLORS[plane.class], true),
      translation: plane.center,
      rotation: plane.rotation_quaternion,
      scale: [Math.max(plane.size[0], 1e-3), Math.max(plane.size[1], 1e-3), 1],
      extras: { id: plane.id, class: plane.class }
    })),
    ...objects.flatMap((object) => object.instances.map((instance) => ({
      name: instance.id,
      shape: "box",
      material: materialFor(object.object_id, object.color, false),
      translation: instance.center,
      rotation: instance.rotation_quaternion,
      scale: instance.size.map((value) => Math.max(value, 1e-3)),
      extras: { id: instance.id, class: "object", object_id: object.object_id, name: object.name, size_basis: instance.size_basis }
    })))
  ];
  // Lights: emissive quads for light sources plus a KHR_lights_punctual directional light along the
  // fitted dominant direction (intensities are relative, not photometric).
  const lights = [];
  for (const emitter of emitterPrimitives) {
    const material = materialFor("light-source", [255, 236, 170], true);
    materials[material].emissive = emitter.color_linear_rgb.map((c) => Math.min(1, c));
    nodes.push({
      name: emitter.id,
      shape: "quad",
      material,
      translation: emitter.center,
      rotation: emitter.rotation_quaternion,
      scale: [Math.max(emitter.size[0], 1e-3), Math.max(emitter.size[1], 1e-3), 1],
      extras: { id: emitter.id, class: "light_source", on_surface: emitter.on_surface }
    });
  }
  if (lighting.status === "fitted" && lighting.dominant_light.confidence !== "low") {
    const z = lighting.dominant_light.direction;
    const x = Math.abs(z[1]) > 0.999 ? [1, 0, 0] : normalize(cross([0, 1, 0], z));
    lights.push({ name: "dominant-light", type: "directional", color: lighting.light_color.linear_rgb.map((c) => Math.min(1, c)), intensity: 1 });
    nodes.push({
      name: "dominant-light",
      light: 0,
      translation: roundVec(add([0, 1.5, -3], scale(z, 3))),
      rotation: roundVec(quaternionFromBasis(x, cross(z, x), z), 6),
      extras: { class: "dominant_light", direction: z, directionality: lighting.dominant_light.directionality }
    });
  }

  await writeFile(files.layout_glb, writeLayoutGlb({
    nodes,
    materials,
    lights,
    camera: {
      name: "source-camera",
      yfov: fovY,
      aspectRatio: width / height,
      translation: roundVec(cameraPosition),
      rotation: roundVec(cameraQuaternion, 6),
      extras: { image_size: [width, height] }
    },
    extras: { world, layout_index: index, units: "meters" }
  }));

  // Text summaries for prompt-driven models.
  const fmt = (value) => round(value, 2);
  const structureLines = [
    `camera ${fmt(cameraHeight)} m above the ${floorPlane ? "floor" : "lowest visible ground"}, pitched ${fmt(Math.abs(pitch))} deg ${pitch < 0 ? "down" : "up"}, horizontal field of view ${round(degrees(fovX), 0)} deg.`
  ];
  for (const plane of structure) {
    if (plane.class === "floor") {
      const zs = plane.corners.map((corner) => -corner[2]);
      structureLines.push(`floor visible from ${fmt(Math.min(...zs))} m to ${fmt(Math.max(...zs))} m ahead, ${fmt(plane.size[0])} m across.`);
    } else if (plane.class === "ceiling") {
      structureLines.push(`ceiling ${fmt(plane.center[1])} m above the floor.`);
    } else if (plane.class === "wall" || plane.class === "vertical_surface") {
      const facing = wallFacing(plane.normal);
      const where = plane.normal[2] > 0.7 ? `${fmt(-plane.center[2])} m ahead` : `${fmt(Math.abs(plane.center[0]))} m to the ${plane.center[0] < 0 ? "left" : "right"}`;
      structureLines.push(`${plane.class === "wall" ? "wall" : "vertical surface"} ${where}, ${facing}, ${fmt(plane.size[0])} m wide, ${fmt(plane.size[1])} m tall visible.`);
    } else if (plane.class === "horizontal_surface") {
      structureLines.push(`flat horizontal surface ${fmt(plane.center[1])} m above the floor, ${fmt(plane.size[0])} x ${fmt(plane.size[1])} m, ${directionWords(plane.center)}.`);
    } else {
      structureLines.push(`inclined surface ${directionWords(plane.center)}, ${fmt(plane.size[0])} x ${fmt(plane.size[1])} m.`);
    }
  }
  const objectLines = objects.flatMap((object) => object.instances.map((instance) => {
    const [sx, sy, sz] = instance.size;
    const lower = instance.size_basis.x === "visible_lower_bound" ? "width" : "depth";
    const supportText = instance.support === "floor" || instance.support === "floor_estimate"
      ? "on the floor"
      : instance.support === "none_detected" ? `bottom ${fmt(instance.bottom_y_m)} m above the floor` : `on ${instance.support}`;
    return `${object.name}: ${fmt(sx)} m wide x ${fmt(sy)} m tall x ${fmt(sz)} m deep (${lower} is a visible lower bound), ${supportText}, ${directionWords(instance.center)}.`;
  }));

  const layout = {
    schema_version: 1,
    kind: "layout",
    world,
    index,
    created_at: new Date().toISOString(),
    source_image: depthRequest.input_files?.[0],
    units: "meters",
    frame: {
      handedness: "right",
      up: "+Y",
      forward: "-Z",
      right: "+X",
      origin: "floor point directly below the source camera",
      rotations: "quaternions are [x, y, z, w]; yaw_deg is rotation about +Y",
      plane_quaternion: "maps unit quad local +X to u_axis, +Y to v_axis, +Z to normal",
      box_quaternion: "maps box local axes to the frame; size is [x, y, z] in local axes"
    },
    camera: {
      image_size: [width, height],
      intrinsics_px: { fx: round(intrinsics.fx, 2), fy: round(intrinsics.fy, 2), cx: round(intrinsics.cx, 2), cy: round(intrinsics.cy, 2) },
      fov_x_deg: round(degrees(fovX), 2),
      fov_y_deg: round(degrees(fovY), 2),
      height_m: round(cameraHeight),
      pitch_deg: round(pitch, 2),
      roll_deg: round(roll, 2),
      angle_convention: "pitch_deg > 0 looks up; roll_deg > 0 when the image right side is raised",
      position: roundVec(cameraPosition),
      rotation_quaternion: roundVec(cameraQuaternion, 6),
      rotation_convention: "OpenGL camera: looks down local -Z, local +Y up",
      matrix_world_from_camera_opencv: worldFromCameraOpenCv,
      matrix_convention: "row-major 4x4; maps OpenCV camera coordinates (x right, y down, z forward) to the layout frame"
    },
    gravity: {
      source: gravitySource,
      floor_source: floorSource,
      up_in_camera_opencv: roundVec(up, 5)
    },
    depth: {
      provider: depthRequest.endpoint,
      model: depthRequest.input?.model,
      valid_fraction: round(validCount / cellCount, 4),
      grid_size: [gw, gh],
      grid_step_px: step,
      point_frame_check: { signs: frameCheck.signs, basis: frameCheck.basis, candidates: frameCheck.candidates }
    },
    structure,
    objects,
    lighting,
    labels: {
      file: files.labels_png,
      size: [gw, gh],
      background: [0, 0, 0],
      palette
    },
    files: {
      layout_glb: files.layout_glb,
      labels_png: files.labels_png,
      guide_png: files.guide_png,
      depth_png: files.depth_png,
      depth_png_encoding: "uint16 millimeters along the camera optical axis, 0 = no data",
      depth_control_png: files.depth_control_png,
      depth_control_encoding: "uint8 normalized inverse depth, near = white",
      ...((await pathExists(files.depth_mesh)) ? {
        depth_mesh_glb: files.depth_mesh,
        depth_mesh_frame: "provider camera frame, not the layout frame"
      } : {}),
      points_ply: files.points
    },
    solver_params: params,
    prompts: {
      structure: structureLines.join(" "),
      objects: objectLines.join(" "),
      lighting: lightingPrompt(lighting)
    },
    warnings
  };
  const outputPath = artifactPath(dir, index, "layout", ".json");
  await writeJson(outputPath, layout);

  return {
    world,
    index,
    layout_json: outputPath,
    files: Object.fromEntries(Object.entries(files).filter(([key]) => !key.includes("encoding"))),
    camera: { height_m: layout.camera.height_m, pitch_deg: layout.camera.pitch_deg, fov_x_deg: layout.camera.fov_x_deg },
    structure: structure.map((plane) => `${plane.id} ${plane.size.join("x")} m`),
    objects: objects.map((object) => `${object.object_id}: ${object.instances.length} instance(s)`),
    lighting: lighting.status === "fitted"
      ? { dominant: lighting.dominant_light, sh_fit_r2: lighting.irradiance_sh.fit_r2, emitters: lighting.emitters.map((e) => `${e.id} ${e.size.join("x")} m on ${e.on_surface}`) }
      : lighting,
    warnings
  };
}

async function latestLayoutIndex(dir) {
  const requests = await requestMetadataFiles(dir, { slug: "layout", scope: "depth" });
  if (!requests.length) throw new Error(`No layout depth requests found in ${dir}.`);
  return requests[0].index;
}

async function main() {
  const { flags } = parseArgs();
  const world = one(flags, "world");
  if (!world) throw new Error("Usage: node .claude/scripts/layout/build-layout.mjs --world <world> [--index N]");
  const explicit = one(flags, "index");
  const index = explicit !== undefined ? Number(explicit) : await latestLayoutIndex(layoutDir(world));
  const params = one(flags, "params") ? JSON.parse(one(flags, "params")) : {};
  const unknown = Object.keys(params).filter((key) => !(key in DEFAULT_PARAMS));
  if (unknown.length) throw new Error(`Unknown layout params: ${unknown.join(", ")}`);
  console.log(JSON.stringify(await buildLayout({ world, index, params }), null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}
