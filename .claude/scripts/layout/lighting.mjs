import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { pathExists, readJson } from "../asset-pipeline/fal-queue.mjs";
import { parseIndexedName } from "../asset-pipeline/request-metadata.mjs";
import { add, cross, degrees, dot, fitPlane, mulMat3Vec, normalize, percentile, quaternionFromBasis, scale, sortedValues, sub } from "./geometry.mjs";
import { decodePng } from "./png.mjs";

const LUMA = [0.2126, 0.7152, 0.0722];
const SH_BASIS = "[1, y, z, x, x*y, y*z, 3*z*z - 1, x*z, x*x - y*y] of the unit surface normal in the layout frame";
const round = (value, digits = 3) => Number(value.toFixed(digits));
const roundVec = (vector, digits = 3) => vector.map((value) => round(value, digits));

const shBasis = ([x, y, z]) => [1, y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y];

// Best single distant light for shading = ambient + k * max(0, n . l): sphere search, closed-form
// ambient and k per direction, then local refinement. Unbiased by which normals the photo happens to show.
function fitDirectionalLight(samples, prior) {
  const stride = Math.max(1, Math.floor(samples.length / 20000));
  const subset = samples.filter((_, i) => i % stride === 0);
  const evaluate = (l) => {
    let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (const s of subset) {
      const x = Math.max(0, s.n[0] * l[0] + s.n[1] * l[1] + s.n[2] * l[2]);
      sx += x; sy += s.luma; sxx += x * x; sxy += x * s.luma; syy += s.luma * s.luma;
    }
    const n = subset.length;
    const varX = sxx - (sx * sx) / n;
    if (varX < 1e-9) return undefined;
    const k = (sxy - (sx * sy) / n) / varX;
    if (k <= 0) return undefined;
    const ambient = (sy - k * sx) / n;
    const sse = syy - 2 * ambient * sy - 2 * k * sxy + n * ambient * ambient + 2 * ambient * k * sx + k * k * sxx;
    return { l, k, ambient, sse, meanX: sx / n, meanY: sy / n, tss: syy - (sy * sy) / n };
  };
  let best;
  const evaluated = [];
  const count = 4000;
  for (let i = 0; i < count; i += 1) {
    const y = 1 - (2 * (i + 0.5)) / count;
    const r = Math.sqrt(1 - y * y);
    const phi = i * Math.PI * (3 - Math.sqrt(5));
    const candidate = evaluate([r * Math.cos(phi), y, r * Math.sin(phi)]);
    if (!candidate) continue;
    evaluated.push(candidate);
    if (!best || candidate.sse < best.sse) best = candidate;
  }
  if (!best) return undefined;
  for (let radius = 0.05; radius > 0.0005; radius /= 2) {
    for (let improved = true; improved;) {
      improved = false;
      for (const delta of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
        const candidate = evaluate(normalize(add(best.l, scale(delta, radius))));
        if (candidate && candidate.sse < best.sse) { best = candidate; improved = true; }
      }
    }
  }
  // Directions the pixels cannot tell apart (R^2 within 0.005 of the best) form the ambiguity set.
  // Inside it, the direction closest to the smooth SH field's band-1 direction is chosen.
  const r2Of = (candidate) => 1 - candidate.sse / Math.max(candidate.tss, 1e-12);
  const bestR2 = r2Of(best);
  const ambiguous = [best, ...evaluated.filter((candidate) => r2Of(candidate) >= bestR2 - 0.005)];
  const chosen = prior ? ambiguous.reduce((a, b) => (dot(b.l, prior) > dot(a.l, prior) ? b : a)) : best;
  const spread = Math.max(...ambiguous.map((candidate) => degrees(Math.acos(Math.max(-1, Math.min(1, dot(candidate.l, chosen.l)))))));
  return {
    direction: chosen.l,
    ambient: chosen.ambient,
    intensity: chosen.k,
    r2: r2Of(chosen),
    directional_share: (chosen.k * chosen.meanX) / Math.max(chosen.meanY, 1e-12),
    uncertainty_deg: spread
  };
}

// Gaussian elimination with partial pivoting; columns of B are solved together.
function solve(A, B) {
  const n = A.length;
  const m = A.map((row, i) => [...row, ...B[i]]);
  const width = m[0].length;
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = m[row][col] / m[col][col];
      for (let k = col; k < width; k += 1) m[row][k] -= factor * m[col][k];
    }
  }
  return m.map((row, i) => row.slice(n).map((value) => value / row[i]));
}

// Latest light estimate made from the same source image as the layout.
export async function loadLightEstimate(world, sourceImage) {
  const dir = path.join("worlds", world, "output", "light");
  const entries = await readdir(dir).catch(() => []);
  const candidates = entries
    .map((name) => parseIndexedName(name))
    .filter((parsed) => parsed && !parsed.hidden && parsed.slug === "light" && parsed.extension === ".json")
    .sort((a, b) => b.index - a.index);
  for (const candidate of candidates) {
    const file = path.join(dir, candidate.name);
    const json = await readJson(file);
    if (sourceImage && path.normalize(json.source_image) !== path.normalize(sourceImage)) continue;
    if (!(await pathExists(json.files.shading)) || !(await pathExists(json.files.emitters))) continue;
    const shading = decodePng(await readFile(json.files.shading), { keep16: true });
    const emitters = decodePng(await readFile(json.files.emitters));
    return { file, json, shading, shadingScale: json.encoding.shading["value = sample / 65535 *"] / 65535, emitters };
  }
  return undefined;
}

function directionWords(direction) {
  const azimuth = degrees(Math.atan2(direction[0], -direction[2]));
  const side = Math.abs(azimuth) < 20 ? "ahead of the camera"
    : Math.abs(azimuth) > 160 ? "behind the camera"
    : `${azimuth > 0 ? "right" : "left"}${Math.abs(azimuth) < 70 ? "-front" : Math.abs(azimuth) > 110 ? "-back" : ""}`;
  return { azimuth, side };
}

/**
 * Fits a second-order spherical-harmonic irradiance field to the model's diffuse shading against
 * the depth normals, and lifts the model's light-source regions to 3D area lights.
 */
export function analyzeLighting({ light, grid, normals, hasNormal, planeNormalOf, R, toLayout, labels, palette, structure, width, height, cameraPosition, intrinsics }) {
  const { gw, gh, step, valid, points } = grid;
  const cellCount = gw * gh;
  const sx = light.shading.width / width;
  const sy = light.shading.height / height;
  const pixelOf = (cell) => {
    const px = Math.min(width - 1, (cell % gw) * step + Math.floor(step / 2));
    const py = Math.min(height - 1, Math.floor(cell / gw) * step + Math.floor(step / 2));
    return Math.min(light.shading.height - 1, Math.floor(py * sy)) * light.shading.width + Math.min(light.shading.width - 1, Math.floor(px * sx));
  };
  const emitterOf = new Uint8Array(cellCount);
  const kinds = Object.fromEntries(light.json.emitters.map((emitter) => [emitter.id, emitter]));
  for (let cell = 0; cell < cellCount; cell += 1) emitterOf[cell] = light.emitters.data[pixelOf(cell) * light.emitters.channels];

  // Spherical-harmonic irradiance fit (least squares with a small ridge for unseen normal directions).
  const AtA = Array.from({ length: 9 }, () => new Array(9).fill(0));
  const Atb = Array.from({ length: 9 }, () => [0, 0, 0]);
  const samples = [];
  for (let cell = 0; cell < cellCount; cell += 1) {
    if (!valid[cell] || emitterOf[cell]) continue;
    // A fitted plane's normal (from thousands of points) beats a per-cell normal from noisy depth.
    const planeNormal = planeNormalOf?.get(cell);
    if (!planeNormal && !hasNormal[cell]) continue;
    const normal = planeNormal || normalize(mulMat3Vec(R, [normals[cell * 3], normals[cell * 3 + 1], normals[cell * 3 + 2]]));
    const base = pixelOf(cell) * 3;
    const rgb = [0, 1, 2].map((c) => light.shading.data[base + c] * light.shadingScale);
    const y = shBasis(normal);
    for (let i = 0; i < 9; i += 1) {
      for (let j = 0; j < 9; j += 1) AtA[i][j] += y[i] * y[j];
      for (let c = 0; c < 3; c += 1) Atb[i][c] += y[i] * rgb[c];
    }
    samples.push({ y, n: normal, luma: dot(rgb, LUMA) });
  }
  if (samples.length < 100) return { lighting: { status: "insufficient_normals", samples: samples.length }, emitters: [], emitterOf };
  const ridge = 1e-3 * AtA.reduce((sum, row, i) => sum + row[i], 0) / 9;
  const coefficients = solve(AtA.map((row, i) => row.map((value, j) => value + (i === j ? ridge : 0))), Atb);
  const lumaCoefficients = coefficients.map((rgb) => dot(rgb, LUMA));
  const mean = samples.reduce((sum, s) => sum + s.luma, 0) / samples.length;
  let residual = 0, total = 0;
  for (const s of samples) {
    const predicted = s.y.reduce((sum, value, i) => sum + value * lumaCoefficients[i], 0);
    residual += (s.luma - predicted) ** 2;
    total += (s.luma - mean) ** 2;
  }
  const band1 = [lumaCoefficients[3], lumaCoefficients[1], lumaCoefficients[2]];
  const directional = fitDirectionalLight(samples, normalize(band1));
  const dominant = directional ? directional.direction : normalize(band1);
  const { azimuth, side } = directionWords(dominant);
  const elevation = degrees(Math.asin(Math.max(-1, Math.min(1, dominant[1]))));
  const directionality = Math.hypot(...band1) / Math.max(1e-9, lumaCoefficients[0]);

  // Light sources: group regions by the surface they lie on; regions on a confirmed object are highlights.
  const groups = new Map();
  const highlights = [];
  const sunlitPatches = [];
  for (const emitter of light.json.emitters) {
    const cells = [];
    for (let cell = 0; cell < cellCount; cell += 1) if (emitterOf[cell] === emitter.id && valid[cell]) cells.push(cell);
    if (cells.length < 10) continue;
    const votes = new Map();
    for (const cell of cells) if (labels[cell]) votes.set(labels[cell], (votes.get(labels[cell]) || 0) + 1);
    const [topLabel, topVotes] = [...votes.entries()].sort((a, b) => b[1] - a[1])[0] || [0, 0];
    const owner = topLabel ? palette[topLabel - 1] : undefined;
    if (emitter.kind !== "light_source") continue;
    if (owner?.class === "object" && topVotes >= 0.5 * cells.length) {
      highlights.push({ region: emitter.id, object_id: owner.id });
      continue;
    }
    // An overexposed window has no usable depth of its own; the surface around it (a ring of cells) holds it.
    // Each labeled ring cell votes for its plane or object; neighboring panes and invalid cells carry no evidence.
    // The region lies on the most common label when that label is a structure plane. Objects compete one by one,
    // so a desk, a lamp and a laptop in front of a window do not outvote the wall around it together.
    const seen = new Uint8Array(cellCount);
    for (const cell of cells) seen[cell] = 1;
    const ringVotes = new Map();
    for (const cell of cells) {
      const gx = cell % gw, gy = Math.floor(cell / gw);
      for (let dy = -3; dy <= 3; dy += 1) {
        for (let dx = -3; dx <= 3; dx += 1) {
          const x = gx + dx, y = gy + dy;
          if (x < 0 || y < 0 || x >= gw || y >= gh) continue;
          const other = y * gw + x;
          if (seen[other]) continue;
          seen[other] = 1;
          if (!labels[other]) continue;
          ringVotes.set(labels[other], (ringVotes.get(labels[other]) || 0) + 1);
        }
      }
    }
    const ringTopEntry = palette[([...ringVotes.entries()].sort((a, b) => b[1] - a[1])[0] || [0])[0] - 1];
    const ownPlane = owner && owner.class !== "object" && topVotes >= 0.5 * cells.length ? owner.id : undefined;
    const surfaceId = ringTopEntry && ringTopEntry.class !== "object" ? ringTopEntry.id : ownPlane;
    // Floors and tabletops do not hold light sources: an overexposed patch there is sunlight spill.
    const surfaceClass = structure.find((plane) => plane.id === surfaceId)?.class;
    if (surfaceClass === "floor" || surfaceClass === "horizontal_surface") {
      sunlitPatches.push({ region: emitter.id, on_surface: surfaceId });
      continue;
    }
    const key = surfaceId || `region-${emitter.id}`;
    if (!groups.has(key)) groups.set(key, { surface: key.startsWith("region-") ? undefined : key, cells: [], regions: [], luminance: [] });
    const group = groups.get(key);
    group.cells.push(...cells);
    group.regions.push(emitter.id);
    group.luminance.push(emitter.relative_luminance);
  }

  const roomCenter = add(structure.find((plane) => plane.class === "floor")?.center || [0, 0, -3], [0, 1, 0]);
  const emitters = [];
  for (const group of groups.values()) {
    const surface = structure.find((plane) => plane.id === group.surface);
    // On a known surface, region pixels are placed by intersecting their camera rays with the plane.
    const layoutPoints = group.cells.map((cell) => {
      if (surface && intrinsics) {
        const px = (cell % gw) * step + step / 2, py = Math.floor(cell / gw) * step + step / 2;
        const ray = normalize(mulMat3Vec(R, [(px - intrinsics.cx) / intrinsics.fx, (py - intrinsics.cy) / intrinsics.fy, 1]));
        const denom = dot(surface.normal, ray);
        const t = Math.abs(denom) > 1e-6 ? -(dot(surface.normal, cameraPosition) - dot(surface.normal, surface.center)) / denom : -1;
        if (t > 0) return add(cameraPosition, scale(ray, t));
      }
      return toLayout([points[cell * 3], points[cell * 3 + 1], points[cell * 3 + 2]]);
    });
    let normal;
    if (surface) {
      normal = surface.normal;
    } else {
      const flat = Float32Array.from(layoutPoints.flat());
      normal = fitPlane(flat, layoutPoints.map((_, i) => i)).normal;
      if (dot(normal, sub(cameraPosition, layoutPoints[0])) < 0) normal = scale(normal, -1);
    }
    const u = Math.abs(normal[1]) > 0.999 ? [1, 0, 0] : normalize(cross([0, 1, 0], normal));
    const v = cross(normal, u);
    const origin = scale(layoutPoints.reduce((sum, p) => add(sum, p), [0, 0, 0]), 1 / layoutPoints.length);
    const us = sortedValues(layoutPoints.map((p) => dot(sub(p, origin), u)));
    const vs = sortedValues(layoutPoints.map((p) => dot(sub(p, origin), v)));
    const u0 = percentile(us, 0.02), u1 = percentile(us, 0.98), v0 = percentile(vs, 0.02), v1 = percentile(vs, 0.98);
    const center = add(origin, add(scale(u, (u0 + u1) / 2), scale(v, (v0 + v1) / 2)));
    const size = [u1 - u0, v1 - v0];
    const vertical = Math.abs(normal[1]) < 0.26;
    const label = vertical ? "window" : normal[1] < 0 ? "ceiling-light" : "light";
    const toward = normalize(sub(center, roomCenter));
    emitters.push({
      id: `${label}-${emitters.filter((e) => e.id.startsWith(label)).length + 1}`,
      type: "area_light",
      class: "light_source",
      on_surface: group.surface || null,
      center: roundVec(center),
      normal: roundVec(normal, 4),
      u_axis: roundVec(u, 4),
      v_axis: roundVec(v, 4),
      size: roundVec(size),
      rotation_quaternion: roundVec(quaternionFromBasis(u, v, normal), 5),
      corners: [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => roundVec(add(center, add(scale(u, (a * size[0]) / 2), scale(v, (b * size[1]) / 2))))),
      color_linear_rgb: light.json.light_color.linear_rgb,
      saturated: true,
      relative_luminance_min: round(Math.min(...group.luminance), 2),
      image_regions: group.regions,
      angle_to_dominant_light_deg: round(degrees(Math.acos(Math.max(-1, Math.min(1, dot(toward, dominant))))), 1)
    });
  }
  emitters.sort((a, b) => b.size[0] * b.size[1] - a.size[0] * a.size[1]);

  const lighting = {
    status: "fitted",
    source: light.file,
    model: light.json.model,
    decomposition_r2: light.json.reconstruction_r2,
    light_color: light.json.light_color,
    nondiffuse_fraction: light.json.nondiffuse_fraction,
    irradiance_sh: {
      basis: SH_BASIS,
      units: "linear diffuse shading, image units",
      coefficients_rgb: coefficients.map((rgb) => roundVec(rgb, 5)),
      fit_r2: round(1 - residual / Math.max(total, 1e-12), 4),
      samples: samples.length
    },
    dominant_light: {
      direction: roundVec(dominant, 4),
      direction_meaning: "unit vector from the scene toward the light, layout frame",
      model: "shading = ambient + intensity * max(0, normal . direction), least squares over the sphere",
      fit_r2: directional ? round(directional.r2, 4) : null,
      confidence: !directional || directional.r2 < 0.25 ? "low" : directional.r2 < 0.5 ? "medium" : "high",
      confidence_meaning: "from fit_r2: below 0.25 one distant light does not explain the shading (bounce light, shadows, occlusion dominate)",
      directional_share: directional ? round(directional.directional_share, 3) : null,
      uncertainty_deg: directional ? round(directional.uncertainty_deg, 1) : null,
      uncertainty_meaning: "largest angle to a direction that fits within 0.005 R^2 (the photo's normals cannot separate them)",
      directional_share_meaning: "fraction of mean diffuse shading carried by the directional term",
      sh_band1_direction: roundVec(normalize(band1), 4),
      azimuth_deg: round(azimuth, 1),
      elevation_deg: round(elevation, 1),
      side,
      directionality: round(directionality, 3),
      directionality_meaning: "|SH band 1| / SH band 0; 0 = uniform ambient, larger = stronger single direction"
    },
    emitters,
    highlights_on_objects: highlights,
    sunlit_patches: sunlitPatches
  };
  return { lighting, emitters, emitterOf };
}

export function lightingPrompt(lighting) {
  if (lighting.status !== "fitted") return "";
  const fmt = (value) => round(value, 1);
  const { dominant_light: dominant, light_color: color } = lighting;
  const parts = [];
  const where = (e) => {
    const ahead = -e.center[2];
    const lateral = Math.abs(e.center[0]) < 0.3 ? "centered" : `${fmt(Math.abs(e.center[0]))} m ${e.center[0] < 0 ? "left" : "right"}`;
    return `${fmt(ahead)} m ahead, ${lateral}`;
  };
  for (const emitter of lighting.emitters) {
    const noun = { window: "window", "ceiling-light": "ceiling light", light: "bright light source" }[emitter.id.replace(/-\d+$/, "")];
    parts.push(`${noun} ${fmt(emitter.size[0])} x ${fmt(emitter.size[1])} m, ${where(emitter)}, overexposed.`);
  }
  const share = dominant.directional_share ?? 0;
  const softness = share < 0.25 ? "soft, mostly ambient" : share < 0.5 ? "moderately directional" : "strongly directional";
  parts.push(`light color about ${Math.round(color.cct_kelvin / 100) * 100} K.`);
  if (dominant.confidence === "low") {
    parts.push(`${lighting.emitters.length ? "light mainly enters through the light sources above as" : "mostly"} soft indirect light, no single dominant direction.`);
  } else {
    parts.push(`${softness} light arriving from the ${dominant.side}, ${fmt(dominant.elevation_deg)} deg ${dominant.elevation_deg >= 0 ? "above" : "below"} the horizon.`);
  }
  parts.push(`${Math.round(lighting.nondiffuse_fraction * 100)}% of reflected light is non-diffuse (gloss, glass).`);
  return parts.join(" ");
}
