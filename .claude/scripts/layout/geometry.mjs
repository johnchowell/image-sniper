export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const length = (a) => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a) => {
  const l = length(a);
  return l > 0 ? scale(a, 1 / l) : [0, 0, 0];
};
export const degrees = (radians) => (radians * 180) / Math.PI;
export const radians = (deg) => (deg * Math.PI) / 180;

// Row-major 3x3 helpers.
export const mulMat3Vec = (m, v) => [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
export const transpose3 = (m) => [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]];

export function seededRandom(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function percentile(sorted, q) {
  if (sorted.length === 0) return NaN;
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.min(sorted.length - 1, low + 1);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

export function sortedValues(values) {
  return Float64Array.from(values).sort();
}

// Eigen decomposition of a symmetric 3x3 matrix (Jacobi). Eigenvalues ascending, vectors as unit rows.
export function symmetricEigen3(matrix) {
  const a = matrix.map((row) => [...row]);
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep += 1) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-12) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-15) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k += 1) {
        const akp = a[k][p];
        const akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k += 1) {
        const apk = a[p][k];
        const aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k += 1) {
        const vkp = v[k][p];
        const vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  return [0, 1, 2]
    .map((i) => ({ value: a[i][i], vector: normalize([v[0][i], v[1][i], v[2][i]]) }))
    .sort((x, y) => x.value - y.value);
}

// Least-squares plane through points[indices] (flat xyz array). Returns unit normal n and offset d with n.p + d = 0.
export function fitPlane(points, indices) {
  let cx = 0, cy = 0, cz = 0;
  for (const i of indices) {
    cx += points[i * 3];
    cy += points[i * 3 + 1];
    cz += points[i * 3 + 2];
  }
  const n = indices.length;
  cx /= n; cy /= n; cz /= n;
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (const i of indices) {
    const x = points[i * 3] - cx;
    const y = points[i * 3 + 1] - cy;
    const z = points[i * 3 + 2] - cz;
    xx += x * x; xy += x * y; xz += x * z; yy += y * y; yz += y * z; zz += z * z;
  }
  const normal = symmetricEigen3([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])[0].vector;
  const centroid = [cx, cy, cz];
  return { normal, d: -dot(normal, centroid), centroid };
}

// Rotation matrix with columns c0, c1, c2 to quaternion [x, y, z, w].
export function quaternionFromBasis(c0, c1, c2) {
  const m00 = c0[0], m10 = c0[1], m20 = c0[2];
  const m01 = c1[0], m11 = c1[1], m21 = c1[2];
  const m02 = c2[0], m12 = c2[1], m22 = c2[2];
  const trace = m00 + m11 + m22;
  let x, y, z, w;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    w = 0.25 / s; x = (m21 - m12) * s; y = (m02 - m20) * s; z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s;
  }
  const l = Math.hypot(x, y, z, w);
  return [x / l, y / l, z / l, w / l];
}

// Sequential RANSAC plane extraction over candidate point indices with per-point normals.
export function extractPlanes(points, normals, candidates, options = {}) {
  const {
    maxPlanes = 10,
    minPoints = 200,
    iterations = 500,
    sampleSize = 30000,
    normalAngleDeg = 25,
    baseTolerance = 0.01,
    relativeTolerance = 0.01,
    gridWidth,
    regionReach = 2,
    minRegionFraction = 0.25,
    dominantRegionFraction = 0.5,
    seed = 7
  } = options;
  let rejected = 0;
  const random = seededRandom(seed);
  const cosLimit = Math.cos(radians(normalAngleDeg));
  const distance = new Float32Array(points.length / 3);
  for (let i = 0; i < distance.length; i += 1) {
    distance[i] = Math.hypot(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
  }
  const isInlier = (i, normal, d) => {
    const residual = Math.abs(normal[0] * points[i * 3] + normal[1] * points[i * 3 + 1] + normal[2] * points[i * 3 + 2] + d);
    if (residual > baseTolerance + relativeTolerance * distance[i]) return false;
    const alignment = Math.abs(normal[0] * normals[i * 3] + normal[1] * normals[i * 3 + 1] + normal[2] * normals[i * 3 + 2]);
    return alignment >= cosLimit;
  };

  let remaining = Int32Array.from(candidates);
  const planes = [];
  while (planes.length < maxPlanes && remaining.length >= minPoints) {
    const sample = remaining.length <= sampleSize
      ? remaining
      : Int32Array.from({ length: sampleSize }, () => remaining[Math.floor(random() * remaining.length)]);
    let best = { score: 0 };
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const a = sample[Math.floor(random() * sample.length)];
      const b = sample[Math.floor(random() * sample.length)];
      const c = sample[Math.floor(random() * sample.length)];
      const pa = [points[a * 3], points[a * 3 + 1], points[a * 3 + 2]];
      const pb = [points[b * 3], points[b * 3 + 1], points[b * 3 + 2]];
      const pc = [points[c * 3], points[c * 3 + 1], points[c * 3 + 2]];
      const raw = cross(sub(pb, pa), sub(pc, pa));
      if (length(raw) < 1e-9) continue;
      const normal = normalize(raw);
      const d = -dot(normal, pa);
      let score = 0;
      for (const i of sample) if (isInlier(i, normal, d)) score += 1;
      if (score > best.score) best = { score, normal, d };
    }
    if (!best.normal) break;

    let plane = { normal: best.normal, d: best.d };
    let inliers = [];
    for (let refine = 0; refine < 3; refine += 1) {
      inliers = [];
      for (const i of remaining) if (isInlier(i, plane.normal, plane.d)) inliers.push(i);
      if (inliers.length < 3) break;
      plane = fitPlane(points, inliers);
    }
    if (inliers.length < minPoints) break;

    // A surface is a coherent image region: drop scattered fragments that only happen to lie near the plane.
    const regions = gridWidth ? connectedRegions(inliers, gridWidth, regionReach) : [inliers];
    const minRegion = Math.max(50, Math.round(minRegionFraction * minPoints));
    const coherent = regions.filter((region) => region.length >= minRegion).flat();
    const consumed = new Uint8Array(distance.length);
    if (coherent.length < minPoints || regions[0].length < dominantRegionFraction * minPoints) {
      for (const i of inliers) consumed[i] = 1;
      remaining = remaining.filter((i) => !consumed[i]);
      rejected += 1;
      if (rejected >= 3) break;
      continue;
    }
    rejected = 0;
    plane = fitPlane(points, coherent);
    for (const i of coherent) consumed[i] = 1;
    planes.push({ normal: plane.normal, d: plane.d, inliers: Int32Array.from(coherent) });
    remaining = remaining.filter((i) => !consumed[i]);
  }
  return mergeCoplanar(points, distance, planes, { normalAngleDeg: 5, baseTolerance, relativeTolerance });
}

// Regions of grid cells linked within `reach` cells (bridges noise speckle inside one surface), largest first.
function connectedRegions(cells, gridWidth, reach = 2) {
  const member = new Set(cells);
  const seen = new Set();
  const regions = [];
  for (const start of cells) {
    if (seen.has(start)) continue;
    const region = [];
    const stack = [start];
    seen.add(start);
    while (stack.length) {
      const cell = stack.pop();
      region.push(cell);
      const x = cell % gridWidth;
      for (let dy = -reach; dy <= reach; dy += 1) {
        for (let dx = -reach; dx <= reach; dx += 1) {
          if ((dx === 0 && dy === 0) || x + dx < 0 || x + dx >= gridWidth) continue;
          const next = cell + dy * gridWidth + dx;
          if (member.has(next) && !seen.has(next)) {
            seen.add(next);
            stack.push(next);
          }
        }
      }
    }
    regions.push(region);
  }
  return regions.sort((a, b) => b.length - a.length);
}

// Joins planes that describe the same surface (noise left part of it outside the first fit's tolerance band).
function mergeCoplanar(points, distance, planes, { normalAngleDeg, baseTolerance, relativeTolerance }) {
  const cosLimit = Math.cos(radians(normalAngleDeg));
  const meanDistance = (plane) => plane.inliers.reduce((sum, i) => sum + distance[i], 0) / plane.inliers.length;
  let merged = [...planes];
  for (let changed = true; changed;) {
    changed = false;
    for (let a = 0; a < merged.length && !changed; a += 1) {
      for (let b = a + 1; b < merged.length && !changed; b += 1) {
        const pa = merged[a];
        const pb = merged[b];
        const sign = dot(pa.normal, pb.normal) < 0 ? -1 : 1;
        if (sign * dot(pa.normal, pb.normal) < cosLimit) continue;
        const tolerance = 2 * (baseTolerance + relativeTolerance * Math.max(meanDistance(pa), meanDistance(pb)));
        if (Math.abs(pa.d - sign * pb.d) > tolerance) continue;
        const inliers = Int32Array.from([...pa.inliers, ...pb.inliers]);
        const fit = fitPlane(points, inliers);
        merged = [...merged.slice(0, a), { normal: fit.normal, d: fit.d, inliers }, ...merged.slice(a + 1, b), ...merged.slice(b + 1)];
        changed = true;
      }
    }
  }
  return merged;
}

// Minimum-area yaw rectangle of 2D points (x, z) using robust percentile extents.
// Returns yaw (radians, rotation about +Y), center (x, z), and sizes along local X and local Z.
export function minAreaRect(xs, zs, low = 0.02, high = 0.98) {
  let best;
  for (let step = -45; step < 45; step += 1) {
    const yaw = radians(step);
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const along = new Float64Array(xs.length);
    const across = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i += 1) {
      along[i] = xs[i] * c - zs[i] * s;
      across[i] = xs[i] * s + zs[i] * c;
    }
    along.sort();
    across.sort();
    const a0 = percentile(along, low), a1 = percentile(along, high);
    const b0 = percentile(across, low), b1 = percentile(across, high);
    const area = (a1 - a0) * (b1 - b0);
    if (!best || area < best.area - 1e-9) best = { area, yaw, a0, a1, b0, b1 };
  }
  const c = Math.cos(best.yaw);
  const s = Math.sin(best.yaw);
  const sc = (best.a0 + best.a1) / 2;
  const tc = (best.b0 + best.b1) / 2;
  return {
    yaw: best.yaw,
    center: [sc * c + tc * s, -sc * s + tc * c],
    sizeX: best.a1 - best.a0,
    sizeZ: best.b1 - best.b0
  };
}
