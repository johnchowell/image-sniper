// Minimal glTF 2.0 binary writer for layout blockouts: a shared unit cube and a unit quad,
// one material per primitive class, one node per primitive, and the source camera.

function unitCube() {
  const faces = [
    { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] },
    { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
    { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] },
    { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
    { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] }
  ];
  const positions = [];
  const normals = [];
  const indices = [];
  for (const { n, u, v } of faces) {
    const base = positions.length / 3;
    for (const [su, sv] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      positions.push(...[0, 1, 2].map((k) => 0.5 * (n[k] + su * u[k] + sv * v[k])));
      normals.push(...n);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { positions, normals, indices };
}

function unitQuad() {
  return {
    positions: [-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0],
    normals: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    indices: [0, 1, 2, 0, 2, 3]
  };
}

function bounds(positions) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k += 1) {
      min[k] = Math.min(min[k], positions[i + k]);
      max[k] = Math.max(max[k], positions[i + k]);
    }
  }
  return { min, max };
}

function pad4(buffer, fill = 0) {
  const remainder = buffer.length % 4;
  return remainder ? Buffer.concat([buffer, Buffer.alloc(4 - remainder, fill)]) : buffer;
}

/**
 * nodes: [{ name, shape: "box" | "quad", material, translation, rotation, scale, extras }]
 * materials: [{ name, color: [r, g, b] (0-1), doubleSided }]
 * camera: { name, yfov, aspectRatio, znear, translation, rotation, extras } or undefined
 */
export function writeLayoutGlb({ nodes, materials, camera, extras }) {
  const shapes = { box: unitCube(), quad: unitQuad() };
  const chunks = [];
  const bufferViews = [];
  const accessors = [];
  let byteOffset = 0;
  const pushView = (typed, target) => {
    const data = pad4(Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength));
    bufferViews.push({ buffer: 0, byteOffset, byteLength: typed.byteLength, target });
    chunks.push(data);
    byteOffset += data.length;
    return bufferViews.length - 1;
  };

  const shapeAccessors = {};
  for (const [name, shape] of Object.entries(shapes)) {
    const positions = Float32Array.from(shape.positions);
    const normals = Float32Array.from(shape.normals);
    const indices = Uint16Array.from(shape.indices);
    const { min, max } = bounds(shape.positions);
    accessors.push({ bufferView: pushView(positions, 34962), componentType: 5126, count: positions.length / 3, type: "VEC3", min, max });
    const position = accessors.length - 1;
    accessors.push({ bufferView: pushView(normals, 34962), componentType: 5126, count: normals.length / 3, type: "VEC3" });
    const normal = accessors.length - 1;
    accessors.push({ bufferView: pushView(indices, 34963), componentType: 5123, count: indices.length, type: "SCALAR" });
    shapeAccessors[name] = { position, normal, indices: accessors.length - 1 };
  }

  const meshes = [];
  const meshIndex = new Map();
  const meshFor = (shape, material) => {
    const key = `${shape}:${material}`;
    if (!meshIndex.has(key)) {
      const access = shapeAccessors[shape];
      meshes.push({
        name: `${shape}-${materials[material].name}`,
        primitives: [{ attributes: { POSITION: access.position, NORMAL: access.normal }, indices: access.indices, material }]
      });
      meshIndex.set(key, meshes.length - 1);
    }
    return meshIndex.get(key);
  };

  const gltfNodes = nodes.map((node) => ({
    name: node.name,
    mesh: meshFor(node.shape, node.material),
    translation: node.translation,
    rotation: node.rotation,
    scale: node.scale,
    ...(node.extras ? { extras: node.extras } : {})
  }));

  const cameras = [];
  if (camera) {
    cameras.push({
      name: camera.name,
      type: "perspective",
      perspective: { yfov: camera.yfov, aspectRatio: camera.aspectRatio, znear: camera.znear ?? 0.01 }
    });
    gltfNodes.push({
      name: camera.name,
      camera: 0,
      translation: camera.translation,
      rotation: camera.rotation,
      ...(camera.extras ? { extras: camera.extras } : {})
    });
  }

  const binary = Buffer.concat(chunks);
  const gltf = {
    asset: { version: "2.0", generator: "image-blaster layout" },
    scene: 0,
    scenes: [{ name: "layout", nodes: gltfNodes.map((_, index) => index), ...(extras ? { extras } : {}) }],
    nodes: gltfNodes,
    meshes,
    materials: materials.map((material) => ({
      name: material.name,
      doubleSided: Boolean(material.doubleSided),
      pbrMetallicRoughness: { baseColorFactor: [...material.color, 1], metallicFactor: 0, roughnessFactor: 1 }
    })),
    ...(cameras.length ? { cameras } : {}),
    accessors,
    bufferViews,
    buffers: [{ byteLength: binary.length }]
  };

  const json = pad4(Buffer.from(JSON.stringify(gltf), "utf8"), 0x20);
  const header = Buffer.alloc(12);
  header.writeUInt32LE(0x46546c67, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + json.length + 8 + binary.length, 8);
  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(json.length, 0);
  jsonHeader.writeUInt32LE(0x4e4f534a, 4);
  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(binary.length, 0);
  binHeader.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([header, jsonHeader, json, binHeader, binary]);
}
