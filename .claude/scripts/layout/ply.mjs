const TYPES = {
  char: ["Int8", 1], int8: ["Int8", 1],
  uchar: ["Uint8", 1], uint8: ["Uint8", 1],
  short: ["Int16", 2], int16: ["Int16", 2],
  ushort: ["Uint16", 2], uint16: ["Uint16", 2],
  int: ["Int32", 4], int32: ["Int32", 4],
  uint: ["Uint32", 4], uint32: ["Uint32", 4],
  float: ["Float32", 4], float32: ["Float32", 4],
  double: ["Float64", 8], float64: ["Float64", 8]
};

function parseHeader(buffer) {
  const end = buffer.indexOf("end_header");
  if (buffer.toString("ascii", 0, 3) !== "ply" || end === -1) throw new Error("Not a PLY file.");
  const newline = buffer.indexOf(0x0a, end);
  const lines = buffer.toString("ascii", 0, end).split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  let format;
  const elements = [];
  for (const line of lines) {
    const parts = line.split(/\s+/);
    if (parts[0] === "format") format = parts[1];
    else if (parts[0] === "element") elements.push({ name: parts[1], count: Number(parts[2]), properties: [] });
    else if (parts[0] === "property") {
      const element = elements.at(-1);
      if (parts[1] === "list") element.properties.push({ list: true, name: parts[4] });
      else element.properties.push({ name: parts[2], type: parts[1] });
    }
  }
  return { format, elements, bodyOffset: newline + 1 };
}

// Reads the vertex element of a PLY file: positions and optional 8-bit RGB colors, in file order.
export function parsePly(buffer) {
  const { format, elements, bodyOffset } = parseHeader(buffer);
  const vertexIndex = elements.findIndex((element) => element.name === "vertex");
  if (vertexIndex === -1) throw new Error("PLY has no vertex element.");
  const vertex = elements[vertexIndex];
  const names = vertex.properties.map((property) => property.name);
  const ix = names.indexOf("x");
  const iy = names.indexOf("y");
  const iz = names.indexOf("z");
  if (ix < 0 || iy < 0 || iz < 0) throw new Error("PLY vertex element has no x/y/z.");
  const colorIndex = ["red", "green", "blue"].map((name) => names.indexOf(name));
  const hasColor = colorIndex.every((index) => index >= 0);
  const colorScale = hasColor && vertex.properties[colorIndex[0]].type?.startsWith("float") ? 255 : 1;

  const count = vertex.count;
  const positions = new Float32Array(count * 3);
  const colors = hasColor ? new Uint8Array(count * 3) : null;
  const store = (i, values) => {
    positions[i * 3] = values[ix];
    positions[i * 3 + 1] = values[iy];
    positions[i * 3 + 2] = values[iz];
    if (colors) {
      for (let c = 0; c < 3; c += 1) {
        colors[i * 3 + c] = Math.max(0, Math.min(255, Math.round(values[colorIndex[c]] * colorScale)));
      }
    }
  };

  if (format === "ascii") {
    const lines = buffer.toString("ascii", bodyOffset).split(/\r?\n/);
    let line = 0;
    for (let e = 0; e < vertexIndex; e += 1) line += elements[e].count;
    for (let i = 0; i < count; i += 1) store(i, lines[line + i].trim().split(/\s+/).map(Number));
    return { count, positions, colors };
  }

  if (format !== "binary_little_endian" && format !== "binary_big_endian") {
    throw new Error(`Unsupported PLY format ${format}.`);
  }
  const littleEndian = format === "binary_little_endian";
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let offset = bodyOffset;
  for (let e = 0; e < vertexIndex; e += 1) {
    if (elements[e].properties.some((property) => property.list)) {
      throw new Error(`PLY element "${elements[e].name}" before vertex has list properties.`);
    }
    offset += elements[e].count * elements[e].properties.reduce((sum, property) => sum + TYPES[property.type][1], 0);
  }
  if (vertex.properties.some((property) => property.list || !TYPES[property.type])) {
    throw new Error("PLY vertex element has unsupported properties.");
  }
  const readers = vertex.properties.map((property) => {
    const [method, size] = TYPES[property.type];
    return { getter: `get${method}`, size };
  });
  const values = new Array(readers.length);
  for (let i = 0; i < count; i += 1) {
    for (let p = 0; p < readers.length; p += 1) {
      values[p] = view[readers[p].getter](offset, littleEndian);
      offset += readers[p].size;
    }
    store(i, values);
  }
  return { count, positions, colors };
}
