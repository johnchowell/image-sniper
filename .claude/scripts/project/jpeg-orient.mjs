import { readFile, writeFile } from "node:fs/promises";

// Lossless EXIF orientation for baseline JPEGs: the quantized DCT coefficients are transposed and mirrored block by
// block (as jpegtran does), so the stored pixel grid becomes the upright image and the Orientation tag becomes 1.
// Every consumer then sees the same image: the browser applies the tag, PIL and the model endpoints do not.
// A mirror along an axis whose size is not a whole number of MCUs drops the partial edge MCU (as jpegtran -trim).

// Natural index (row * 8 + column) of each zigzag position.
const ZIGZAG = (() => {
  const out = [];
  for (let s = 0; s < 15; s += 1) {
    const rows = [];
    for (let row = Math.max(0, s - 7); row <= Math.min(s, 7); row += 1) rows.push(row);
    if (s % 2 === 0) rows.reverse();
    for (const row of rows) out.push(row * 8 + (s - row));
  }
  return out;
})();

// Orientation -> operations on the stored image, applied in order.
const OPERATIONS = {
  2: ["flipH"],
  3: ["flipH", "flipV"],
  4: ["flipV"],
  5: ["transpose"],
  6: ["transpose", "flipH"],
  7: ["transpose", "flipH", "flipV"],
  8: ["transpose", "flipV"]
};

export async function normalizeJpegOrientation(file) {
  const buffer = await readFile(file);
  const result = orientJpeg(buffer);
  if (result.data) await writeFile(file, result.data);
  return result.info;
}

// Returns { data, info }: data is null when nothing changes (no tag, orientation 1) or the file is not supported.
export function orientJpeg(buffer) {
  if (buffer.length < 4 || buffer.readUInt16BE(0) !== 0xffd8) return { data: null, info: { status: "not_jpeg" } };
  const segments = readSegments(buffer);
  const exif = segments.find((s) => s.marker === 0xe1 && buffer.toString("latin1", s.start, s.start + 6) === "Exif\0\0");
  const tag = exif ? orientationTag(buffer, exif.start + 6) : null;
  if (!tag || tag.value === 1) return { data: null, info: { status: "upright", orientation: tag?.value ?? null } };
  if (!OPERATIONS[tag.value]) return { data: null, info: { status: "unsupported", orientation: tag.value, reason: "invalid orientation value" } };
  let image;
  try {
    image = decode(buffer, segments);
  } catch (error) {
    return { data: null, info: { status: "unsupported", orientation: tag.value, reason: error.message } };
  }
  const size = [image.width, image.height];
  for (const op of OPERATIONS[tag.value]) TRANSFORMS[op](image);
  // The Orientation tag becomes 1 (top-left); the MPF index (offsets of images after this one) no longer applies.
  const patched = Buffer.from(buffer);
  if (tag.little) patched.writeUInt16LE(1, tag.at);
  else patched.writeUInt16BE(1, tag.at);
  const keep = segments.filter((s) => s.marker >= 0xe0 && s.marker <= 0xef &&
    !(s.marker === 0xe2 && buffer.toString("latin1", s.start, s.start + 4) === "MPF\0"));
  const data = encode(image, keep.map((s) => patched.subarray(s.start - 4, s.end)));
  return {
    data,
    info: { status: "oriented", orientation: tag.value, stored_size: size, size: [image.width, image.height], lossless: true }
  };
}

function readSegments(buffer) {
  const segments = [];
  let at = 2;
  while (at + 4 <= buffer.length) {
    if (buffer[at] !== 0xff) throw new Error("bad marker");
    const marker = buffer[at + 1];
    if (marker === 0xff) { at += 1; continue; }
    if (marker === 0xd9) break;
    const length = buffer.readUInt16BE(at + 2);
    const segment = { marker, start: at + 4, end: at + 2 + length };
    segments.push(segment);
    if (marker === 0xda) {
      // Entropy-coded data runs to the next marker that is not a restart or a stuffed 0xFF00.
      let end = segment.end;
      while (end + 1 < buffer.length && !(buffer[end] === 0xff && buffer[end + 1] !== 0 && (buffer[end + 1] < 0xd0 || buffer[end + 1] > 0xd7))) end += 1;
      segment.dataEnd = end;
      at = end;
    } else {
      at = segment.end;
    }
  }
  return segments;
}

function orientationTag(buffer, tiff) {
  const little = buffer.toString("latin1", tiff, tiff + 2) === "II";
  const u16 = (at) => (little ? buffer.readUInt16LE(at) : buffer.readUInt16BE(at));
  const u32 = (at) => (little ? buffer.readUInt32LE(at) : buffer.readUInt32BE(at));
  const ifd = tiff + u32(tiff + 4);
  if (ifd + 2 > buffer.length) return null;
  for (let i = 0; i < u16(ifd); i += 1) {
    const entry = ifd + 2 + i * 12;
    if (entry + 12 > buffer.length) break;
    if (u16(entry) === 0x0112 && u16(entry + 2) === 3) return { value: u16(entry + 8), at: entry + 8, little };
  }
  return null;
}

// ---- decoding (baseline and extended sequential Huffman, one scan with every component) ----

function huffmanTable(counts, symbols) {
  // maxcode[l] / valptr[l] / mincode[l] per code length (ITU T.81 F.2.2.3)
  const maxcode = new Int32Array(18).fill(-1);
  const mincode = new Int32Array(17);
  const valptr = new Int32Array(17);
  let code = 0;
  let k = 0;
  for (let l = 1; l <= 16; l += 1) {
    valptr[l] = k;
    mincode[l] = code;
    code += counts[l - 1];
    k += counts[l - 1];
    maxcode[l] = counts[l - 1] ? code - 1 : -1;
    code <<= 1;
  }
  maxcode[17] = 0x7fffffff;
  return { maxcode, mincode, valptr, symbols };
}

function decode(buffer, segments) {
  const quant = [];
  const dc = [];
  const ac = [];
  let frame = null;
  let restart = 0;
  let scan = null;
  for (const s of segments) {
    if (s.marker === 0xdb) {
      for (let at = s.start; at < s.end;) {
        const pq = buffer[at] >> 4;
        const tq = buffer[at] & 15;
        const table = new Uint16Array(64);
        for (let k = 0; k < 64; k += 1) table[k] = pq ? buffer.readUInt16BE(at + 1 + 2 * k) : buffer[at + 1 + k];
        quant[tq] = { precision: pq, zigzag: table };
        at += 1 + (pq ? 128 : 64);
      }
    } else if (s.marker === 0xc4) {
      for (let at = s.start; at < s.end;) {
        const tc = buffer[at] >> 4;
        const th = buffer[at] & 15;
        const counts = [...buffer.subarray(at + 1, at + 17)];
        const total = counts.reduce((a, b) => a + b, 0);
        const table = huffmanTable(counts, buffer.subarray(at + 17, at + 17 + total));
        (tc ? ac : dc)[th] = table;
        at += 17 + total;
      }
    } else if (s.marker === 0xdd) {
      restart = buffer.readUInt16BE(s.start);
    } else if (s.marker === 0xc0 || s.marker === 0xc1) {
      if (buffer[s.start] !== 8) throw new Error("12-bit samples");
      const height = buffer.readUInt16BE(s.start + 1);
      const width = buffer.readUInt16BE(s.start + 3);
      const count = buffer[s.start + 5];
      const components = [];
      for (let i = 0; i < count; i += 1) {
        const at = s.start + 6 + 3 * i;
        components.push({ id: buffer[at], h: buffer[at + 1] >> 4, v: buffer[at + 1] & 15, tq: buffer[at + 2] });
      }
      frame = { width, height, components };
    } else if (s.marker >= 0xc2 && s.marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(s.marker)) {
      throw new Error(s.marker === 0xc2 ? "progressive JPEG" : `JPEG process SOF${s.marker - 0xc0}`);
    } else if (s.marker === 0xda) {
      if (scan) throw new Error("more than one scan");
      scan = s;
    }
  }
  if (!frame || !scan) throw new Error("no frame or scan");
  if (frame.height === 0) throw new Error("height defined by DNL");
  if (![1, 3].includes(frame.components.length)) throw new Error(`${frame.components.length} components`);
  const ns = buffer[scan.start];
  if (ns !== frame.components.length) throw new Error("scan without every component");
  for (let i = 0; i < ns; i += 1) {
    const comp = frame.components.find((c) => c.id === buffer[scan.start + 1 + 2 * i]);
    if (!comp) throw new Error("scan names an unknown component");
    comp.dc = dc[buffer[scan.start + 2 + 2 * i] >> 4];
    comp.ac = ac[buffer[scan.start + 2 + 2 * i] & 15];
    comp.order = i;
    if (!comp.dc || !comp.ac) throw new Error("missing Huffman table");
  }
  const ss = buffer[scan.start + 1 + 2 * ns];
  const se = buffer[scan.start + 2 + 2 * ns];
  if (ss !== 0 || se !== 63) throw new Error("spectral selection in a sequential scan");
  frame.components.sort((a, b) => a.order - b.order);

  const image = { width: frame.width, height: frame.height, components: frame.components };
  layout(image);
  for (const comp of image.components) {
    const q = quant[comp.tq];
    if (!q) throw new Error("missing quantization table");
    comp.quant = { precision: q.precision, natural: fromZigzag(q.zigzag) };
    comp.blocks = new Int16Array(comp.bw * comp.bh * 64);
  }

  const reader = bitReader(buffer, scan.end, scan.dataEnd);
  const decodeHuffman = (table) => {
    let code = 0;
    for (let l = 1; l <= 16; l += 1) {
      code = (code << 1) | reader.bit();
      if (code <= table.maxcode[l]) return table.symbols[table.valptr[l] + code - table.mincode[l]];
    }
    throw new Error("bad Huffman code");
  };
  const receiveExtend = (s) => {
    if (!s) return 0;
    let v = 0;
    for (let i = 0; i < s; i += 1) v = (v << 1) | reader.bit();
    return v < 1 << (s - 1) ? v - (1 << s) + 1 : v;
  };
  const decodeBlock = (comp, offset) => {
    const diff = receiveExtend(decodeHuffman(comp.dc));
    comp.pred += diff;
    comp.blocks[offset] = comp.pred;
    for (let k = 1; k < 64;) {
      const rs = decodeHuffman(comp.ac);
      const r = rs >> 4;
      const s = rs & 15;
      if (!s) {
        if (r !== 15) break;
        k += 16;
        continue;
      }
      k += r;
      if (k > 63) throw new Error("coefficient index out of range");
      comp.blocks[offset + ZIGZAG[k]] = receiveExtend(s);
      k += 1;
    }
  };
  for (const comp of image.components) comp.pred = 0;
  forEachBlock(image, (comp, offset) => decodeBlock(comp, offset), (unit) => {
    if (restart && unit > 0 && unit % restart === 0) {
      reader.restart();
      for (const comp of image.components) comp.pred = 0;
    }
  });
  return image;
}

function bitReader(buffer, start, end) {
  let at = start;
  let byte = 0;
  let left = 0;
  return {
    bit() {
      if (!left) {
        if (at >= end) byte = 0xff;
        else if (buffer[at] !== 0xff) byte = buffer[at++];
        else if (buffer[at + 1] === 0) { byte = 0xff; at += 2; } // stuffed byte
        else byte = 0; // a marker: feed zeros and stay on it
        left = 8;
      }
      left -= 1;
      return (byte >> left) & 1;
    },
    restart() {
      left = 0;
      while (at + 1 < end && !(buffer[at] === 0xff && buffer[at + 1] >= 0xd0 && buffer[at + 1] <= 0xd7)) at += 1;
      at += 2;
    }
  };
}

// Block grid of each component: whole MCUs, so mirroring and transposing keep every block.
function layout(image) {
  const hmax = Math.max(...image.components.map((c) => c.h));
  const vmax = Math.max(...image.components.map((c) => c.v));
  image.hmax = hmax;
  image.vmax = vmax;
  image.mcux = Math.ceil(image.width / (8 * hmax));
  image.mcuy = Math.ceil(image.height / (8 * vmax));
  for (const comp of image.components) {
    comp.bw = image.mcux * comp.h;
    comp.bh = image.mcuy * comp.v;
  }
}

// Visits the blocks in scan order: interleaved MCUs, or for one component the blocks that hold samples.
function forEachBlock(image, visit, beforeUnit) {
  let unit = 0;
  if (image.components.length === 1) {
    const comp = image.components[0];
    const cw = Math.ceil(Math.ceil((image.width * comp.h) / image.hmax) / 8);
    const ch = Math.ceil(Math.ceil((image.height * comp.v) / image.vmax) / 8);
    for (let by = 0; by < ch; by += 1) {
      for (let bx = 0; bx < cw; bx += 1) {
        beforeUnit(unit);
        visit(comp, (by * comp.bw + bx) * 64);
        unit += 1;
      }
    }
    return;
  }
  for (let my = 0; my < image.mcuy; my += 1) {
    for (let mx = 0; mx < image.mcux; mx += 1) {
      beforeUnit(unit);
      for (const comp of image.components) {
        for (let y = 0; y < comp.v; y += 1) {
          for (let x = 0; x < comp.h; x += 1) visit(comp, ((my * comp.v + y) * comp.bw + mx * comp.h + x) * 64);
        }
      }
      unit += 1;
    }
  }
}

function fromZigzag(zigzag) {
  const natural = new Uint16Array(64);
  for (let k = 0; k < 64; k += 1) natural[ZIGZAG[k]] = zigzag[k];
  return natural;
}

// ---- transforms on the coefficient image (natural order: index = v * 8 + u, v vertical frequency) ----

const TRANSFORMS = {
  transpose(image) {
    for (const comp of image.components) {
      const out = new Int16Array(comp.blocks.length);
      for (let by = 0; by < comp.bh; by += 1) {
        for (let bx = 0; bx < comp.bw; bx += 1) {
          const from = (by * comp.bw + bx) * 64;
          const to = (bx * comp.bh + by) * 64;
          for (let v = 0; v < 8; v += 1) for (let u = 0; u < 8; u += 1) out[to + u * 8 + v] = comp.blocks[from + v * 8 + u];
        }
      }
      comp.blocks = out;
      [comp.bw, comp.bh, comp.h, comp.v] = [comp.bh, comp.bw, comp.v, comp.h];
      const q = new Uint16Array(64);
      for (let v = 0; v < 8; v += 1) for (let u = 0; u < 8; u += 1) q[u * 8 + v] = comp.quant.natural[v * 8 + u];
      comp.quant = { ...comp.quant, natural: q };
    }
    [image.width, image.height, image.hmax, image.vmax, image.mcux, image.mcuy] =
      [image.height, image.width, image.vmax, image.hmax, image.mcuy, image.mcux];
  },
  flipH(image) {
    const span = 8 * image.hmax;
    const mcux = Math.floor(image.width / span);
    if (!mcux) throw new Error("image narrower than one MCU");
    image.width = mcux * span; // the partial edge MCU column cannot move losslessly
    image.mcux = mcux;
    for (const comp of image.components) {
      const bw = mcux * comp.h;
      const out = new Int16Array(bw * comp.bh * 64);
      for (let by = 0; by < comp.bh; by += 1) {
        for (let bx = 0; bx < bw; bx += 1) {
          const from = (by * comp.bw + (bw - 1 - bx)) * 64;
          const to = (by * bw + bx) * 64;
          for (let i = 0; i < 64; i += 1) out[to + i] = i & 1 ? -comp.blocks[from + i] : comp.blocks[from + i];
        }
      }
      comp.blocks = out;
      comp.bw = bw;
    }
  },
  flipV(image) {
    const span = 8 * image.vmax;
    const mcuy = Math.floor(image.height / span);
    if (!mcuy) throw new Error("image shorter than one MCU");
    image.height = mcuy * span;
    image.mcuy = mcuy;
    for (const comp of image.components) {
      const bh = mcuy * comp.v;
      const out = new Int16Array(comp.bw * bh * 64);
      for (let by = 0; by < bh; by += 1) {
        for (let bx = 0; bx < comp.bw; bx += 1) {
          const from = ((bh - 1 - by) * comp.bw + bx) * 64;
          const to = (by * comp.bw + bx) * 64;
          for (let i = 0; i < 64; i += 1) out[to + i] = (i >> 3) & 1 ? -comp.blocks[from + i] : comp.blocks[from + i];
        }
      }
      comp.blocks = out;
      comp.bh = bh;
    }
  }
};

// ---- encoding (baseline, one interleaved scan, optimal Huffman tables) ----

function bitLength(value) {
  let n = 0;
  for (let v = Math.abs(value); v; v >>= 1) n += 1;
  return n;
}

// Code lengths for the symbol frequencies, limited to 16 bits (ITU T.81 K.2).
function optimalTable(freq) {
  const f = [...freq, 1]; // index 256 reserves the all-ones code
  const codesize = new Array(257).fill(0);
  const others = new Array(257).fill(-1);
  for (;;) {
    let c1 = -1;
    let c2 = -1;
    let v = Infinity;
    for (let i = 0; i <= 256; i += 1) if (f[i] && f[i] <= v) { v = f[i]; c1 = i; }
    v = Infinity;
    for (let i = 0; i <= 256; i += 1) if (f[i] && f[i] <= v && i !== c1) { v = f[i]; c2 = i; }
    if (c2 < 0) break;
    f[c1] += f[c2];
    f[c2] = 0;
    codesize[c1] += 1;
    while (others[c1] >= 0) { c1 = others[c1]; codesize[c1] += 1; }
    others[c1] = c2;
    codesize[c2] += 1;
    while (others[c2] >= 0) { c2 = others[c2]; codesize[c2] += 1; }
  }
  const bits = new Array(33).fill(0);
  for (let i = 0; i <= 256; i += 1) if (codesize[i]) bits[codesize[i]] += 1;
  for (let i = 32; i > 16; i -= 1) {
    while (bits[i] > 0) {
      let j = i - 2;
      while (bits[j] === 0) j -= 1;
      bits[i] -= 2;
      bits[i - 1] += 1;
      bits[j + 1] += 2;
      bits[j] -= 1;
    }
  }
  let i = 16;
  while (bits[i] === 0) i -= 1;
  bits[i] -= 1;
  const symbols = [];
  for (let l = 1; l <= 32; l += 1) for (let s = 0; s < 256; s += 1) if (codesize[s] === l) symbols.push(s);
  const counts = bits.slice(1, 17);
  const code = new Int32Array(256);
  const size = new Uint8Array(256);
  let next = 0;
  let k = 0;
  for (let l = 1; l <= 16; l += 1) {
    for (let n = 0; n < counts[l - 1]; n += 1) {
      code[symbols[k]] = next;
      size[symbols[k]] = l;
      next += 1;
      k += 1;
    }
    next <<= 1;
  }
  return { counts, symbols, code, size };
}

function encode(image, appSegments) {
  const comps = image.components;
  const tableOf = (comp) => (comp === comps[0] ? 0 : 1);
  const visitAll = (emitSymbol) => {
    for (const comp of comps) comp.pred = 0;
    forEachBlock(image, (comp, offset) => {
      const t = tableOf(comp);
      const diff = comp.blocks[offset] - comp.pred;
      comp.pred = comp.blocks[offset];
      const s = bitLength(diff);
      emitSymbol("dc", t, s, diff, s);
      let run = 0;
      for (let k = 1; k < 64; k += 1) {
        const value = comp.blocks[offset + ZIGZAG[k]];
        if (!value) { run += 1; continue; }
        while (run > 15) { emitSymbol("ac", t, 0xf0, 0, 0); run -= 16; }
        const size = bitLength(value);
        emitSymbol("ac", t, (run << 4) | size, value, size);
        run = 0;
      }
      if (run) emitSymbol("ac", t, 0x00, 0, 0);
    }, () => {});
  };
  const tableCount = comps.length > 1 ? 2 : 1;
  const freq = { dc: [], ac: [] };
  for (let t = 0; t < tableCount; t += 1) { freq.dc.push(new Array(256).fill(0)); freq.ac.push(new Array(256).fill(0)); }
  visitAll((kind, t, symbol) => { freq[kind][t][symbol] += 1; });
  const tables = { dc: freq.dc.map(optimalTable), ac: freq.ac.map(optimalTable) };

  const bytes = [];
  let acc = 0;
  let bits = 0;
  const put = (value, size) => {
    for (let i = size - 1; i >= 0; i -= 1) {
      acc = (acc << 1) | ((value >> i) & 1);
      bits += 1;
      if (bits === 8) {
        bytes.push(acc);
        if (acc === 0xff) bytes.push(0);
        acc = 0;
        bits = 0;
      }
    }
  };
  visitAll((kind, t, symbol, value, size) => {
    const table = tables[kind][t];
    put(table.code[symbol], table.size[symbol]);
    if (size) put(value < 0 ? value - 1 + (1 << size) : value, size);
  });
  if (bits) put((1 << (8 - bits)) - 1, 8 - bits);

  const out = [Buffer.from([0xff, 0xd8]), ...appSegments];
  const segment = (marker, body) => {
    const head = Buffer.alloc(4);
    head.writeUInt16BE(0xff00 | marker, 0);
    head.writeUInt16BE(body.length + 2, 2);
    out.push(head, body);
  };
  const quantIds = [...new Set(comps.map((c) => c.tq))];
  for (const tq of quantIds) {
    const comp = comps.find((c) => c.tq === tq);
    const { precision, natural } = comp.quant;
    const body = Buffer.alloc(1 + (precision ? 128 : 64));
    body[0] = (precision << 4) | tq;
    for (let k = 0; k < 64; k += 1) {
      if (precision) body.writeUInt16BE(natural[ZIGZAG[k]], 1 + 2 * k);
      else body[1 + k] = natural[ZIGZAG[k]];
    }
    segment(0xdb, body);
  }
  const sof = Buffer.alloc(6 + 3 * comps.length);
  sof[0] = 8;
  sof.writeUInt16BE(image.height, 1);
  sof.writeUInt16BE(image.width, 3);
  sof[5] = comps.length;
  comps.forEach((c, i) => { sof[6 + 3 * i] = c.id; sof[7 + 3 * i] = (c.h << 4) | c.v; sof[8 + 3 * i] = c.tq; });
  segment(0xc0, sof);
  for (const [kind, tc] of [["dc", 0], ["ac", 1]]) {
    tables[kind].forEach((table, th) => segment(0xc4, Buffer.from([(tc << 4) | th, ...table.counts, ...table.symbols])));
  }
  const sos = Buffer.alloc(4 + 2 * comps.length);
  sos[0] = comps.length;
  comps.forEach((c, i) => { sos[1 + 2 * i] = c.id; sos[2 + 2 * i] = (tableOf(c) << 4) | tableOf(c); });
  sos[1 + 2 * comps.length] = 0;
  sos[2 + 2 * comps.length] = 63;
  sos[3 + 2 * comps.length] = 0;
  segment(0xda, sos);
  out.push(Buffer.from(bytes), Buffer.from([0xff, 0xd9]));
  return Buffer.concat(out);
}
