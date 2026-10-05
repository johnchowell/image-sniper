import { deflateSync, inflateSync } from "node:zlib";

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}

function unfilter(raw, width, height, bitsPerPixel) {
  const stride = Math.ceil((width * bitsPerPixel) / 8);
  const bpp = Math.max(1, Math.ceil(bitsPerPixel / 8));
  const out = Buffer.alloc(stride * height);
  let offset = 0;
  for (let y = 0; y < height; y += 1) {
    const filter = raw[offset];
    offset += 1;
    const row = y * stride;
    const prev = row - stride;
    for (let x = 0; x < stride; x += 1) {
      const value = raw[offset + x];
      const left = x >= bpp ? out[row + x - bpp] : 0;
      const up = y > 0 ? out[prev + x] : 0;
      const upLeft = y > 0 && x >= bpp ? out[prev + x - bpp] : 0;
      let result;
      switch (filter) {
        case 0: result = value; break;
        case 1: result = value + left; break;
        case 2: result = value + up; break;
        case 3: result = value + ((left + up) >> 1); break;
        case 4: result = value + paeth(left, up, upLeft); break;
        default: throw new Error(`Unsupported PNG filter type ${filter}.`);
      }
      out[row + x] = result & 0xff;
    }
    offset += stride;
  }
  return { data: out, stride };
}

// Decodes a non-interlaced PNG into 8-bit samples per channel (gray, gray+alpha, RGB, or RGBA).
export function decodePng(buffer) {
  if (!buffer.subarray(0, 8).equals(SIGNATURE)) throw new Error("Not a PNG file.");
  let offset = 8;
  let header;
  let palette;
  let transparency;
  const idat = [];
  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("ascii", offset + 4, offset + 8);
    const chunk = buffer.subarray(offset + 8, offset + 8 + length);
    offset += 12 + length;
    if (type === "IHDR") {
      header = {
        width: chunk.readUInt32BE(0),
        height: chunk.readUInt32BE(4),
        bitDepth: chunk[8],
        colorType: chunk[9],
        interlace: chunk[12]
      };
    } else if (type === "PLTE") {
      palette = chunk;
    } else if (type === "tRNS") {
      transparency = chunk;
    } else if (type === "IDAT") {
      idat.push(chunk);
    } else if (type === "IEND") {
      break;
    }
  }
  if (!header) throw new Error("PNG has no IHDR chunk.");
  if (header.interlace !== 0) throw new Error("Interlaced PNG is not supported.");

  const { width, height, bitDepth, colorType } = header;
  const sourceChannels = CHANNELS[colorType];
  if (!sourceChannels) throw new Error(`Unsupported PNG color type ${colorType}.`);
  const { data, stride } = unfilter(inflateSync(Buffer.concat(idat)), width, height, sourceChannels * bitDepth);

  const sample = (row, index) => {
    if (bitDepth === 8) return data[row + index];
    if (bitDepth === 16) return data[row + index * 2];
    const bitOffset = index * bitDepth;
    const value = (data[row + (bitOffset >> 3)] >> (8 - bitDepth - (bitOffset & 7))) & ((1 << bitDepth) - 1);
    return colorType === 3 ? value : Math.round((value * 255) / ((1 << bitDepth) - 1));
  };

  const channels = colorType === 3 ? (transparency ? 4 : 3) : sourceChannels;
  const pixels = new Uint8Array(width * height * channels);
  for (let y = 0; y < height; y += 1) {
    const row = y * stride;
    for (let x = 0; x < width; x += 1) {
      const out = (y * width + x) * channels;
      if (colorType === 3) {
        const entry = sample(row, x);
        pixels[out] = palette[entry * 3];
        pixels[out + 1] = palette[entry * 3 + 1];
        pixels[out + 2] = palette[entry * 3 + 2];
        if (channels === 4) pixels[out + 3] = entry < transparency.length ? transparency[entry] : 255;
      } else {
        for (let c = 0; c < channels; c += 1) pixels[out + c] = sample(row, x * channels + c);
      }
    }
  }
  return { width, height, channels, data: pixels };
}

// Binary mask from a decoded PNG: alpha when the alpha channel carries information, otherwise luminance.
export function maskFromPng(png) {
  const { width, height, channels, data } = png;
  const count = width * height;
  const mask = new Uint8Array(count);
  const hasAlpha = channels === 2 || channels === 4;
  let alphaUsed = false;
  if (hasAlpha) {
    for (let i = 0; i < count; i += 1) {
      if (data[i * channels + channels - 1] !== 255) {
        alphaUsed = true;
        break;
      }
    }
  }
  for (let i = 0; i < count; i += 1) {
    const base = i * channels;
    let value;
    if (alphaUsed) value = data[base + channels - 1];
    else if (channels >= 3) value = Math.max(data[base], data[base + 1], data[base + 2]);
    else value = data[base];
    mask[i] = value > 127 ? 1 : 0;
  }
  return { width, height, mask };
}

function chunk(type, payload) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(payload.length);
  const typed = Buffer.concat([Buffer.from(type, "ascii"), payload]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(typed));
  return Buffer.concat([length, typed, crc]);
}

// Encodes 8-bit gray (channels 1), 8-bit RGB (channels 3), or 16-bit gray (Uint16Array, channels 1).
export function encodePng({ width, height, channels, data }) {
  const sixteen = data instanceof Uint16Array;
  const colorType = channels === 3 ? 2 : 0;
  if (sixteen && channels !== 1) throw new Error("16-bit PNG output supports gray only.");
  const bytesPerPixel = channels * (sixteen ? 2 : 1);
  const stride = width * bytesPerPixel;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const row = y * (stride + 1);
    raw[row] = 0;
    for (let x = 0; x < width * channels; x += 1) {
      const value = data[y * width * channels + x];
      if (sixteen) raw.writeUInt16BE(value, row + 1 + x * 2);
      else raw[row + 1 + x] = value;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = sixteen ? 16 : 8;
  ihdr[9] = colorType;
  return Buffer.concat([
    SIGNATURE,
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 6 })),
    chunk("IEND", Buffer.alloc(0))
  ]);
}
