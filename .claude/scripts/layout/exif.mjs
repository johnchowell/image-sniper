import { readFile } from "node:fs/promises";

// Horizontal field of view (deg) of the stored pixel grid from a JPEG's EXIF focal length, with its basis; null when
// the file has no usable focal length. Same rules as exif_fov_x in .claude/scripts/local/common.py.
export async function exifFovX(file) {
  let buffer;
  try {
    buffer = await readFile(file);
  } catch {
    return null;
  }
  if (buffer.length < 4 || buffer.readUInt16BE(0) !== 0xffd8) return null;
  let size = null;
  let tiff = null;
  for (let offset = 2; offset + 4 <= buffer.length && buffer[offset] === 0xff;) {
    const marker = buffer[offset + 1];
    const length = buffer.readUInt16BE(offset + 2);
    if (marker === 0xe1 && buffer.toString("latin1", offset + 4, offset + 10) === "Exif\0\0") tiff = offset + 10;
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
      size = { height: buffer.readUInt16BE(offset + 5), width: buffer.readUInt16BE(offset + 7) };
    }
    if (marker === 0xda || (size && tiff !== null)) break;
    offset += 2 + length;
  }
  if (!size || tiff === null) return null;
  const tags = exifTags(buffer, tiff);
  if (!tags) return null;
  const { width: W, height: H } = size;
  let half;
  let basis;
  const f35 = tags[0xa405] || 0;
  if (f35 > 0) {
    // 35 mm equivalent: matched on the diagonal of a 36 x 24 mm frame
    half = (43.2666 / (2 * f35)) * W / Math.hypot(W, H);
    basis = `FocalLengthIn35mmFilm ${f35} mm`;
  } else {
    const f = tags[0x920a] || 0;
    const resolution = tags[0xa20e] || 0;
    const unitMm = { 2: 25.4, 3: 10, 4: 1, 5: 0.001 }[tags[0xa210] || 2];
    if (!(f > 0 && resolution > 0 && unitMm)) return null;
    // focal length in stored pixels; the long-side ratio corrects a resize and is the same for a rotated grid
    const sampled = Math.max(tags[0xa002] || 0, tags[0xa003] || 0) || Math.max(W, H);
    const fPx = ((f * resolution) / unitMm) * Math.max(W, H) / sampled;
    half = W / (2 * fPx);
    basis = `FocalLength ${f} mm, ${fPx.toFixed(0)} px`;
  }
  const fov = (2 * Math.atan(half) * 180) / Math.PI;
  return fov >= 15 && fov <= 130 ? { fov_x_deg: Math.round(fov * 100) / 100, basis } : null;
}

// Numeric tags of the EXIF sub-IFD (rationals as numbers).
function exifTags(buffer, tiff) {
  const little = buffer.toString("latin1", tiff, tiff + 2) === "II";
  const u16 = (at) => (little ? buffer.readUInt16LE(at) : buffer.readUInt16BE(at));
  const u32 = (at) => (little ? buffer.readUInt32LE(at) : buffer.readUInt32BE(at));
  const i32 = (at) => (little ? buffer.readInt32LE(at) : buffer.readInt32BE(at));
  const readIfd = (start) => {
    const out = {};
    if (start + 2 > buffer.length) return out;
    const count = u16(start);
    for (let i = 0; i < count; i += 1) {
      const entry = start + 2 + i * 12;
      if (entry + 12 > buffer.length) break;
      const tag = u16(entry);
      const type = u16(entry + 2);
      if (type === 3) out[tag] = u16(entry + 8);
      else if (type === 4) out[tag] = u32(entry + 8);
      else if (type === 9) out[tag] = i32(entry + 8);
      else if (type === 11) out[tag] = little ? buffer.readFloatLE(entry + 8) : buffer.readFloatBE(entry + 8);
      else if (type === 5 || type === 10 || type === 12) {
        const at = tiff + u32(entry + 8);
        if (at + 8 > buffer.length) continue;
        if (type === 12) out[tag] = little ? buffer.readDoubleLE(at) : buffer.readDoubleBE(at);
        else {
          const [num, den] = type === 5 ? [u32(at), u32(at + 4)] : [i32(at), i32(at + 4)];
          out[tag] = den ? num / den : 0;
        }
      }
    }
    return out;
  };
  try {
    const ifd0 = readIfd(tiff + u32(tiff + 4));
    return ifd0[0x8769] ? readIfd(tiff + ifd0[0x8769]) : null;
  } catch {
    return null;
  }
}
