// ReGolazo block numerals: an original athletic numeral set drawn from polygons (no font dependency).
// Each digit is built from chamfered blocks on a 6 x 10 grid.

type Seg = "a" | "b" | "c" | "d" | "e" | "f" | "g";

const DIGITS: Record<string, Seg[]> = {
  "0": ["a", "b", "c", "d", "e", "f"],
  "1": ["b", "c"],
  "2": ["a", "b", "g", "e", "d"],
  "3": ["a", "b", "g", "c", "d"],
  "4": ["f", "g", "b", "c"],
  "5": ["a", "f", "g", "c", "d"],
  "6": ["a", "f", "g", "e", "c", "d"],
  "7": ["a", "b", "c"],
  "8": ["a", "b", "c", "d", "e", "f", "g"],
  "9": ["a", "b", "c", "d", "f", "g"],
};

const W = 6;
const H = 10;
const T = 1.55; // stroke thickness in grid units
const C = 0.55; // chamfer

/** Polygon (grid units) for one segment. Horizontal segments span the full width so joins are solid. */
function segPoly(s: Seg): [number, number][] {
  const m = H / 2;
  const rect = (x0: number, y0: number, x1: number, y1: number, chamfer: [boolean, boolean, boolean, boolean]) => {
    // chamfer flags: top-left, top-right, bottom-right, bottom-left
    const c = C;
    const pts: [number, number][] = [];
    pts.push(chamfer[0] ? [x0 + c, y0] : [x0, y0]);
    pts.push(chamfer[1] ? [x1 - c, y0] : [x1, y0]);
    if (chamfer[1]) pts.push([x1, y0 + c]);
    pts.push(chamfer[2] ? [x1, y1 - c] : [x1, y1]);
    if (chamfer[2]) pts.push([x1 - c, y1]);
    pts.push(chamfer[3] ? [x0 + c, y1] : [x0, y1]);
    if (chamfer[3]) pts.push([x0, y1 - c]);
    if (chamfer[0]) pts.push([x0, y0 + c]);
    return pts;
  };
  switch (s) {
    case "a":
      return rect(0, 0, W, T, [true, true, false, false]);
    case "d":
      return rect(0, H - T, W, H, [false, false, true, true]);
    case "g":
      return rect(0, m - T / 2, W, m + T / 2, [false, false, false, false]);
    case "f":
      return rect(0, 0, T, m + T / 2, [true, false, false, false]);
    case "b":
      return rect(W - T, 0, W, m + T / 2, [false, true, false, false]);
    case "e":
      return rect(0, m - T / 2, T, H, [false, false, false, true]);
    case "c":
      return rect(W - T, m - T / 2, W, H, [false, false, true, false]);
  }
}

function digitPolys(d: string): [number, number][][] {
  const segs = DIGITS[d] ?? [];
  const polys = segs.map(segPoly);
  if (d === "1") {
    // a flag serif and a centred stem make the 1 read as a kit numeral, not a calculator digit
    return [
      [
        [W / 2 - T / 2, 0],
        [W / 2 + T / 2, 0],
        [W / 2 + T / 2, H],
        [W / 2 - T / 2, H],
      ],
      [
        [W / 2 - T / 2, 0],
        [W / 2 - T / 2 - 1.6, 1.4],
        [W / 2 - T / 2 - 1.6, 1.4 + T * 0.8],
        [W / 2 - T / 2, T * 0.8],
      ],
    ];
  }
  return polys;
}

/**
 * Draw `text` (digits) centred at (cx, cy) with height h.
 * Outline is drawn first (wider stroke) so the fill sits on top.
 */
export function drawNumber(
  ctx: CanvasRenderingContext2D,
  text: string,
  cx: number,
  cy: number,
  h: number,
  fill: string,
  outline: string | null,
  outlineWidth = 0.08,
): void {
  const u = h / H;
  const gap = 1.1;
  const widths = [...text].map((d) => (d === "1" ? W * 0.62 : W));
  const total = widths.reduce((a, b) => a + b, 0) + gap * (text.length - 1);
  let x = cx - (total * u) / 2;
  const y = cy - h / 2;
  const paths: Path2D[] = [];
  [...text].forEach((d, i) => {
    const off = d === "1" ? -W * 0.19 : 0;
    for (const poly of digitPolys(d)) {
      const p = new Path2D();
      poly.forEach(([px, py], k) => {
        const X = x + (px + off) * u;
        const Y = y + py * u;
        if (k === 0) p.moveTo(X, Y);
        else p.lineTo(X, Y);
      });
      p.closePath();
      paths.push(p);
    }
    x += (widths[i]! + gap) * u;
  });
  if (outline) {
    ctx.strokeStyle = outline;
    ctx.lineJoin = "miter";
    ctx.lineWidth = h * outlineWidth;
    for (const p of paths) ctx.stroke(p);
  }
  ctx.fillStyle = fill;
  for (const p of paths) ctx.fill(p);
}
