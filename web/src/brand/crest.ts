// Generated club crests from CrestSpec: flat, two-colour, geometric (docs/art_direction.md section 3).
import type { CrestSpec } from "../types.ts";

export const DISPLAY_FONT = `"Barlow Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif`;

function shapePath(shape: CrestSpec["shape"], s: number): Path2D {
  const p = new Path2D();
  const c = s / 2;
  switch (shape) {
    case "shield": {
      p.moveTo(s * 0.12, s * 0.1);
      p.lineTo(s * 0.88, s * 0.1);
      p.lineTo(s * 0.88, s * 0.5);
      p.bezierCurveTo(s * 0.88, s * 0.74, s * 0.7, s * 0.86, c, s * 0.95);
      p.bezierCurveTo(s * 0.3, s * 0.86, s * 0.12, s * 0.74, s * 0.12, s * 0.5);
      p.closePath();
      break;
    }
    case "diamond":
      p.moveTo(c, s * 0.04);
      p.lineTo(s * 0.96, c);
      p.lineTo(c, s * 0.96);
      p.lineTo(s * 0.04, c);
      p.closePath();
      break;
    case "hexagon":
      for (let i = 0; i < 6; i++) {
        const a = Math.PI / 6 + (i * Math.PI) / 3;
        const x = c + Math.cos(a) * s * 0.46;
        const y = c + Math.sin(a) * s * 0.46;
        if (i === 0) p.moveTo(x, y);
        else p.lineTo(x, y);
      }
      p.closePath();
      break;
    default:
      p.arc(c, c, s * 0.46, 0, Math.PI * 2);
  }
  return p;
}

function emblem(ctx: CanvasRenderingContext2D, kind: CrestSpec["emblem"], cx: number, cy: number, r: number, col: string): void {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.fillStyle = col;
  ctx.strokeStyle = col;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  switch (kind) {
    case "anchor": {
      ctx.lineWidth = r * 0.16;
      ctx.beginPath();
      ctx.arc(0, -r * 0.72, r * 0.16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, -r * 0.56);
      ctx.lineTo(0, r * 0.78);
      ctx.moveTo(-r * 0.36, -r * 0.34);
      ctx.lineTo(r * 0.36, -r * 0.34);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, r * 0.16, r * 0.62, Math.PI * 0.12, Math.PI * 0.88);
      ctx.stroke();
      for (const sx of [-1, 1]) {
        const ax = Math.cos(sx > 0 ? Math.PI * 0.12 : Math.PI * 0.88) * r * 0.62;
        const ay = r * 0.16 + Math.sin(Math.PI * 0.12) * r * 0.62;
        ctx.beginPath();
        ctx.moveTo(ax + sx * r * 0.06, ay - r * 0.3);
        ctx.lineTo(ax + sx * r * 0.14, ay + r * 0.04);
        ctx.lineTo(ax - sx * r * 0.2, ay + r * 0.0);
        ctx.closePath();
        ctx.fill();
      }
      break;
    }
    case "bolt": {
      ctx.beginPath();
      ctx.moveTo(r * 0.18, -r * 0.95);
      ctx.lineTo(-r * 0.46, r * 0.12);
      ctx.lineTo(-r * 0.04, r * 0.12);
      ctx.lineTo(-r * 0.22, r * 0.95);
      ctx.lineTo(r * 0.48, -r * 0.16);
      ctx.lineTo(r * 0.06, -r * 0.16);
      ctx.closePath();
      ctx.fill();
      break;
    }
    case "star": {
      ctx.beginPath();
      for (let i = 0; i < 10; i++) {
        const a = -Math.PI / 2 + (i * Math.PI) / 5;
        const rr = i % 2 === 0 ? r * 0.85 : r * 0.36;
        if (i === 0) ctx.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
        else ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr);
      }
      ctx.closePath();
      ctx.fill();
      break;
    }
    case "wave": {
      ctx.lineWidth = r * 0.16;
      for (let k = -1; k <= 1; k++) {
        ctx.beginPath();
        for (let i = 0; i <= 24; i++) {
          const x = -r * 0.8 + (i / 24) * r * 1.6;
          const y = k * r * 0.38 + Math.sin((i / 24) * Math.PI * 2) * r * 0.14;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      break;
    }
    case "tower": {
      ctx.fillRect(-r * 0.42, -r * 0.4, r * 0.84, r * 1.2);
      for (let i = 0; i < 3; i++) ctx.fillRect(-r * 0.42 + i * r * 0.32, -r * 0.62, r * 0.2, r * 0.24);
      ctx.globalCompositeOperation = "destination-out";
      ctx.beginPath();
      ctx.arc(0, r * 0.6, r * 0.17, Math.PI, 0);
      ctx.rect(-r * 0.17, r * 0.6, r * 0.34, r * 0.2);
      ctx.fill();
      break;
    }
    case "bird": {
      ctx.beginPath();
      ctx.moveTo(-r * 0.9, -r * 0.3);
      ctx.quadraticCurveTo(-r * 0.3, -r * 0.1, 0, r * 0.35);
      ctx.quadraticCurveTo(r * 0.3, -r * 0.1, r * 0.9, -r * 0.3);
      ctx.quadraticCurveTo(r * 0.3, r * 0.15, 0, r * 0.7);
      ctx.quadraticCurveTo(-r * 0.3, r * 0.15, -r * 0.9, -r * 0.3);
      ctx.fill();
      break;
    }
    case "lion": {
      ctx.beginPath();
      for (let i = 0; i < 16; i++) {
        const a = (i * Math.PI * 2) / 16;
        const rr = i % 2 === 0 ? r * 0.85 : r * 0.62;
        if (i === 0) ctx.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
        else ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr);
      }
      ctx.closePath();
      ctx.fill();
      break;
    }
    case "oak": {
      ctx.beginPath();
      ctx.ellipse(0, -r * 0.1, r * 0.42, r * 0.72, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = r * 0.12;
      ctx.beginPath();
      ctx.moveTo(0, r * 0.4);
      ctx.lineTo(0, r * 0.95);
      ctx.stroke();
      break;
    }
  }
  ctx.restore();
}

/** Draws a crest into a square of side `s` at (x, y). */
export function drawCrest(ctx: CanvasRenderingContext2D, spec: CrestSpec, x: number, y: number, s: number): void {
  ctx.save();
  ctx.translate(x, y);
  const outer = shapePath(spec.shape, s);
  ctx.fillStyle = spec.secondary;
  ctx.fill(outer);
  // inner field: same shape scaled in
  ctx.save();
  ctx.translate(s * 0.09, s * 0.09);
  ctx.scale(0.82, 0.82);
  ctx.fillStyle = spec.primary;
  ctx.fill(shapePath(spec.shape, s));
  if (spec.shape === "roundel") {
    ctx.strokeStyle = spec.secondary;
    ctx.lineWidth = s * 0.035;
    ctx.beginPath();
    ctx.arc(s / 2, s / 2, s * 0.34, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
  const emblemCol = spec.shape === "roundel" ? spec.secondary : "#ffffff";
  emblem(ctx, spec.emblem, s / 2, s * 0.46, s * 0.22, emblemCol);
  ctx.fillStyle = "#ffffff";
  ctx.font = `700 ${Math.round(s * 0.15)}px ${DISPLAY_FONT}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(spec.monogram, s / 2, spec.shape === "shield" ? s * 0.76 : s * 0.78);
  ctx.restore();
}

export function crestCanvas(spec: CrestSpec, size = 256): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  drawCrest(ctx, spec, 0, 0, size);
  return c;
}
