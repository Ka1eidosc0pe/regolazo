// Canvas textures for kits: per-player back (name + number) and per-team front (crest + sponsor).
import * as THREE from "three";
import { crestCanvas, DISPLAY_FONT } from "../brand/crest.ts";
import { drawNumber } from "../brand/numerals.ts";
import type { KitColors, TeamMeta } from "../types.ts";

export const SPONSORS: Record<string, string> = {
  home: "TIDEWATER",
  away: "KESTREL AIR",
};

function tex(c: HTMLCanvasElement): THREE.CanvasTexture {
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  return t;
}

/** 512 x 512: name in the top 20 %, block number below. Transparent background. */
export function backTexture(name: string, number: number | null, kit: KitColors): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 512;
  const ctx = c.getContext("2d")!;
  if (name) {
    ctx.fillStyle = kit.number;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const label = name.toUpperCase();
    let size = 78;
    ctx.font = `700 ${size}px ${DISPLAY_FONT}`;
    while (ctx.measureText(label).width > 440 && size > 30) {
      size -= 4;
      ctx.font = `700 ${size}px ${DISPLAY_FONT}`;
    }
    ctx.save();
    ctx.translate(256, 56);
    ctx.scale(0.86, 1); // condensed
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }
  if (number !== null) drawNumber(ctx, String(number), 256, 306, 330, kit.number, kit.accent, 0.045);
  return tex(c);
}

/** 1024 x 256: crest in the left square, sponsor wordmark in the remaining 768 x 256. */
export function frontTexture(team: TeamMeta, kit: KitColors, sponsor: string): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 1024;
  c.height = 256;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(crestCanvas(team.crest, 256), 0, 0);
  ctx.fillStyle = kit.number;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.save();
  ctx.translate(640, 128);
  ctx.transform(1, 0, -0.18, 1, 0, 0); // italic lean
  let size = 150;
  ctx.font = `800 ${size}px ${DISPLAY_FONT}`;
  while (ctx.measureText(sponsor).width > 700 && size > 40) {
    size -= 6;
    ctx.font = `800 ${size}px ${DISPLAY_FONT}`;
  }
  ctx.fillText(sponsor, 0, 6);
  ctx.restore();
  return tex(c);
}

export function blankTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 4;
  return tex(c);
}
