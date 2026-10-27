// Bundled display font (Barlow Condensed, SIL Open Font License 1.1, see public/fonts/OFL.txt). Every canvas
// texture with text (kits, boards, fascia, crests) is drawn only after this resolves, so text is identical on
// every machine instead of falling back to whatever condensed sans is installed.
export const FONT_FAMILY = "Barlow Condensed";

let loading: Promise<void> | null = null;

export function loadFonts(base = "/fonts/"): Promise<void> {
  loading ??= (async () => {
    const faces = [
      new FontFace(FONT_FAMILY, `url(${base}BarlowCondensed-Bold.ttf)`, { weight: "700" }),
      new FontFace(FONT_FAMILY, `url(${base}BarlowCondensed-ExtraBold.ttf)`, { weight: "800" }),
    ];
    for (const f of faces) document.fonts.add(await f.load());
    await Promise.all([document.fonts.load(`700 40px "${FONT_FAMILY}"`), document.fonts.load(`800 40px "${FONT_FAMILY}"`)]);
  })();
  return loading;
}

/**
 * Draw `text` centred at (cx, cy), shrinking the font until it fits `maxWidth` (never relying on the browser's
 * horizontal squash). Returns the drawn width.
 */
export function fitText(ctx: CanvasRenderingContext2D, text: string, cx: number, cy: number, maxWidth: number, weight: number, size: number): number {
  let s = size;
  ctx.font = `${weight} ${s}px "${FONT_FAMILY}"`;
  while (ctx.measureText(text).width > maxWidth && s > 8) {
    s -= 2;
    ctx.font = `${weight} ${s}px "${FONT_FAMILY}"`;
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, cx, cy);
  return ctx.measureText(text).width;
}
