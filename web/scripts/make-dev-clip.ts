// Writes the scripted development clip (MatchMeta fixture + ~27 s of 25 Hz SimStates) to public/dev/.
//   node scripts/make-dev-clip.ts
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { buildScriptedClip } from "../src/dev/scriptedClip.ts";
import type { MatchMeta } from "../src/types.ts";

const repo = (p: string) => fileURLToPath(new URL(`../../${p}`, import.meta.url));
const meta = JSON.parse(readFileSync(repo("shared/fixtures/MatchMeta.json"), "utf8")) as MatchMeta;
const clip = buildScriptedClip(meta);
const out = repo("web/public/dev/clip_demo.json");
mkdirSync(fileURLToPath(new URL("../public/dev/", import.meta.url)), { recursive: true });
writeFileSync(out, JSON.stringify(clip));
const first = clip.frames[0]!;
const last = clip.frames[clip.frames.length - 1]!;
console.log(`wrote ${out}: ${clip.frames.length} frames, t ${first.t}..${last.t}, score ${last.score.home}-${last.score.away}`);
