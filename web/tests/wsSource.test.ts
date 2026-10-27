// WsSource against the shared/schemas/stream.py protocol: hello, frames, pause, seek back/forward, speed.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import Ajv from "ajv";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildScriptedClip } from "../src/dev/scriptedClip";
import { WsSource } from "../src/source/WsSource";
import type { ClientMessage, ClockStatus, MatchMeta, ServerMessage, SimState } from "../src/types";
import type { ViewerProfile } from "../src/types/generated/contracts";

const repo = (p: string) => fileURLToPath(new URL(`../../${p}`, import.meta.url));
const readJson = (p: string): unknown => JSON.parse(readFileSync(repo(p), "utf8"));
const meta = readJson("shared/fixtures/MatchMeta.json") as MatchMeta;
const profile = readJson("shared/fixtures/ViewerProfile.json") as ViewerProfile;
const frames = buildScriptedClip(meta).frames;
const ajv = new Ajv({ allErrors: true, strict: false });
ajv.addSchema(readJson("shared/schemas/json/contracts.schema.json") as object, "contracts");
const validProtocol = (server: ServerMessage | null, client: ClientMessage | null) => {
  const v = ajv.getSchema("contracts#/$defs/StreamProtocol")!;
  // the container needs both halves; pair each message with a fixed valid counterpart
  const fixture = readJson("shared/fixtures/StreamProtocol.json") as { server: unknown; client: unknown };
  const ok = v({ server: server ?? fixture.server, client: client ?? fixture.client });
  return ok ? true : JSON.stringify(v.errors);
};

class FakeSocket {
  static OPEN = 1;
  static last: FakeSocket | null = null;
  readyState = 1;
  sent: ClientMessage[] = [];
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeSocket.last = this;
  }
  send(data: string) {
    const m = JSON.parse(data) as ClientMessage;
    expect(validProtocol(null, m)).toBe(true);
    this.sent.push(m);
  }
  close() {
    this.readyState = 3;
  }
  deliver(m: ServerMessage) {
    expect(validProtocol(m, null)).toBe(true);
    this.onmessage?.({ data: JSON.stringify(m) });
  }
}

const clock = (t: number, playing = true, speed = 1): ClockStatus => ({ t, tick: Math.round(t * 25), playing, speed, duration_s: 5400 });
const frame = (s: SimState): ServerMessage => ({ type: "frame", state: s });

let saved: unknown;
beforeEach(() => {
  saved = (globalThis as { WebSocket?: unknown }).WebSocket;
  (globalThis as { WebSocket?: unknown }).WebSocket = FakeSocket;
});
afterEach(() => {
  (globalThis as { WebSocket?: unknown }).WebSocket = saved;
});

async function connected(): Promise<{ src: WsSource; ws: FakeSocket }> {
  const src = new WsSource("ws://test/ws");
  const p = src.connect();
  const ws = FakeSocket.last!;
  ws.deliver({ type: "hello", schema_version: "1.0.0", session_id: "sess_t", meta, profile, clock: clock(frames[0]!.t) });
  await expect(p).resolves.toEqual(meta);
  return { src, ws };
}

/** Deliver frames i0..i1 and advance the render clock in step with them (25 Hz). */
function play(src: WsSource, ws: FakeSocket, i0: number, i1: number): number {
  let t = 0;
  for (let i = i0; i <= i1; i++) {
    ws.deliver(frame(frames[i]!));
    t = src.advance(0.04);
  }
  return t;
}

describe("WsSource", () => {
  it("hello resolves the meta and frames are rendered a little behind the newest tick", async () => {
    const { src, ws } = await connected();
    const t = play(src, ws, 0, 100);
    expect(src.buffer.length).toBe(101);
    expect(frames[100]!.t - t).toBeGreaterThan(0.05);
    expect(frames[100]!.t - t).toBeLessThan(0.3);
    expect(src.sample(t)).not.toBeNull();
  });

  it("pause keeps the buffer and freezes the clock, play resumes", async () => {
    const { src, ws } = await connected();
    play(src, ws, 0, 200); // 8 s in: the clock message is far from the last clock message (hello at t0)
    const before = src.advance(0);
    src.pause();
    expect(ws.sent.at(-1)).toEqual({ type: "pause" });
    ws.deliver({ type: "clock", clock: clock(frames[200]!.t, false) });
    expect(src.buffer.length).toBe(201);
    expect(src.advance(1.0)).toBeCloseTo(before, 6);
    expect(src.advance(1.0)).toBeCloseTo(before, 6);
    src.play();
    ws.deliver({ type: "clock", clock: clock(frames[200]!.t, true) });
    expect(src.buffer.length).toBe(201);
    const after = play(src, ws, 201, 210);
    expect(after).toBeGreaterThan(before);
  });

  it("a seek back (even under 1 s) clears the buffer and restarts from the answered time", async () => {
    const { src, ws } = await connected();
    play(src, ws, 0, 200);
    const target = frames[190]!.t; // 0.4 s back
    src.seek(target);
    expect(ws.sent.at(-1)).toEqual({ type: "seek", t: target });
    ws.deliver({ type: "clock", clock: clock(target, true) });
    expect(src.buffer.length).toBe(0);
    expect(src.advance(0)).toBeCloseTo(target, 6);
    const t = play(src, ws, 190, 220);
    expect(t).toBeGreaterThan(target);
    expect(t).toBeLessThan(frames[220]!.t);
  });

  it("a seek forward clears the buffer", async () => {
    const { src, ws } = await connected();
    play(src, ws, 0, 50);
    const target = frames[400]!.t;
    src.seek(target);
    ws.deliver({ type: "clock", clock: clock(target, true) });
    expect(src.buffer.length).toBe(0);
    play(src, ws, 400, 420);
    expect(src.buffer.first!.t).toBeCloseTo(target, 6);
  });

  it("a speed change keeps the buffer and scales the clock", async () => {
    const { src, ws } = await connected();
    play(src, ws, 0, 100);
    src.setSpeed(2);
    expect(ws.sent.at(-1)).toEqual({ type: "speed", speed: 2 });
    ws.deliver({ type: "clock", clock: clock(frames[100]!.t, true, 2) });
    expect(src.buffer.length).toBe(101);
    expect(src.clock().speed).toBe(2);
  });

  it("a seek from another client (server time jumps) clears the buffer", async () => {
    const { src, ws } = await connected();
    play(src, ws, 0, 100);
    ws.deliver({ type: "clock", clock: clock(frames[500]!.t, true) });
    expect(src.buffer.length).toBe(0);
  });
});
