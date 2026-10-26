import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import * as fixtures from "./generated/fixtures";
import { PASS_TECHNIQUE, SCHEMA_VERSION } from "../src/types/generated/enums";
import type { ActionIntent, StreamProtocol, ViewerProfile } from "../src/types/generated/contracts";

type ClientMessage = StreamProtocol["client"];

const repo = (p: string) => fileURLToPath(new URL(`../../${p}`, import.meta.url));
const readJson = (p: string): unknown => JSON.parse(readFileSync(repo(p), "utf8"));

const ajv = new Ajv({ allErrors: true, strict: false });
const bundle = readJson("shared/schemas/json/contracts.schema.json") as { required: string[] };
ajv.addSchema(bundle, "contracts");
const validatorFor = (name: string) => {
  const validate = ajv.getSchema(`contracts#/$defs/${name}`);
  if (!validate) throw new Error(`no schema for ${name}`);
  return validate;
};
const expectValid = (name: string, value: unknown) => {
  const validate = validatorFor(name);
  expect(validate(value), `${name}: ${JSON.stringify(validate.errors)}`).toBe(true);
};

const typedFixtures: Record<string, unknown> = Object.fromEntries(
  Object.entries(fixtures)
    .filter(([key]) => key.endsWith("Fixture"))
    .map(([key, value]) => [key.replace(/Fixture$/, "").replace(/^./, (c) => c.toUpperCase()), value]),
);

describe("contracts (Python -> JSON -> TS)", () => {
  it("has a typed TS fixture for every root contract", () => {
    expect(Object.keys(typedFixtures).sort()).toEqual([...bundle.required].sort());
  });

  for (const name of bundle.required) {
    it(`${name}: JSON fixture equals the typed TS fixture and validates`, () => {
      const json = readJson(`shared/fixtures/${name}.json`);
      expect(json).toEqual(typedFixtures[name]);
      expectValid(name, json);
    });
  }

  it("every event-qualifier and cue-payload example is typed and validates", () => {
    const events = readJson("shared/fixtures/examples/MatchEvent.json") as unknown[];
    const cues = readJson("shared/fixtures/examples/OverlayCue.json") as unknown[];
    expect(events).toEqual(fixtures.matchEventExamples);
    expect(cues).toEqual(fixtures.overlayCueExamples);
    events.forEach((e) => expectValid("MatchEvent", e));
    cues.forEach((c) => expectValid("OverlayCue", c));
    const kinds = new Set(fixtures.matchEventExamples.map((e) => e.qualifiers?.kind).filter(Boolean));
    expect(kinds.size).toBe(21);
    expect(new Set(fixtures.overlayCueExamples.map((c) => c.payload.type)).size).toBe(11);
  });

  it("rejects structurally malformed data via the exported schema", () => {
    const broken = structuredClone(fixtures.simStateFixture) as unknown as { ball: { pos: number[] } };
    broken.ball.pos = [1, 2];
    expect(validatorFor("SimState")(broken)).toBe(false);
  });
});

describe("contracts (TS -> JSON -> Python)", () => {
  // Built only from the generated TS types. Written to shared/fixtures/from_ts/ where a pytest parses them with
  // Pydantic; CI fails if the committed file differs from what this test writes.
  const intents: ActionIntent[] = [
    {
      kind: "shot",
      player: "H09",
      target_player: null,
      target: [52.5, -3.1, 1.9],
      path: [],
      curve: 0.4,
      power: 0.85,
      pass_technique_override: null,
      shot_technique_override: "finesse",
      source: "fan",
    },
    {
      kind: "pass_space",
      player: "H10",
      target_player: null,
      target: [38, -4, 0],
      path: [
        [18.4, 6.1],
        [30, 0],
        [38, -4],
      ],
      curve: 0.35,
      power: 0.6,
      pass_technique_override: "curled",
      shot_technique_override: null,
      source: "fan",
    },
  ];
  const profile: ViewerProfile = {
    schema_version: SCHEMA_VERSION,
    session_id: "sess_test",
    mode: "casual",
    favorite_club: "away",
    language: "es",
    voice_enabled: true,
    voice: null,
    show_numbers_first: false,
  };
  const commands: ClientMessage[] = [
    { type: "play" },
    { type: "pause" },
    { type: "seek", t: 2330 },
    { type: "speed", speed: 2 },
    { type: "profile", profile },
  ];

  it("client-built messages validate against the Python-exported schema and are written for pytest", () => {
    intents.forEach((i) => expectValid("ActionIntent", i));
    expectValid("ViewerProfile", profile);
    const out = { ActionIntent: intents, ViewerProfile: [profile], ClientMessage: commands };
    const path = repo("shared/fixtures/from_ts/client_messages.json");
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, JSON.stringify(out, null, 2) + "\n");
    expect(PASS_TECHNIQUE).toContain("curled");
  });
});
