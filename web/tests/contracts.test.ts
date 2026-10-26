import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import * as fixtures from "./generated/fixtures";
import { PASS_TECHNIQUE, SCHEMA_VERSION } from "../src/types/generated/enums";
import type { ActionIntent, ViewerProfile } from "../src/types/generated/contracts";

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

const typed: Record<string, unknown> = Object.fromEntries(
  Object.entries(fixtures).map(([key, value]) => [key.replace(/Fixture$/, "").replace(/^./, (c) => c.toUpperCase()), value]),
);

describe("contracts (Python -> JSON -> TS)", () => {
  it("has a typed TS fixture for every root contract", () => {
    expect(Object.keys(typed).sort()).toEqual([...bundle.required].sort());
  });

  for (const name of bundle.required) {
    it(`${name}: JSON fixture equals the typed TS fixture and validates`, () => {
      const json = readJson(`shared/fixtures/${name}.json`);
      expect(json).toEqual(typed[name]);
      const validate = validatorFor(name);
      expect(validate(json), JSON.stringify(validate.errors)).toBe(true);
    });
  }

  it("rejects malformed data via the exported schema", () => {
    const validate = validatorFor("SimState");
    const broken = structuredClone(fixtures.simStateFixture) as unknown as { ball: { pos: number[] } };
    broken.ball.pos = [1, 2];
    expect(validate(broken)).toBe(false);
  });

  it("client-built messages (TS -> JSON) validate against the Python-exported schema", () => {
    const intent: ActionIntent = {
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
    };
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
    const validateIntent = validatorFor("ActionIntent");
    expect(validateIntent(JSON.parse(JSON.stringify(intent))), JSON.stringify(validateIntent.errors)).toBe(true);
    expect(validatorFor("ViewerProfile")(JSON.parse(JSON.stringify(profile)))).toBe(true);
    expect(PASS_TECHNIQUE).toContain("curled");
  });
});
