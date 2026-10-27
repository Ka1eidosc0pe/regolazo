# ReGolazo art direction

**One line:** a night match on TV where the players are premium figurines. The world is shot like a real broadcast (stadium bowl, floodlights, haze, a wet pitch that catches the light, a main camera on the gantry). The athletes are sculpted, smooth and faceless, wearing detailed kits. Everything is original: no real players, clubs, crests, kit makers or sponsors.

Reference screenshots live in [`docs/art/`](art/) and are regenerated with `npm run art` in `web/` (see section 9).

**Status (2026-10-26):** the user approved the stadium, lighting and kit direction. Before merge they asked for athlete polish: smooth sculpted bodies without visible mannequin joints, hair silhouettes, and crowd variety. That polish is in place (sections 4 and 5) and the screenshots were re-captured for review.

## 1. Mood

- **Night, under floodlights.** The sky is close to black. A ring of LED floodlights along the roof edge is the only strong light source. The pitch is the brightest surface in the frame and the stands fall off into shadow.
- **Light haze.** A thin blue-grey haze hangs over the bowl: the far stand and roof read softer and cooler than the pitch, and the floodlight heads bloom into soft stars. The haze also hides the low detail of the background, so the eye stays on the pitch.
- **Wet pitch.** The grass has been watered before kick-off. It is not a mirror, but it has a soft sheen: broad floodlight highlights slide across the stripes as the camera pans, and the stripes swap brightness depending on the camera's angle to the mowing direction (as real grass does).
- **Floodlight shadows.** Floodlights at four corners light the pitch; the two near-side banks cast shadows, so every player throws a faint two-way shadow away from the main camera, the signature of a night match. A tight contact shadow grounds the feet and the ball.
- **Colour temperature.** Cool white floodlights (about 5,600 K), cooler shadows, slightly warm hospitality glass and LED boards. Kits stay saturated; the world stays restrained.

## 2. Palette

| Role | Name | Hex | Notes |
|---|---|---|---|
| Sky / UI ink | Midnight | `#05070d` | Base of the sky dome and of UI panels |
| Sky gradient | Deep navy | `#0b1020` | Upper sky |
| Horizon haze | Sodium dusk | `#2a2433` | City glow under the haze, low on the horizon |
| Haze / fog | Mist | `#1c2433` | Fog colour, distance falloff |
| Floodlight | Lumen | `#f4f7ff` | Emissive lamp heads, HDR > 1 so they bloom |
| Grass, light stripe | Sward | `#3f7f34` | Linear-space tone is tuned in the shader |
| Grass, dark stripe | Fairway | `#2e6428` | |
| Line paint | Chalk | `#f1f4ee` | Lines are slightly off-white, never pure white |
| Run-off track | Asphalt teal | `#16302c` | Synthetic surface between boards and stands |
| Seats | Slate | `#1d2532` | Under the crowd |
| Rewind ghost | Rewind cyan | `#38e1ff` | Reserved for the holographic Rewind look (Step 9) |
| Highlight | Volt | `#d7ff3a` | Brand accent: live markers, focus rings |
| Alert | Flare | `#ff7a1a` | Danger, shot alerts |

Kit colours come from `MatchMeta` (`TeamMeta.kit`, `gk_kit`); the renderer never hard-codes club colours.

## 3. Clubs, kits and crests

Both clubs are original (fixture `shared/fixtures/MatchMeta.json`).

### Harbour City Mariners (home, `HCM`)
- A port-city club. Calm, technical, nautical.
- **Kit:** deep navy shirt (`#0b3d91`) with ice-blue (`#4cc9f0`) raglan shoulder panels and side panels, white collar; white shorts (`#f2f4f8`) with a navy-and-ice side stripe; navy socks with an ice-blue band. White numbers and names.
- **Goalkeeper:** signal green (`#2dc653`) with dark navy (`#0d1b2a`) shorts, long sleeves and gloves.
- **Crest:** shield, navy field with an ice-blue rim, a white anchor emblem and the monogram **HC**.
- **Shirt sponsor (fictional):** TIDEWATER.

### Redcliffe Forge (away, `RFG`)
- An industrial-town club. Direct, physical, loud.
- **Kit:** red shirt (`#c8102e`) with a bold black chevron (`#1a1a1a`) across the chest, piped in forge gold (`#f5b700`); black shorts; red socks with a black-and-gold band. White numbers and names.
- **Goalkeeper:** violet (`#7b2cbf`) with dark navy shorts.
- **Crest:** roundel, red ring around a gold lightning bolt, monogram **RF**.
- **Shirt sponsor (fictional):** KESTREL AIR.

### Referee
- Charcoal (`#23262d`) shirt and shorts with a volt (`#d7ff3a`) collar and cuffs; black socks. No number.

### Kit construction rules (all teams)
- Shirt = `primary`, shorts = `secondary`, socks = `primary`, trims and piping = `accent`, numbers and names = `number`.
- The pattern (`plain`, `stripes`, `hoops`, `sash`, `halves`, `chevron`) is drawn on the front of the shirt; the back stays plain behind the number.
- Back: player name in condensed capitals above a large block number. Front: crest on the left chest, sponsor across the chest, small number on the left leg of the shorts.
- Numbers use ReGolazo's own **block numerals** (drawn from polygons in code, not a font), white with a thin accent outline.
- Fabric reads as fabric: a soft sheen at grazing angles, a fine knit texture in close-ups, low specular. Boots are glossy.

### Crest style
Flat, two-colour, geometric. One outer shape from `CrestSpec.shape` (shield, circle, roundel, diamond, hexagon), a thick rim in `secondary`, one simple emblem (`star`, `wave`, `bird`, `lion`, `tower`, `bolt`, `anchor`, `oak`) and the monogram. No gradients, no fine detail: a crest must read at 24 px on a scorebug and as a small patch on a shirt. Crests are generated from `CrestSpec` by `web/src/brand/crest.ts`.

## 4. Athletes

- **Body:** a smooth sculpted figurine generated for the Mixamo skeleton (`scripts/assets/figurine.mjs`): tapered capsules and ellipsoids on the bind-pose bones (V-shaped torso, deltoids, biceps, forearms, strong thighs and calves, mitten hands, boots), smoothly unioned into one signed distance field, polygonised with marching cubes (about 40k triangles) and skinned by distance to each bone's shapes. No joints, no segments, no seams. Scaled per player from `height_cm`. The Mixamo clips drive it unchanged.
- **Surface:** a single kit shader decides per pixel, in bind-pose space, what each part is (skin, shirt, sleeves, collar, shorts, socks, boots, gloves). That keeps all 22 players on one shader program while every player has their own colours, number and name.
- **Skin:** smooth satin, like painted vinyl, with a slight clear coat. Six skin tones, assigned deterministically per player.
- **Head:** a smooth egg with a soft jaw and no face. Hair is a sculpted shell on the cranium (crop, fade, curls, bun, ponytail, shaved, quiff) that meets the scalp tangentially at a shaped hairline (forehead, temples, sideburns, nape), with combed grooves or curl bumps in the shader and short sides fading into the skin tone, so it reads as hair rather than a helmet. Style, hair colour and skin tone are picked deterministically from the player id.
- **Boots:** glossy, one colour per player from a small set (black, white, volt, flare, ice, red).
- **Goalkeepers:** long sleeves and gloves.

## 5. Stadium: Lantern Park

- **Bowl:** a continuous two-tier bowl on a rounded-rectangle plan. Lower tier from the pitch-side wall up to a concourse band; a glowing hospitality strip (warm glass) between tiers; a steeper upper tier; a thin cantilevered roof.
- **Floodlights:** a continuous ring of LED lamp heads along the inner roof edge, angled down to the pitch, rendered as HDR emissive so they bloom. Haze makes their light visible as soft cones.
- **Crowd:** instanced, about 30,000 shader cards: seated, broad seated and standing silhouettes (standing fans in the away end and behind the goals, never in the back rows below the LED fascia), size and width variety, some caps and striped scarves, skin-tone heads. Home colours dominate; the away end is red, black and gold. Figures bob and sway out of phase, raise their arms after a goal; a few phone lights twinkle. Distant stands fade to a soft texture; replays soften them further with depth of field.
- **Pitch surround:** grass run-off, then LED advertising boards on all four sides with **fictional sponsors** (NOVA LUME, TIDEWATER, KESTREL AIR, ORBIT FIZZ, HALCYON MOTORS, BLUEFIN, MERIDIAN, PIXEL HARVEST), then a teal synthetic track to the stands. The boards scroll slowly and glow.
- **Pitch:** 105 x 68 m, 20 mowing stripes across the length, shader-drawn markings (sharp at every distance), subtle wear in the goalmouths, grass micro-detail in close-ups.
- **Goals:** white round posts and crossbar, a dark back frame, a fine net that bulges when the ball hits it.
- **Ball:** an original panel design (white with navy and ice-blue panels and a flare stripe). It spins with its simulated angular velocity, casts a contact shadow and leaves a short light trail on hard strikes.

## 6. Lighting

- Four directional lights from the four corners at about 50 degrees elevation (the floodlight banks); the two near-side banks cast shadows (four shadow maps cost about 20 % of the frame budget), plus a dim cool hemisphere light (sky versus grass bounce).
- An environment map built in code from a dark dome with a bright ring of lamp heads: it gives the grass its wet sheen and the athletes' skin, boots and hair their highlights.
- ACES filmic tone mapping, bloom on HDR emissives only (floodlights, LED boards, hospitality glass), a subtle vignette, gentle cool-shadow / warm-highlight grade, very light film grain. Depth of field only on replays.

## 7. Camera language

| Camera | Use | Behaviour |
|---|---|---|
| **Broadcast** (default, recording) | Live play, Spotter training video | On the main gantry at the halfway line, about 21 m high and 30 m back from the near touchline. Pans, tilts and zooms only (no dolly), like a real main camera, which also keeps the Spotter's homography simple. Aims at the ball plus a lead of about 0.6 s of its velocity and drifts toward the attacking direction. Zoom follows the spread of nearby players and tightens in the final third. All motion is critically damped; no snapping. |
| **Tactical** | Analyst mode, Rewind freeze | High and behind the near touchline, about 65 m up, looking down at roughly 60 degrees, the whole width of the pitch in frame. |
| **Replay / cinematic** | Goals, Rewind playback, close-ups | Low (1.6 to 4 m), long lens (vertical FOV 12 to 20 degrees), slowly orbiting the ball carrier; depth of field on. |

Transitions between cameras are eased blends of position, orientation and field of view over 1 to 1.5 s (no hard cuts during open play). Replays may cut.

## 8. Overlay typography (for Step 10; mount point `#overlay-root`)

- **Display / numbers:** a condensed sports face (Barlow Condensed, SIL OFL), semi-bold for names, bold tabular figures for the scorebug, clock and stats.
- **Text:** Inter (SIL OFL) for captions and commentary.
- Fonts must be bundled with the app (no CDN) in Step 10. Until then overlays fall back to the system condensed sans.
- **Panels:** midnight glass (`#05070d` at 72 % opacity, 12 px backdrop blur), 2 px volt or club-colour key line, 6 px corner radius, no drop shadows. Text is off-white `#e8ecf4`; secondary text `#9aa6bd`.
- **Scorebug:** top-left, compact: crest, three-letter code, score, clock. Club colours as thin edge bars, never as full fills.
- **Motion:** overlays slide 12 px and fade over 180 ms; nothing bounces.

## 9. How the screenshots are made

`web/scripts/capture-art.ts` starts the app with the dev clip, freezes it at fixed match times and saves JPEGs to `docs/art/` through Playwright (headless Chromium on the Metal GPU, 1920 x 1080):

| File | View |
|---|---|
| `ref_broadcast_wide.jpg` | Main camera, build-up through midfield |
| `ref_broadcast_attack.jpg` | Main camera, the right winger attacking the box (away end behind) |
| `ref_tactical.jpg` | Tactical camera |
| `ref_closeup_players.jpg` | Low close-up of the winger on the ball |
| `ref_closeup_box.jpg` | Low close-up: striker attacking the cross, keeper set |
| `ref_replay_goal.jpg` | Replay camera (depth of field) on the finish: name and number on the back, slide tackle, keeper |
| `ref_replay_celebration.jpg` | Replay camera on the celebration |
| `ref_stadium_night.jpg` | The bowl: roof floodlight ring, tiers, hospitality glass, LED fascia |

## 10. Renderer tooling and gate results (Step 5)

All commands run in `web/`.

| Command | What it does |
|---|---|
| `npm run assets` | Rebuilds `public/assets/athlete.glb` + `athlete_clips.json` from the raw Mixamo FBX files (gitignored) |
| `npm run dev-clip` | Rebuilds the scripted dev clip `public/dev/clip_demo.json` |
| `npm run art` | Reference screenshots to `docs/art/` |
| `node scripts/record.ts` | Recording mode: frames + one `FrameLabels` JSON per frame to `$REGOLAZO_DATA_DIR/renders/<match>-<clip>/` |
| `node scripts/check-labels.ts` | Label projection test; overlays and `report.json` to `docs/art/labels/` |
| `node scripts/fps.ts [--unthrottled]` | Frame-rate check of the live view (headed Chromium, 1920 x 1080 viewport) |
| `npx playwright test --project visual` | Visual regression against `e2e/__screenshots__/` (GPU baselines, skipped on CI) |

Measured on the M3 Pro (ANGLE Metal), 2026-10-26, after the athlete polish (figurine bodies, hair, crowd variety):

- **Frame rate, live view at 1920 x 1080** (`node scripts/fps.ts [--unthrottled] --extra cam=...`, headed Chromium, dev clip playing, 2x MSAA, two 2048 shadow maps, half-res bloom, ~30k crowd cards, 15 s of `requestAnimationFrame` timestamps):

  | Camera | vsync on (120 Hz display) | vsync off |
  |---|---|---|
  | Broadcast | 112.3 fps mean, p50 8.3 ms, p95 16.6 ms, p99 16.9 ms | 134.4 fps, p50 7.0 ms, p95 14.0 ms, p99 14.6 ms |
  | Replay (DOF on, close to the players) | 95.1 fps mean, p50 8.4 ms, p95 16.7 ms, p99 17.5 ms | 111.8 fps, p50 8.3 ms, p95 16.8 ms, p99 26.4 ms |

  With vsync, frame times are whole refresh intervals (8.3 or 16.7 ms), so p95 at 16.7 ms means no frame took longer than two intervals. The biggest wins were the crowd as shader cards, the figurine body at about 40k triangles (replay p95 27 to 16.5 ms unthrottled), two shadow-casting banks instead of four, depth of field from the scene depth texture, and half-res bloom. Fixed-size renders (recording, screenshots) use 4x MSAA.
- **Labels** (`node scripts/check-labels.ts`, 676 frames, 8,664 objects): `K [R | t]` reprojection vs the Three.js camera, max 0.007 px; projected box vs GPU silhouette IoU mean 0.965, 5th percentile 0.940, min 0.880 (unoccluded objects at least 20 px tall); ball centre inside its box and ball-shadow pixel at or below the ball in 676 / 676 frames. All 676 label files validate with `FrameLabels.model_validate_json`.
- **Determinism** (Playwright `visual` project): two recordings in the same page, and recordings in two fresh pages, give identical frames and labels; after other playback in the same page the labels are identical (a few GPU edge pixels can differ).
- **Visual tests**: 5 baselines (broadcast x 3, tactical, replay). A 0.2 s time shift changes 7 % of pixels and fails the test; the tolerance is 1 %.
