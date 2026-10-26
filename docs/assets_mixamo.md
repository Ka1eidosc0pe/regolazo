# Character & animation assets (Mixamo)

Mixamo (https://www.mixamo.com, free with an Adobe account) provides the base athlete and animation clips.

## Licensing rule for this repo
- Raw downloads go in `assets/third_party/mixamo/` — this folder is **gitignored** and never committed (Mixamo files may not be redistributed as standalone assets).
- The build pipeline converts them into optimized in-app assets (GLB with merged clips) used by the renderer; only those processed game assets are shipped with the app. README must credit Mixamo.

## Download settings
- **Character:** FBX Binary (.fbx), **With Skin**, T-pose. Save as `character_ybot.fbx`.
- **Animations:** FBX Binary (.fbx), **Without Skin**, **30 fps**, keyframe reduction **none**. Tick **In Place** whenever the option appears (locomotion clips).
- File name = the clip name in snake_case, e.g. `soccer_pass.fbx`.

## What to download
Mixamo's exact clip names vary; search the term and pick the best-looking match. Multiple variants are welcome (they add variety).

**Character**
- [ ] **Y Bot** — smooth, faceless mannequin; fits the stylized-broadcast look (kits are applied by our shaders).

**Locomotion** (search: idle, jog, run, sprint, strafe, turn)
- [ ] Soccer idle / breathing idle
- [ ] Jog forward
- [ ] Run forward
- [ ] Fast run / sprint
- [ ] Jog backward
- [ ] Jog strafe left, jog strafe right
- [ ] Running turn (180)

**On-ball** (search: soccer, kick, header, pass)
- [ ] Soccer pass (short pass)
- [ ] Long pass / soccer kick
- [ ] Shot: penalty kick / strike forward / soccer kick (power shot)
- [ ] Soccer header (standing) and jumping header
- [ ] Soccer trap / receive (first touch)
- [ ] Soccer dribble (if available)
- [ ] Throw in

**Defending & contact** (search: tackle, slide, trip, fall, get up)
- [ ] Soccer tackle
- [ ] Slide tackle
- [ ] Soccer trip / falling forward
- [ ] Getting up

**Goalkeeper** (search: goalkeeper)
- [ ] Goalkeeper idle
- [ ] Goalkeeper diving save (left and right, if both exist; otherwise one is mirrored)
- [ ] Goalkeeper catch
- [ ] Goalkeeper body block
- [ ] Goalkeeper drop kick
- [ ] Goalkeeper overhand throw
- [ ] Goalkeeper scoop

**Emotion** (search: celebration, cheer, victory, disappointed)
- [ ] 2–3 celebrations (e.g. victory, cheering, fist pump)
- [ ] Disappointed / defeated reaction

When done, put everything in `assets/third_party/mixamo/` in this project.

## Inventory (downloaded and verified 2026-10-26)
37 FBX files in `assets/third_party/mixamo/`. Verified by parsing each with Three.js `FBXLoader` (r170): all load, all clips are 30 fps, animation files contain no meshes, and every animated bone name (`mixamorig*`) exists in the character rig (65 bones), so clips apply to the character directly with no retargeting.

| Group | Files |
|---|---|
| Character | `character_ybot.fbx` (2 meshes: body + joints) |
| Locomotion | `Breathing_Idle`, `Jog_Forward`, `Jog_Strafe_Left`, `Jog_Strafe_Right`, `Run_forward`, `Sprint`, `Running_Backward`, `Running_Turn_180` |
| On-ball | `Soccer_Pass`, `long_pass`, `Strike_Foward` (shot), `standing_Header`, `jumping_Header`, `Receive_ground_pass`, `Receive_aerial_pass`, `Dribble`, `Throw_In` |
| Defending / contact | `Soccer_Tackle`, `Slide_Tackle`, `Soccer_Trip`, `Getting_Up` |
| Goalkeeper | `Goalkeeper_Idle`, `Goalkeeper_Diving_Save_left`, `Goalkeeper_Diving_Save_Right`, `Goalkeeper_Body_Block_left`, `Goalkeeper_Body_Block_right`, `Goalkeeper_Catch_highshot`, `Goalkeeper_Catch_lowshot`, `Goalkeeper_Drop_Kick`, `Goalkeeper_Overhand_Throw`, `Goalkeeper_Scoop` |
| Emotion | `celebration_Cheering`, `celebration_Fist_Pump`, `celebration_Victory`, `Disappointed` |

**Notes for the renderer:**
- **Root motion:** the looping locomotion clips are in place, except **`Sprint`** (~3.2 m hip drift per loop). Many one-shot clips also carry root motion (e.g. `long_pass` ~5 m, `Dribble` ~4.6 m, `Slide_Tackle` ~4.4 m, GK dives ~2.8 m). The simulation owns player positions, so the asset pipeline must strip horizontal (XZ) hip translation from every clip. Keep vertical (Y) motion for dives, jumps and slides. Optionally keep the extracted root path to sync a clip's timing with the sim's movement.
- **Coverage gaps:** there is a single shot clip and no dedicated chip, volley, curler or cross clip. Derive these from `Strike_Foward`, `long_pass` and `Soccer_Pass` by varying playback speed, blending, and additive leg/upper-body adjustments. Use the left/right GK pairs directly and mirror other clips where a side-specific version is needed.
- Kit colors, numbers and names are applied by our own shaders or materials on the Y Bot body mesh. The Y Bot's original material is not used as-is.
