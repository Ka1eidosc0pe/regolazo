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
