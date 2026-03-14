---
name: rti-tuner
description: >
  Debug, tune, and validate Realtime Traffic Intelligence (RTI) across the
  Waze/OpenWebNinja daemon, threat matching and deduping, planner slowdown
  path, and RTI HUD/settings UI. Use when RTI is offline, missing threats,
  choosing the wrong road or direction, slowing too early or too late, ignoring
  posted-speed handoff, publishing wrong `rtiStateSP` fields, or rendering RTI
  alerts incorrectly in the UI. Triggers include `rtid`, `ThreatDetector`,
  `RTIController`, `rtiStateSP`, `RTIForwardSlowdownRange`, `RTIDecelRate`,
  `RTIHUDEnabled`, RTI duplicate-collapse radii, and RTI threat arrows/cards.
---

# RTI Tuner

## Guardrails

- Treat branch code and tests as ground truth. `docs/chauffeur/rti/` has useful captures and plans, but some planning notes are stale relative to the current branch.
- Do not try to "fix RTI" in Qt if `rtiStateSP` semantics are already wrong upstream.
- Do not casually increase RTI API polling. `rtid` intentionally fetches on a 30 second cadence and relies on caching to stay within provider quota.
- Keep secrets and raw captures out of git. `RTIManualApiKey` is sensitive, and ad hoc captures belong under `.cache/` or device-local paths.
- For offroad RTI settings work, prefer `*SP` widgets from `selfdrive/ui/sunnypilot/qt/widgets/controls.h` and follow `docs/chauffeur/ui/bsg/offroad/offroad_settings_bsg.md`.

## Quick Loop

- Daemon / API / publish path:
  edit `sunnypilot/rtid/rtid.py`, `sunnypilot/rtid/waze_api_client.py`, and `sunnypilot/rtid/api_key_manager.py`; verify with `sunnypilot/rtid/tests/test_rtid.py` before touching planner or UI code.
- Matching / direction / dedupe path:
  edit `sunnypilot/rtid/threat_detector.py` plus `sunnypilot/rtid/street_name_matcher.py` when needed; verify with `sunnypilot/rtid/tests/test_threat_detector.py` and `sunnypilot/rtid/tests/test_integration_flow.py`.
- Planner slowdown / ramp / resume path:
  edit `sunnypilot/selfdrive/controls/lib/rti_controller.py` and `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`; verify first with `sunnypilot/selfdrive/controls/lib/tests/rti/test_rti_rampdown_unit.py`.
- HUD / offroad settings path:
  edit `selfdrive/ui/sunnypilot/qt/onroad/hud.cc`, `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_control.cc`, and `selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.cc`; do a targeted `scons` build before broader tests.

## Current RTI Facts (Code-Verified)

- `rtiStateSP` is the contract between RTID, the planner path, and the RTI HUD.
- `rtid` fetches traffic data on a 30 second interval and treats cached data older than 300 seconds as stale.
- `ThreatDetector` sorts processed threats by distance and only publishes the closest five for HUD consumption.
- Planner-side RTI in `posted` mode prefers the posted speed limit already selected by SLC (`self.slc.speed_limit`) before falling back to threat-carried limits.
- RTID explicitly handles the case where neither map nor dashboard speed-limit input has produced a usable posted speed yet; missing posted speed at a snapshot is not by itself a fault.
- `ThreatDetector` and `SpeedRecommendationEngine` still cache many params in `__init__`, and `RTIController` also caches most speed and distance params on init. `RTIEnabled` is checked live, and RTI threat filtering gets an extra live read in the controller. Do not assume a param is live-tuneable without checking its actual read site.
- Current RTI offroad range controls display miles and store meters; custom speed reduction displays mph and stores km/h. Do not assume older metric or imperial plans or broad historical tests match the current panel behavior.

## Workflow Decision Tree

- No alerts or `apiStatus=offline`:
  start at `sunnypilot/rtid/rtid.py`, `api_key_manager.py`, and `waze_api_client.py`; verify the API key source and the 30 second fetch gate before changing any threat logic.
- No alerts or no posted speed limit yet, but `apiStatus` is healthy:
  do not assume a bug from one snapshot. Threat presence and posted speed can both legitimately be absent until the provider side, dashboard TSR, or OSM path has actually produced data; during live monitoring, wait through at least one RTID fetch cycle before escalating.
- `apiStatus=connected` but `rtiStateSP.threats` stays empty:
  confirm the current `RTIThreatFilter` and the raw provider alert types before debugging same-road or planner logic; a police-only filter with provider-side `jam` alerts legitimately publishes nothing.
- Threat appears on HUD but the car does not slow:
  inspect `recommendedSpeed`, `speedLimitMs`, `isCausingRecommendation`, `onSameRoad`, and `direction`; visual-only alerts with no usable speed limit are expected in some cases.
- Threat is visible and carries a usable posted speed, but `onSameRoad=false` and `isCausingRecommendation=false`:
  inspect `logMessage` for `RTI street mismatch` plus the current map road name before blaming planner windows. Generic map names like `Route 50` vs Waze `US-50 W`, or placeholder road names like `'None'`, can block same-road matching upstream so RTI never enters slowdown control.
- Slowdown happens for the wrong alert or the wrong road:
  debug `ThreatDetector` and `RoadMatcher` first; do not paper over a same-road or direction bug in the HUD.
- Duplicate police or hazard pins clutter the HUD:
  tune duplicate-collapse behavior in `ThreatDetector`; the HUD mostly renders the ranked threats it receives.
- A param change seems ignored:
  verify whether the param is live-read or init-cached and whether `rtid` or `plannerd` must be restarted before concluding the tuning has no effect.
- UI looks wrong but message content is already wrong:
  fix backend semantics first, then re-check Qt rendering.

## Recommended Verification Commands

Use exact commands rather than vague "run RTI tests."

### Daemon, schema, and threat-processing checks

```bash
.venv/bin/python -m pytest -o addopts='' \
  sunnypilot/rtid/tests/test_rtid.py -q
```

```bash
.venv/bin/python -m pytest -o addopts='' \
  sunnypilot/rtid/tests/test_threat_detector.py -q
```

```bash
.venv/bin/python -m pytest -o addopts='' \
  sunnypilot/rtid/tests/test_schema_synchronization.py -q
```

```bash
.venv/bin/python -m pytest -o addopts='' \
  sunnypilot/rtid/tests/test_integration_flow.py -q
```

### Planner-side rampdown and slowdown behavior

```bash
.venv/bin/python -m pytest --confcutdir=sunnypilot/selfdrive/controls/lib/tests -q \
  sunnypilot/selfdrive/controls/lib/tests/rti/test_rti_rampdown_unit.py
```

Use `selfdrive/test/test_rti_integration.py` as a broader smoke check, not as the sole source of truth for current RTI UI semantics:

```bash
.venv/bin/python -m pytest -o addopts='' \
  selfdrive/test/test_rti_integration.py -q
```

### RTI UI and HUD build checks

```bash
scons -j$(nproc) selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/rti_settings_panel.o
```

```bash
scons -j$(nproc) selfdrive/ui/sunnypilot/qt/onroad/hud.o
```

### Live inspection helpers

```bash
python3 sunnypilot/rtid/monitor_rtid.py
```

```bash
python3 docs/chauffeur/rti/tests/check_rti_content.py
```

## Diagnostic Heuristics

- If the HUD is empty, inspect `rtiStateSP` before editing Qt.
- If posted-mode slowdown feels wrong, inspect the SLC handoff into `RTIController.update(... posted_speed_limit=self.slc.speed_limit)` before retuning threat logic.
- If posted speed or threat presence is missing at the instant you inspect logs or a live session, first treat that as "not acquired yet" rather than "broken"; if you are watching in real time, give dashboard/OSM acquisition and the next 30 second RTID fetch a chance to populate.
- If raw API access is healthy but RTI stays empty, check whether the current `RTIThreatFilter` excludes the provider payload (for example police-only while the area only has `jam` alerts) before chasing backend matching bugs.
- If `threat.speedLimitMs > 0` but `threatAhead` never goes true and `recommendedSpeed` stays `0`, verify whether `onSameRoad` is being rejected by street-name matching before retuning slowdown or resume distances.
- If a threat is rendered but `recommendedSpeed == 0`, confirm whether RTI intentionally classified it as visual-only.
- If side-street alerts still slow the car, debug street matching and heading gating before changing the duplicate-collapse radii or HUD sorting.
- Do not assume `RTIDataSource` or `RTIAggressiveness` are active runtime levers in the current branch without code search; they are easy to over-credit from legacy tests and UI surface area.

## References

- Read `references/pipeline.md` when you need the end-to-end RTI ownership map, key file paths, or a compact list of branch facts that are easy to forget.

## Skill Maintenance

After any real RTI debug or tuning session:
- compare new evidence against existing guidance before finishing
- correct stale bullets instead of appending contradictory notes
- keep exact file paths, params, and verification commands current
- prefer one clear proven failure mode and one reliable validation command over speculative lists
- add new gotchas only for real, repeatable footguns
- if a session teaches a reusable RTI heuristic, add it only when it is durable and locally verifiable
- keep only durable workflow guidance in `SKILL.md`; move detailed examples or branch-specific notes to `references/` or `.cache/`
- update `agents/openai.yaml` only if the skill’s scope or trigger wording changed materially

Treat this skill as a recursive kaizen loop:
- every real invocation should leave it more accurate, more actionable, more compact, or all three
- recursively self-improve by reconciling new facts with old guidance in the same pass so the next invocation starts smarter
- if new evidence proves an older bullet wrong, incomplete, redundant, or noisy, replace, tighten, or delete it instead of stacking another warning
- prefer editing and pruning over adding line after line; a shorter, sharper skill beats a longer noisier one
- when a debugging session teaches a durable pattern, capture the minimum proven lesson, not the full story
