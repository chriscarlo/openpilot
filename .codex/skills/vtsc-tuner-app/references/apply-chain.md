# Apply chain contract

Background-thread pipeline in `apply.rs`. The UI spawns `apply::Task::spawn(action, params, bands, repo_root, tune_file)` which returns immediately; the worker streams `ApplyEvent`s back via `mpsc::channel`. Each step emits a `Running` event at start and an `Ok` or `Err` event at end — both tagged with the same `id: u32` so the UI collapses them into one row.

## Steps (by id)

| id | Action | Plain-English progress text | Detail payload |
|---|---|---|---|
| 1 | `write_tune_json` | "Saving the tune file…" → "Saved the tune file" | `→ <path>` |
| 2 | `patch_source` | "Updating the openpilot source files…" → "Updated the openpilot source files" | newline-joined list of touched absolute paths |
| 3 | `git_commit` | "Committing the change to the local repo…" → "Committed the change to the local repo" | git stdout |
| 4 | `git_push` | "Pushing the current branch to the remote…" → "Pushed the current branch to the remote" | git stdout+stderr |
| 5 | `pick_tici_profile` | "Looking for the tici on the network…" → "Found the tici via `<profile>`" | empty on success |
| 6 | `ssh_pull_and_reboot` | "Pulling the new tune on the tici and rebooting it…" → "Pulled on the tici and sent the reboot signal" | git pull output + reboot hint |
| 7 | `validate_mapd_config` | "Reading the mapd config…" → "Mapd config looks good" | path + summary; **failure detail prints example contents** |
| 8 | `wait_for_tici_and_discover_regions` | "Waiting for the tici to come back…" → "Found N cached regions on the tici" | newline-joined region list |
| 9 | `tici_free_mb_check` | "Checking free space on the tici…" → "X MB free, need ~Y MB" | `df -BM` output |
| 10 | `build_mapd_with_earthly` | "Building mapd locally…" → "mapd binary ready" / "Skipping (binary already exists)" | earthly stdout (or skip note) |
| 100+i | `generate_one_region` | "Generating tile <lat,lon>…" → "Wrote tile <lat,lon>" | `mapd --generate` log + bytes written |
| 200+i | `rsync_region_to_tici` | "Pushing tile <lat,lon> to tici…" → "Tile <lat,lon> on tici" | rsync stats |
| 999 | `ssh_reboot_only` | "Rebooting the tici to load fresh tiles…" → "Reboot signal sent" | empty on success |

Steps 3–999 are conditional on `Action`:

```
Action::Local                  → steps 1, 2
Action::Commit                 → 1, 2, 3
Action::Push                   → 1, 2, 3, 4
Action::PullOnTici             → 1, 2, 3, 4, 5, 6
Action::RebuildTilesAndReboot  → 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100..N, 200..N, 999
```

If step 2 fails, the chain aborts (can't commit a broken patch). If step 3 has nothing to commit (source was already in sync), the id-3 row reports "Nothing to commit — tune is already on this branch" with `Ok` status and the chain continues. Any other failure aborts.

**Cancellation** (RebuildTilesAndReboot only) is honoured at region boundaries — the worker checks `Task::is_cancel_requested()` between regions. Mid-`mapd --generate` is opaque, so an in-flight region runs to completion. The cancel button only renders for this action while running.

**Two-reboot ordering.** Step 6 reboots after `git pull` lands the new openpilot Python (so `vision_turn_controller.py` with new constants is live before mapd publishes baked params with the matching hash). Step 999 reboots after rsync because mapd loads tiles only at startup — no inotify, no polling. Skipping the step-6 reboot risks the device briefly publishing baked speeds the openpilot binary can't consume safely.

## Source patch contract

`patch_source(repo_root, params, bands)` returns `Result<Vec<PathBuf>, String>` — the list of files actually changed. An empty vec means "no rewrite needed" (source already matched); that's still `Ok`. This feeds `git add` in step 3, which is skipped when the list is empty.

Touched files (when changed):
- `<repo>/sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`
- `<repo>/sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py`

`patch_physics` formats constants:
- A, B, C, D → `{:.6}`
- MIN_LAT, MAX_LAT → `{:.4}`

`patch_q_curve` writes:
- `Q_CURVE_ENABLED = True` iff there are any enabled bands.
- `Q_CURVE_POINTS: list[tuple[float, float]] = [` followed by 64 entries of `  (<kappa:.6e>, <q:.4>),` and a closing `]`, one-per-line.
- If bands are empty, writes `Q_CURVE_POINTS: list[tuple[float, float]] = []\n`.

Multi-line existing lists are detected by scanning forward from the `Q_CURVE_POINTS` line until a line ending with `]` — those lines are swallowed.

## Git contract

Working directory: `repo_root` (detected via `detect_repo_root` — walks up from the binary location looking for `.git/`).

Commands:
- `git add <relative_path>` (for each touched file)
- `git commit -m "vtsc: tune sigmoid via vtsc_tuner"`
- `git push` (no args — pushes current branch to its configured upstream)

All three failure-paths surface stdout+stderr to the UI's detail panel. The commit message is hard-coded for now; if you add a user-editable note field, pass it through instead.

## SSH contract

### Detection order (`pick_tici_profile`)

1. `apply::detect_ssid()` — tries `iwgetid -r` first, then PowerShell `(Get-NetConnectionProfile | Where IPv4Connectivity=Internet | First 1).Name` as a WSL fallback.
2. `profile_order_for_ssid(&ssid)`:
   - SSID contains "comma" (case-insensitive) → `[commaCar, commaHome, commaAdb]`
   - else → `[commaHome, commaCar, commaAdb]`
3. `probe_profile(name)` each: `ssh -o ConnectTimeout=2 -o BatchMode=yes -o StrictHostKeyChecking=accept-new <name> true`. First zero-exit wins.

### Remote execution (`ssh_pull_and_reboot`)

Two calls:
1. `ssh <name> "cd /data/openpilot && git pull"` with `ConnectTimeout=10`. Non-zero exit returns Err with stdout+stderr.
2. `ssh <name> "sudo reboot"` — fire-and-forget. Connection dies mid-command (that's fine). `.output()` result ignored, 200 ms sleep so the signal is definitely sent before the function returns.

## Caveats the user has to know

- **Branch state (as of 2026-04-19)**: dev machine and tici both track `chauffeur-exp01`. The branch-mismatch warning the modal still shows is now mostly defensive — the actual mismatch only matters if a future agent repoints either side. Verify with `ssh commaCar 'cd /data/openpilot && git branch --show-current'` rather than trusting any static doc claim.
- **Destructive reboot**: SIGTERM-style shutdown of the driver stack. Don't run mid-drive — the user must be parked. Step 6 fires one reboot, step 999 fires a second.
- **SSH first-connection**: `StrictHostKeyChecking=accept-new` allows unknown hosts but refuses changed ones. If the host key changes (e.g. a reimaged tici), `ssh-keygen -R <host>` and re-probe.
- **PBF prep is NOT in the apply chain** — it's a one-time setup the user does before their first `RebuildTilesAndReboot`. Step 7 only validates that `pbf_path` and `mapd_repo_path` exist; it can't tell whether the PBF has been processed with `osmium add-locations-to-ways`. A bad PBF produces 26+ tiles each ~57 bytes (zero ways) and the chain reports `Ok` for everything. See SKILL.md Landmines for the symptom + recovery.
- **Step 10 skips the build if `mapd_binary_path` exists.** This is intentional — `earthly` may not be installed. The user is expected to either (a) install earthly, or (b) keep a pre-built `build/mapd_amd64` (cross-compile via `docker run --platform=linux/amd64 golang:1.24-alpine3.21 sh -c 'go mod download && CGO_ENABLED=0 go build …'`). The skip-message in step 10's detail panel makes the gate visible.
- **The on-device binary is independent.** `RebuildTilesAndReboot` doesn't touch the tici's `third_party/mapd/mapd`; that's installed by `mapd_installer.py` on boot from the chriscarlo/mapd release URL. If the on-device binary is missing post-reboot but `MapdVersion` says it's installed, you've hit the installer silent-failure bug — recovery is `scp mapd_repo/openpilot-mapd/build/mapd commaCar:/data/openpilot/third_party/mapd/mapd && ssh commaCar sudo reboot`.

## Real-world timing (2026-04-19 first end-to-end)

For a single user, single dev box (WSL2 amd64, ~1 Gbps home Wi-Fi → ~5 MB/s car-hotspot SSH):

| Phase | Duration |
|---|---|
| Step 6 (ssh pull + reboot) | ~75 s (network pull + boot) |
| Step 7-9 (config + region + df) | <2 s |
| Step 10 (mapd build, skipped because binary cached) | <0.5 s |
| Steps 100..N+135 (bake all 36 regions, 1638 tiles, 483 MB) | ~36 s |
| Steps 200..N+135 (rsync 448 MB to tici over hotspot) | ~82 s |
| Step 999 (final reboot + mapd reload) | ~75 s |
| **Total wall-clock** | **~4.5 minutes** |

If the PBF must also be re-prepped (geofabrik refresh): add ~1 min for the docker `osmium tags-filter` + `add-locations-to-ways` pass on a 1.3 GB PBF.

## Adding a step

1. Pick an unused id (current max is 6).
2. Add a `start(tx, id, "Gerund phrase…")` + `end_ok` / `end_err` pair in `run_chain`, gated by the action kind.
3. The UI renders automatically — no changes needed in `app::show_apply_modal`.

Keep the text style consistent:
- Running: present participle (`"Verbing the thing…"`)
- Ok: past tense (`"Verbed the thing"`)
- Err: failure phrase (`"Couldn't verb the thing"` / `"Thing failed"`)
