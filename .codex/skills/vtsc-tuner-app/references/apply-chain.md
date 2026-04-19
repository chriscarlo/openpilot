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

Steps 3–6 are conditional on `Action`:

```
Action::Local       → steps 1, 2
Action::Commit      → 1, 2, 3
Action::Push        → 1, 2, 3, 4
Action::PullOnTici  → 1, 2, 3, 4, 5, 6
```

If step 2 fails, the chain aborts (can't commit a broken patch). If step 3 has nothing to commit (source was already in sync), the id-3 row reports "Nothing to commit — tune is already on this branch" with `Ok` status and the chain continues. Any other failure aborts.

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

- **Branch mismatch**: the dev machine typically runs `chauffeur-exp01`; the device tracks `chauffeur-dev4`. `git pull` on the tici pulls whatever it tracks — if it's a different branch than what was pushed, the new tune won't be seen. The confirmation modal warns about this for Push / PullOnTici actions.
- **Destructive reboot**: SIGTERM-style shutdown of the driver stack. Don't run mid-drive.
- **SSH first-connection**: `StrictHostKeyChecking=accept-new` allows unknown hosts but refuses changed ones. If the host key changes (e.g. a reimaged tici), `ssh-keygen -R <host>` and re-probe.

## Adding a step

1. Pick an unused id (current max is 6).
2. Add a `start(tx, id, "Gerund phrase…")` + `end_ok` / `end_err` pair in `run_chain`, gated by the action kind.
3. The UI renders automatically — no changes needed in `app::show_apply_modal`.

Keep the text style consistent:
- Running: present participle (`"Verbing the thing…"`)
- Ok: past tense (`"Verbed the thing"`)
- Err: failure phrase (`"Couldn't verb the thing"` / `"Thing failed"`)
