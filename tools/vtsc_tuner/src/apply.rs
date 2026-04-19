//! Apply pipeline: write tune file, patch repo source, optionally commit /
//! push / SSH the device.  Runs on a background thread and streams progress
//! events back to the UI through an `mpsc::Receiver`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::{Duration, Instant};

use crate::mapd_config::{MapdConfig, RegionBox};
use crate::sigmoid::{Band, SigmoidParams, bands_as_q_curve_points};

// ---------------------------------------------------------------------------
// Action kinds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
  /// Write tune JSON + patch source files. No git, no SSH.
  Local,
  /// Local + git add + git commit on the current branch.
  Commit,
  /// Commit + git push (current branch → origin).
  Push,
  /// Push + ssh <tici-profile> "cd /data/openpilot && git pull && sudo reboot".
  PullOnTici,
  /// PullOnTici + regenerate sigmoid-baked map tiles locally and rsync them
  /// to the tici (skipping any public CDN), then a final reboot. See
  /// `mapd_config.rs` for the required `~/.config/vtsc_tuner/mapd.json`.
  RebuildTilesAndReboot,
}

impl Action {
  pub const ALL: [Self; 5] = [
    Self::Local,
    Self::Commit,
    Self::Push,
    Self::PullOnTici,
    Self::RebuildTilesAndReboot,
  ];

  pub fn label(&self) -> &'static str {
    match self {
      Self::Local => "Apply locally",
      Self::Commit => "Apply + commit",
      Self::Push => "Apply + commit + push",
      Self::PullOnTici => "Apply + commit + push + pull on tici",
      Self::RebuildTilesAndReboot => "Apply + push + pull + rebuild map tiles",
    }
  }

  pub fn description(&self) -> &'static str {
    match self {
      Self::Local => {
        "Save the tune file and rewrite the PHYSICS_* constants in \
         vision_turn_controller.py (and Q_CURVE_POINTS in vtsc_curve_tuning.py \
         if any bands are set).  No git activity."
      }
      Self::Commit => {
        "Everything in 'Apply locally', plus a single git commit on the \
         current branch."
      }
      Self::Push => {
        "Apply + commit, then `git push` the current branch to origin."
      }
      Self::PullOnTici => {
        "Apply + commit + push, then SSH the tici (auto-detected by \
         reachable profile) and run `git pull && sudo reboot`."
      }
      Self::RebuildTilesAndReboot => {
        "Apply + push + pull + reboot, then locally regenerate sigmoid-baked \
         map tiles for whatever 2°×2° regions the tici has cached, rsync them \
         over SSH, and reboot the tici a second time so mapd reloads the new \
         tiles. Requires ~/.config/vtsc_tuner/mapd.json with pbf_path and \
         mapd_repo_path."
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Events streamed from the worker thread to the UI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
  Running,
  Ok,
  Err,
}

#[derive(Debug, Clone)]
pub struct StepEvent {
  /// Stable id so start/finish events for the same step collapse into one row.
  pub id: u32,
  pub status: StepStatus,
  /// Plain-English line shown to the user.  Phrased for the current status:
  ///   Running: "Saving the tune file…"
  ///   Ok:      "Saved the tune file"
  ///   Err:     "Couldn't save the tune file"
  pub text: String,
  /// Raw command output (git stdout/stderr, SSH output, paths touched).
  /// Empty while running.
  pub detail: String,
}

#[derive(Debug, Clone)]
pub enum ApplyEvent {
  Step(StepEvent),
  /// Final event — `ok` is the AND of all step results.
  Done { ok: bool },
}

// ---------------------------------------------------------------------------
// Public task handle held by the UI
// ---------------------------------------------------------------------------

pub struct Task {
  pub action: Action,
  pub events: Vec<StepEvent>,
  pub running: bool,
  pub success: Option<bool>,
  rx: Receiver<ApplyEvent>,
  cancel: Arc<AtomicBool>,
}

impl Task {
  pub fn spawn(
    action: Action,
    params: SigmoidParams,
    bands: Vec<Band>,
    repo_root: PathBuf,
    tune_file: PathBuf,
  ) -> Self {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_worker = cancel.clone();
    thread::spawn(move || {
      run_chain(action, &params, &bands, &repo_root, &tune_file, &tx, &cancel_worker);
    });
    Self {
      action,
      events: Vec::new(),
      running: true,
      success: None,
      rx,
      cancel,
    }
  }

  /// Signal the worker to stop at the next region boundary. The current
  /// `mapd --generate` invocation can't be interrupted mid-run, so cancel
  /// takes effect after the in-flight region completes (or fails).
  pub fn request_cancel(&self) {
    self.cancel.store(true, Ordering::Relaxed);
  }

  pub fn is_cancel_requested(&self) -> bool {
    self.cancel.load(Ordering::Relaxed)
  }

  /// Drain pending events into `self.events`.  Call once per UI frame.
  /// Start / finish events for the same step id collapse into a single row —
  /// the latest status / text / detail win.
  pub fn poll(&mut self) {
    while let Ok(ev) = self.rx.try_recv() {
      match ev {
        ApplyEvent::Step(s) => {
          if let Some(existing) = self.events.iter_mut().find(|e| e.id == s.id) {
            *existing = s;
          } else {
            self.events.push(s);
          }
        }
        ApplyEvent::Done { ok } => {
          self.running = false;
          self.success = Some(ok);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Worker — orchestrates each step and decides when to stop on failure
// ---------------------------------------------------------------------------

fn run_chain(
  action: Action,
  params: &SigmoidParams,
  bands: &[Band],
  repo_root: &Path,
  tune_file: &Path,
  tx: &Sender<ApplyEvent>,
  cancel: &AtomicBool,
) {
  let mut all_ok = true;

  // 1) Write tune JSON.
  start(tx, 1, "Saving the tune file…");
  let r = write_tune_json(params, bands, tune_file);
  let tune_note = format!("→ {}", tune_file.display());
  match r {
    Ok(()) => end_ok(tx, 1, "Saved the tune file", &tune_note),
    Err(e) => {
      end_err(tx, 1, "Couldn't save the tune file", &e);
      all_ok = false;
    }
  }

  // 2) Patch openpilot source files with the new sigmoid + Q-curve.
  start(tx, 2, "Updating the openpilot source files…");
  let patch_result = patch_source(repo_root, params, bands);
  let touched = patch_result
    .as_ref()
    .map(|paths| paths.clone())
    .unwrap_or_default();
  match &patch_result {
    Ok(paths) if paths.is_empty() => end_ok(
      tx,
      2,
      "Source already matched the tune — nothing to rewrite",
      "",
    ),
    Ok(paths) => {
      let detail = paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n");
      end_ok(tx, 2, "Updated the openpilot source files", &detail);
    }
    Err(e) => {
      end_err(tx, 2, "Couldn't update the openpilot source files", e);
      finish(tx, false);
      return;
    }
  }

  if matches!(action, Action::Local) {
    finish(tx, all_ok);
    return;
  }

  // 3) Git commit.
  if touched.is_empty() {
    start(tx, 3, "Checking for changes to commit…");
    end_ok(tx, 3, "Nothing to commit — tune is already on this branch", "");
  } else {
    start(tx, 3, "Committing the change to the local repo…");
    let r = git_commit(repo_root, &touched);
    match r {
      Ok(detail) => end_ok(tx, 3, "Committed the change to the local repo", &detail),
      Err(e) => {
        end_err(tx, 3, "Couldn't commit the change", &e);
        finish(tx, false);
        return;
      }
    }
  }

  if matches!(action, Action::Commit) {
    finish(tx, all_ok);
    return;
  }

  // 4) Git push.
  start(tx, 4, "Pushing the current branch to the remote…");
  match git_push(repo_root) {
    Ok(detail) => end_ok(tx, 4, "Pushed the current branch to the remote", &detail),
    Err(e) => {
      end_err(tx, 4, "Couldn't push to the remote", &e);
      finish(tx, false);
      return;
    }
  }

  if matches!(action, Action::Push) {
    finish(tx, all_ok);
    return;
  }

  // 5) Find a reachable tici and pull on it.
  start(tx, 5, "Looking for the tici on the network…");
  let profile = match pick_tici_profile() {
    Some(p) => {
      end_ok(tx, 5, &format!("Found the tici via `{p}`"), "");
      p
    }
    None => {
      end_err(
        tx,
        5,
        "Couldn't reach the tici",
        "tried commaHome / commaCar / commaAdb — none responded within 2 s",
      );
      finish(tx, false);
      return;
    }
  };

  start(
    tx,
    6,
    "Pulling the new tune on the tici and rebooting it…",
  );
  match ssh_pull_and_reboot(profile) {
    Ok(detail) => end_ok(
      tx,
      6,
      "Pulled on the tici and sent the reboot signal",
      &detail,
    ),
    Err(e) => {
      end_err(tx, 6, "Tici pull / reboot failed", &e);
      all_ok = false;
    }
  }

  if matches!(action, Action::PullOnTici) {
    finish(tx, all_ok);
    return;
  }

  // Steps 7..M+1: rebuild sigmoid-baked map tiles + push them, second reboot.
  // The chain is intentionally a superset of PullOnTici because the openpilot
  // Python code that reads MapPreCurveSpeeds must land on the device BEFORE
  // mapd starts publishing the new param — that's what step 6's reboot ensures.
  if !all_ok {
    // Don't try to rebuild on a tici we couldn't reach for git pull. The
    // user can retry once the issue is fixed.
    finish(tx, all_ok);
    return;
  }

  if cancel.load(Ordering::Relaxed) {
    end_err(tx, 7, "Cancelled before rebuild started", "");
    finish(tx, false);
    return;
  }

  // 7) Validate the local mapd config.
  start(tx, 7, "Reading the mapd rebuild config…");
  let mapd_cfg = match MapdConfig::load() {
    Ok(c) => c,
    Err(e) => {
      end_err(
        tx,
        7,
        "Couldn't load ~/.config/vtsc_tuner/mapd.json",
        &format!("{e}\n\nExample contents:\n{{\n  \"pbf_path\": \"/path/to/region.osm.pbf\",\n  \"mapd_repo_path\": \"{}\"\n}}",
          repo_root.join("mapd_repo/openpilot-mapd").display()),
      );
      finish(tx, false);
      return;
    }
  };
  if let Err(e) = mapd_cfg.validate() {
    end_err(tx, 7, "mapd config has unusable paths", &e);
    finish(tx, false);
    return;
  }
  end_ok(
    tx,
    7,
    "Read the mapd rebuild config",
    &format!(
      "pbf={}\nmapd_repo={}\nbinary={}",
      mapd_cfg.pbf_path.display(),
      mapd_cfg.mapd_repo_path.display(),
      mapd_cfg.binary().display()
    ),
  );

  // 8) Wait for tici to come back online after step 6's reboot, then
  //    discover which 2°×2° regions it has cached (or use the override).
  start(tx, 8, "Waiting for the tici to come back online…");
  if !wait_for_tici(profile, Duration::from_secs(120), cancel) {
    end_err(
      tx,
      8,
      if cancel.load(Ordering::Relaxed) {
        "Cancelled while waiting for the tici"
      } else {
        "Tici did not come back within 120 s"
      },
      "verify the tici rebooted and is on the same SSH profile",
    );
    finish(tx, false);
    return;
  }
  let regions = match mapd_cfg.regions_override.clone() {
    Some(rs) if !rs.is_empty() => {
      end_ok(
        tx,
        8,
        &format!("Using {} region(s) from regions_override", rs.len()),
        &format_regions_for_detail(&rs),
      );
      rs
    }
    _ => match discover_tici_regions(profile) {
      Ok(rs) if rs.is_empty() => {
        end_err(
          tx,
          8,
          "Tici has no cached map regions to rebake",
          "/data/media/0/osm/offline/ is empty — download a region in the offroad UI first, then retry",
        );
        finish(tx, false);
        return;
      }
      Ok(rs) => {
        end_ok(
          tx,
          8,
          &format!("Found {} cached region(s) on the tici", rs.len()),
          &format_regions_for_detail(&rs),
        );
        rs
      }
      Err(e) => {
        end_err(tx, 8, "Couldn't discover tici regions", &e);
        finish(tx, false);
        return;
      }
    },
  };

  // 9) Pre-flight free-space check on /data/media/0.
  start(tx, 9, "Checking free disk on the tici…");
  let est_mb_per_region: u64 = 50;
  let est_mb = (regions.len() as u64) * est_mb_per_region * 3 / 2; // 1.5× headroom
  match tici_free_mb(profile) {
    Ok(free_mb) if free_mb < est_mb => {
      end_err(
        tx,
        9,
        "Not enough free space on /data/media/0",
        &format!("estimated need ≈ {est_mb} MB; tici has {free_mb} MB free"),
      );
      finish(tx, false);
      return;
    }
    Ok(free_mb) => end_ok(
      tx,
      9,
      &format!("Tici has {free_mb} MB free (need ≈ {est_mb} MB)"),
      "",
    ),
    Err(e) => {
      end_err(tx, 9, "Couldn't read tici disk usage", &e);
      finish(tx, false);
      return;
    }
  }

  if cancel.load(Ordering::Relaxed) {
    end_err(tx, 10, "Cancelled before mapd build", "");
    finish(tx, false);
    return;
  }

  // 10) Build the mapd binary if missing. We don't try to upgrade an
  //     existing binary — assume the user manages that separately so we
  //     don't accidentally rebuild every run.
  start(tx, 10, "Checking for the local mapd binary…");
  let mapd_bin = mapd_cfg.binary();
  if mapd_bin.exists() {
    end_ok(
      tx,
      10,
      &format!("Local mapd binary exists at {}", mapd_bin.display()),
      "",
    );
  } else {
    match build_mapd_with_earthly(&mapd_cfg.mapd_repo_path) {
      Ok(detail) => end_ok(
        tx,
        10,
        &format!("Built mapd binary at {}", mapd_bin.display()),
        &detail,
      ),
      Err(e) => {
        end_err(tx, 10, "Could not build mapd binary", &e);
        finish(tx, false);
        return;
      }
    }
  }

  // 11..N) Generate one region per step into a staging dir.
  let staging = mapd_cfg.mapd_repo_path.join("osm/offline.staging");
  let _ = std::fs::remove_dir_all(&staging); // best-effort cleanup
  for (i, region) in regions.iter().enumerate() {
    if cancel.load(Ordering::Relaxed) {
      end_err(tx, 100 + i as u32, "Cancelled before region gen", "");
      finish(tx, false);
      return;
    }
    let id = 100 + i as u32;
    start(
      tx,
      id,
      &format!(
        "Generating tiles for {}/{} ({}°,{}°)…",
        i + 1,
        regions.len(),
        region.min_lat,
        region.min_lon
      ),
    );
    match generate_one_region(&mapd_cfg, &mapd_bin, region) {
      Ok(detail) => end_ok(
        tx,
        id,
        &format!("Generated tiles for ({}°,{}°)", region.min_lat, region.min_lon),
        &detail,
      ),
      Err(e) => {
        end_err(
          tx,
          id,
          &format!("Tile gen failed for ({}°,{}°)", region.min_lat, region.min_lon),
          &e,
        );
        finish(tx, false);
        return;
      }
    }
  }

  // N+1..M) rsync each region to the tici. Ordering matters: rsync ALL
  // regions before the final reboot so a partial push doesn't get loaded.
  for (i, region) in regions.iter().enumerate() {
    if cancel.load(Ordering::Relaxed) {
      end_err(tx, 200 + i as u32, "Cancelled before rsync", "");
      finish(tx, false);
      return;
    }
    let id = 200 + i as u32;
    start(
      tx,
      id,
      &format!(
        "Pushing tiles for {}/{} ({}°,{}°) to the tici…",
        i + 1,
        regions.len(),
        region.min_lat,
        region.min_lon
      ),
    );
    match rsync_region_to_tici(&mapd_cfg, profile, region) {
      Ok(detail) => end_ok(
        tx,
        id,
        &format!("Pushed tiles for ({}°,{}°)", region.min_lat, region.min_lon),
        &detail,
      ),
      Err(e) => {
        end_err(
          tx,
          id,
          &format!("rsync failed for ({}°,{}°)", region.min_lat, region.min_lon),
          &e,
        );
        finish(tx, false);
        return;
      }
    }
  }

  // M+1) Final reboot so mapd reloads the new tile contents at startup.
  start(tx, 999, "Rebooting the tici a second time to reload mapd…");
  match ssh_reboot_only(profile) {
    Ok(_) => end_ok(
      tx,
      999,
      "Sent the second reboot signal",
      "tici will be back in ~60 s with new sigmoid-baked tiles",
    ),
    Err(e) => {
      end_err(tx, 999, "Final reboot failed", &e);
      all_ok = false;
    }
  }

  finish(tx, all_ok);
}

fn finish(tx: &Sender<ApplyEvent>, ok: bool) {
  let _ = tx.send(ApplyEvent::Done { ok });
}

fn start(tx: &Sender<ApplyEvent>, id: u32, text: &str) {
  let _ = tx.send(ApplyEvent::Step(StepEvent {
    id,
    status: StepStatus::Running,
    text: text.into(),
    detail: String::new(),
  }));
}

fn end_ok(tx: &Sender<ApplyEvent>, id: u32, text: &str, detail: &str) {
  let _ = tx.send(ApplyEvent::Step(StepEvent {
    id,
    status: StepStatus::Ok,
    text: text.into(),
    detail: detail.into(),
  }));
}

fn end_err(tx: &Sender<ApplyEvent>, id: u32, text: &str, detail: &str) {
  let _ = tx.send(ApplyEvent::Step(StepEvent {
    id,
    status: StepStatus::Err,
    text: text.into(),
    detail: detail.into(),
  }));
}

// ---------------------------------------------------------------------------
// Step implementations
// ---------------------------------------------------------------------------

fn write_tune_json(
  params: &SigmoidParams,
  bands: &[Band],
  path: &Path,
) -> Result<(), String> {
  let tune = crate::io::Tune {
    schema: 1,
    created: chrono::Local::now(),
    note: String::new(),
    params: *params,
    bands: bands.to_vec(),
  };
  tune.save_to(path).map_err(|e| e.to_string())
}

/// Returns the list of files actually changed.  If both files already had the
/// requested values, returns an empty vec — caller treats that as "nothing to
/// commit" rather than an error.
fn patch_source(
  repo_root: &Path,
  params: &SigmoidParams,
  bands: &[Band],
) -> Result<Vec<PathBuf>, String> {
  let mut changed = Vec::new();

  let phys = repo_root.join("sunnypilot/selfdrive/controls/lib/vision_turn_controller.py");
  if patch_physics(&phys, params)? {
    changed.push(phys);
  }

  let qcurve = repo_root.join("sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py");
  if patch_q_curve(&qcurve, params, bands)? {
    changed.push(qcurve);
  }

  Ok(changed)
}

fn patch_physics(path: &Path, params: &SigmoidParams) -> Result<bool, String> {
  let content = std::fs::read_to_string(path)
    .map_err(|e| format!("read {}: {}", path.display(), e))?;

  let mut updated = content.clone();
  updated = replace_constant(&updated, "PHYSICS_A", &format!("{:.6}", params.a));
  updated = replace_constant(&updated, "PHYSICS_B", &format!("{:.6}", params.b));
  updated = replace_constant(&updated, "PHYSICS_C", &format!("{:.6}", params.c));
  updated = replace_constant(&updated, "PHYSICS_D", &format!("{:.6}", params.d));
  updated = replace_constant(&updated, "PHYSICS_MIN_LAT_ACCEL", &format!("{:.4}", params.min_lat));
  updated = replace_constant(&updated, "PHYSICS_MAX_LAT_ACCEL", &format!("{:.4}", params.max_lat));

  if updated == content {
    return Ok(false);
  }
  std::fs::write(path, updated).map_err(|e| format!("write {}: {}", path.display(), e))?;
  Ok(true)
}

/// Replace the value on a line of the form `<NAME> = <number>` (only the first
/// match — these constants live at module scope and are unique).  Preserves
/// indentation and trailing comment if present.
fn replace_constant(text: &str, name: &str, value: &str) -> String {
  let mut out = String::with_capacity(text.len());
  let mut replaced = false;
  for line in text.split_inclusive('\n') {
    if !replaced {
      // Match an unindented `NAME = ...` line.
      if let Some(rest) = line.strip_prefix(name) {
        let trimmed = rest.trim_start();
        if let Some(rest2) = trimmed.strip_prefix('=') {
          let after_eq = rest2.trim_start();
          // Find the trailing newline / comment to preserve them.
          let (val_part, tail) = split_value_and_tail(after_eq);
          let _ = val_part; // discarded — replaced with `value`
          out.push_str(name);
          out.push_str(" = ");
          out.push_str(value);
          out.push_str(tail);
          replaced = true;
          continue;
        }
      }
    }
    out.push_str(line);
  }
  out
}

/// Split a value-bearing line tail into (value, trailing).  The trailing
/// portion includes any inline comment + the newline.
fn split_value_and_tail(s: &str) -> (&str, &str) {
  let mut idx = 0;
  for (i, c) in s.char_indices() {
    if c.is_whitespace() && c != ' ' && c != '\t' {
      idx = i;
      return (&s[..idx], &s[idx..]);
    }
    if c == '#' || c == '\n' {
      idx = i;
      return (&s[..idx], &s[idx..]);
    }
    idx = i + c.len_utf8();
  }
  (s, "")
}

fn patch_q_curve(
  path: &Path,
  params: &SigmoidParams,
  bands: &[Band],
) -> Result<bool, String> {
  let content = std::fs::read_to_string(path)
    .map_err(|e| format!("read {}: {}", path.display(), e))?;

  let enabled = !bands.is_empty();
  let points = if enabled {
    bands_as_q_curve_points(params, bands, 256)
  } else {
    Vec::new()
  };

  let mut updated = String::new();
  let mut in_points_block = false;
  let mut wrote_points = false;
  for line in content.split_inclusive('\n') {
    if in_points_block {
      // Skip until we see the closing ']' line.
      if line.trim_end().ends_with(']') {
        in_points_block = false;
      }
      continue;
    }
    if line.starts_with("Q_CURVE_ENABLED") {
      let val = if enabled { "True" } else { "False" };
      updated.push_str(&format!("Q_CURVE_ENABLED = {val}\n"));
      continue;
    }
    if line.starts_with("Q_CURVE_POINTS") {
      // Replace the entire (possibly multi-line) list literal.
      let header = "Q_CURVE_POINTS: list[tuple[float, float]] = [";
      updated.push_str(header);
      if points.is_empty() {
        updated.push_str("]\n");
      } else {
        updated.push('\n');
        for (k, q) in &points {
          updated.push_str(&format!("  ({:.6e}, {:.4}),\n", k, q));
        }
        updated.push_str("]\n");
      }
      wrote_points = true;
      // If the original list spans multiple lines we need to swallow them.
      if !line.trim_end().ends_with(']') {
        in_points_block = true;
      }
      continue;
    }
    updated.push_str(line);
  }
  if !wrote_points {
    return Err(format!(
      "could not find Q_CURVE_POINTS line in {}",
      path.display()
    ));
  }
  if updated == content {
    return Ok(false);
  }
  std::fs::write(path, updated).map_err(|e| format!("write {}: {}", path.display(), e))?;
  Ok(true)
}

// ---------------------------------------------------------------------------
// Git
// ---------------------------------------------------------------------------

fn git_commit(repo_root: &Path, paths: &[PathBuf]) -> Result<String, String> {
  for p in paths {
    let rel = p
      .strip_prefix(repo_root)
      .map(|p| p.to_path_buf())
      .unwrap_or_else(|_| p.clone());
    let out = Command::new("git")
      .current_dir(repo_root)
      .arg("add")
      .arg(&rel)
      .output()
      .map_err(|e| format!("git add: {e}"))?;
    if !out.status.success() {
      return Err(format!(
        "git add {}: {}",
        rel.display(),
        String::from_utf8_lossy(&out.stderr).trim()
      ));
    }
  }
  let msg = "vtsc: tune sigmoid via vtsc_tuner";
  let out = Command::new("git")
    .current_dir(repo_root)
    .args(["commit", "-m", msg])
    .output()
    .map_err(|e| format!("git commit: {e}"))?;
  if !out.status.success() {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    return Err(format!(
      "git commit failed:\n{}{}",
      stdout.trim(),
      stderr.trim()
    ));
  }
  Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn git_push(repo_root: &Path) -> Result<String, String> {
  let out = Command::new("git")
    .current_dir(repo_root)
    .args(["push"])
    .output()
    .map_err(|e| format!("git push: {e}"))?;
  if !out.status.success() {
    return Err(format!(
      "git push failed:\n{}{}",
      String::from_utf8_lossy(&out.stdout).trim(),
      String::from_utf8_lossy(&out.stderr).trim()
    ));
  }
  let stdout = String::from_utf8_lossy(&out.stdout);
  let stderr = String::from_utf8_lossy(&out.stderr);
  Ok(format!("{}{}", stdout.trim(), stderr.trim()))
}

// ---------------------------------------------------------------------------
// SSID detection + tici profile probing
// ---------------------------------------------------------------------------

/// Try Linux first, then PowerShell on WSL.  Returns the SSID string or None
/// if none of the probes worked.
pub fn detect_ssid() -> Option<String> {
  if let Ok(out) = Command::new("iwgetid").arg("-r").output() {
    if out.status.success() {
      let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
      if !s.is_empty() {
        return Some(s);
      }
    }
  }
  // WSL fallback — ask Windows for the active connection's SSID.
  if let Ok(out) = Command::new("powershell.exe")
    .args([
      "-NoProfile",
      "-Command",
      "(Get-NetConnectionProfile | Where-Object {$_.IPv4Connectivity -eq 'Internet'} | Select-Object -First 1).Name",
    ])
    .output()
  {
    if out.status.success() {
      let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
      if !s.is_empty() {
        return Some(s);
      }
    }
  }
  None
}

/// Order-of-attempt the SSH profiles, biasing on whatever SSID we detected.
/// Profile names match the entries in ~/.ssh/config in this repo's CLAUDE.md.
fn profile_order_for_ssid(ssid: &str) -> [&'static str; 3] {
  let lc = ssid.to_lowercase();
  // The car hotspot is typically named with "comma" in it.
  if lc.contains("comma") {
    ["commaCar", "commaHome", "commaAdb"]
  } else {
    ["commaHome", "commaCar", "commaAdb"]
  }
}

fn probe_profile(name: &str) -> bool {
  let out = Command::new("ssh")
    .args([
      "-o",
      "ConnectTimeout=2",
      "-o",
      "BatchMode=yes",
      "-o",
      "StrictHostKeyChecking=accept-new",
      name,
      "true",
    ])
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .output();
  match out {
    Ok(o) => o.status.success(),
    Err(_) => false,
  }
}

fn pick_tici_profile() -> Option<&'static str> {
  let ssid = detect_ssid().unwrap_or_default();
  for p in profile_order_for_ssid(&ssid) {
    if probe_profile(p) {
      return Some(p);
    }
  }
  None
}

fn ssh_pull_and_reboot(profile: &str) -> Result<String, String> {
  // Two-step so we can attribute the failure to pull vs reboot.
  let out = Command::new("ssh")
    .args([
      "-o",
      "ConnectTimeout=10",
      "-o",
      "StrictHostKeyChecking=accept-new",
      profile,
      "cd /data/openpilot && git pull",
    ])
    .output()
    .map_err(|e| format!("ssh: {e}"))?;
  let stdout = String::from_utf8_lossy(&out.stdout);
  let stderr = String::from_utf8_lossy(&out.stderr);
  if !out.status.success() {
    return Err(format!(
      "git pull on tici failed:\n{}{}",
      stdout.trim(),
      stderr.trim()
    ));
  }
  // Reboot — fire-and-forget, the connection drops mid-command.
  let _ = Command::new("ssh")
    .args([
      "-o",
      "ConnectTimeout=5",
      "-o",
      "StrictHostKeyChecking=accept-new",
      profile,
      "sudo reboot",
    ])
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .output();
  // Tiny wait so the reboot signal is sent before this function returns.
  thread::sleep(Duration::from_millis(200));
  Ok(format!(
    "{}{}\n(reboot signal sent — tici will be back in ~60 s)",
    stdout.trim(),
    stderr.trim()
  ))
}

fn ssh_reboot_only(profile: &str) -> Result<(), String> {
  let _ = Command::new("ssh")
    .args([
      "-o",
      "ConnectTimeout=5",
      "-o",
      "StrictHostKeyChecking=accept-new",
      profile,
      "sudo reboot",
    ])
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .output()
    .map_err(|e| format!("ssh reboot: {e}"))?;
  thread::sleep(Duration::from_millis(200));
  Ok(())
}

// ---------------------------------------------------------------------------
// Tile-rebuild helpers (RebuildTilesAndReboot only)
// ---------------------------------------------------------------------------

/// Poll `ssh <profile> true` every 5 s until either it succeeds or the
/// deadline elapses or cancellation is requested.
fn wait_for_tici(profile: &str, max_wait: Duration, cancel: &AtomicBool) -> bool {
  let deadline = Instant::now() + max_wait;
  // Initial wait — the ssh just-rebooted-it isn't going to answer for ~30 s.
  thread::sleep(Duration::from_secs(15));
  while Instant::now() < deadline {
    if cancel.load(Ordering::Relaxed) {
      return false;
    }
    if probe_profile(profile) {
      return true;
    }
    thread::sleep(Duration::from_secs(5));
  }
  false
}

/// `ls /data/media/0/osm/offline/` → list of (group_lat, group_lon).
/// Tiles are stored at `<lat_group>/<lon_group>/...` (group sizes are 2°×2°
/// per generate_offline.go's GROUP_AREA_BOX_DEGREES).
fn discover_tici_regions(profile: &str) -> Result<Vec<RegionBox>, String> {
  let out = Command::new("ssh")
    .args([
      "-o",
      "ConnectTimeout=10",
      "-o",
      "StrictHostKeyChecking=accept-new",
      profile,
      // Walk two levels deep, print "lat lon" per pair.
      "for d in /data/media/0/osm/offline/*/; do \
         lat=$(basename \"$d\"); \
         for sd in \"$d\"*/; do \
           [ -d \"$sd\" ] && echo \"$lat $(basename \"$sd\")\"; \
         done; \
       done 2>/dev/null",
    ])
    .output()
    .map_err(|e| format!("ssh ls: {e}"))?;
  if !out.status.success() {
    return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
  }
  let mut regions = Vec::new();
  for line in String::from_utf8_lossy(&out.stdout).lines() {
    let mut parts = line.split_whitespace();
    let lat = parts.next().and_then(|s| s.parse::<i32>().ok());
    let lon = parts.next().and_then(|s| s.parse::<i32>().ok());
    if let (Some(lat), Some(lon)) = (lat, lon) {
      regions.push(RegionBox { min_lat: lat, min_lon: lon });
    }
  }
  Ok(regions)
}

fn format_regions_for_detail(regions: &[RegionBox]) -> String {
  let mut sorted: Vec<_> = regions.to_vec();
  sorted.sort_by_key(|r| (r.min_lat, r.min_lon));
  sorted
    .iter()
    .map(|r| format!("({:>4}°, {:>5}°)", r.min_lat, r.min_lon))
    .collect::<Vec<_>>()
    .join("  ")
}

/// `df -BM /data/media/0` → free megabytes. Linux-flavoured df output.
fn tici_free_mb(profile: &str) -> Result<u64, String> {
  let out = Command::new("ssh")
    .args([
      "-o",
      "ConnectTimeout=10",
      "-o",
      "StrictHostKeyChecking=accept-new",
      profile,
      "df -BM /data/media/0 | tail -1",
    ])
    .output()
    .map_err(|e| format!("ssh df: {e}"))?;
  if !out.status.success() {
    return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
  }
  // Format: "/dev/...  100M  20M  80M  20%  /data/media/0" → field 3 (0-idx) is free.
  let line = String::from_utf8_lossy(&out.stdout);
  let trimmed = line.trim();
  let fields: Vec<&str> = trimmed.split_whitespace().collect();
  if fields.len() < 4 {
    return Err(format!("unexpected df output: {trimmed:?}"));
  }
  let raw = fields[3];
  raw
    .trim_end_matches('M')
    .parse::<u64>()
    .map_err(|e| format!("could not parse free MB from {raw:?}: {e}"))
}

fn build_mapd_with_earthly(repo: &Path) -> Result<String, String> {
  let out = Command::new("earthly")
    .arg("+build")
    .current_dir(repo)
    .output()
    .map_err(|e| format!("earthly invocation failed: {e} (is earthly installed and on PATH?)"))?;
  if !out.status.success() {
    return Err(format!(
      "earthly +build exited {}\nstderr:\n{}",
      out.status,
      String::from_utf8_lossy(&out.stderr).trim()
    ));
  }
  Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn generate_one_region(
  cfg: &MapdConfig,
  binary: &Path,
  region: &RegionBox,
) -> Result<String, String> {
  // mapd --generate writes to $CWD/osm/offline/... The Earthfile build expects
  // the PBF to be at $CWD/map.osm.pbf — symlink it in (idempotent).
  let pbf_link = cfg.mapd_repo_path.join("map.osm.pbf");
  let _ = std::fs::remove_file(&pbf_link);
  std::os::unix::fs::symlink(&cfg.pbf_path, &pbf_link)
    .map_err(|e| format!("could not symlink {} → {}: {}", cfg.pbf_path.display(), pbf_link.display(), e))?;

  let max_lat = region.min_lat + 2;
  let max_lon = region.min_lon + 2;
  let out = Command::new(binary)
    .args([
      "--generate",
      &format!("--minlat={}", region.min_lat),
      &format!("--minlon={}", region.min_lon),
      &format!("--maxlat={max_lat}"),
      &format!("--maxlon={max_lon}"),
    ])
    .current_dir(&cfg.mapd_repo_path)
    .output()
    .map_err(|e| format!("mapd --generate spawn failed: {e}"))?;
  if !out.status.success() {
    return Err(format!(
      "mapd --generate exited {}\nstderr:\n{}",
      out.status,
      String::from_utf8_lossy(&out.stderr).trim()
    ));
  }

  // mapd writes to BOUNDS_DIR which is `/data/media/0/osm/offline` on tici
  // BUT `./osm/offline` on the dev box (see generate_offline.go:60-70's
  // GetBaseOpPath which falls back to `.` when /data/media/0 doesn't exist).
  // Verify the per-region directory got created so we can rsync from it.
  let region_dir = cfg.mapd_repo_path.join(format!(
    "osm/offline/{}/{}",
    region.min_lat, region.min_lon
  ));
  if !region_dir.exists() {
    return Err(format!(
      "mapd --generate succeeded but no tiles landed at {} (PBF doesn't cover this bbox?)",
      region_dir.display()
    ));
  }
  let count = std::fs::read_dir(&region_dir)
    .map(|it| it.count())
    .unwrap_or(0);
  Ok(format!("{} tile file(s) under {}", count, region_dir.display()))
}

fn rsync_region_to_tici(
  cfg: &MapdConfig,
  profile: &str,
  region: &RegionBox,
) -> Result<String, String> {
  let local = cfg.mapd_repo_path.join(format!(
    "osm/offline/{}/{}/",
    region.min_lat, region.min_lon
  ));
  let remote = format!(
    "{profile}:/data/media/0/osm/offline/{}/{}/",
    region.min_lat, region.min_lon
  );
  let out = Command::new("rsync")
    .args([
      "-av",
      "--partial",
      "--mkpath",
      // Tici's sshd is plain OpenSSH on the comma profiles; rsync over ssh works.
      "-e",
      "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
    ])
    .arg(&local)
    .arg(&remote)
    .output()
    .map_err(|e| format!("rsync spawn failed: {e}"))?;
  if !out.status.success() {
    return Err(format!(
      "rsync exited {}\nstderr:\n{}",
      out.status,
      String::from_utf8_lossy(&out.stderr).trim()
    ));
  }
  Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

// ---------------------------------------------------------------------------
// Repo root discovery
// ---------------------------------------------------------------------------

/// Walk up from this binary's location looking for a `.git` directory.
pub fn detect_repo_root() -> Option<PathBuf> {
  let exe = std::env::current_exe().ok()?;
  let mut cur: &Path = exe.parent()?;
  loop {
    if cur.join(".git").exists() {
      return Some(cur.to_path_buf());
    }
    cur = cur.parent()?;
  }
}
