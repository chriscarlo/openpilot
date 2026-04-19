//! Per-user configuration for the `RebuildTilesAndReboot` apply action.
//!
//! Lives at `~/.config/vtsc_tuner/mapd.json` (alongside `current.tune.json`).
//! Never auto-created — explicit config is the line between "I know which OSM
//! PBF I'm baking against" and "I just bricked my tici's tiles with the wrong
//! continent". Step 7 of the rebuild action surfaces a plain-English error
//! when the file is missing.
//!
//! Sample config:
//! ```json
//! {
//!   "pbf_path": "/projects/mapd_data/us-northeast-latest.osm.pbf",
//!   "mapd_repo_path": "/projects/chauffeur/data/openpilot/mapd_repo/openpilot-mapd",
//!   "mapd_binary_path": null,
//!   "regions_override": null
//! }
//! ```
//!
//! `regions_override` is an optional `[{min_lat, min_lon}, ...]` list of
//! 2°×2° group boxes. When present, the rebuild bakes ONLY these boxes
//! instead of probing the tici for cached regions. Useful for forcing a
//! rebake of regions you haven't downloaded yet.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapdConfig {
  pub pbf_path: PathBuf,
  pub mapd_repo_path: PathBuf,
  /// When `None`, defaults to `mapd_repo_path/build/mapd` (Earthly +build
  /// output target). The on-device installer drops the binary at
  /// `/data/openpilot/third_party/mapd/mapd` — that path is for the
  /// device, not the dev box.
  #[serde(default)]
  pub mapd_binary_path: Option<PathBuf>,
  /// When `None`, the rebuild step queries the tici for cached regions.
  #[serde(default)]
  pub regions_override: Option<Vec<RegionBox>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegionBox {
  /// Both fields are the *south-west corner* of a 2°×2° group box and must be
  /// even integer multiples of 2 (matching `GROUP_AREA_BOX_DEGREES` in
  /// generate_offline.go). max_lat = min_lat + 2; max_lon = min_lon + 2.
  pub min_lat: i32,
  pub min_lon: i32,
}

impl MapdConfig {
  pub fn default_path() -> PathBuf {
    crate::io::default_dir().join("mapd.json")
  }

  pub fn load() -> Result<Self, String> {
    let path = Self::default_path();
    let text = std::fs::read_to_string(&path).map_err(|e| {
      format!(
        "could not read {} ({}). Create the file with `pbf_path` and `mapd_repo_path` to enable tile rebuilds.",
        path.display(),
        e
      )
    })?;
    serde_json::from_str::<MapdConfig>(&text)
      .map_err(|e| format!("could not parse {}: {}", path.display(), e))
  }

  /// Sanity-check both required paths exist on disk and look sensible.
  /// The binary path is allowed to be missing (we'll build it in step 10).
  pub fn validate(&self) -> Result<(), String> {
    if !self.pbf_path.exists() {
      return Err(format!(
        "pbf_path does not exist: {}",
        self.pbf_path.display()
      ));
    }
    let capnp = self.mapd_repo_path.join("offline.capnp");
    if !capnp.exists() {
      return Err(format!(
        "mapd_repo_path does not look like an openpilot-mapd checkout (no offline.capnp at {}): {}",
        capnp.display(),
        self.mapd_repo_path.display()
      ));
    }
    Ok(())
  }

  /// Effective binary path used by the apply chain. Caller still needs to
  /// check `.exists()` and trigger a build if missing.
  pub fn binary(&self) -> PathBuf {
    self
      .mapd_binary_path
      .clone()
      .unwrap_or_else(|| self.mapd_repo_path.join("build/mapd"))
  }
}

/// Walk up from the openpilot repo root to find a sibling directory that
/// looks like an openpilot-mapd checkout (contains `offline.capnp`). Used
/// only for surfacing a hint in error messages — never silently substituted
/// for an explicit user-provided `mapd_repo_path`.
#[allow(dead_code)] // wired into the friendly error message path
pub fn detect_mapd_repo(openpilot_root: &Path) -> Option<PathBuf> {
  // First check the in-tree vendored copy: this is the canonical source of
  // truth in the chauffeur fork.
  let vendored = openpilot_root.join("mapd_repo/openpilot-mapd");
  if vendored.join("offline.capnp").exists() {
    return Some(vendored);
  }
  // Then walk up looking for a sibling `mapd` dir.
  let mut cur: &Path = openpilot_root;
  loop {
    if cur.join("mapd/offline.capnp").exists() {
      return Some(cur.join("mapd"));
    }
    cur = cur.parent()?;
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn binary_defaults_to_repo_build_dir() {
    let cfg = MapdConfig {
      pbf_path: PathBuf::from("/tmp/nope.pbf"),
      mapd_repo_path: PathBuf::from("/x/mapd"),
      mapd_binary_path: None,
      regions_override: None,
    };
    assert_eq!(cfg.binary(), PathBuf::from("/x/mapd/build/mapd"));
  }

  #[test]
  fn binary_override_takes_precedence() {
    let cfg = MapdConfig {
      pbf_path: PathBuf::from("/tmp/nope.pbf"),
      mapd_repo_path: PathBuf::from("/x/mapd"),
      mapd_binary_path: Some(PathBuf::from("/usr/local/bin/mapd")),
      regions_override: None,
    };
    assert_eq!(cfg.binary(), PathBuf::from("/usr/local/bin/mapd"));
  }

  #[test]
  fn load_missing_yields_friendly_error() {
    // Path won't exist (default ~/.config/vtsc_tuner/mapd.json under HOME=/nonexistent).
    let prev_home = std::env::var_os("HOME");
    // SAFETY: test serial — Cargo runs tests in parallel, so this can race
    // with anything else that reads HOME. Restored at end.
    unsafe { std::env::set_var("HOME", "/nonexistent_home_for_test") };
    let result = MapdConfig::load();
    if let Some(prev) = prev_home {
      unsafe { std::env::set_var("HOME", prev) };
    } else {
      unsafe { std::env::remove_var("HOME") };
    }
    let err = result.expect_err("should error on missing config");
    assert!(err.contains("Create the file"), "actual: {err}");
  }
}
