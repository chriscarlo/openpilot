//! Tune file I/O.  The on-disk schema is a thin JSON blob pairing the raw
//! sigmoid params with the stack of EQ bands so the tune is reproducible.

use std::{
  fs,
  path::{Path, PathBuf},
};

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

use crate::sigmoid::{Band, SigmoidParams};

const CURRENT_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tune {
  pub schema: u32,
  pub created: DateTime<Local>,
  pub note: String,
  pub params: SigmoidParams,
  pub bands: Vec<Band>,
}

impl Default for Tune {
  fn default() -> Self {
    Self {
      schema: CURRENT_SCHEMA,
      created: Local::now(),
      note: String::new(),
      params: SigmoidParams::default(),
      bands: Vec::new(),
    }
  }
}

impl Tune {
  pub fn save_to(&self, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(self)?;
    fs::write(path, text)
  }

  pub fn load_from(path: &Path) -> std::io::Result<Self> {
    let text = fs::read_to_string(path)?;
    let parsed: Tune = serde_json::from_str(&text)?;
    Ok(parsed)
  }
}

/// `~/.config/vtsc_tuner/` (or the platform equivalent).
pub fn default_dir() -> PathBuf {
  dirs::config_dir()
    .unwrap_or_else(|| PathBuf::from("."))
    .join("vtsc_tuner")
}

pub fn default_file() -> PathBuf {
  default_dir().join("current.tune.json")
}

// ---------------------------------------------------------------------------
// SSH push — stub for v1.  The shape mirrors the `commaHome` / `commaCar` /
// `commaAdb` SSH profiles documented in this repo's CLAUDE.md so the UI can
// hint the user at which profile to pick.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceProfile {
  CommaHome,
  CommaCar,
  CommaAdb,
}

impl DeviceProfile {
  pub const ALL: [Self; 3] = [Self::CommaHome, Self::CommaCar, Self::CommaAdb];
  pub fn label(&self) -> &'static str {
    match self {
      Self::CommaHome => "home Wi-Fi",
      Self::CommaCar => "car hotspot",
      Self::CommaAdb => "USB (adb)",
    }
  }
  pub fn ssh_host(&self) -> &'static str {
    match self {
      Self::CommaHome => "commaHome",
      Self::CommaCar => "commaCar",
      Self::CommaAdb => "commaAdb",
    }
  }
}

#[allow(dead_code)] // populated when the actual SSH push is wired up
#[derive(Debug, Clone)]
pub struct PushResult {
  pub ok: bool,
  pub message: String,
}

/// Eventually: write each sigmoid param into `/dev/shm/params/d/<Key>` on
/// the device via `ssh <profile> "tee ..."`. For v1 we return a preview of
/// the shell commands the real implementation would run — that way the UI
/// can show the user exactly what will be sent before we wire up the
/// subprocess call.
pub fn plan_push(profile: DeviceProfile, params: &SigmoidParams) -> Vec<String> {
  let host = profile.ssh_host();
  let writes: &[(&str, f64)] = &[
    ("VisionTurnSpeedControlPhysicsAmplitude", params.a),
    ("VisionTurnSpeedControlPhysicsSteepness", params.b),
    ("VisionTurnSpeedControlPhysicsCenter", params.c),
    ("VisionTurnSpeedControlPhysicsBaseline", params.d),
    ("VisionTurnSpeedControlPhysicsMinLatAccel", params.min_lat),
    ("VisionTurnSpeedControlPhysicsMaxLatAccel", params.max_lat),
  ];
  writes
    .iter()
    .map(|(key, val)| {
      format!(
        "ssh {host} \"printf '%s' '{val}' | sudo tee /data/params/d/{key} >/dev/null\""
      )
    })
    .collect()
}
