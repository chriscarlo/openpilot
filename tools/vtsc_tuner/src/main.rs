// Suppress the secondary console on Windows when launched from a shortcut;
// harmless on Linux/Mac.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod apply;
mod io;
mod knob;
mod mapd_config;
mod params;
mod plot;
mod sigmoid;
mod theme;

use eframe::egui;
use std::io::Write;

fn diag(msg: &str) {
  if let Ok(mut f) = std::fs::OpenOptions::new()
    .create(true)
    .append(true)
    .open("/tmp/vtsc_tuner.diag")
  {
    let _ = writeln!(
      f,
      "[{}] pid={} ppid={} {}",
      chrono::Local::now().format("%H:%M:%S%.3f"),
      std::process::id(),
      std::os::unix::process::parent_id(),
      msg
    );
  }
}

fn main() -> eframe::Result<()> {
  // Always write a backtrace on panic — the app runs detached from a terminal
  // via WSLg / launcher, so stderr is often the only place to see failures.
  if std::env::var_os("RUST_BACKTRACE").is_none() {
    // SAFETY: single-threaded at this point, before any threads are spawned.
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
  }
  diag(&format!(
    "main start  DISPLAY={:?} WAYLAND_DISPLAY={:?} XDG_RUNTIME_DIR={:?} stdin_tty={} stdout_tty={}",
    std::env::var("DISPLAY").ok(),
    std::env::var("WAYLAND_DISPLAY").ok(),
    std::env::var("XDG_RUNTIME_DIR").ok(),
    is_tty(0),
    is_tty(1),
  ));
  let default_hook = std::panic::take_hook();
  std::panic::set_hook(Box::new(move |info| {
    diag(&format!("PANIC {info}"));
    eprintln!("---- vtsc_tuner panic ----");
    default_hook(info);
    eprintln!("--------------------------");
  }));

  let options = eframe::NativeOptions {
    viewport: egui::ViewportBuilder::default()
      .with_title("VTSC Sigmoid Tuner")
      // Logical-points (post-DPI scaling).  egui then renders 2× over that
      // for readable text — so this gives a ~1300×850 logical canvas with
      // 2600×1700-equivalent pixel density.  Min size keeps the side panel
      // from squeezing out the plot.
      .with_inner_size([1300.0, 850.0])
      .with_min_inner_size([900.0, 620.0]),
    // WSLg's wayland event loop crashes during corner-resize (calloop "Broken
    // pipe").  XWayland handles resize cleanly, so force the X11 backend at
    // the winit level rather than relying on env-var hints.
    #[cfg(all(unix, not(target_os = "macos")))]
    event_loop_builder: Some(Box::new(|builder| {
      use winit::platform::x11::EventLoopBuilderExtX11;
      builder.with_x11();
      diag("event_loop_builder: forced X11");
    })),
    ..Default::default()
  };
  diag("about to run_native");
  let result = eframe::run_native(
    "vtsc_tuner",
    options,
    Box::new(|cc| {
      diag("CreationContext callback fired");
      Ok(Box::new(app::TunerApp::new(cc)))
    }),
  );
  diag(&format!("run_native returned: {:?}", result.as_ref().err()));
  result
}

fn is_tty(fd: i32) -> bool {
  // SAFETY: isatty is async-signal-safe and takes any int.
  unsafe { libc::isatty(fd) != 0 }
}
